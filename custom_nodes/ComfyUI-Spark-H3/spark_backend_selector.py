"""按显卡算力在运行时选择 spark-h3-kitchen 的 CUDA 后端。

云端镜像在构建时不知道会分配到哪种显卡。因此这里不在安装时固定 wheel，
而是把三套架构的内核随节点包放在 ``kernels/`` 下，等 ComfyUI 启动或真正
拿到张量时，再按当前设备的 SM 版本把对应目录加入 ``sys.path`` 并导入。

节点包放进 ``custom_nodes/`` 即可，不需要 install.py、解包命令或启动脚本。
同一个镜像可以同时服务 RTX 30 / 40 / 50 系。
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
from pathlib import Path

# 消费级与工作站显卡的计算能力 -> 随包提供的架构目录。
# 8.0 (A100/A800) 没有对应的预编译 wheel，sm86 的 cubin 也不能在 8.0 上运行，
# 因此这里不把它们映射到 sm86。
ARCH_BY_CAPABILITY = {
    (8, 6): "sm86",   # GeForce RTX 30 系
    (8, 9): "sm89",   # GeForce RTX 40 系
    (12, 0): "sm120",  # GeForce RTX 50 系、RTX PRO 5000/6000 Blackwell
}

SUPPORTED_CAPABILITIES = frozenset(ARCH_BY_CAPABILITY)
ARCHITECTURES = frozenset(ARCH_BY_CAPABILITY.values())

# 内核默认随节点包一起放在 kernels/ 下，复制 custom_nodes 即可使用。
# 环境变量优先，其次是包内目录，最后才回退到镜像级共享目录。
PACKAGE_ROOT = Path(__file__).resolve().parent
PACKAGED_KERNEL_ROOT = PACKAGE_ROOT / "kernels"
FALLBACK_KERNEL_ROOT = "/opt/spark-h3/kernels"

_lock = threading.RLock()
_state: dict[str, object] = {
    "arch": None,
    "capability": None,
    "path": None,
    "module": None,
}


def kernel_root() -> Path:
    """返回内核根目录。

    优先级：``SPARK_H3_KERNEL_ROOT`` 环境变量 → 节点包内的 ``kernels/``
    → ``/opt/spark-h3/kernels``。默认情况下随节点包走，无需任何安装步骤。
    """

    override = os.environ.get("SPARK_H3_KERNEL_ROOT")
    if override:
        return Path(override)
    if any((PACKAGED_KERNEL_ROOT / arch).is_dir() for arch in ARCHITECTURES):
        return PACKAGED_KERNEL_ROOT
    return Path(FALLBACK_KERNEL_ROOT)


def _torch():
    import torch

    return torch


def capability_of(device=None) -> tuple[int, int]:
    """返回设备的 (major, minor)；``device`` 可以是 int、str 或 torch.device。"""
    torch = _torch()
    if device is None:
        device = torch.cuda.current_device()
    elif getattr(device, "type", None) != "cuda":
        device = torch.cuda.current_device()
    return tuple(int(part) for part in torch.cuda.get_device_capability(device))


def architecture_for(capability) -> str:
    capability = tuple(int(part) for part in capability)
    arch = ARCH_BY_CAPABILITY.get(capability)
    if arch is None:
        supported = ", ".join(
            f"SM{major}{minor}" for major, minor in sorted(ARCH_BY_CAPABILITY)
        )
        raise RuntimeError(
            "Spark-H3 内核不支持当前显卡："
            f"SM{capability[0]}{capability[1]}；镜像内置的架构是 {supported}"
        )
    return arch


def _backend_dir(arch: str) -> Path:
    return kernel_root() / arch


def _validate_layout(arch: str, path: Path) -> None:
    package = path / "spark_h3_kitchen"
    cuda = package / "backends" / "cuda"
    extension = cuda / ("_C.abi3.pyd" if sys.platform == "win32" else "_C.abi3.so")
    if not cuda.is_dir() or not extension.is_file():
        raise RuntimeError(
            f"Spark-H3 的 {arch} 内核目录不完整：{path}；"
            "期望包含 spark_h3_kitchen/backends/cuda/_C.abi3"
            + (".pyd" if sys.platform == "win32" else ".so")
        )


def _loaded_arch() -> str | None:
    module = sys.modules.get("spark_h3_kitchen")
    if module is None:
        return None
    location = getattr(module, "__file__", "") or ""
    for arch in ARCHITECTURES:
        if f"/{arch}/" in location.replace("\\", "/"):
            return arch
    return None


def ensure_backend(device=None):
    """确保当前进程加载了与 ``device`` 架构匹配的内核，返回包模块。

    一个进程只能绑定一种架构：CUDA 扩展一旦加载就无法卸载。云端一卡一实例
    （``--cuda-device`` / ``CUDA_VISIBLE_DEVICES``）时不会有问题。
    """

    torch = _torch()
    if not torch.cuda.is_available():
        raise RuntimeError("Spark-H3 需要可用的 NVIDIA CUDA 显卡")

    capability = capability_of(device)
    arch = architecture_for(capability)

    with _lock:
        module = _state["module"]
        if module is not None:
            if _state["arch"] != arch:
                raise RuntimeError(
                    "Spark-H3 已在当前进程加载了 "
                    f"{_state['arch']} 内核，无法切换到 {arch}。"
                    "同一 ComfyUI 进程只能使用一种显卡架构，"
                    "异构多卡请为每张卡单独启动一个实例。"
                )
            return module

        existing = sys.modules.get("spark_h3_kitchen")
        if existing is not None:
            loaded = _loaded_arch()
            location = getattr(existing, "__file__", "") or "<unknown>"
            if loaded != arch:
                raise RuntimeError(
                    "Spark-H3 检测到已加载的 spark_h3_kitchen 与当前显卡不匹配："
                    f"已加载 {loaded or location}，当前需要 {arch}。"
                    "请卸载 pip 安装的 spark-h3-kitchen，改用镜像内置的多架构内核目录。"
                )
            _state.update(
                arch=loaded, capability=capability, module=existing,
                path=_backend_dir(loaded),
            )
            return existing

        path = _backend_dir(arch)
        _validate_layout(arch, path)

        # 路径常驻 sys.path：包内存在运行期延迟导入，移除会破坏它们。
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        try:
            module = importlib.import_module("spark_h3_kitchen")
        except OSError as error:
            raise RuntimeError(
                f"Spark-H3 无法加载 {arch} 内核（SM{capability[0]}{capability[1]}）：{error}"
            ) from error

        _state.update(arch=arch, capability=capability, path=path, module=module)
        print(
            f"[Spark-H3] GPU SM{capability[0]}{capability[1]} -> 使用 {arch} 内核：{path}",
            flush=True,
        )
        return module


def loaded_summary() -> dict:
    """返回当前进程已加载内核的信息，便于日志与排障。"""
    return {
        "architecture": _state["arch"],
        "capability": _state["capability"],
        "path": str(_state["path"]) if _state["path"] else None,
    }


def warmup() -> dict:
    """ComfyUI 启动时调用：探测算力并预加载对应内核。

    CUDA 尚不可用（例如离线导入、构建期静态检查）时静默跳过，不阻塞启动。
    """

    torch = _torch()
    if not torch.cuda.is_available():
        print("[Spark-H3] 未检测到 CUDA 设备，跳过内核预加载", flush=True)
        return loaded_summary()

    # 异构多卡时不在启动阶段绑定架构：等首次推理拿到真实张量再选，
    # 避免预加载了错误的卡。同构多卡仍直接预加载。
    try:
        count = torch.cuda.device_count()
        capabilities = {capability_of(index) for index in range(count)}
    except Exception:
        capabilities = set()
    if len(capabilities) > 1:
        supported = ", ".join(
            f"SM{major}{minor}" for major, minor in sorted(capabilities)
        )
        print(
            f"[Spark-H3] 检测到异构多卡（{supported}），延迟到首次推理时选择内核",
            flush=True,
        )
        return loaded_summary()

    try:
        ensure_backend(None)
    except Exception as error:  # 启动阶段不因内核缺失而中断 ComfyUI
        print(f"[Spark-H3] 内核预加载失败：{error}", flush=True)
    return loaded_summary()


__all__ = [
    "ARCH_BY_CAPABILITY",
    "SUPPORTED_CAPABILITIES",
    "architecture_for",
    "capability_of",
    "ensure_backend",
    "kernel_root",
    "loaded_summary",
    "warmup",
]
