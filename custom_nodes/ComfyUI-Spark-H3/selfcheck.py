#!/usr/bin/env python3
"""容器内自检：确认显卡、算力和选中的内核一致。

    python /path/to/custom_nodes/ComfyUI-Spark-H3/selfcheck.py

需要真实 GPU 环境。它会真正加载对应架构的 CUDA 扩展。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_NODE_DIR = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-dir", default=DEFAULT_NODE_DIR, type=Path)
    args = parser.parse_args(argv)

    node_dir = args.node_dir.resolve()
    if not node_dir.is_dir():
        raise SystemExit(f"节点目录不存在：{node_dir}")
    sys.path.insert(0, str(node_dir))

    import torch

    print(f"torch          : {torch.__version__}")
    print(f"torch CUDA     : {torch.version.cuda}")
    print(f"CUDA 可用      : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("没有可用显卡，无法继续")
        return 1

    from spark_backend_selector import (
        architecture_for,
        capability_of,
        ensure_backend,
        kernel_root,
        loaded_summary,
    )

    print(f"内核根目录     : {kernel_root()}")
    for index in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(index)
        capability = capability_of(index)
        try:
            arch = architecture_for(capability)
        except RuntimeError as error:
            arch = f"不支持（{error}）"
        print(f"GPU {index}        : {name}  SM{capability[0]}{capability[1]} -> {arch}")

    for arch in ("sm86", "sm89", "sm120"):
        extension = kernel_root() / arch / "spark_h3_kitchen" / "backends" / "cuda" / "_C.abi3.so"
        state = "OK" if extension.is_file() else "缺失"
        size = f"{extension.stat().st_size / 1048576:.1f} MB" if extension.is_file() else "-"
        print(f"  {arch:<6}      : {state} {size}")

    kitchen = ensure_backend(None)
    print(f"已加载内核     : {loaded_summary()}")
    print(f"INT8 attention : {kitchen.int8_attention_is_available()}")
    print("自检完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
