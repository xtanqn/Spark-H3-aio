# Spark-H3-aio

Spark-H3 的**通用整合包**：同一个节点目录，在 RTX 30 / 40 / 50 系显卡上都能直接运行，
不需要按显卡架构分别安装，也不需要安装脚本或启动脚本。

基于 [zechengtang/Spark-H3](https://github.com/zechengtang/Spark-H3) v0.2.0（Apache-2.0）制作。

## 快速开始

```bash
cp -r custom_nodes/ComfyUI-Spark-H3  <你的 ComfyUI>/custom_nodes/
```

重启 ComfyUI。就这两步。

Dockerfile：

```dockerfile
COPY custom_nodes/ComfyUI-Spark-H3 /opt/ComfyUI/custom_nodes/ComfyUI-Spark-H3
```

不需要 `install.py`，不需要解包命令，不需要 entrypoint 或开机启动项。
`custom_nodes` 只读也可以运行。

## 为什么通用

目录内已经放好三套 CUDA 内核（`.so` 已解包）：

```
custom_nodes/ComfyUI-Spark-H3/
├── spark_backend_selector.py     ← 运行时按算力选内核
├── kernels/
│   ├── sm86/...    RTX 30 系
│   ├── sm89/...    RTX 40 系
│   └── sm120/...   RTX 50 系、RTX PRO 5000/6000 Blackwell
├── comfyui_nodes.py
├── comfyui_backend.py
├── selfcheck.py
├── workflows/                    ← 5 个示例，已使用 Auto 节点
└── web/
```

ComfyUI 启动加载节点包时立即探测：

```
torch.cuda.get_device_capability()
  → (8,6) sm86 / (8,9) sm89 / (12,0) sm120
  → 把对应目录加入 sys.path 并 import
  → [Spark-H3] GPU SM89 -> 使用 sm89 内核：.../kernels/sm89
```

首次推理时还会用张量所在设备复核一次，`--cuda-device 1` 也能选对。

## 节点

新增自动节点：

```
MiniMax H3 Spark Attention (Auto · RTX 30/40/50)
```

接线与官方一致：

```
UNETLoader ──MODEL──> Spark Attention (Auto) ──MODEL──> BasicGuider
                                                 └────> BasicScheduler
```

原有的 `(SM86)` / `(SM89)` / `(SM120)` 节点保留，用于锁定架构排查。
通用部署请使用 Auto 节点；自带示例工作流已全部改为 Auto。

旧工作流迁移：

```bash
sed -i 's/MiniMaxH3SparkAttentionSM120/MiniMaxH3SparkAttentionAuto/g' 你的工作流.json
```

## 内核版本

- `spark_h3_kitchen` **0.2.37**（作者推荐的默认实现，发布测试中速度和稳定性优于 0.2.39）
- 与 ComfyUI 官方 `comfy-kitchen` 完全解耦：使用独立 `spark_h3_kitchen` 命名空间，
  不替换、不降级、不锁定官方包，也不校验其版本
- 内核来源：官方 `ComfyUI-Spark-H3-0.2.0-*-cu130.zip`，未修改任何 CUDA 内核

## 前提条件

- ComfyUI 0.38.x–0.40.x 或 0.42.x–0.44.x，且带原生 MiniMax-H3 支持
- Python 3.12+
- PyTorch 为 CUDA 13.0 构建（`torch.version.cuda >= 13.0`）
- NVIDIA SM86 / SM89 / SM120 显卡

## 验证

启动日志：

```
[Spark-H3] GPU SM89 -> 使用 sm89 内核：/.../custom_nodes/ComfyUI-Spark-H3/kernels/sm89
```

容器内自检：

```bash
python /path/to/custom_nodes/ComfyUI-Spark-H3/selfcheck.py
```

## 限制

1. 一个 ComfyUI 进程只能绑定一种架构（CUDA 扩展加载后无法卸载）。
   一卡一实例没问题；同一进程混用 40 系和 50 系需要拆成多个实例。
2. 不支持 A100 / A800（SM80），官方没有对应预编译内核。
3. 内核要求 CUDA 13.0，这是官方 wheel 的构建条件。

## 排错

| 现象 | 原因 | 处理 |
|---|---|---|
| `内核不支持当前显卡：SM80` | A100/A800 | 不支持 |
| `内核目录不完整` | `kernels/` 缺失 | 重新复制节点目录 |
| `检测到已加载的 spark_h3_kitchen 与当前显卡不匹配` | 残留旧 `install.py` 安装的包 | `pip uninstall -y spark-h3-kitchen` |
| 节点列表看不到 Auto | 目录位置不对 | 确认 `custom_nodes/ComfyUI-Spark-H3/__init__.py` 存在 |

## 可选环境变量

| 变量 | 作用 |
|---|---|
| `SPARK_H3_KERNEL_ROOT` | 改用外部内核目录，优先于包内 `kernels/` |
| `H3_SPARK_PRODUCER_CHUNK` | Q/K/V 分块大小，默认 16384 |

## 许可

- 本项目基于 [Spark-H3](https://github.com/zechengtang/Spark-H3)，Apache-2.0
- 内置的 `spark_h3_kitchen` 二进制来自 ComfyUI `comfy-kitchen`，Apache-2.0，
  许可证见 `custom_nodes/ComfyUI-Spark-H3/kernels/LICENSE.spark_h3_kitchen`
  与 `NOTICE.spark_h3_kitchen`

本仓库只做了运行时架构选择与节点入口适配，未修改任何 CUDA 内核。
