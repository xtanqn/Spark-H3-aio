# Spark-H3 通用节点（RTX 30 / 40 / 50 系共用，Linux + Windows）

把一个目录放进 `custom_nodes/`，ComfyUI 启动后即可使用。
**不需要 install.py、不需要解包命令、不需要 entrypoint 或开机脚本。**

同一个目录同时包含 Linux 的 `.so` 和 Windows 的 `.pyd`，按操作系统自动加载对应的那一个。

## 使用

```bash
cp -r ComfyUI-Spark-H3  <你的 ComfyUI>/custom_nodes/
```

重启 ComfyUI。就这两步。

Dockerfile 里就是一行：

```dockerfile
COPY custom_nodes/ComfyUI-Spark-H3 /opt/ComfyUI/custom_nodes/ComfyUI-Spark-H3
```

## 为什么 30 / 40 / 50 系都能用

目录里已经放好了三套架构的 CUDA 内核（`.so` 是提前解包好的）：

```
ComfyUI-Spark-H3/
├── __init__.py
├── spark_backend_selector.py     ← 运行时按算力选内核
├── comfyui_nodes.py
├── comfyui_backend.py
├── selfcheck.py
├── kernels/                      ← 三套内核，约 73 MB
│   ├── sm86/spark_h3_kitchen/...    RTX 30 系
│   ├── sm89/spark_h3_kitchen/...    RTX 40 系
│   └── sm120/spark_h3_kitchen/...   RTX 50 系、RTX PRO 5000/6000
├── workflows/                    ← 5 个示例，已全部使用 Auto 节点
└── web/
```

每个架构目录下同时有两种扩展，ComfyUI 按操作系统自动选择：

```
backends/cuda/
├── _C.abi3.so    Linux
└── _C.abi3.pyd   Windows
```

两者的 Python 层完全一致，只有扩展文件的二进制不同。

ComfyUI 启动加载这个节点包时，`__init__.py` 立即执行探测：

```
torch.cuda.get_device_capability()
  → (8,6) 选 sm86 / (8,9) 选 sm89 / (12,0) 选 sm120
  → 把对应目录加入 sys.path 并 import
  → 日志：[Spark-H3] GPU SM89 -> 使用 sm89 内核：.../kernels/sm89
```

首次真正推理时还会用**张量所在设备**复核一次，`--cuda-device 1` 也能选对。

整个过程只读文件，不解包、不写磁盘，所以 `custom_nodes` 只读也能跑。

## 节点

新增：

```
MiniMax H3 Spark Attention (Auto · RTX 30/40/50)
```

接线和以前一样：

```
UNETLoader ──MODEL──> Spark Attention (Auto) ──MODEL──> BasicGuider
                                                 └────> BasicScheduler
```

原有的 `(SM86)` / `(SM89)` / `(SM120)` 节点保留，用于锁定架构排查。
通用镜像里请用 Auto 节点。

自带 5 个示例工作流已全部改成 Auto，拖进画布即可。

旧工作流迁移：

```bash
sed -i 's/MiniMaxH3SparkAttentionSM120/MiniMaxH3SparkAttentionAuto/g' 你的工作流.json
```

## 验证

启动日志：

```
[Spark-H3] GPU SM89 -> 使用 sm89 内核：/.../custom_nodes/ComfyUI-Spark-H3/kernels/sm89
```

容器内自检：

```bash
python /path/to/custom_nodes/ComfyUI-Spark-H3/selfcheck.py
```

输出会列出每张卡的型号、算力、对应内核、三套 `.so` 是否齐全，以及 INT8 支持情况。

## 前提条件

- ComfyUI 0.38.x–0.40.x 或 0.42.x–0.44.x，且带原生 MiniMax-H3 支持
- Python 3.12+（Linux 与 Windows x86_64 均支持）
- PyTorch 为 CUDA 13.0 构建（`torch.version.cuda >= 13.0`）
- NVIDIA 显卡：SM86 / SM89 / SM120

Windows 下用便携版时，`custom_nodes` 位于：

```
ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Spark-H3
```

## 限制

1. **一个 ComfyUI 进程只能绑定一种架构**。CUDA 扩展加载后无法卸载。
   一卡一实例（`CUDA_VISIBLE_DEVICES`）没问题；同一进程里 40 系和 50 系混用
   需要拆成多个实例。启动时检测到异构多卡会打印警告并延迟到首次推理再选。
2. 不支持 A100 / A800（SM80）。官方没有 SM80 的预编译内核。
3. 内核要求 CUDA 13.0，这是官方 wheel 的硬条件，与架构选择无关。

## 排错

| 现象 | 原因 | 处理 |
|---|---|---|
| 日志出现 `内核不支持当前显卡：SM80` | A100/A800 | 不支持 |
| `内核目录不完整` | `kernels/` 缺失或被误删 | 重新复制节点目录 |
| `检测到已加载的 spark_h3_kitchen 与当前显卡不匹配` | 镜像里残留官方 `install.py` 装的包 | `pip uninstall -y spark-h3-kitchen` |
| 节点列表里看不到 Auto | 节点目录没放对位置 | 确认路径是 `custom_nodes/ComfyUI-Spark-H3/__init__.py` |
| 启动变慢 1–3 秒 | 预加载 13 MB CUDA 扩展 | 正常现象 |

## 环境变量（可选）

| 变量 | 作用 |
|---|---|
| `SPARK_H3_KERNEL_ROOT` | 改用外部内核目录，优先于包内 `kernels/` |
| `H3_SPARK_PRODUCER_CHUNK` | Q/K/V 分块大小，默认 16384 |

## 来源与许可

节点与内核来自 <https://github.com/zechengtang/Spark-H3>（Apache-2.0）。
本目录只做了运行时架构选择与节点入口适配，未修改任何 CUDA 内核。
