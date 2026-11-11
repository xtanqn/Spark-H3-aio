"""ComfyUI entry point for the Spark-H3 custom node.

通用版：一个镜像同时携带 sm86 / sm89 / sm120 三套内核，ComfyUI 启动时按
当前显卡算力选择，不需要为不同显卡更换安装包或节点。
"""

if __package__:
    from .comfyui_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:  # pytest/importlib may load a custom-node root as ``__init__``.
    from comfyui_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


def _probe_kernel():
    """启动时探测算力并预加载对应内核，失败不阻塞 ComfyUI 启动。"""

    try:
        if __package__:
            from .spark_backend_selector import warmup
        else:
            from spark_backend_selector import warmup
    except ImportError as error:
        print(f"[Spark-H3] 内核选择器不可用：{error}", flush=True)
        return
    warmup()


_probe_kernel()
