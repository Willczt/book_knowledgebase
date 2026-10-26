"""调试模式的运行时开关。

配置来自 .env（DEBUG_ENABLED / DEBUG_LAYERS），但允许在调试面板里热改，
不必重启服务：只影响后续请求的采集，不改变检索/生成逻辑。
"""

from __future__ import annotations

from typing import Any, Iterable

from config import get_settings

# 五个可独立开关的观测层
LAYER_NAMES = ("query", "retrieval", "rerank", "generation", "performance")

LAYER_LABELS = {
    "query": "Query 处理",
    "retrieval": "检索召回",
    "rerank": "重排",
    "generation": "生成 / Prompt",
    "performance": "性能耗时",
}


def normalize_layers(layers: str | Iterable[str] | None) -> set[str]:
    """把逗号分隔字符串或列表收敛成合法层集合。"""
    if layers is None:
        return set()
    if isinstance(layers, str):
        raw = layers.split(",")
    else:
        raw = [str(item) for item in layers]
    return {part.strip() for part in raw if part.strip() in LAYER_NAMES}


class DebugRuntime:
    """进程内调试开关。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        settings = get_settings()
        self.enabled: bool = bool(settings.debug_enabled)
        self.layers: set[str] = normalize_layers(settings.debug_layers)

    def on(self, layer: str) -> bool:
        return self.enabled and layer in self.layers

    def configure(
        self,
        *,
        enabled: bool | None = None,
        layers: str | Iterable[str] | None = None,
    ) -> dict[str, Any]:
        if enabled is not None:
            self.enabled = bool(enabled)
        if layers is not None:
            self.layers = normalize_layers(layers)
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        settings = get_settings()
        return {
            "enabled": self.enabled,
            "layers": sorted(self.layers),
            "available_layers": [
                {"name": name, "label": LAYER_LABELS.get(name, name)} for name in LAYER_NAMES
            ],
            "defaults": {
                "enabled": bool(settings.debug_enabled),
                "layers": sorted(normalize_layers(settings.debug_layers)),
            },
            "max_traces": settings.debug_max_traces,
            "persist": settings.debug_persist,
        }


_runtime = DebugRuntime()


def get_runtime() -> DebugRuntime:
    return _runtime


def reset_runtime() -> None:
    """测试用：回到配置文件的默认值"""
    _runtime.reset()
