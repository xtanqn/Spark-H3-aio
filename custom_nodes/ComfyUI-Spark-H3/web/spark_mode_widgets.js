import { app } from "../../../scripts/app.js";

const SPARK_NODES = new Set([
    "MiniMaxH3SparkAttentionAuto",
    "MiniMaxH3SparkAttentionSM86",
    "MiniMaxH3SparkAttentionSM89",
    "MiniMaxH3SparkAttentionSM120",
    "MiniMaxH3SolAttentionSM120",
]);
const MODE_DEFAULTS = {
    mode: "fidelity",
    warmup_mode: "warmup_ratio",
    warmup_ratio: 0.2,
    warmup_steps: 4,
    topk_mode: "topk_ratio",
    topk_blocks: 228,
    sparse_scope: "target-video",
};

const MODE_PRESETS = {
    speed: { warmup_mode: "warmup_ratio", warmup_ratio: 0.0 },
    fidelity: { warmup_mode: "warmup_ratio", warmup_ratio: 0.2 },
};

function applyModePreset(node) {
    const widgets = new Map((node.widgets ?? []).map((widget) => [widget.name, widget]));
    const preset = MODE_PRESETS[widgets.get("mode")?.value];
    if (!preset) return;
    for (const [name, value] of Object.entries(preset)) {
        const widget = widgets.get(name);
        if (widget && !modeIsLinked(node, name)) widget.value = value;
    }
}

function presetMatches(node) {
    const widgets = new Map((node.widgets ?? []).map((widget) => [widget.name, widget]));
    const mode = widgets.get("mode")?.value;
    const preset = MODE_PRESETS[mode];
    if (!preset) return true;
    if (modeIsLinked(node, "warmup_mode") || modeIsLinked(node, "warmup_ratio")) {
        return false;
    }
    return widgets.get("warmup_mode")?.value === preset.warmup_mode &&
        Math.abs(Number(widgets.get("warmup_ratio")?.value) - preset.warmup_ratio) < 1e-9;
}

function markCustomIfNeeded(node, notify = true) {
    const mode = node.widgets?.find((widget) => widget.name === "mode");
    if (!mode || modeIsLinked(node, "mode") || !MODE_PRESETS[mode.value] || presetMatches(node)) {
        return;
    }
    const previousValue = mode.value;
    mode.value = "custom";
    if (notify) node.onWidgetChanged?.("mode", mode.value, previousValue, mode);
}

function modeIsLinked(node, name) {
    const input = node.inputs?.find((item) => item.name === name);
    return input?.link != null;
}

function setVisible(widget, visible) {
    if (!widget) return;
    widget.hidden = !visible;
    widget.options ??= {};
    widget.options.hidden = !visible;
}

function installTwoStageHint(node) {
    if (typeof node.addWidget !== "function" || node._sparkTwoStageHint) return;
    const hint = node.addWidget(
        "button",
        "双采高分辨率二阶段：建议使用 mode=speed → 一键切换",
        null,
        () => {
            const mode = node.widgets?.find((widget) => widget.name === "mode");
            if (!mode || modeIsLinked(node, "mode")) return;
            const previousValue = mode.value;
            mode.value = "speed";
            applyModePreset(node);
            node.onWidgetChanged?.("mode", mode.value, previousValue, mode);
            syncModeWidgets(node);
        },
        { serialize: false },
    );
    hint.serialize = false;
    hint.options ??= {};
    hint.options.serialize = false;
    node._sparkTwoStageHint = hint;
}

function installDenseBackendHint(node) {
    const backend = node.widgets?.find((widget) => widget.name === "dense_backend");
    if (!backend || typeof node.addWidget !== "function" || node._sparkDenseBackendHint) return;
    const hint = node.addWidget(
        "button",
        "Dense warmup 加速：可使用 dense_backend=kitchen → 一键切换",
        null,
        () => {
            if (modeIsLinked(node, "dense_backend")) return;
            const previousValue = backend.value;
            backend.value = "kitchen";
            backend.callback?.(backend.value);
            node.onWidgetChanged?.("dense_backend", backend.value, previousValue, backend);
            syncModeWidgets(node);
        },
        { serialize: false },
    );
    hint.serialize = false;
    hint.options ??= {};
    hint.options.serialize = false;
    node._sparkDenseBackendHint = hint;
}

function installRef2VAHint(node) {
    const scope = node.widgets?.find((widget) => widget.name === "sparse_scope");
    if (!scope || typeof node.addWidget !== "function" || node._sparkRef2VAHint) return;
    const hint = node.addWidget(
        "button",
        "带视频条件的 Ref2VA：建议使用 sparse_scope=joint-video → 一键切换",
        null,
        () => {
            if (modeIsLinked(node, "sparse_scope")) return;
            const previousValue = scope.value;
            scope.value = "joint-video";
            scope.callback?.(scope.value);
            node.onWidgetChanged?.("sparse_scope", scope.value, previousValue, scope);
            syncModeWidgets(node);
        },
        { serialize: false },
    );
    hint.serialize = false;
    hint.options ??= {};
    hint.options.serialize = false;
    node._sparkRef2VAHint = hint;
}

function syncModeWidgets(node) {
    const widgets = new Map((node.widgets ?? []).map((widget) => [widget.name, widget]));
    const topkMode = widgets.get("topk_mode")?.value ?? "topk_ratio";
    const warmupMode = widgets.get("warmup_mode")?.value ?? "warmup_ratio";

    // A linked mode is decided at execution time, so keep both values editable.
    const topkLinked = modeIsLinked(node, "topk_mode");
    const warmupLinked = modeIsLinked(node, "warmup_mode");
    const modeLinked = modeIsLinked(node, "mode");
    setVisible(widgets.get("topk_ratio"), topkLinked || topkMode === "topk_ratio");
    setVisible(widgets.get("topk_blocks"), topkLinked || topkMode === "topk_blocks");
    setVisible(widgets.get("warmup_ratio"), warmupLinked || warmupMode === "warmup_ratio");
    setVisible(widgets.get("warmup_steps"), warmupLinked || warmupMode === "warmup_steps");
    setVisible(node._sparkTwoStageHint, !modeLinked && widgets.get("mode")?.value === "fidelity");
    setVisible(
        node._sparkDenseBackendHint,
        !modeIsLinked(node, "dense_backend") && widgets.get("dense_backend")?.value === "inherit",
    );
    setVisible(
        node._sparkRef2VAHint,
        !modeIsLinked(node, "sparse_scope") && widgets.get("sparse_scope")?.value === "target-video",
    );

    if (typeof node.computeSize === "function" && typeof node.setSize === "function") {
        const [width, height] = node.computeSize();
        node.setSize([Math.max(node.size[0], width), height]);
    }
    node.setDirtyCanvas?.(true, true);
}

function migrateSparseScope(node) {
    const scope = node.widgets?.find((widget) => widget.name === "sparse_scope");
    if (scope?.value === "target") scope.value = "target-video";
    if (scope?.value === "ref+target") scope.value = "joint-video";
}

app.registerExtension({
    name: "MiniMaxH3.SparkModeWidgets",
    nodeCreated(node) {
        const comfyClass = node.comfyClass ?? node.constructor?.comfyClass;
        if (!SPARK_NODES.has(comfyClass)) return;
        if (node._sparkModeWidgetsInstalled) return;
        node._sparkModeWidgetsInstalled = true;
        installTwoStageHint(node);
        installDenseBackendHint(node);
        installRef2VAHint(node);

        for (const name of ["mode", "topk_mode", "warmup_mode", "warmup_ratio", "warmup_steps", "dense_backend", "sparse_scope"]) {
            const widget = node.widgets?.find((item) => item.name === name);
            if (!widget) continue;
            const previousCallback = widget.callback;
            widget.callback = function (...args) {
                const result = previousCallback?.apply(this, args);
                if (name === "mode") applyModePreset(node);
                markCustomIfNeeded(node);
                syncModeWidgets(node);
                return result;
            };
        }

        const previousConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function (...args) {
            const result = previousConnectionsChange?.apply(this, args);
            markCustomIfNeeded(this);
            syncModeWidgets(this);
            return result;
        };
        const previousConfigure = node.onConfigure;
        node.onConfigure = function (serialized) {
            const result = previousConfigure?.call(this, serialized);
            // Older workflows can restore widget values by position after the
            // schema order changes. Use their saved names to recover the values.
            const saved = serialized?.widgets_values_named;
            if (saved) {
                for (const widget of this.widgets ?? []) {
                    if (widget.name === "mode" &&
                        !Object.prototype.hasOwnProperty.call(saved, "mode") &&
                        saved.enabled === false) {
                        widget.value = "dense";
                    } else if (widget.name === "warmup_ratio" &&
                        !Object.prototype.hasOwnProperty.call(saved, "warmup_ratio") &&
                        Object.prototype.hasOwnProperty.call(saved, "warmup_percent")) {
                        widget.value = Number(saved.warmup_percent) / 100.0;
                    } else if (widget.name === "warmup_mode" && saved.warmup_mode === "warmup_percent") {
                        widget.value = "warmup_ratio";
                    } else if (Object.prototype.hasOwnProperty.call(saved, widget.name)) {
                        widget.value = saved[widget.name];
                    } else if (Object.prototype.hasOwnProperty.call(MODE_DEFAULTS, widget.name)) {
                        widget.value = MODE_DEFAULTS[widget.name];
                    }
                }
            } else {
                // Positional workflows put the old percent value in the same
                // slot now occupied by warmup_ratio.
                const widgets = new Map((this.widgets ?? []).map((widget) => [widget.name, widget]));
                const mode = widgets.get("warmup_mode");
                const ratio = widgets.get("warmup_ratio");
                const legacySchema = serialized?.inputs?.some((input) => input.name === "warmup_percent");
                if (legacySchema && ratio) {
                    ratio.value = Number(ratio.value) / 100.0;
                }
                if (mode?.value === "warmup_percent") mode.value = "warmup_ratio";
            }
            migrateSparseScope(this);
            markCustomIfNeeded(this, false);
            syncModeWidgets(this);
            return result;
        };
        syncModeWidgets(node);
    },
    loadedGraphNode(node) {
        const comfyClass = node.comfyClass ?? node.constructor?.comfyClass;
        if (SPARK_NODES.has(comfyClass)) {
            migrateSparseScope(node);
            markCustomIfNeeded(node, false);
            syncModeWidgets(node);
        }
    },
});
