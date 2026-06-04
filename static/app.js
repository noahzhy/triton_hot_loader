const configInput = document.getElementById("config-input");
const pruneMissing = document.getElementById("prune-missing");
const forceReload = document.getElementById("force-reload");
const resultPanel = document.getElementById("result-panel");
const statusSummary = document.getElementById("status-summary");
const managedImageGrid = document.getElementById("managed-image-grid");
const modelGroupsPanel = document.getElementById("model-groups-panel");
const tritonModelBody = document.getElementById("triton-model-body");
const selectAllModels = document.getElementById("select-all-models");
const modelFilterInput = document.getElementById("model-filter-input");
const clearModelFilterBtn = document.getElementById("clear-model-filter-btn");
const modelSelectionSummary = document.getElementById("model-selection-summary");
const themeLightBtn = document.getElementById("theme-light-btn");
const themeDarkBtn = document.getElementById("theme-dark-btn");
const gpuMonitorPanel = document.getElementById("gpu-monitor-panel");
const gpuMetricsUpdated = document.getElementById("gpu-metrics-updated");
const tritonUrlInput = document.getElementById("triton-url-input");
const metricsPortInput = document.getElementById("metrics-port-input");
const saveTritonUrlBtn = document.getElementById("save-triton-url-btn");
const resetTritonUrlBtn = document.getElementById("reset-triton-url-btn");
const tritonUrlStatus = document.getElementById("triton-url-status");

const operationBadge = document.getElementById("operation-badge");
const operationTitle = document.getElementById("operation-title");
const operationElapsed = document.getElementById("operation-elapsed");
const operationProgressFill = document.getElementById("operation-progress-fill");
const operationProgressText = document.getElementById("operation-progress-text");
const operationStageText = document.getElementById("operation-stage-text");
const operationInlineTip = document.getElementById("operation-inline-tip");
const operationTipList = document.getElementById("operation-tip-list");
const operationLogList = document.getElementById("operation-log-list");

const THEME_STORAGE_KEY = "hot_triton_theme";
const TRITON_URL_STORAGE_KEY = "hot_triton_triton_url";
const METRICS_PORT_STORAGE_KEY = "hot_triton_metrics_port";
const STATUS_POLL_INTERVAL_MS = 10000;
const GPU_METRICS_POLL_INTERVAL_MS = 1000;

const BUTTON_IDS = [
    "apply-config-btn",
    "refresh-all-btn",
    "unload-models-btn",
    "reload-models-btn",
    "load-sample-btn",
    "format-json-btn",
    "save-triton-url-btn",
    "reset-triton-url-btn",
    "clear-result-btn",
];

const BUTTON_BUSY_TEXT = {
    "apply-config-btn": "热加载中...",
    "refresh-all-btn": "刷新中...",
    "unload-models-btn": "卸载中...",
    "reload-models-btn": "重载中...",
    "load-sample-btn": "加载中...",
    "format-json-btn": "格式化中...",
};

const ACTION_PROFILES = {
    idle: {
        label: "等待新的操作",
        tips: [
            "热加载会依次执行配置校验、提交请求、等待镜像处理和刷新状态。",
            "若页面长时间停在处理中，优先检查 Docker 拉镜像速度和 Triton 日志。",
            "出现 `polling enabled` 或 `non-explicit` 报错时，请检查 Triton 启动参数。",
        ],
    },
    init: {
        label: "初始化页面",
        maxProgress: 86,
        tick: 8,
        stages: [
            {
                progress: 8,
                text: "正在连接 hot_triton API",
                tip: "如果这里失败，请先确认 8090 服务已经启动。",
            },
            {
                progress: 34,
                text: "正在拉取 Triton 状态与模型列表",
                tip: "初始化会同时刷新状态卡、已管理镜像和模型版本视图。",
            },
            {
                progress: 68,
                text: "正在加载示例 JSON 配置",
                tip: "示例配置会自动填入左侧文本框，方便直接试跑。",
            },
        ],
        tips: [
            "页面初始化时会自动检查 Triton 连通性。",
            "如果初始化失败，请优先确认 `http://127.0.0.1:8090/api/status` 是否可达。",
            "首次打开页面时自动载入示例 JSON，且会自动忽略 mlman_config 项。",
        ],
        successTip: "初始化完成，可以直接执行热加载或热卸载。",
        failureTip: "请检查 UI 服务是否已启动，以及 API 地址是否正确。",
    },
    refresh: {
        label: "刷新状态",
        maxProgress: 88,
        tick: 10,
        stages: [
            {
                progress: 14,
                text: "正在请求状态数据",
                tip: "这里会同时刷新 Triton Ready 状态和当前模型列表。",
            },
            {
                progress: 64,
                text: "正在更新页面内容",
                tip: "刷新完成后可以继续执行热加载、热卸载或重载操作。",
            },
        ],
        tips: [
            "刷新只会读取状态，不会改动任何模型。",
            "如果 Triton Ready 不是 `READY`，请先检查 Triton 容器日志。",
        ],
        successTip: "状态已刷新，下面的镜像摘要和模型版本视图已经是最新数据。",
        failureTip: "刷新失败时，通常是 UI 服务或 Triton 服务不可达。",
    },
    apply: {
        label: "执行热加载",
        maxProgress: 94,
        tick: 4,
        stages: [
            {
                progress: 8,
                text: "正在校验 JSON 配置",
                tip: "会先检查 JSON 格式；系统会忽略 key，只读取镜像 value。",
            },
            {
                progress: 18,
                text: "正在向 hot loader 提交热加载请求",
                tip: "服务端将依次拉取镜像、提取 `/trt_models` 并同步 model repository。",
            },
            {
                progress: 46,
                text: "等待镜像拉取与模型提取完成",
                tip: "首次拉取大镜像耗时会明显更长，这个阶段进度条会缓慢前进。",
            },
            {
                progress: 76,
                text: "等待 Triton 执行版本 load / reload",
                tip: "此阶段会通过 Triton repository API 激活目标版本，请保持页面打开。",
            },
            {
                progress: 92,
                text: "正在刷新镜像与模型列表",
                tip: "完成后页面会自动更新已管理镜像、模型版本视图和 Triton 当前模型。",
            },
        ],
        tips: [
            "勾选“自动卸载”后，本次 JSON 中缺失的镜像会被自动下线。",
            "如果镜像地址未变化但你仍想重载，可勾选“强制重载”。",
            "所有 `mlman_config / mlmanconfig` 相关项会被自动忽略。",
            "若返回 Docker 或镜像仓库错误，请检查本机是否已执行 `docker login`。",
        ],
        successTip: "热加载完成后，可以重点检查已管理镜像与模型版本视图是否符合预期。",
        failureTip: "热加载失败时，请查看结果面板中的详细错误，常见原因包括镜像地址、Docker 权限或 Triton 控制模式。",
    },
    unloadModels: {
        label: "按模型名/版本热卸载",
        maxProgress: 92,
        tick: 7,
        stages: [
            {
                progress: 12,
                text: "正在校验选中的模型或版本",
                tip: "系统会确认你至少勾选了一个 Triton model name 或 model@version。",
            },
            {
                progress: 44,
                text: "正在提交模型/版本卸载请求",
                tip: "若选中了具体版本，服务端会改写 repository 并触发 Triton reload。",
            },
            {
                progress: 88,
                text: "正在刷新模型列表",
                tip: "完成后你可以检查目标版本是否消失，或模型是否切到新的激活版本。",
            },
        ],
        tips: [
            "按模型名卸载适合整体摘除一个模型。",
            "按带版本的行卸载时，会只下线该版本，并尽量保留其它版本继续服务。",
            "卸载后，来源镜像摘要和模型分组视图都会一起刷新。",
        ],
        successTip: "模型/版本卸载完成，建议再刷新一次确认 Triton 当前状态。",
        failureTip: "若卸载失败，请检查模型名、版本号以及 Triton 是否处于 EXPLICIT 模式。",
    },
    reloadModels: {
        label: "按模型名重载",
        maxProgress: 92,
        tick: 7,
        stages: [
            {
                progress: 12,
                text: "正在校验选中的模型",
                tip: "系统会确认你至少勾选了一个要重载的模型。",
            },
            {
                progress: 46,
                text: "正在调用 Triton load 触发版本重载",
                tip: "重载会直接触发 Triton reload，并应用共享目录中的当前目标版本。",
            },
            {
                progress: 88,
                text: "正在刷新模型列表",
                tip: "完成后可以检查模型状态是否恢复为 READY。",
            },
        ],
        tips: [
            "重载适合目录内容已更新、但模型名未变化的场景。",
            "如果重载失败，优先检查共享目录内模型结构是否完整。",
        ],
        successTip: "模型重载完成后，可继续通过业务接口验证效果。",
        failureTip: "请检查共享目录、模型配置以及 Triton 日志中的具体报错。",
    },
    sample: {
        label: "填充示例 JSON",
        maxProgress: 82,
        tick: 14,
        stages: [
            {
                progress: 20,
                text: "正在请求示例配置",
                tip: "会从后端读取 `sample_config.json` 并写入左侧输入框。",
            },
            {
                progress: 72,
                text: "正在填充示例 JSON",
                tip: "填充完成后可以直接点“执行热加载”进行测试。",
            },
        ],
        tips: [
            "示例 JSON 适合作为第一轮联调或演示配置。",
            "正式使用前，请记得替换为你自己的镜像地址。",
        ],
        successTip: "示例配置已填充，可以继续格式化或直接执行热加载。",
        failureTip: "若示例加载失败，请检查 `/api/sample-config` 是否可用。",
    },
    format: {
        label: "格式化 JSON",
        maxProgress: 84,
        tick: 16,
        stages: [
            {
                progress: 18,
                text: "正在解析输入内容",
                tip: "这里会验证左侧文本框里的内容是否为合法 JSON。",
            },
            {
                progress: 76,
                text: "正在回写格式化结果",
                tip: "格式化只会调整排版，不会修改配置语义。",
            },
        ],
        tips: [
            "格式化前请先确认文本框内容是合法 JSON。",
            "如果格式化失败，通常是少了逗号、引号或大括号。",
        ],
        successTip: "JSON 已格式化，现在更适合继续检查或提交。",
        failureTip: "请修正 JSON 语法后再试。",
    },
};

const buttonTextCache = new Map(
    BUTTON_IDS
        .map((id) => {
            const button = document.getElementById(id);
            return button ? [id, button.textContent] : null;
        })
        .filter(Boolean),
);

const state = {
    managed: null,
    tritonModels: [],
    logs: [],
    modelFilter: "",
    selectedModelRefs: new Set(),
    modelSelectionIndex: new Map(),
    modelGroupStats: {
        totalGroups: 0,
        visibleGroups: 0,
    },
    runtime: {
        reportedTritonUrl: "",
        reportedMetricsUrl: "",
    },
    operation: {
        key: "idle",
        status: "idle",
        progress: 0,
        title: "等待新的操作",
        stageText: "点击任意操作按钮后，这里会显示执行阶段。",
        tipText: "提示：首次拉取大镜像时会比较慢，保持页面打开即可。",
        startedAt: null,
        stageIndex: -1,
        progressTimerId: null,
        elapsedTimerId: null,
        activeButtonId: null,
    },
};

let statusPollingTimerId = null;
let gpuMetricsPollingTimerId = null;
let statusPollingInFlight = false;
let gpuMetricsPollingInFlight = false;

function getPreferredTheme() {
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (stored === "light" || stored === "dark") {
        return stored;
    }
    return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    themeLightBtn?.classList.toggle("active", theme === "light");
    themeDarkBtn?.classList.toggle("active", theme === "dark");
}

function normalizeTritonUrl(value) {
    return String(value || "").trim().replace(/\/+$/, "");
}

function withDefaultTritonScheme(value) {
    const normalized = normalizeTritonUrl(value);
    if (!normalized) {
        return "";
    }
    if (/^[a-zA-Z][a-zA-Z\d+.-]*:\/\//.test(normalized)) {
        return normalized;
    }
    return `http://${normalized}`;
}

function parseTritonUrl(value, { strict = false } = {}) {
    const normalized = normalizeTritonUrl(value);
    if (!normalized) {
        if (strict) {
            throw new Error("Triton URL 不能为空");
        }
        return "";
    }

    const candidate = withDefaultTritonScheme(normalized);

    try {
        const parsed = new URL(candidate);
        if (!["http:", "https:"].includes(parsed.protocol)) {
            throw new Error("unsupported protocol");
        }
        return parsed.toString().replace(/\/+$/, "");
    } catch (error) {
        if (strict) {
            throw new Error("Triton URL 格式不正确；可直接输入 127.0.0.1:8000，系统会自动补成 http://127.0.0.1:8000");
        }
        return "";
    }
}

function formatTritonUrlForDisplay(value) {
    const normalized = parseTritonUrl(value);
    if (!normalized) {
        return normalizeTritonUrl(value);
    }

    try {
        const parsed = new URL(normalized);
        const hasExtraParts = Boolean(
            parsed.username
            || parsed.password
            || parsed.search
            || parsed.hash
            || (parsed.pathname && parsed.pathname !== "/"),
        );

        if (parsed.protocol === "http:" && !hasExtraParts) {
            return parsed.host;
        }

        return normalized;
    } catch (error) {
        return normalizeTritonUrl(value);
    }
}

function parseMetricsPort(value, { strict = false, allowEmpty = true } = {}) {
    const normalized = String(value || "").trim();
    if (!normalized) {
        if (allowEmpty) {
            return "";
        }
        if (strict) {
            throw new Error("Metrics 端口不能为空");
        }
        return "";
    }

    if (!/^\d+$/.test(normalized)) {
        if (strict) {
            throw new Error("Metrics 端口必须是 1-65535 的整数");
        }
        return "";
    }

    const port = Number(normalized);
    if (!Number.isInteger(port) || port < 1 || port > 65535) {
        if (strict) {
            throw new Error("Metrics 端口必须在 1-65535 之间");
        }
        return "";
    }

    return String(port);
}

function extractPortFromUrl(value) {
    const normalized = parseTritonUrl(value);
    if (!normalized) {
        return "";
    }

    try {
        return new URL(normalized).port || "";
    } catch (error) {
        return "";
    }
}

function getSavedTritonUrlOverride() {
    let rawValue = "";
    try {
        rawValue = window.localStorage.getItem(TRITON_URL_STORAGE_KEY) || "";
    } catch (error) {
        return "";
    }

    const normalized = parseTritonUrl(rawValue);
    if (rawValue && !normalized) {
        try {
            window.localStorage.removeItem(TRITON_URL_STORAGE_KEY);
        } catch (error) {
            console.debug("failed to clear invalid triton url override", error);
        }
    }
    return normalized;
}

function saveTritonUrlOverride(value) {
    const normalized = parseTritonUrl(value, { strict: true });
    try {
        window.localStorage.setItem(TRITON_URL_STORAGE_KEY, normalized);
    } catch (error) {
        throw new Error("浏览器当前不允许写入本地存储，无法记住 Triton URL");
    }
    return normalized;
}

function getSavedMetricsPortOverride() {
    let rawValue = "";
    try {
        rawValue = window.localStorage.getItem(METRICS_PORT_STORAGE_KEY) || "";
    } catch (error) {
        return "";
    }

    const normalized = parseMetricsPort(rawValue, { allowEmpty: true });
    if (rawValue && !normalized) {
        try {
            window.localStorage.removeItem(METRICS_PORT_STORAGE_KEY);
        } catch (error) {
            console.debug("failed to clear invalid metrics port override", error);
        }
    }
    return normalized;
}

function saveMetricsPortOverride(value) {
    const normalized = parseMetricsPort(value, { strict: true, allowEmpty: true });
    try {
        if (normalized) {
            window.localStorage.setItem(METRICS_PORT_STORAGE_KEY, normalized);
        } else {
            window.localStorage.removeItem(METRICS_PORT_STORAGE_KEY);
        }
    } catch (error) {
        throw new Error("浏览器当前不允许写入本地存储，无法记住 Metrics 端口");
    }
    return normalized;
}

function clearTritonUrlOverride() {
    try {
        window.localStorage.removeItem(TRITON_URL_STORAGE_KEY);
    } catch (error) {
        console.debug("failed to clear triton url override", error);
    }
}

function clearMetricsPortOverride() {
    try {
        window.localStorage.removeItem(METRICS_PORT_STORAGE_KEY);
    } catch (error) {
        console.debug("failed to clear metrics port override", error);
    }
}

function updateTritonUrlEditor(
    reportedUrl = state.runtime.reportedTritonUrl,
    reportedMetricsUrl = state.runtime.reportedMetricsUrl,
) {
    if (!tritonUrlInput || !tritonUrlStatus) {
        return;
    }

    const savedOverride = getSavedTritonUrlOverride();
    const savedMetricsPort = getSavedMetricsPortOverride();
    const normalizedReportedUrl = parseTritonUrl(reportedUrl) || normalizeTritonUrl(reportedUrl);
    const displayReportedUrl = formatTritonUrlForDisplay(normalizedReportedUrl);
    const displaySavedOverride = formatTritonUrlForDisplay(savedOverride);
    const currentValue = normalizeTritonUrl(tritonUrlInput.value);
    const reportedMetricsPort = extractPortFromUrl(reportedMetricsUrl);
    const currentMetricsPort = metricsPortInput
        ? parseMetricsPort(metricsPortInput.value, { allowEmpty: true })
        : "";

    if (!currentValue) {
        tritonUrlInput.value = displaySavedOverride || displayReportedUrl;
    }
    if (metricsPortInput && !currentMetricsPort) {
        metricsPortInput.value = savedMetricsPort || reportedMetricsPort;
    }

    const effectiveValue = parseTritonUrl(tritonUrlInput.value) || normalizeTritonUrl(tritonUrlInput.value);
    const effectiveMetricsPort = metricsPortInput
        ? parseMetricsPort(metricsPortInput.value, { allowEmpty: true })
        : "";

    if (savedOverride || savedMetricsPort) {
        const urlDirty = savedOverride
            ? effectiveValue && effectiveValue !== savedOverride
            : effectiveValue && normalizedReportedUrl && effectiveValue !== normalizedReportedUrl;
        const metricsDirty = effectiveMetricsPort !== savedMetricsPort;
        tritonUrlStatus.textContent = urlDirty || metricsDirty
            ? "已修改，点击保存覆盖当前记住的 endpoint"
            : "已保存到本地，刷新后仍会记住";
    } else {
        const urlDirty = effectiveValue && normalizedReportedUrl && effectiveValue !== normalizedReportedUrl;
        const metricsDirty = effectiveMetricsPort !== reportedMetricsPort;
        tritonUrlStatus.textContent = urlDirty || metricsDirty
            ? "已修改，点击保存后立即生效"
            : "默认读取环境变量 / .env；Metrics 端口留空时自动尝试 Triton URL + 2";
    }

    if (resetTritonUrlBtn) {
        resetTritonUrlBtn.disabled = state.operation.status === "running" || (!savedOverride && !savedMetricsPort);
    }
}

function buildRequestHeaders(extraHeaders = {}) {
    const headers = {
        "Content-Type": "application/json",
        ...extraHeaders,
    };

    const savedTritonUrl = getSavedTritonUrlOverride();
    if (savedTritonUrl) {
        headers["X-Hot-Triton-Url"] = savedTritonUrl;
    }

    const savedMetricsPort = getSavedMetricsPortOverride();
    if (savedMetricsPort) {
        headers["X-Hot-Triton-Metrics-Port"] = savedMetricsPort;
    }

    return headers;
}

function getActionProfile(actionKey) {
    return ACTION_PROFILES[actionKey] || ACTION_PROFILES.idle;
}

function formatResultPayload(payload) {
    return typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
}

function formatBytes(bytes) {
    if (typeof bytes !== "number" || Number.isNaN(bytes)) {
        return "-";
    }

    const gib = bytes / (1024 ** 3);
    if (gib >= 100) {
        return `${gib.toFixed(0)} GiB`;
    }
    if (gib >= 10) {
        return `${gib.toFixed(1)} GiB`;
    }
    return `${gib.toFixed(2)} GiB`;
}

function formatPercent(value) {
    if (typeof value !== "number" || Number.isNaN(value)) {
        return "-";
    }
    if (value >= 100 || value <= 0) {
        return `${value.toFixed(0)}%`;
    }
    if (value < 10) {
        return `${value.toFixed(1)}%`;
    }
    return `${value.toFixed(0)}%`;
}

function formatWatts(value) {
    if (typeof value !== "number" || Number.isNaN(value)) {
        return "-";
    }
    return `${value.toFixed(1)} W`;
}

function formatDateTime(value) {
    if (!value) {
        return "-";
    }

    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return value;
    }

    return date.toLocaleString("zh-CN", {
        hour12: false,
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
    });
}

function shortenId(value) {
    if (!value) {
        return "-";
    }
    if (value.length <= 22) {
        return value;
    }
    return `${value.slice(0, 12)}…${value.slice(-6)}`;
}

function compactImageLabel(imageRef) {
    const value = String(imageRef || "").trim();
    if (!value) {
        return "来源镜像未知";
    }

    const withoutDigest = value.split("@")[0];
    const lastSlashIndex = withoutDigest.lastIndexOf("/");
    return lastSlashIndex >= 0 ? withoutDigest.slice(lastSlashIndex + 1) : withoutDigest;
}

function setResult(title, payload, isError = false) {
    const prefix = isError ? "❌" : "✅";
    resultPanel.textContent = `${prefix} ${title}\n\n${formatResultPayload(payload)}`;
}

function summarizeErrorItems(items) {
    return items
        .map((item) => {
            const parts = [item.alias, item.model, item.image].filter(Boolean);
            if (parts.length > 0) {
                return `${parts.join(" / ")}: ${item.error || "失败"}`;
            }
            return item.error || "失败";
        })
        .join("；");
}

function summarizeOperationFailure(result, fallback = "操作执行失败") {
    if (!result) {
        return fallback;
    }

    if (typeof result.detail === "string" && result.detail.trim()) {
        return result.detail.trim();
    }

    const errors = [];
    if (Array.isArray(result.errors)) {
        errors.push(...result.errors);
    }
    if (Array.isArray(result.alias_result?.errors)) {
        errors.push(...result.alias_result.errors);
    }
    if (Array.isArray(result.model_result?.errors)) {
        errors.push(...result.model_result.errors);
    }

    if (errors.length > 0) {
        return summarizeErrorItems(errors);
    }

    if (typeof result.message === "string" && result.message.trim()) {
        return result.message.trim();
    }

    return fallback;
}

function ensureOperationSucceeded(result, fallback = "操作执行失败") {
    if (result && result.success === false) {
        throw new Error(summarizeOperationFailure(result, fallback));
    }
    return result;
}

async function fetchJson(url, options = {}) {
    const response = await fetch(url, {
        ...options,
        headers: buildRequestHeaders(options.headers || {}),
    });

    let data;
    try {
        data = await response.json();
    } catch (error) {
        data = { detail: await response.text() };
    }

    if (!response.ok) {
        throw new Error(data.detail || JSON.stringify(data));
    }
    return data;
}

function createManagedPayloadFallback(statusPayload, modelPayload) {
    const aliases = modelPayload.managed_aliases || statusPayload.manager?.aliases || {};
    const managedModels = modelPayload.managed_models || statusPayload.manager?.managed_models || [];
    const managedImages = statusPayload.manager?.managed_images || Object.entries(aliases).map(([id, meta]) => ({
        id,
        image: meta.image || id,
        models: meta.models || [],
        model_versions: meta.model_versions || {},
        active_versions: meta.active_versions || {},
        updated_at: meta.updated_at || null,
    }));

    return {
        config: statusPayload.manager?.config || {},
        updated_at: statusPayload.manager?.updated_at || null,
        aliases,
        managed_images: managedImages,
        managed_alias_count:
            statusPayload.manager?.managed_alias_count || managedImages.length,
        managed_image_count:
            statusPayload.manager?.managed_image_count || managedImages.length,
        managed_model_count:
            statusPayload.manager?.managed_model_count || managedModels.length,
        managed_models: managedModels,
    };
}

function normalizeKeyword(value) {
    return String(value || "").trim().toLowerCase();
}

function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
    }[char] || char));
}

function matchesModelGroupKeyword(group, keyword) {
    if (!keyword) {
        return true;
    }

    const fragments = [group.name, group.activeVersion, ...(group.images || [])];
    (group.versions || []).forEach((versionEntry) => {
        fragments.push(
            versionEntry.version,
            versionEntry.image,
            versionEntry.state,
            versionEntry.reason,
            versionEntry.updatedAt,
        );
    });

    return fragments
        .filter(Boolean)
        .join(" ")
        .toLowerCase()
        .includes(keyword);
}

function getVisibleModelCheckboxes() {
    return Array.from(modelGroupsPanel?.querySelectorAll(".model-checkbox") || []);
}

function syncModelSelectionControls() {
    const visibleCheckboxes = getVisibleModelCheckboxes();
    const visibleRefs = visibleCheckboxes.map((checkbox) => checkbox.dataset.ref || checkbox.value);
    const selectedVisibleCount = visibleRefs.filter((ref) => state.selectedModelRefs.has(ref)).length;

    if (selectAllModels) {
        selectAllModels.checked = visibleRefs.length > 0 && selectedVisibleCount === visibleRefs.length;
        selectAllModels.indeterminate = selectedVisibleCount > 0 && selectedVisibleCount < visibleRefs.length;
    }

    if (modelSelectionSummary) {
        const { totalGroups, visibleGroups } = state.modelGroupStats;
        const selectedCount = state.selectedModelRefs.size;
        modelSelectionSummary.textContent = state.modelFilter.trim()
            ? `筛选 ${visibleGroups}/${totalGroups} 个模型 · 已选 ${selectedCount} 项`
            : `共 ${totalGroups} 个模型 · 已选 ${selectedCount} 项`;
    }
}

async function fetchAndRenderDashboard() {
    const [statusPayload, modelPayload] = await Promise.all([
        fetchJson("/api/status"),
        fetchJson("/api/models"),
    ]);

    const managedPayload = modelPayload.managed || createManagedPayloadFallback(statusPayload, modelPayload);

    state.managed = managedPayload;
    state.tritonModels = modelPayload.triton_models || [];

    renderStatus(statusPayload);
    renderManagedImages(managedPayload);
    renderModelGroups(managedPayload, state.tritonModels);
    renderTritonModels(state.tritonModels);

    return {
        statusPayload,
        modelPayload,
        managedPayload,
    };
}

async function refreshStatusOnly() {
    const statusPayload = await fetchJson("/api/status");
    renderStatus(statusPayload);
    return statusPayload;
}

async function refreshGpuMetricsOnly() {
    const metricsPayload = await fetchJson("/api/gpu-metrics");
    renderGpuMonitor(metricsPayload);
    return metricsPayload;
}

function getCheckedValues(selector) {
    return Array.from(document.querySelectorAll(selector))
        .filter((node) => node.checked)
        .map((node) => node.value);
}

function getCheckedModelSelections() {
    return Array.from(state.selectedModelRefs)
        .map((ref) => state.modelSelectionIndex.get(ref))
        .filter(Boolean)
        .map((entry) => ({
            model: entry.model,
            version: entry.version || "",
            ref: entry.ref,
        }));
}

function toggleCheckboxGroup(selector, checked) {
    document.querySelectorAll(selector).forEach((checkbox) => {
        checkbox.checked = checked;
    });
}

function renderStatus(statusPayload) {
    const manager = statusPayload.manager || {};
    const triton = statusPayload.triton || {};
    const metrics = triton.metrics || {};
    const managerConfig = manager.config || {};
    const managedImageCount = manager.managed_image_count || manager.managed_alias_count || 0;
    const managedModelCount = manager.managed_model_count || 0;
    const lastUpdated = manager.updated_at || metrics.updated_at || null;
    const readyClass = triton.ready ? "ready" : "not-ready";
    const readyText = triton.ready ? "READY" : "NOT READY";
    const metricsSummary = metrics.available
        ? `${metrics.summary?.device_count || 0} GPU · ${formatBytes(metrics.summary?.used_bytes)} / ${formatBytes(metrics.summary?.total_bytes)}`
        : (metrics.detail || "未连接");
    const debugItems = [
        { label: "Triton URL", value: triton.url || "-" },
        { label: "Metrics Endpoint", value: metrics.url || (metrics.candidate_urls || [])[0] || "-" },
        { label: "Model Repository", value: managerConfig.model_repository || "-" },
        { label: "State File", value: managerConfig.state_file || "-" },
    ];

    state.runtime.reportedTritonUrl = triton.url || "";
    state.runtime.reportedMetricsUrl = metrics.url || (metrics.candidate_urls || [])[0] || "";
    updateTritonUrlEditor(state.runtime.reportedTritonUrl, state.runtime.reportedMetricsUrl);

    statusSummary.innerHTML = `
        <div class="status-item status-item-hero">
            <div class="status-item-row">
                <strong>Triton</strong>
                <div class="badge ${readyClass}">${readyText}</div>
            </div>
            <span>${triton.detail || "-"}</span>
        </div>
        <div class="status-item status-item-compact">
            <strong>管理中</strong>
            <span>${managedModelCount} 个模型 · ${managedImageCount} 个镜像</span>
        </div>
        <div class="status-item status-item-compact">
            <strong>GPU / Metrics</strong>
            <span>${metricsSummary}</span>
        </div>
        <div class="status-item status-item-compact">
            <strong>最近更新</strong>
            <span>${lastUpdated ? formatDateTime(lastUpdated) : "-"}</span>
        </div>
        <details class="status-debug-panel">
            <summary>环境路径 / Endpoint</summary>
            <div class="status-debug-grid">
                ${debugItems.map((item) => `
                    <div class="status-debug-item">
                        <strong>${escapeHtml(item.label)}</strong>
                        <span class="mono">${escapeHtml(item.value)}</span>
                    </div>
                `).join("")}
            </div>
        </details>
    `;

    renderGpuMonitor(metrics);
}

function renderGpuMonitor(metricsPayload) {
    if (!gpuMonitorPanel) {
        return;
    }

    const metrics = metricsPayload || {};
    if (gpuMetricsUpdated) {
        gpuMetricsUpdated.textContent = metrics.updated_at
            ? `更新于 ${formatDateTime(metrics.updated_at)}`
            : "等待拉取";
    }

    if (!metrics.available) {
        gpuMonitorPanel.innerHTML = `
            <div class="gpu-error">
                ${metrics.detail || "显存指标暂不可用；请确认 Triton metrics 端口可达，并开启 --allow-gpu-metrics。"}
            </div>
        `;
        return;
    }

    const summary = metrics.summary || {};
    const gpus = Array.isArray(metrics.gpus) ? metrics.gpus : [];

    if (!gpus.length) {
        gpuMonitorPanel.innerHTML = '<div class="empty">当前没有可展示的 GPU 指标</div>';
        return;
    }

    const summaryHtml = `
        <div class="gpu-summary-strip">
            <div class="gpu-kpi">
                <strong>${formatBytes(summary.used_bytes)} / ${formatBytes(summary.total_bytes)}</strong>
                <span>总显存 · ${formatPercent(summary.used_percent)}</span>
            </div>
            <div class="gpu-kpi">
                <strong>${summary.device_count || 0}</strong>
                <span>GPU 数量</span>
            </div>
            <div class="gpu-kpi">
                <strong>${formatPercent(summary.average_utilization_percent)}</strong>
                <span>平均利用率</span>
            </div>
            <div class="gpu-kpi">
                <strong>${formatWatts(summary.total_power_usage_watts)}</strong>
                <span>总功耗</span>
            </div>
        </div>
    `;

    const cardsHtml = gpus
        .map((gpu) => {
            const usedRatio = typeof gpu.used_ratio === "number"
                ? Math.max(0, Math.min(1, gpu.used_ratio))
                : 0;

            return `
                <article class="gpu-card">
                    <div class="gpu-card-head">
                        <div>
                            <strong class="gpu-card-name">${gpu.label || "GPU"}</strong>
                            <div class="gpu-card-uuid mono">${shortenId(gpu.gpu_uuid || gpu.gpu_bus_id || "unknown")}</div>
                        </div>
                        <span class="mini-badge active">${formatPercent(gpu.used_percent)}</span>
                    </div>
                    <div class="gpu-bar" aria-label="${gpu.label || 'GPU'} 显存占用">
                        <span class="gpu-bar-fill" style="width: ${Math.round(usedRatio * 100)}%"></span>
                    </div>
                    <div class="gpu-card-stats">
                        <div class="gpu-stat">
                            <strong>${formatBytes(gpu.used_bytes)} / ${formatBytes(gpu.total_bytes)}</strong>
                            <span>显存使用</span>
                        </div>
                        <div class="gpu-stat">
                            <strong>${formatPercent(gpu.utilization_percent)}</strong>
                            <span>GPU 利用率</span>
                        </div>
                        <div class="gpu-stat">
                            <strong>${formatWatts(gpu.power_usage_watts)}</strong>
                            <span>功耗</span>
                        </div>
                        <div class="gpu-stat">
                            <strong>${gpu.gpu_bus_id || shortenId(gpu.gpu_uuid || "-")}</strong>
                            <span>设备标识</span>
                        </div>
                    </div>
                </article>
            `;
        })
        .join("");

    gpuMonitorPanel.innerHTML = `${summaryHtml}<div class="gpu-card-grid">${cardsHtml}</div>`;
}

function renderManagedImages(managedPayload) {
    const managedImages = managedPayload.managed_images || [];

    if (!managedImageGrid) {
        return;
    }

    if (!managedImages.length) {
        managedImageGrid.innerHTML = '<div class="empty">暂无已管理镜像</div>';
        return;
    }

    managedImageGrid.innerHTML = managedImages
        .map((entry) => {
            const activeVersions = entry.active_versions || {};
            const models = (entry.models || [])
                .map((model) => {
                    const activeVersion = activeVersions[model];
                    const label = activeVersion ? `${model}@${activeVersion}` : model;
                    return `<span class="tag">${label}</span>`;
                })
                .join("");

            return `
                <article class="image-card">
                    <div class="image-card-head">
                        <strong class="image-card-title">${entry.image || "-"}</strong>
                        <span class="image-card-meta mono">${entry.updated_at || "-"}</span>
                    </div>
                    <div class="image-card-body">
                        <div class="tag-list">${models || '<span class="empty-chip">未发现模型</span>'}</div>
                    </div>
                </article>
            `;
        })
        .join("");
}

function buildModelGroups(managedPayload, tritonModels) {
    const groups = new Map();

    function ensureGroup(modelName) {
        if (!groups.has(modelName)) {
            groups.set(modelName, {
                name: modelName,
                activeVersion: null,
                images: new Set(),
                versions: new Map(),
                updatedAt: null,
                externalOnly: true,
            });
        }
        return groups.get(modelName);
    }

    function ensureVersion(group, version) {
        const versionKey = version || "-";
        if (!group.versions.has(versionKey)) {
            group.versions.set(versionKey, {
                model: group.name,
                version: versionKey,
                ref: version && version !== "-" ? `${group.name}@${version}` : group.name,
                image: "",
                state: "",
                reason: "",
                loaded: false,
                isActive: false,
                updatedAt: null,
            });
        }
        return group.versions.get(versionKey);
    }

    (managedPayload.managed_images || []).forEach((entry) => {
        const activeVersions = entry.active_versions || {};
        const modelVersions = entry.model_versions || {};

        (entry.models || []).forEach((modelName) => {
            const group = ensureGroup(modelName);
            group.externalOnly = false;
            if (entry.image) {
                group.images.add(entry.image);
            }
            if (entry.updated_at && (!group.updatedAt || entry.updated_at > group.updatedAt)) {
                group.updatedAt = entry.updated_at;
            }
            if (activeVersions[modelName]) {
                group.activeVersion = activeVersions[modelName];
            }

            const versions = Array.isArray(modelVersions[modelName]) ? modelVersions[modelName] : [];
            versions.forEach((version) => {
                const versionEntry = ensureVersion(group, version);
                versionEntry.image = entry.image || versionEntry.image;
                versionEntry.updatedAt = entry.updated_at || versionEntry.updatedAt;
                versionEntry.isActive = activeVersions[modelName] === version;
            });
        });
    });

    (tritonModels || []).forEach((row) => {
        if (!row.name) {
            return;
        }

        const versionKey = row.version || "-";
        const existingGroup = groups.get(row.name);
        const knownManagedVersion = existingGroup?.versions.has(versionKey) || false;
        const tritonState = String(row.state || "").toUpperCase();
        const tritonReason = String(row.reason || "").toLowerCase();
        const isStaleUnloadedVersion =
            !knownManagedVersion &&
            tritonState === "UNAVAILABLE" &&
            tritonReason.includes("unloaded");

        if (isStaleUnloadedVersion) {
            return;
        }

        const group = ensureGroup(row.name);
        const versionEntry = ensureVersion(group, versionKey);
        versionEntry.state = row.state || versionEntry.state;
        versionEntry.reason = row.reason || versionEntry.reason;
        versionEntry.loaded = true;
    });

    return Array.from(groups.values())
        .map((group) => ({
            ...group,
            images: Array.from(group.images).sort(),
            versions: Array.from(group.versions.values()).sort((a, b) => {
                const aVersion = /^\d+$/.test(a.version) ? Number(a.version) : -1;
                const bVersion = /^\d+$/.test(b.version) ? Number(b.version) : -1;
                if (aVersion !== bVersion) {
                    return bVersion - aVersion;
                }
                return String(a.version).localeCompare(String(b.version));
            }),
        }))
        .sort((a, b) => a.name.localeCompare(b.name));
}

function renderModelGroups(managedPayload, tritonModels) {
    if (!modelGroupsPanel) {
        return;
    }

    const groups = buildModelGroups(managedPayload, tritonModels);
    const selectionIndex = new Map();
    groups.forEach((group) => {
        selectionIndex.set(group.name, { model: group.name, version: "", ref: group.name });
        group.versions.forEach((versionEntry) => {
            selectionIndex.set(versionEntry.ref, {
                model: group.name,
                version: versionEntry.version === "-" ? "" : versionEntry.version,
                ref: versionEntry.ref,
            });
        });
    });
    state.modelSelectionIndex = selectionIndex;

    for (const ref of Array.from(state.selectedModelRefs)) {
        if (!selectionIndex.has(ref)) {
            state.selectedModelRefs.delete(ref);
        }
    }

    const keyword = normalizeKeyword(state.modelFilter);
    const visibleGroups = groups.filter((group) => matchesModelGroupKeyword(group, keyword));
    state.modelGroupStats = {
        totalGroups: groups.length,
        visibleGroups: visibleGroups.length,
    };

    if (!groups.length) {
        modelGroupsPanel.innerHTML = '<div class="empty">暂无模型版本信息</div>';
        syncModelSelectionControls();
        return;
    }

    if (!visibleGroups.length) {
        modelGroupsPanel.innerHTML = `<div class="empty">没有匹配关键字 “${escapeHtml(state.modelFilter.trim())}” 的模型</div>`;
        syncModelSelectionControls();
        return;
    }

    modelGroupsPanel.innerHTML = visibleGroups
        .map((group) => {
            const modelChecked = state.selectedModelRefs.has(group.name) ? "checked" : "";
            const sourceSummary = group.images.length === 0
                ? "来源镜像未知"
                : group.images.length === 1
                    ? compactImageLabel(group.images[0])
                    : `${compactImageLabel(group.images[0])} 等 ${group.images.length} 个镜像`;
            const sourceTitle = group.images.length ? group.images.join("\n") : "来源镜像未知";

            const versionRows = group.versions
                .map((versionEntry) => {
                    const versionClass = versionEntry.isActive ? "version-row active" : "version-row";
                    const loadedBadge = versionEntry.loaded
                        ? '<span class="mini-badge ready">已加载</span>'
                        : '<span class="mini-badge muted">未加载</span>';
                    const activeBadge = versionEntry.isActive
                        ? '<span class="mini-badge active">当前激活</span>'
                        : "";
                    const versionChecked = state.selectedModelRefs.has(versionEntry.ref) ? "checked" : "";
                    const versionMetaParts = [];

                    if (group.images.length !== 1 || !group.images[0]) {
                        versionMetaParts.push(`<span class="mono">${escapeHtml(compactImageLabel(versionEntry.image || ""))}</span>`);
                    }

                    const versionMetaHtml = versionMetaParts.length
                        ? `<div class="version-row-meta">${versionMetaParts.join("")}</div>`
                        : "";

                    return `
                        <label class="${versionClass}">
                            <input class="model-checkbox" type="checkbox" value="${versionEntry.ref}" data-model="${group.name}" data-version="${versionEntry.version === '-' ? '' : versionEntry.version}" data-ref="${versionEntry.ref}" ${versionChecked}>
                            <div class="version-row-main">
                                <div class="version-row-top">
                                    <div class="version-row-title">
                                        <span class="version-label mono">${versionEntry.version}</span>
                                        ${activeBadge}
                                        ${loadedBadge}
                                    </div>
                                    <span class="${stateClassForModel(versionEntry)}">${versionEntry.state || (versionEntry.loaded ? 'READY' : '-')}</span>
                                </div>
                                ${versionMetaHtml}
                                ${versionEntry.reason ? `<div class="version-row-reason">${versionEntry.reason}</div>` : ""}
                            </div>
                        </label>
                    `;
                })
                .join("");

            return `
                <article class="model-card">
                    <div class="model-card-header">
                        <div class="model-card-title-wrap">
                            <label class="select-toggle model-select-toggle">
                                <input class="model-checkbox" type="checkbox" value="${group.name}" data-model="${group.name}" data-version="" data-ref="${group.name}" ${modelChecked}>
                            </label>
                            <div>
                                <h3>${group.name}</h3>
                                <p title="${escapeHtml(sourceTitle)}">${group.versions.length} 个版本 · ${escapeHtml(sourceSummary)}</p>
                            </div>
                        </div>
                        <div class="model-card-badges">
                            <span class="mini-badge active">当前 ${group.activeVersion || '-'}</span>
                            ${group.externalOnly ? '<span class="mini-badge warning">仅 Triton 可见</span>' : ''}
                        </div>
                    </div>
                    <div class="version-row-list">${versionRows}</div>
                </article>
            `;
        })
        .join("");

    syncModelSelectionControls();
}

function stateClassForModel(model) {
    const currentState = (model.state || "").toUpperCase();
    if (currentState === "READY") {
        return "state-ready";
    }
    if (currentState === "UNAVAILABLE") {
        return "state-error";
    }
    return "state-other";
}

function isStaleTritonRow(model) {
    const currentState = String(model?.state || "").toUpperCase();
    const currentReason = String(model?.reason || "").toLowerCase();
    return currentState === "UNAVAILABLE" && currentReason.includes("unloaded");
}

function renderTritonModels(tritonModels) {
    if (!tritonModels.length) {
        tritonModelBody.innerHTML = '<tr><td colspan="4" class="empty">暂无 Triton 模型信息</td></tr>';
        return;
    }

    tritonModelBody.innerHTML = tritonModels
        .map((model) => {
            const staleRow = isStaleTritonRow(model);
            const stateClass = staleRow ? "state-stale" : stateClassForModel(model);
            const reasonHtml = staleRow
                ? `<span class="mono">${model.reason || "-"}</span><span class="stale-note">历史残留（不可操作）</span>`
                : (model.reason || "-");

            return `
            <tr class="${staleRow ? "stale-triton-row" : ""}">
                <td class="mono">${model.name || "-"}</td>
                <td class="mono">${model.version || "-"}</td>
                <td class="${stateClass}">${model.state || "-"}</td>
                <td>${reasonHtml}</td>
            </tr>
        `;
        })
        .join("");
}

function startStatusPolling() {
    if (statusPollingTimerId) {
        window.clearInterval(statusPollingTimerId);
    }

    statusPollingTimerId = window.setInterval(async () => {
        if (document.hidden || state.operation.status === "running" || statusPollingInFlight) {
            return;
        }

        statusPollingInFlight = true;
        try {
            await refreshStatusOnly();
        } catch (error) {
            console.debug("status polling failed", error);
        } finally {
            statusPollingInFlight = false;
        }
    }, STATUS_POLL_INTERVAL_MS);
}

function startGpuMetricsPolling() {
    if (gpuMetricsPollingTimerId) {
        window.clearInterval(gpuMetricsPollingTimerId);
    }

    gpuMetricsPollingTimerId = window.setInterval(async () => {
        if (gpuMetricsPollingInFlight) {
            return;
        }

        gpuMetricsPollingInFlight = true;
        try {
            await refreshGpuMetricsOnly();
        } catch (error) {
            console.debug("gpu metrics polling failed", error);
        } finally {
            gpuMetricsPollingInFlight = false;
        }
    }, GPU_METRICS_POLL_INTERVAL_MS);
}

function parseConfigText() {
    const value = configInput.value.trim();
    if (!value) {
        throw new Error("请先输入 JSON 配置");
    }
    return JSON.parse(value);
}

function formatElapsed(startedAt) {
    if (!startedAt) {
        return "耗时 --";
    }

    const totalSeconds = Math.max(0, Math.round((Date.now() - startedAt) / 1000));
    if (totalSeconds < 60) {
        return `耗时 ${totalSeconds}s`;
    }

    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `耗时 ${minutes}m ${seconds}s`;
}

function formatLogTime(date) {
    return date.toLocaleTimeString("zh-CN", { hour12: false });
}

function renderOperationTips() {
    const profile = getActionProfile(state.operation.key);
    const tips = [];

    if (state.operation.tipText) {
        tips.push(state.operation.tipText);
    }

    (profile.tips || []).forEach((tip) => {
        if (tip && !tips.includes(tip)) {
            tips.push(tip);
        }
    });

    const visibleTips = tips.slice(0, 3);
    operationTipList.innerHTML = "";

    if (!visibleTips.length) {
        const emptyItem = document.createElement("li");
        emptyItem.className = "empty";
        emptyItem.textContent = "暂无操作提示";
        operationTipList.appendChild(emptyItem);
        return;
    }

    visibleTips.forEach((tip, index) => {
        const item = document.createElement("li");
        if (index === 0) {
            item.classList.add("highlight");
        }
        item.textContent = tip;
        operationTipList.appendChild(item);
    });
}

function renderOperationLogs() {
    operationLogList.innerHTML = "";

    if (!state.logs.length) {
        const emptyItem = document.createElement("li");
        emptyItem.className = "empty";
        emptyItem.textContent = "暂无操作记录";
        operationLogList.appendChild(emptyItem);
        return;
    }

    state.logs.forEach((entry) => {
        const item = document.createElement("li");
        item.className = `operation-log-item ${entry.tone}`;

        const time = document.createElement("span");
        time.className = "operation-log-time mono";
        time.textContent = formatLogTime(entry.time);

        const message = document.createElement("span");
        message.className = "operation-log-message";
        message.textContent = entry.message;

        item.appendChild(time);
        item.appendChild(message);
        operationLogList.appendChild(item);
    });
}

function renderOperationState() {
    const labels = {
        idle: "空闲中",
        running: "进行中",
        success: "已完成",
        failed: "执行失败",
    };

    operationBadge.textContent = labels[state.operation.status] || "空闲中";
    operationBadge.className = `badge operation-badge ${state.operation.status}`;
    operationTitle.textContent = state.operation.title;
    operationElapsed.textContent = formatElapsed(state.operation.startedAt);
    operationProgressFill.style.width = `${Math.max(0, Math.min(100, state.operation.progress))}%`;
    operationProgressFill.className = `progress-fill ${state.operation.status}`;
    operationProgressText.textContent = `${Math.round(state.operation.progress)}%`;
    operationStageText.textContent = state.operation.stageText;
    operationInlineTip.textContent = state.operation.tipText;
    operationInlineTip.className = `inline-tip ${state.operation.status === "failed" ? "failed" : state.operation.status === "success" ? "success" : "info"}`;

    renderOperationTips();
    renderOperationLogs();
    updateTritonUrlEditor(state.runtime.reportedTritonUrl);
}

function pushOperationLog(message, tone = "info") {
    const lastEntry = state.logs[0];
    if (lastEntry && lastEntry.message === message && lastEntry.tone === tone) {
        return;
    }

    state.logs.unshift({
        message,
        tone,
        time: new Date(),
    });
    state.logs = state.logs.slice(0, 4);
    renderOperationLogs();
}

function clearOperationTimers() {
    if (state.operation.progressTimerId) {
        window.clearInterval(state.operation.progressTimerId);
    }
    if (state.operation.elapsedTimerId) {
        window.clearInterval(state.operation.elapsedTimerId);
    }

    state.operation.progressTimerId = null;
    state.operation.elapsedTimerId = null;
}

function setControlsBusy(isBusy, activeButtonId = null) {
    BUTTON_IDS.forEach((id) => {
        const button = document.getElementById(id);
        if (!button) {
            return;
        }

        button.disabled = isBusy;
        const originalText = buttonTextCache.get(id) || button.textContent;
        button.textContent = isBusy && id === activeButtonId ? (BUTTON_BUSY_TEXT[id] || originalText) : originalText;
    });
}

function applyProfileStage(profile, stageIndex, log = true) {
    const stage = profile.stages?.[stageIndex];
    if (!stage) {
        return;
    }

    const hasChanged = state.operation.stageIndex !== stageIndex;
    state.operation.stageIndex = stageIndex;
    state.operation.progress = Math.max(state.operation.progress, stage.progress);
    state.operation.stageText = stage.text;
    state.operation.tipText = stage.tip || state.operation.tipText;

    if (hasChanged && log) {
        pushOperationLog(stage.text, "info");
    }
}

function advanceStageByProgress(profile) {
    if (!Array.isArray(profile.stages)) {
        return;
    }

    let nextStageIndex = state.operation.stageIndex;
    profile.stages.forEach((stage, index) => {
        if (state.operation.progress >= stage.progress) {
            nextStageIndex = index;
        }
    });

    if (nextStageIndex > state.operation.stageIndex) {
        applyProfileStage(profile, nextStageIndex);
    }
}

function bumpOperationProgress() {
    const profile = getActionProfile(state.operation.key);
    const maxProgress = profile.maxProgress || 90;
    const tick = profile.tick || 4;

    if (state.operation.progress < maxProgress) {
        state.operation.progress = Math.min(state.operation.progress + tick, maxProgress);
    }

    advanceStageByProgress(profile);
    renderOperationState();
}

function startOperation(actionKey, activeButtonId = null) {
    const profile = getActionProfile(actionKey);

    clearOperationTimers();
    state.operation.key = actionKey;
    state.operation.status = "running";
    state.operation.progress = 0;
    state.operation.title = profile.label;
    state.operation.stageText = "正在准备执行...";
    state.operation.tipText = (profile.tips && profile.tips[0]) || "请稍候...";
    state.operation.startedAt = Date.now();
    state.operation.stageIndex = -1;
    state.operation.activeButtonId = activeButtonId;

    setControlsBusy(true, activeButtonId);
    pushOperationLog(`${profile.label}已开始`, "info");

    if (Array.isArray(profile.stages) && profile.stages.length > 0) {
        applyProfileStage(profile, 0, false);
    } else {
        state.operation.progress = 10;
    }

    renderOperationState();
    state.operation.progressTimerId = window.setInterval(bumpOperationProgress, 1100);
    state.operation.elapsedTimerId = window.setInterval(renderOperationState, 1000);
}

function markOperation(progress, stageText, tipText, { log = true } = {}) {
    const profile = getActionProfile(state.operation.key);
    state.operation.progress = Math.max(state.operation.progress, Math.min(progress, 99));

    if (stageText) {
        const changed = stageText !== state.operation.stageText;
        state.operation.stageText = stageText;
        if (changed && log) {
            pushOperationLog(stageText, "info");
        }
    }

    if (tipText) {
        state.operation.tipText = tipText;
    }

    advanceStageByProgress(profile);
    renderOperationState();
}

function finishOperation(actionKey, success, summaryText, tipText = null) {
    const profile = getActionProfile(actionKey);

    clearOperationTimers();
    state.operation.key = actionKey;
    state.operation.status = success ? "success" : "failed";
    state.operation.progress = 100;
    state.operation.title = success ? `${profile.label}已完成` : `${profile.label}失败`;
    state.operation.stageText = summaryText || (success ? `${profile.label}完成` : `${profile.label}失败`);
    state.operation.tipText = tipText || (success ? (profile.successTip || state.operation.tipText) : (profile.failureTip || "请查看结果面板中的错误详情。"));

    setControlsBusy(false);
    pushOperationLog(state.operation.stageText, success ? "success" : "error");
    renderOperationState();
}

async function runTrackedAction(actionKey, activeButtonId, executor, { successTitle, failureTitle } = {}) {
    startOperation(actionKey, activeButtonId);

    try {
        const result = await executor({
            mark: (progress, stageText, tipText, options) => markOperation(progress, stageText, tipText, options),
            refreshDashboard: fetchAndRenderDashboard,
        });

        const successText = typeof successTitle === "function" ? successTitle(result) : successTitle;
        finishOperation(actionKey, true, successText);
        return result;
    } catch (error) {
        const failureText = typeof failureTitle === "function" ? failureTitle(error) : failureTitle;
        finishOperation(actionKey, false, failureText || error.message);
        throw error;
    }
}

async function refreshAll(showToast = false) {
    try {
        return await runTrackedAction(
            "refresh",
            "refresh-all-btn",
            async ({ mark, refreshDashboard }) => {
                mark(18, "正在请求状态数据", "这里会同时刷新 Triton Ready 状态、已管理镜像和模型版本视图。");
                const payloads = await refreshDashboard();
                mark(90, "状态数据已返回，正在更新页面", "刷新完成后可以继续执行热加载、热卸载或重载操作。");

                if (showToast) {
                    setResult("状态已刷新", payloads.statusPayload, false);
                }

                return payloads;
            },
            {
                successTitle: "状态刷新完成",
                failureTitle: (error) => `状态刷新失败：${error.message}`,
            },
        );
    } catch (error) {
        setResult("状态刷新失败", { error: error.message }, true);
        return null;
    }
}

async function applyConfig() {
    try {
        const result = await runTrackedAction(
            "apply",
            "apply-config-btn",
            async ({ mark, refreshDashboard }) => {
                mark(10, "正在校验 JSON 配置", "会先验证 JSON 格式；系统会忽略 key、只读取镜像 value。", { log: false });
                const config = parseConfigText();

                const payload = {
                    config,
                    prune_missing: pruneMissing.checked,
                    force: forceReload.checked,
                };

                mark(22, "正在提交热加载请求", "服务端会开始拉取镜像、提取模型并同步到共享 repository。", { log: true });
                const result = await fetchJson("/api/apply-config", {
                    method: "POST",
                    body: JSON.stringify(payload),
                });

                ensureOperationSucceeded(result, "热加载执行失败");
                mark(92, "热加载请求已返回，正在刷新页面状态", "即将更新已管理镜像、模型版本视图和 Triton 当前模型列表。", { log: true });
                setResult("热加载执行完成", result, false);
                await refreshDashboard();
                return result;
            },
            {
                successTitle: (result) => {
                    const applied = result.applied?.length || 0;
                    const skipped = result.skipped?.length || 0;
                    return `热加载完成：${applied} 个镜像更新，${skipped} 个镜像跳过`;
                },
                failureTitle: (error) => `热加载失败：${error.message}`,
            },
        );

        return result;
    } catch (error) {
        setResult("热加载执行失败", { error: error.message }, true);
        return null;
    }
}

async function unloadSelectedModels() {
    try {
        const result = await runTrackedAction(
            "unloadModels",
            "unload-models-btn",
            async ({ mark, refreshDashboard }) => {
                mark(10, "正在校验选中的模型或版本", "系统会先确认你至少勾选了一个 Triton model name 或 model@version。", { log: false });
                const selections = getCheckedModelSelections();

                if (!selections.length) {
                    throw new Error("请至少勾选一个模型或版本");
                }

                const versions = Array.from(new Set(selections.filter((item) => item.version).map((item) => item.ref)));
                const models = Array.from(new Set(selections.filter((item) => !item.version).map((item) => item.model)));

                mark(36, "正在提交模型/版本卸载请求", "若选中了具体版本，服务端会改写 repository 并触发 Triton reload。", { log: true });
                const result = await fetchJson("/api/unload", {
                    method: "POST",
                    body: JSON.stringify({ aliases: [], models, versions }),
                });

                ensureOperationSucceeded(result, "模型/版本卸载失败");
                mark(90, "模型/版本卸载请求已返回，正在刷新页面状态", "完成后可确认目标版本是否消失，或模型是否切到新的激活版本。", { log: true });
                setResult("模型/版本卸载完成", result, false);
                await refreshDashboard();
                return result;
            },
            {
                successTitle: (result) => {
                    const removedModels = result.model_result?.removed_models?.length || 0;
                    const removedVersions = result.version_result?.removed_versions?.length || 0;
                    return `卸载完成：${removedModels} 个模型、${removedVersions} 个版本已处理`;
                },
                failureTitle: (error) => `模型/版本卸载失败：${error.message}`,
            },
        );

        return result;
    } catch (error) {
        setResult("模型/版本卸载失败", { error: error.message }, true);
        return null;
    }
}

async function reloadSelectedModels() {
    try {
        const result = await runTrackedAction(
            "reloadModels",
            "reload-models-btn",
            async ({ mark, refreshDashboard }) => {
                mark(10, "正在校验选中的模型", "系统会先确认你至少勾选了一个要重载的模型。", { log: false });
                const models = Array.from(new Set(getCheckedModelSelections().map((item) => item.model)));

                if (!models.length) {
                    throw new Error("请至少勾选一个模型");
                }

                mark(40, "正在调用 Triton load 触发版本重载", "模型会直接触发 Triton reload，并应用共享目录中的当前目标版本。", { log: true });
                const result = await fetchJson("/api/reload", {
                    method: "POST",
                    body: JSON.stringify({ models }),
                });

                ensureOperationSucceeded(result, "模型重载失败");
                mark(90, "模型重载请求已返回，正在刷新页面状态", "完成后建议检查模型状态是否恢复为 READY。", { log: true });
                setResult("模型重载完成", result, false);
                await refreshDashboard();
                return result;
            },
            {
                successTitle: (result) => `模型重载完成：${result.reloaded_models?.length || 0} 个模型已重载`,
                failureTitle: (error) => `模型重载失败：${error.message}`,
            },
        );

        return result;
    } catch (error) {
        setResult("模型重载失败", { error: error.message }, true);
        return null;
    }
}

async function loadSampleConfig() {
    try {
        const sample = await runTrackedAction(
            "sample",
            "load-sample-btn",
            async ({ mark }) => {
                mark(24, "正在请求示例配置", "会从后端读取 `sample_config.json` 并填入左侧文本框。", { log: false });
                const sample = await fetchJson("/api/sample-config");
                mark(78, "正在填充示例 JSON", "填充完成后可以直接点击“执行热加载”进行测试。", { log: true });
                configInput.value = JSON.stringify(sample, null, 2);
                setResult("已填充示例 JSON", sample, false);
                return sample;
            },
            {
                successTitle: "示例 JSON 已填充",
                failureTitle: (error) => `示例 JSON 加载失败：${error.message}`,
            },
        );

        return sample;
    } catch (error) {
        setResult("示例 JSON 加载失败", { error: error.message }, true);
        return null;
    }
}

async function formatJson() {
    try {
        const payload = await runTrackedAction(
            "format",
            "format-json-btn",
            async ({ mark }) => {
                mark(22, "正在解析输入内容", "这里会验证左侧文本框里的内容是否为合法 JSON。", { log: false });
                const payload = parseConfigText();
                mark(78, "正在回写格式化结果", "格式化只会调整排版，不会修改配置语义。", { log: true });
                configInput.value = JSON.stringify(payload, null, 2);
                setResult("JSON 已格式化", payload, false);
                return payload;
            },
            {
                successTitle: "JSON 已格式化",
                failureTitle: (error) => `JSON 格式化失败：${error.message}`,
            },
        );

        return payload;
    } catch (error) {
        setResult("JSON 格式化失败", { error: error.message }, true);
        return null;
    }
}

async function saveTritonUrlSetting() {
    if (!tritonUrlInput) {
        return null;
    }

    try {
        const validatedUrl = parseTritonUrl(
            tritonUrlInput.value || state.runtime.reportedTritonUrl,
            { strict: true },
        );
        const validatedMetricsPort = parseMetricsPort(metricsPortInput?.value || "", {
            strict: true,
            allowEmpty: true,
        });

        const savedUrl = saveTritonUrlOverride(validatedUrl);
        const savedMetricsPort = saveMetricsPortOverride(validatedMetricsPort);
        tritonUrlInput.value = formatTritonUrlForDisplay(savedUrl);
        if (metricsPortInput) {
            metricsPortInput.value = savedMetricsPort;
        }
        updateTritonUrlEditor(state.runtime.reportedTritonUrl, state.runtime.reportedMetricsUrl);
        setResult(
            "Endpoint 设置已保存",
            {
                triton_url: savedUrl,
                metrics_port: savedMetricsPort || null,
                persistence: "browser localStorage",
                default_source: "环境变量 / .env / 启动参数",
            },
            false,
        );
        await refreshAll(false);
        return savedUrl;
    } catch (error) {
        setResult("Triton URL 保存失败", { error: error.message }, true);
        return null;
    }
}

async function resetTritonUrlSetting() {
    clearTritonUrlOverride();
    clearMetricsPortOverride();
    if (tritonUrlInput) {
        tritonUrlInput.value = "";
    }
    if (metricsPortInput) {
        metricsPortInput.value = "";
    }
    updateTritonUrlEditor("", "");

    setResult(
        "Endpoint 设置已恢复默认",
        {
            default_source: "环境变量 / .env / 启动参数",
            metrics_port: null,
        },
        false,
    );

    return refreshAll(false);
}

document.getElementById("apply-config-btn")?.addEventListener("click", applyConfig);
document.getElementById("refresh-all-btn")?.addEventListener("click", () => {
    refreshAll(true);
});
document.getElementById("unload-models-btn")?.addEventListener("click", unloadSelectedModels);
document.getElementById("reload-models-btn")?.addEventListener("click", reloadSelectedModels);
document.getElementById("load-sample-btn")?.addEventListener("click", loadSampleConfig);
document.getElementById("format-json-btn")?.addEventListener("click", formatJson);
document.getElementById("clear-result-btn")?.addEventListener("click", () => {
    resultPanel.textContent = "等待操作...";
});
saveTritonUrlBtn?.addEventListener("click", saveTritonUrlSetting);
resetTritonUrlBtn?.addEventListener("click", resetTritonUrlSetting);

tritonUrlInput?.addEventListener("input", () => {
    updateTritonUrlEditor(state.runtime.reportedTritonUrl, state.runtime.reportedMetricsUrl);
});

metricsPortInput?.addEventListener("input", () => {
    updateTritonUrlEditor(state.runtime.reportedTritonUrl, state.runtime.reportedMetricsUrl);
});

tritonUrlInput?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        event.preventDefault();
        saveTritonUrlSetting();
    }
});

metricsPortInput?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        event.preventDefault();
        saveTritonUrlSetting();
    }
});

selectAllModels?.addEventListener("change", (event) => {
    getVisibleModelCheckboxes().forEach((checkbox) => {
        const ref = checkbox.dataset.ref || checkbox.value;
        checkbox.checked = event.target.checked;
        if (event.target.checked) {
            state.selectedModelRefs.add(ref);
        } else {
            state.selectedModelRefs.delete(ref);
        }
    });
    syncModelSelectionControls();
});

modelGroupsPanel?.addEventListener("change", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement) || !target.classList.contains("model-checkbox")) {
        return;
    }

    const ref = target.dataset.ref || target.value;
    if (target.checked) {
        state.selectedModelRefs.add(ref);
    } else {
        state.selectedModelRefs.delete(ref);
    }
    syncModelSelectionControls();
});

modelFilterInput?.addEventListener("input", (event) => {
    state.modelFilter = event.target.value || "";
    renderModelGroups(state.managed || { managed_images: [] }, state.tritonModels || []);
});

clearModelFilterBtn?.addEventListener("click", () => {
    state.modelFilter = "";
    if (modelFilterInput) {
        modelFilterInput.value = "";
        modelFilterInput.focus();
    }
    renderModelGroups(state.managed || { managed_images: [] }, state.tritonModels || []);
});

themeLightBtn?.addEventListener("click", () => applyTheme("light"));
themeDarkBtn?.addEventListener("click", () => applyTheme("dark"));

window.addEventListener("DOMContentLoaded", async () => {
    applyTheme(getPreferredTheme());
    updateTritonUrlEditor("");
    renderOperationState();
    startStatusPolling();
    startGpuMetricsPolling();

    try {
        await runTrackedAction(
            "init",
            null,
            async ({ mark, refreshDashboard }) => {
                mark(18, "正在连接 hot_triton API", "如果这里失败，请先确认 8090 服务已经启动。", { log: false });
                await refreshDashboard();
                mark(68, "正在加载示例 JSON 配置", "示例配置会自动填入左侧文本框，方便直接试跑。", { log: true });
                const sample = await fetchJson("/api/sample-config");
                configInput.value = JSON.stringify(sample, null, 2);
                return sample;
            },
            {
                successTitle: "页面初始化完成",
                failureTitle: (error) => `页面初始化失败：${error.message}`,
            },
        );
    } catch (error) {
        setResult("初始化页面失败", { error: error.message }, true);
    }
});
