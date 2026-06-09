const resultPanel = document.getElementById("result-panel");
const statusSummary = document.getElementById("status-summary");
const managedModelsPanel = document.getElementById("managed-models-panel");
const jobListPanel = document.getElementById("job-list-panel");
const gpuMonitorPanel = document.getElementById("gpu-monitor-panel");
const gpuMetricsUpdated = document.getElementById("gpu-metrics-updated");
const tritonModelBody = document.getElementById("triton-model-body");
const tritonModelFilterInput = document.getElementById("triton-model-filter-input");
const clearTritonModelFilterBtn = document.getElementById("clear-triton-model-filter-btn");
const selectAllTritonModelsCheckbox = document.getElementById("select-all-triton-models");
const bulkUnloadTritonBtn = document.getElementById("bulk-unload-triton-btn");
const tritonSelectionSummary = document.getElementById("triton-selection-summary");
const batchInput = document.getElementById("batch-input");
const singleImageInput = document.getElementById("single-image");
const jobNameInput = document.getElementById("job-name-input");
const actionModelNameInput = document.getElementById("action-model-name");
const tritonUrlInput = document.getElementById("triton-url-input");
const metricsPortInput = document.getElementById("metrics-port-input");
const operationStatusPanel = document.getElementById("operation-status-panel");
const operationStatusCaption = document.getElementById("operation-status-caption");
const operationStatusBadge = document.getElementById("operation-status-badge");
const operationStatusTitle = document.getElementById("operation-status-title");
const operationStatusDetail = document.getElementById("operation-status-detail");

const TRITON_URL_STORAGE_KEY = "hot_triton_triton_url";
const METRICS_PORT_STORAGE_KEY = "hot_triton_metrics_port";
const AUTO_REFRESH_INTERVAL_MS = 5000;
const MAX_JOB_LIST_ITEMS = 50;
const API_ROUTES = {
    status: "/api/status",
    modelsOverview: "/api/models",
    gpuStatus: "/api/gpu-status",
    loadModel: "/api/models/load",
    loadModelBatch: "/api/models/load-batch",
    unloadModel: "/api/models/unload",
    reloadModel: "/api/models/reload",
    unloadSelection: "/api/models/unload-batch",
    jobStatus: (jobName) => `/api/jobs/${encodeURIComponent(jobName)}`,
};
let tritonRepositoryModels = [];
const selectedTritonModels = new Set();
let refreshInFlight = false;
let autoRefreshTimerId = null;
let operationInFlightCount = 0;
let operationTimerId = null;
let operationStartedAt = 0;
let currentOperationLabel = "";
let currentOperationButton = null;
let collapsiblePanelsInitialized = false;

const INTERACTIVE_BUTTON_IDS = [
    "refresh-all-btn",
    "load-model-btn",
    "load-batch-btn",
    "job-status-btn",
    "unload-model-btn",
    "reload-model-btn",
    "load-sample-btn",
    "save-triton-url-btn",
    "reset-triton-url-btn",
    "bulk-unload-triton-btn",
];

function getOverrideHeaders() {
    const headers = {};
    const tritonUrl = localStorage.getItem(TRITON_URL_STORAGE_KEY)?.trim();
    const metricsPort = localStorage.getItem(METRICS_PORT_STORAGE_KEY)?.trim();
    if (tritonUrl) {
        headers["x-hot-triton-url"] = tritonUrl;
    }
    if (metricsPort) {
        headers["x-hot-triton-metrics-port"] = metricsPort;
    }
    return headers;
}

async function fetchJson(url, options = {}) {
    const headers = {
        "Content-Type": "application/json",
        ...getOverrideHeaders(),
        ...(options.headers || {}),
    };
    const response = await fetch(url, { ...options, headers });
    const payload = await response.json().catch(() => ({ success: false, detail: "非 JSON 响应" }));
    if (!response.ok) {
        throw new Error(payload.detail || payload.error || `请求失败: ${response.status}`);
    }
    return payload;
}

function renderJson(payload) {
    resultPanel.textContent = JSON.stringify(payload, null, 2);
}

function assertBusinessSuccess(result, fallbackMessage = "操作失败") {
    if (!result || result.success !== false) {
        return;
    }
    const errors = Array.isArray(result.errors)
        ? result.errors
            .map((item) => item?.error || item?.detail || item?.status || "")
            .filter(Boolean)
        : [];
    const error = new Error(errors.join("；") || result.detail || result.error || fallbackMessage);
    error.payload = result;
    throw error;
}

function clearFieldOptimistically(field) {
    if (!(field instanceof HTMLInputElement) && !(field instanceof HTMLTextAreaElement)) {
        return () => {};
    }

    const previousValue = field.value;
    field.value = "";
    return () => {
        field.value = previousValue;
    };
}

function formatDurationLabel(startedAt) {
    if (!startedAt) {
        return "0s";
    }

    const seconds = Math.max(0, Math.floor((Date.now() - startedAt) / 1000));
    if (seconds < 60) {
        return `${seconds}s`;
    }

    const minutes = Math.floor(seconds / 60);
    const remainSeconds = seconds % 60;
    return `${minutes}m ${remainSeconds}s`;
}

function toFiniteNumber(value) {
    return Number.isFinite(value) ? value : null;
}

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function formatCount(value, suffix = "") {
    const normalized = toFiniteNumber(value);
    return normalized === null ? "-" : `${normalized.toLocaleString("zh-CN")}${suffix}`;
}

function formatPercent(value, digits = 0) {
    const normalized = toFiniteNumber(value);
    return normalized === null ? "-" : `${normalized.toFixed(digits)}%`;
}

function formatWatts(value, digits = 1) {
    const normalized = toFiniteNumber(value);
    return normalized === null ? "-" : `${normalized.toFixed(digits)} W`;
}

function formatMemoryMb(value) {
    const normalized = toFiniteNumber(value);
    return normalized === null ? "-" : `${normalized.toLocaleString("zh-CN")} MB`;
}

function bytesToMb(value) {
    const normalized = toFiniteNumber(value);
    return normalized === null ? null : normalized / (1024 * 1024);
}

function formatMemoryGb(value, digits = 1) {
    const normalized = toFiniteNumber(value);
    if (normalized === null) {
        return "-";
    }

    return `${(normalized / 1024).toLocaleString("zh-CN", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
    })} GB`;
}

function formatTimestamp(value) {
    if (!value) {
        return "-";
    }

    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return String(value);
    }

    return date.toLocaleString("zh-CN", { hour12: false });
}

function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => {
        const entities = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;",
        };
        return entities[char] || char;
    });
}

function getStatusTone(value) {
    const normalized = String(value ?? "").trim().toUpperCase();
    if (!normalized || normalized === "-") {
        return "neutral";
    }

    if (normalized === "READY" || normalized === "OK" || normalized === "MODEL_READY") {
        return "positive";
    }

    return "negative";
}

function renderStatusValue(value) {
    const text = String(value ?? "-").trim() || "-";
    const tone = getStatusTone(text);
    if (tone === "neutral") {
        return escapeHtml(text);
    }

    return `<span class="status-indicator status-${tone}">${escapeHtml(text)}</span>`;
}

function getFilteredTritonModels() {
    const query = tritonModelFilterInput?.value.trim().toLowerCase() || "";
    if (!query) {
        return tritonRepositoryModels;
    }

    return tritonRepositoryModels.filter((item) => {
        return [item.name, item.version, item.state, item.reason].some((value) =>
            String(value || "").toLowerCase().includes(query),
        );
    });
}

function getUniqueModelNames(models) {
    return Array.from(
        new Set(
            models
                .map((item) => String(item.name || "").trim())
                .filter(Boolean),
        ),
    );
}

function syncSelectedTritonModels() {
    const availableModels = new Set(getUniqueModelNames(tritonRepositoryModels));
    selectedTritonModels.forEach((modelName) => {
        if (!availableModels.has(modelName)) {
            selectedTritonModels.delete(modelName);
        }
    });
}

function updateTritonSelectionControls(visibleModels) {
    const visibleModelNames = getUniqueModelNames(visibleModels);
    const selectedVisibleCount = visibleModelNames.filter((modelName) => selectedTritonModels.has(modelName)).length;

    if (tritonSelectionSummary) {
        tritonSelectionSummary.textContent = `当前筛选 ${visibleModels.length} 行 / ${visibleModelNames.length} 个模型 · 已选 ${selectedTritonModels.size} 个模型`;
    }

    if (bulkUnloadTritonBtn) {
        bulkUnloadTritonBtn.disabled = selectedTritonModels.size === 0;
    }

    if (selectAllTritonModelsCheckbox) {
        selectAllTritonModelsCheckbox.disabled = visibleModelNames.length === 0;
        selectAllTritonModelsCheckbox.checked =
            visibleModelNames.length > 0 && selectedVisibleCount === visibleModelNames.length;
        selectAllTritonModelsCheckbox.indeterminate =
            selectedVisibleCount > 0 && selectedVisibleCount < visibleModelNames.length;
    }
}

function setInteractiveButtonsDisabled(disabled) {
    INTERACTIVE_BUTTON_IDS.forEach((id) => {
        const element = document.getElementById(id);
        if (!(element instanceof HTMLButtonElement)) {
            return;
        }
        element.disabled = disabled;
    });
}

function clearPendingButtonState() {
    if (!(currentOperationButton instanceof HTMLButtonElement)) {
        currentOperationButton = null;
        return;
    }

    if (currentOperationButton.dataset.originalLabel) {
        currentOperationButton.textContent = currentOperationButton.dataset.originalLabel;
        delete currentOperationButton.dataset.originalLabel;
    }
    currentOperationButton.classList.remove("is-pending");
    currentOperationButton = null;
}

function setPendingButtonState(buttonId) {
    clearPendingButtonState();

    const button = buttonId ? document.getElementById(buttonId) : null;
    if (!(button instanceof HTMLButtonElement)) {
        return;
    }

    currentOperationButton = button;
    button.dataset.originalLabel = button.textContent || "";
    button.textContent = `${button.dataset.originalLabel}...`;
    button.classList.add("is-pending");
}

function stopOperationTicker() {
    if (operationTimerId !== null) {
        window.clearInterval(operationTimerId);
        operationTimerId = null;
    }
}

function setCollapsiblePanelExpanded(button, expanded) {
    if (!(button instanceof HTMLButtonElement)) {
        return;
    }

    const targetId = button.dataset.collapseTarget || "";
    const target = targetId ? document.getElementById(targetId) : null;
    const panel = button.closest(".panel-collapsible");
    const label = button.querySelector(".panel-collapse-label");

    button.setAttribute("aria-expanded", expanded ? "true" : "false");
    if (target) {
        target.hidden = !expanded;
    }
    if (panel) {
        panel.classList.toggle("is-collapsed", !expanded);
    }
    if (label) {
        label.textContent = expanded ? "收起" : "展开";
    }
}

function initCollapsiblePanels() {
    if (collapsiblePanelsInitialized) {
        return;
    }

    document.querySelectorAll(".panel-collapse-toggle").forEach((button) => {
        if (!(button instanceof HTMLButtonElement)) {
            return;
        }

        setCollapsiblePanelExpanded(button, false);
        button.addEventListener("click", () => {
            const expanded = button.getAttribute("aria-expanded") === "true";
            setCollapsiblePanelExpanded(button, !expanded);
        });
    });

    collapsiblePanelsInitialized = true;
}

function renderIdleOperationState() {
    stopOperationTicker();
    currentOperationLabel = "";
    clearPendingButtonState();
    if (operationStatusPanel) {
        operationStatusPanel.classList.remove("is-running", "is-success", "is-error");
        operationStatusPanel.classList.add("is-idle");
    }
    if (operationStatusCaption) {
        operationStatusCaption.textContent = "等待操作。";
    }
    if (operationStatusBadge) {
        operationStatusBadge.textContent = "Idle";
        operationStatusBadge.className = "mini-badge muted";
    }
    if (operationStatusTitle) {
        operationStatusTitle.textContent = "暂无进行中的任务";
    }
    if (operationStatusDetail) {
        operationStatusDetail.textContent = "提交加载、卸载或查询后，这里会显示当前进度。";
    }
}

function renderRunningOperationState(label) {
    currentOperationLabel = label;
    operationStartedAt = Date.now();
    if (operationStatusPanel) {
        operationStatusPanel.classList.remove("is-idle", "is-success", "is-error");
        operationStatusPanel.classList.add("is-running");
    }
    if (operationStatusCaption) {
        operationStatusCaption.textContent = "请求已发出，正在等待服务端完成。";
    }
    if (operationStatusBadge) {
        operationStatusBadge.textContent = "Running";
        operationStatusBadge.className = "mini-badge active";
    }
    if (operationStatusTitle) {
        operationStatusTitle.textContent = label;
    }
    if (operationStatusDetail) {
        operationStatusDetail.textContent = `已运行 ${formatDurationLabel(operationStartedAt)}`;
    }

    stopOperationTicker();
    operationTimerId = window.setInterval(() => {
        if (operationStatusDetail) {
            operationStatusDetail.textContent = `已运行 ${formatDurationLabel(operationStartedAt)}`;
        }
    }, 1000);
}

function renderFinishedOperationState(type, detail) {
    stopOperationTicker();
    if (operationStatusPanel) {
        operationStatusPanel.classList.remove("is-idle", "is-running", "is-success", "is-error");
        operationStatusPanel.classList.add(type === "success" ? "is-success" : "is-error");
    }
    if (operationStatusCaption) {
        operationStatusCaption.textContent = type === "success" ? "本次操作已完成。" : "本次操作失败。";
    }
    if (operationStatusBadge) {
        operationStatusBadge.textContent = type === "success" ? "Done" : "Error";
        operationStatusBadge.className = `mini-badge ${type === "success" ? "active" : "error"}`;
    }
    if (operationStatusTitle) {
        operationStatusTitle.textContent = currentOperationLabel || (type === "success" ? "操作完成" : "操作失败");
    }
    if (operationStatusDetail) {
        operationStatusDetail.textContent = detail || (type === "success" ? "已完成。" : "请求失败。");
    }
}

function buildGpuStatusPayloadFromMetrics(metrics) {
    const normalizedMetrics = metrics && typeof metrics === "object" ? metrics : {};
    const available = Boolean(normalizedMetrics.available);
    const gpus = Array.isArray(normalizedMetrics.gpus) ? normalizedMetrics.gpus : [];

    return {
        status: available ? "OK" : "UNAVAILABLE",
        detail: normalizedMetrics.detail || (available ? "OK" : "未获取到 GPU 指标"),
        source_url: normalizedMetrics.url || "",
        updated_at: normalizedMetrics.updated_at || "",
        gpus: gpus.map((gpu) => ({
            gpu_index: gpu.index,
            gpu_uuid: gpu.gpu_uuid,
            gpu_bus_id: gpu.gpu_bus_id,
            memory_total_mb: bytesToMb(gpu.total_bytes),
            memory_used_mb: bytesToMb(gpu.used_bytes),
            memory_free_mb:
                toFiniteNumber(gpu.total_bytes) !== null && toFiniteNumber(gpu.used_bytes) !== null
                    ? bytesToMb(gpu.total_bytes - gpu.used_bytes)
                    : null,
            memory_used_percent: gpu.used_percent,
            gpu_utilization_percent: gpu.utilization_percent,
            power_draw_w: gpu.power_usage_watts,
            power_limit_w: null,
            temperature_c: null,
        })),
    };
}

function renderGpuStatus(payload) {
    const status = payload?.status || "UNAVAILABLE";
    const gpus = Array.isArray(payload?.gpus) ? payload.gpus : [];
    const detail = payload?.detail || "-";
    const sourceUrl = payload?.source_url || "-";
    const updatedAt = payload?.updated_at || "";

    if (gpuMetricsUpdated) {
        gpuMetricsUpdated.textContent = updatedAt ? `更新于 ${formatTimestamp(updatedAt)}` : "等待指标...";
    }

    if (!gpuMonitorPanel) {
        return;
    }

    if (status !== "OK") {
        gpuMonitorPanel.innerHTML = `
            <div class="gpu-error">
                <strong>GPU 指标不可用</strong>
                <span>${escapeHtml(detail)}</span>
                <span class="mono">Source: ${escapeHtml(sourceUrl)}</span>
            </div>
        `;
        return;
    }

    if (!gpus.length) {
        gpuMonitorPanel.innerHTML = `
            <div class="gpu-error">
                <strong>未发现 GPU 设备</strong>
                <span>${escapeHtml(detail)}</span>
                <span class="mono">Source: ${escapeHtml(sourceUrl)}</span>
            </div>
        `;
        return;
    }

    const deviceCount = gpus.length;
    const totalMemoryMb = gpus.reduce((sum, gpu) => sum + (toFiniteNumber(gpu.memory_total_mb) || 0), 0);
    const usedMemoryMb = gpus.reduce((sum, gpu) => sum + (toFiniteNumber(gpu.memory_used_mb) || 0), 0);
    const avgGpuUtilization = gpus.reduce((sum, gpu) => sum + (toFiniteNumber(gpu.gpu_utilization_percent) || 0), 0) / deviceCount;
    const totalPowerDraw = gpus.reduce((sum, gpu) => sum + (toFiniteNumber(gpu.power_draw_w) || 0), 0);
    const memoryUsedPercent = totalMemoryMb > 0 ? (usedMemoryMb / totalMemoryMb) * 100 : null;

    gpuMonitorPanel.innerHTML = `
        <div class="gpu-summary-strip">
            <article class="gpu-kpi gpu-kpi-primary">
                <strong>${formatMemoryGb(usedMemoryMb)} / ${formatMemoryGb(totalMemoryMb)}</strong>
                <span>总显存占用 · ${formatPercent(memoryUsedPercent, 1)}</span>
            </article>
            <article class="gpu-kpi gpu-kpi-count">
                <strong>${deviceCount} 张 GPU 在线</strong>
                <span>${escapeHtml(detail)}</span>
                <span class="mono">Source: ${escapeHtml(sourceUrl)}</span>
            </article>
            <article class="gpu-kpi">
                <strong>${formatPercent(avgGpuUtilization, 1)}</strong>
                <span>平均 GPU 利用率</span>
            </article>
            <article class="gpu-kpi">
                <strong>${formatWatts(totalPowerDraw, 1)}</strong>
                <span>总功耗</span>
            </article>
        </div>
        <div class="gpu-card-grid">
            ${gpus
                .map((gpu) => {
                    const gpuIndex = gpu.gpu_index ?? "-";
                    const uuid = gpu.gpu_uuid || gpu.gpu_bus_id || "-";
                    const usedPercent = toFiniteNumber(gpu.memory_used_percent);
                    const barWidth = usedPercent === null ? 0 : clamp(usedPercent, 0, 100);
                    const gpuUtilization = toFiniteNumber(gpu.gpu_utilization_percent);
                    return `
                        <article class="gpu-card">
                            <div class="gpu-card-head">
                                <div>
                                    <div class="gpu-card-name">GPU ${escapeHtml(gpuIndex)}</div>
                                    <div class="gpu-card-uuid mono">${escapeHtml(uuid)}</div>
                                </div>
                                <span class="mini-badge ${gpuUtilization !== null && gpuUtilization > 0 ? "active" : "muted"}">
                                    利用率 ${formatPercent(gpuUtilization, 1)}
                                </span>
                            </div>
                            <div class="gpu-bar">
                                <span class="gpu-bar-fill" style="width: ${barWidth}%;"></span>
                            </div>
                            <div class="gpu-card-stats">
                                <div class="gpu-stat">
                                    <strong>${formatMemoryGb(gpu.memory_used_mb)} / ${formatMemoryGb(gpu.memory_total_mb)}</strong>
                                    <span>显存占用</span>
                                </div>
                                <div class="gpu-stat">
                                    <strong>${formatMemoryGb(gpu.memory_free_mb)}</strong>
                                    <span>剩余显存</span>
                                </div>
                                <div class="gpu-stat">
                                    <strong>${formatPercent(gpu.gpu_utilization_percent, 1)}</strong>
                                    <span>GPU 利用率</span>
                                </div>
                                <div class="gpu-stat">
                                    <strong>${formatWatts(gpu.power_draw_w, 1)}</strong>
                                    <span>功耗</span>
                                </div>
                            </div>
                        </article>
                    `;
                })
                .join("")}
        </div>
    `;
}

function renderStatusSummary(payload) {
    const triton = payload.triton || {};
    const manager = payload.manager || {};
    const config = manager.config || {};
    const activeJobs = manager.active_jobs || [];
    const repositoryModels = Array.isArray(triton.repository_models) ? triton.repository_models : [];
    const runningModelCount = new Set(
        repositoryModels
            .filter((item) => String(item?.state || "").toUpperCase() === "READY")
            .map((item) => String(item?.name || "").trim())
            .filter(Boolean),
    ).size;

    statusSummary.innerHTML = `
        <div class="summary-card">
            <h3>Triton</h3>
            <p><strong>URL:</strong> ${triton.url || "-"}</p>
            <p><strong>Ready:</strong> ${renderStatusValue(triton.ready ? "READY" : "NOT READY")}</p>
            <p><strong>运行模型:</strong> ${runningModelCount}</p>
            <p><strong>Detail:</strong> ${renderStatusValue(triton.detail || "-")}</p>
        </div>
        <div class="summary-card">
            <h3>Controller</h3>
            <p><strong>Namespace:</strong> ${config.k8s_namespace || "-"}</p>
            <p><strong>PVC:</strong> ${config.triton_repository_pvc || "-"}</p>
            <p><strong>活动 Job:</strong> ${activeJobs.length}</p>
        </div>
        <div class="summary-card">
            <h3>模型仓库</h3>
            <p><strong>路径:</strong> ${config.model_repository || "-"}</p>
            <p><strong>目标挂载:</strong> ${config.model_target_path || "-"}</p>
            <p><strong>已管理模型:</strong> ${manager.managed_model_count || 0}</p>
        </div>
    `;
}

function renderManagedModels(payload) {
    const managed = payload.managed || payload.manager || payload;
    const images = managed.managed_images || [];
    if (!images.length) {
        managedModelsPanel.innerHTML = '<div class="empty">暂无已管理模型</div>';
        return;
    }

    managedModelsPanel.innerHTML = images
        .map((item) => {
            const models = (item.models || []).map((name) => `<li>${name}</li>`).join("");
            return `
                <article class="managed-card">
                    <h3>${item.image || "-"}</h3>
                    <p><strong>更新时间:</strong> ${item.updated_at || "-"}</p>
                    <ul>${models}</ul>
                </article>
            `;
        })
        .join("");
}

function renderJobs(payload) {
    const jobs = payload.manager?.jobs || payload.managed?.jobs || payload.jobs || {};
    const entries = Object.entries(jobs).sort((a, b) => {
        const aTime = a[1]?.updated_at || "";
        const bTime = b[1]?.updated_at || "";
        return aTime < bTime ? 1 : -1;
    });
    if (!entries.length) {
        jobListPanel.innerHTML = '<div class="empty">暂无 Job 记录</div>';
        return;
    }

    const visibleEntries = entries.slice(0, MAX_JOB_LIST_ITEMS);
    const hiddenCount = Math.max(entries.length - visibleEntries.length, 0);

    jobListPanel.innerHTML = `
        ${
            hiddenCount
                ? `<div class="job-list-limit-note">仅显示最近 ${MAX_JOB_LIST_ITEMS} 条，另有 ${hiddenCount} 条未展示</div>`
                : ""
        }
        ${visibleEntries
        .map(([jobName, meta]) => {
            const detail = meta.detail || meta.error || "-";
            return `
                <article class="managed-card">
                    <h3>${jobName}</h3>
                    <p><strong>模型:</strong> ${meta.model_name || "-"}</p>
                    <p><strong>状态:</strong> ${renderStatusValue(meta.status || "-")}</p>
                    <p><strong>Pod:</strong> ${meta.pod_name || "-"}</p>
                    <p><strong>详情:</strong> ${detail}</p>
                </article>
            `;
        })
        .join("")}
    `;
}

function renderTritonModels(payload) {
    if (payload) {
        const models = payload.triton_models || payload.triton?.repository_models || [];
        tritonRepositoryModels = Array.isArray(models) ? models : [];
        syncSelectedTritonModels();
    }

    const visibleModels = getFilteredTritonModels();
    updateTritonSelectionControls(visibleModels);

    if (!tritonRepositoryModels.length) {
        tritonModelBody.innerHTML = '<tr><td colspan="5" class="empty">暂无 Triton 模型信息</td></tr>';
        return;
    }

    if (!visibleModels.length) {
        tritonModelBody.innerHTML = '<tr><td colspan="5" class="empty">没有匹配的 Triton 模型</td></tr>';
        return;
    }

    tritonModelBody.innerHTML = visibleModels
        .map((item) => {
            const modelName = String(item.name || "").trim();
            const encodedModelName = encodeURIComponent(modelName);
            const isSelected = modelName ? selectedTritonModels.has(modelName) : false;
            const state = String(item.state || "-");
            return `
                <tr class="${isSelected ? "triton-row-selected" : ""}">
                    <td class="triton-select-cell">
                        <input
                            class="triton-checkbox triton-row-checkbox"
                            type="checkbox"
                            data-model-name="${encodedModelName}"
                            ${isSelected ? "checked" : ""}
                            ${modelName ? "" : "disabled"}
                        >
                    </td>
                    <td>${escapeHtml(item.name || "-")}</td>
                    <td>${escapeHtml(item.version || "-")}</td>
                    <td>${renderStatusValue(state)}</td>
                    <td>${escapeHtml(item.reason || "-")}</td>
                </tr>
            `;
        })
        .join("");
}

async function refreshAll({ background = false } = {}) {
    if (refreshInFlight) {
        return;
    }

    refreshInFlight = true;
    try {
        const statusPayload = await fetchJson(API_ROUTES.status);
        renderStatusSummary(statusPayload);
        renderJobs(statusPayload);
        renderManagedModels(statusPayload);
        renderTritonModels(statusPayload);
        renderGpuStatus(buildGpuStatusPayloadFromMetrics(statusPayload?.triton?.metrics));
        return statusPayload;
    } catch (error) {
        if (!background) {
            throw error;
        }
    } finally {
        refreshInFlight = false;
    }
}

function startAutoRefreshLoop() {
    const tick = async () => {
        if (operationInFlightCount === 0 && !refreshInFlight) {
            await refreshAll({ background: true });
        }
        autoRefreshTimerId = window.setTimeout(tick, AUTO_REFRESH_INTERVAL_MS);
    };

    if (autoRefreshTimerId !== null) {
        window.clearTimeout(autoRefreshTimerId);
    }
    autoRefreshTimerId = window.setTimeout(tick, AUTO_REFRESH_INTERVAL_MS);
}

async function submitSingleLoad() {
    const image = singleImageInput.value.trim();
    if (!image) {
        throw new Error("请先输入 image");
    }

    const restoreInput = clearFieldOptimistically(singleImageInput);
    try {
        const result = await fetchJson(API_ROUTES.loadModel, {
            method: "POST",
            body: JSON.stringify({ image, wait_for_ready: false }),
        });
        if (result.job_name) {
            jobNameInput.value = result.job_name;
        }
        renderJson(result);
        await refreshAll();
    } catch (error) {
        restoreInput();
        throw error;
    }
}

async function submitBatchLoad() {
    const payload = {
        ...JSON.parse(batchInput.value),
        wait_for_ready: false,
    };
    const restoreInput = clearFieldOptimistically(batchInput);
    try {
        const result = await fetchJson(API_ROUTES.loadModelBatch, {
            method: "POST",
            body: JSON.stringify(payload),
        });
        assertBusinessSuccess(result, "批量任务提交失败");
        renderJson(result);
        await refreshAll();
    } catch (error) {
        restoreInput();
        throw error;
    }
}

async function queryJobStatus() {
    const jobName = jobNameInput.value.trim();
    if (!jobName) {
        throw new Error("请先输入 job_name");
    }
    const result = await fetchJson(API_ROUTES.jobStatus(jobName));
    renderJson(result);
    await refreshAll();
}

async function unloadModel() {
    const modelName = actionModelNameInput.value.trim();
    if (!modelName) {
        throw new Error("请先输入 model_name");
    }

    const restoreInput = clearFieldOptimistically(actionModelNameInput);
    try {
        const result = await fetchJson(API_ROUTES.unloadModel, {
            method: "POST",
            body: JSON.stringify({ model_name: modelName }),
        });
        renderJson(result);
        await refreshAll();
    } catch (error) {
        restoreInput();
        throw error;
    }
}

async function unloadSelectedTritonModels() {
    const models = Array.from(selectedTritonModels).sort();
    if (!models.length) {
        throw new Error("请先选择要热卸载的模型");
    }

    if (typeof window !== "undefined") {
        const confirmed = window.confirm(`确认热卸载这 ${models.length} 个模型吗？`);
        if (!confirmed) {
            return;
        }
    }

    const result = await fetchJson(API_ROUTES.unloadSelection, {
        method: "POST",
        body: JSON.stringify({ models }),
    });
    assertBusinessSuccess(result, "批量热卸载失败");
    selectedTritonModels.clear();
    renderJson(result);
    await refreshAll();
}

async function reloadModel() {
    const modelName = actionModelNameInput.value.trim();
    if (!modelName) {
        throw new Error("请先输入 model_name");
    }

    const restoreInput = clearFieldOptimistically(actionModelNameInput);
    try {
        const result = await fetchJson(API_ROUTES.reloadModel, {
            method: "POST",
            body: JSON.stringify({ model_name: modelName }),
        });
        renderJson(result);
        await refreshAll();
    } catch (error) {
        restoreInput();
        throw error;
    }
}

function loadSampleBatch() {
    batchInput.value = JSON.stringify(
        {
            models: [
                {
                    image: "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430",
                },
                {
                    image: "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_hanging_product_yolov5-20230620",
                },
            ],
        },
        null,
        2,
    );
}

function saveOverrides() {
    localStorage.setItem(TRITON_URL_STORAGE_KEY, tritonUrlInput.value.trim());
    localStorage.setItem(METRICS_PORT_STORAGE_KEY, metricsPortInput.value.trim());
}

function resetOverrides() {
    localStorage.removeItem(TRITON_URL_STORAGE_KEY);
    localStorage.removeItem(METRICS_PORT_STORAGE_KEY);
    tritonUrlInput.value = "";
    metricsPortInput.value = "";
}

async function withResult(action, pendingMessage = "", options = {}) {
    const {
        operationLabel = pendingMessage || "正在执行操作...",
        buttonId = null,
        renderActionResult = false,
        renderPendingResult = Boolean(pendingMessage),
        successDetail = "服务端已经返回结果，页面状态已同步刷新。",
    } = options;
    operationInFlightCount += 1;
    setPendingButtonState(buttonId);
    setInteractiveButtonsDisabled(true);
    renderRunningOperationState(operationLabel);
    if (renderPendingResult && pendingMessage) {
        renderJson({ success: true, detail: pendingMessage });
    }
    try {
        const actionResult = await action();
        if (renderActionResult && actionResult !== undefined) {
            renderJson(actionResult);
        }
        renderFinishedOperationState("success", successDetail);
    } catch (error) {
        renderJson(error.payload || { success: false, detail: error.message });
        renderFinishedOperationState("error", error.message);
    } finally {
        operationInFlightCount = Math.max(0, operationInFlightCount - 1);
        setInteractiveButtonsDisabled(false);
        clearPendingButtonState();
        updateTritonSelectionControls(getFilteredTritonModels());
    }
}

document.getElementById("refresh-all-btn")?.addEventListener("click", () =>
    withResult(() => refreshAll(), "", {
        operationLabel: "刷新控制器与 Triton 状态",
        buttonId: "refresh-all-btn",
        renderActionResult: true,
        successDetail: "控制器与 Triton 状态已刷新。",
    }),
);
document.getElementById("load-model-btn")?.addEventListener("click", () =>
    withResult(submitSingleLoad, "正在创建加载任务...", {
        operationLabel: "创建单模型加载任务",
        buttonId: "load-model-btn",
        successDetail: "任务已提交，后续进度会在 Job 列表中继续刷新。",
    }),
);
document.getElementById("load-batch-btn")?.addEventListener("click", () =>
    withResult(submitBatchLoad, "正在批量创建任务...", {
        operationLabel: "创建批量加载任务",
        buttonId: "load-batch-btn",
        successDetail: "批量任务已提交，后续进度会在 Job 列表中继续刷新。",
    }),
);
document.getElementById("job-status-btn")?.addEventListener("click", () =>
    withResult(queryJobStatus, "正在查询 Job 状态...", { operationLabel: "查询指定 Job 状态", buttonId: "job-status-btn" }),
);
document.getElementById("unload-model-btn")?.addEventListener("click", () =>
    withResult(unloadModel, "正在卸载模型...", { operationLabel: "执行模型卸载", buttonId: "unload-model-btn" }),
);
document.getElementById("reload-model-btn")?.addEventListener("click", () =>
    withResult(reloadModel, "正在重载模型...", { operationLabel: "执行模型重载", buttonId: "reload-model-btn" }),
);
document.getElementById("load-sample-btn")?.addEventListener("click", loadSampleBatch);
document.getElementById("save-triton-url-btn")?.addEventListener("click", saveOverrides);
document.getElementById("reset-triton-url-btn")?.addEventListener("click", resetOverrides);
bulkUnloadTritonBtn?.addEventListener("click", () =>
    withResult(unloadSelectedTritonModels, "正在批量热卸载模型...", {
        operationLabel: "执行批量热卸载",
        buttonId: "bulk-unload-triton-btn",
    }),
);
tritonModelFilterInput?.addEventListener("input", () => renderTritonModels());
clearTritonModelFilterBtn?.addEventListener("click", () => {
    tritonModelFilterInput.value = "";
    renderTritonModels();
});
selectAllTritonModelsCheckbox?.addEventListener("change", () => {
    const visibleModelNames = getUniqueModelNames(getFilteredTritonModels());
    visibleModelNames.forEach((modelName) => {
        if (selectAllTritonModelsCheckbox.checked) {
            selectedTritonModels.add(modelName);
        } else {
            selectedTritonModels.delete(modelName);
        }
    });
    renderTritonModels();
});
tritonModelBody?.addEventListener("change", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement) || !target.classList.contains("triton-row-checkbox")) {
        return;
    }

    const modelName = decodeURIComponent(target.dataset.modelName || "");
    if (!modelName) {
        return;
    }

    if (target.checked) {
        selectedTritonModels.add(modelName);
    } else {
        selectedTritonModels.delete(modelName);
    }
    renderTritonModels();
});

tritonUrlInput.value = localStorage.getItem(TRITON_URL_STORAGE_KEY) || "";
metricsPortInput.value = localStorage.getItem(METRICS_PORT_STORAGE_KEY) || "";
loadSampleBatch();
initCollapsiblePanels();
renderIdleOperationState();
refreshAll().catch((error) => {
    renderJson({ success: false, detail: error.message });
    renderFinishedOperationState("error", error.message);
});
startAutoRefreshLoop();
