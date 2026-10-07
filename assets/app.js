"use strict";
const $ = id => document.getElementById(id);
const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
const icon = name => `<i data-icon="${name}" aria-hidden="true"></i>`;

function hydrateIcons(root = document) {
  root.querySelectorAll("[data-icon]").forEach(el => {
    el.style.setProperty("--icon", `url('/assets/icons/${el.dataset.icon}.svg')`);
    el.setAttribute("aria-hidden", "true");
  });
}

function loadStorage(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
}
function saveStorage(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* ignore */ }
}

let currentPdf = { sourceType: "sample", filename: "" };
let yoloCatalog = [], storeFiles = [], lastData = null, currentZoom = 1, activeId = null;
let pageIndex = 0, totalPages = 1, scanning = false, regionsVisible = true, activeTab = "results";
let reviewState = loadStorage("drawing-ocr.reviews", {});
let scanHistory = loadStorage("drawing-ocr.history", []);
const metadata = loadStorage("drawing-ocr.metadata", {});

function setStatus(text, kind = "") {
  if ($("statusText")) {
    $("statusText").textContent = text;
    $("statusText").title = text;
  }
  if ($("statusMain")) $("statusMain").className = `status-chip ${kind}`;
}

async function requestJson(url, payload) {
  const response = await fetch(url, payload === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const data = await response.json();
  if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function currentReviewKey(b) {
  return JSON.stringify([currentPdf.sourceType, currentPdf.filename, pageIndex, b.id, b.x, b.y, b.w, b.h, b._originalText]);
}

function persistBox(b) {
  reviewState[currentReviewKey(b)] = {
    included: b.included, reviewed: b.reviewed, important: b.important,
    edited: b.edited, semantics: b.semantics, ocrText: b.ocrText
  };
  saveStorage("drawing-ocr.reviews", reviewState);
}

function formatYoloLabel(name) {
  if (name === "yolo_best_1class071026.onnx") return "yolo_best_1class071026.onnx (Mới - YOLO11s 07/10)";
  if (name === "yolo_best_1class.onnx") return "yolo_best_1class.onnx (Cũ - best v1)";
  return name;
}

function updateActiveModelBadge() {
  const val = $("yoloSelect")?.value || "";
  if (!$("activeModelBadge")) return;
  if (val.includes("071026")) $("activeModelBadge").textContent = "YOLO11s 07/10";
  else if (val.includes("1class")) $("activeModelBadge").textContent = "best v1";
  else $("activeModelBadge").textContent = val.replace(/\.onnx$/i, "") || "YOLO";
}

function update3RegionLabel() {
  if ($("enable3RegionCheck")) {
    $("enable3RegionCheck").checked = true;
    $("enable3RegionCheck").disabled = true;
  }
  if ($("enable3RegionLabel")) {
    $("enable3RegionLabel").textContent = "Luôn bật — Tự động tách Nominal / Tol+ / Tol− trong từng Blue-Box";
  }
  updateActiveModelBadge();
}

function fillSelect(el, models, selected, isYolo = false) {
  if (!el) return;
  el.replaceChildren();
  for (const model of models) {
    const label = isYolo ? formatYoloLabel(model.name) : model.name;
    el.add(new Option(label, model.name));
  }
  if (models.some(m => m.name === selected)) el.value = selected;
}

function selectDrawing(sourceType, filename) {
  if (!filename) return;
  const value = `${sourceType}:${filename}`;
  if ($("pdfSelect")) {
    if (![...$("pdfSelect").options].some(o => o.value === value)) {
      $("pdfSelect").add(new Option(filename, value), 0);
    }
    $("pdfSelect").value = value;
  }
  currentPdf = { sourceType, filename };
  $("documentTitle").textContent = filename;
  $("documentTitle").title = filename;
  const baseName = filename.replace(/\.[^.]+$/, "");
  if ($("partName")) $("partName").value = metadata[filename]?.partName || baseName;
  if ($("moldName")) $("moldName").value = metadata[filename]?.moldName || "";
}

function updateDocument() {
  const value = $("pdfSelect")?.value || "";
  const colon = value.indexOf(":");
  if (colon < 0) return;
  selectDrawing(value.slice(0, colon), value.slice(colon + 1));
}

function updatePageControls() {
  $("pageText").textContent = `${pageIndex + 1} / ${totalPages}`;
  $("prevPage").disabled = scanning || pageIndex <= 0;
  $("nextPage").disabled = scanning || pageIndex + 1 >= totalPages;
}

function setBusy(value) {
  scanning = value;
  for (const id of ["scanBtn", "uploadBtn", "modelBtn", "storesBtn", "settingsBtn", "yoloSelect", "ppocrSelect"]) {
    if ($(id)) $(id).disabled = value;
  }
  $("scanBtn").classList.toggle("busy", value);
  $("scanBtn").innerHTML = value
    ? `${icon("loader-circle")}<span>Đang quét…</span>`
    : `${icon("play")}<span>Scan</span><kbd>F5</kbd>`;
  $("scanOverlay").hidden = !value;
  $("exportBtn").disabled = value || !lastData?.boxes?.length || lastData.fileName !== currentPdf.filename;
  updatePageControls();
  if (!value) update3RegionLabel();
  hydrateIcons($("scanBtn"));
}

async function runScan() {
  if (scanning) return;
  updateDocument();
  if (!currentPdf.filename || !$("yoloSelect")?.value || !$("ppocrSelect")?.value) {
    setStatus("Hãy kéo thả bản vẽ hoặc chọn trong Stores để quét");
    return;
  }
  setBusy(true);
  setStatus(`Đang quét ${currentPdf.filename}…`);
  try {
    const data = await requestJson("/api/run", {
      ...currentPdf,
      pageIndex,
      yoloModel: $("yoloSelect").value,
      ppocrModel: $("ppocrSelect").value,
      enableStackSplit: true,
      preprocessMode: "raw"
    });
    lastData = data;
    totalPages = data.page?.totalPages || 1;
    activeId = null;
    for (const b of data.boxes || []) {
      b._originalText = b.ocrText || "";
      Object.assign(b, { included: true, reviewed: false, important: false, edited: false });
      const saved = reviewState[currentReviewKey(b)];
      if (saved) {
        b.included = saved.included !== false;
        b.reviewed = !!saved.reviewed;
        b.important = !!saved.important;
        if (saved.edited) Object.assign(b, { edited: true, semantics: saved.semantics, ocrText: saved.ocrText });
      }
    }
    renderResult(data);
    const entry = {
      ...currentPdf,
      pageIndex,
      boxes: data.boxCount,
      timestamp: new Date().toISOString(),
      yolo: $("yoloSelect").value,
      ppocr: $("ppocrSelect").value,
      split: true
    };
    scanHistory = [entry, ...scanHistory].slice(0, 25);
    saveStorage("drawing-ocr.history", scanHistory);
    setStatus(`Hoàn tất · ${data.boxCount} vùng`, "success");
  } catch (error) {
    setStatus(`Lỗi: ${error.message}`, "error");
  } finally {
    setBusy(false);
  }
}

function renderResult(data) {
  const boxes = data.boxes || [];
  $("resultCount").textContent = $("cropCount").textContent = `(${boxes.length})`;
  const timing = value => Number.isFinite(Number(value)) ? `${Number(value).toFixed(0)}ms` : "—";
  $("totalTime").textContent = timing(data.totalElapsedMs);
  $("yoloTime").textContent = timing(data.yolo?.elapsedMs);
  $("ocrTime").textContent = timing(data.ppocr?.elapsedMs);
  $("resultsEmpty").hidden = boxes.length > 0;
  $("canvasEmpty").hidden = true;

  const image = $("pageImg");
  image.onload = () => {
    fitToScreen();
    drawBoxes(boxes);
    if (activeId !== null) selectBox(activeId);
  };
  image.onerror = () => setStatus("Không tải được ảnh bản vẽ", "error");
  image.hidden = false;
  image.src = data.pageImage;

  renderTable();
  renderCrops();
  updateReviewSummary();
  updatePageControls();
  if (boxes.length) selectBox(boxes[0].id);
  else {
    $("regionDetail").hidden = true;
    $("selectedRegionText").textContent = "Không phát hiện vùng";
  }
}

function valueInput(b, field, value, label) {
  return `<input class="value-input" data-field="${field}" value="${escapeHtml(value)}" aria-label="${label} vùng ${b.id}" title="${escapeHtml(value || "Chưa có giá trị")}" placeholder="—" spellcheck="false">`;
}

function renderTable() {
  const body = $("tableBody");
  body.replaceChildren();
  for (const b of lastData?.boxes || []) {
    const sem = b.semantics || {};
    const row = document.createElement("tr");
    row.id = `row-${b.id}`;
    row.classList.toggle("active", b.id === activeId);
    row.innerHTML = `
      <td><input type="checkbox" class="include-box" ${b.included ? "checked" : ""} aria-label="Xuất vùng ${b.id}"></td>
      <td>${b.id}</td>
      <td>${valueInput(b, "nominal", sem.nominal || b.ocrText || "", "Nominal")}</td>
      <td>${valueInput(b, "lowerTol", sem.lowerTol || "", "Dung sai dưới")}</td>
      <td>${valueInput(b, "upperTol", sem.upperTol || "", "Dung sai trên")}</td>
      <td><button class="review-button ${b.reviewed ? "confirmed" : ""}" title="${b.reviewed ? "Bỏ xác nhận" : "Xác nhận đã kiểm tra"}" aria-pressed="${b.reviewed}">${b.reviewed ? icon("check-check") : icon("clock")}<span>${b.reviewed ? "OK" : "Duyệt"}</span></button></td>
      <td><button class="star-button ${b.important ? "active" : ""}" aria-label="Quan trọng vùng ${b.id}" aria-pressed="${b.important}">${icon("star")}</button></td>`;
    row.addEventListener("click", () => selectBox(b.id));
    row.querySelector(".include-box").addEventListener("change", e => {
      b.included = e.target.checked;
      persistBox(b);
      updateReviewSummary();
    });
    row.querySelector(".review-button").addEventListener("click", e => {
      e.stopPropagation();
      b.reviewed = !b.reviewed;
      persistBox(b);
      renderTable();
      selectBox(b.id);
      updateReviewSummary();
    });
    row.querySelector(".star-button").addEventListener("click", e => {
      e.stopPropagation();
      b.important = !b.important;
      persistBox(b);
      renderTable();
      selectBox(b.id);
    });
    row.querySelectorAll(".value-input").forEach(input => input.addEventListener("change", () => {
      b.semantics = { ...b.semantics, [input.dataset.field]: input.value.trim() };
      b.edited = true;
      b.reviewed = false;
      const s = b.semantics;
      b.ocrText = s.upperTol || s.lowerTol ? `${s.nominal || ""} ${s.upperTol || ""}/${s.lowerTol || ""}`.trim() : s.nominal || "";
      persistBox(b);
      renderTable();
      renderCrops();
      drawBoxes(lastData.boxes);
      selectBox(b.id);
      updateReviewSummary();
    }));
    body.appendChild(row);
  }
  hydrateIcons(body);
}

function imageHtml(source, className, alt) {
  return typeof source === "string" && /^data:image\/(png|jpeg|webp);base64,/.test(source)
    ? `<img class="${className}" src="${escapeHtml(source)}" alt="${escapeHtml(alt)}">` : "";
}

function regionImages(b) {
  const labels = { 0: "Danh nghĩa", 1: "Dung sai trên", 2: "Dung sai dưới" };
  const colors = { 0: "#38bdf8", 1: "#c084fc", 2: "#fbbf24" };
  return (b.regionCrops || []).map(r => `<div class="detail-region" style="--region-color:${colors[r.classId] || "#8da6bd"}"><label>${labels[r.classId] || "Vùng"}</label>${imageHtml(r.preview, "region-image", labels[r.classId] || "Vùng")}<code>${escapeHtml(r.text)}</code></div>`).join("");
}

function renderCrops() {
  const view = $("cropsView");
  view.replaceChildren();
  for (const b of lastData?.boxes || []) {
    const row = document.createElement("div");
    row.className = "crop-row";
    row.id = `crop-${b.id}`;
    row.tabIndex = 0;
    row.setAttribute("role", "button");
    row.setAttribute("aria-label", `Đối chiếu vùng ${b.id}`);
    row.innerHTML = `<div class="crop-row-header"><b>Vùng #${b.id}</b><code>${escapeHtml(b.ocrText || "—")}</code></div>${imageHtml(b.processedPreview || b.rawCropPreview, "crop-image", `Ảnh cắt vùng ${b.id}`)}<div class="detail-regions">${regionImages(b)}</div>`;
    row.addEventListener("click", () => selectBox(b.id));
    row.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        selectBox(b.id);
      }
    });
    view.appendChild(row);
  }
  if (!lastData?.boxes?.length) view.innerHTML = '<div class="results-empty"><p>Chưa có ảnh cắt</p></div>';
}

function updateReviewSummary() {
  const boxes = lastData?.boxes || [];
  const checked = boxes.filter(b => b.included).length;
  $("reviewCount").textContent = `${boxes.filter(b => b.reviewed).length} / ${boxes.length} đã duyệt`;
  $("selectAll").checked = boxes.length > 0 && checked === boxes.length;
  $("selectAll").indeterminate = checked > 0 && checked < boxes.length;
  $("exportBtn").disabled = scanning || checked === 0;
}

function drawBoxes(boxes) {
  const layer = $("bboxLayer");
  layer.replaceChildren();
  layer.hidden = !regionsVisible;
  for (const b of boxes) {
    const el = document.createElement("div");
    el.className = `bbox${b.id === activeId ? " active" : ""}`;
    el.id = `bbox-${b.id}`;
    Object.assign(el.style, { left: `${b.x}px`, top: `${b.y}px`, width: `${b.w}px`, height: `${b.h}px` });
    const label = document.createElement("span");
    label.className = "bbox-tag";
    label.textContent = `#${b.id}`;
    el.appendChild(label);
    el.title = `Vùng ${b.id}: ${b.ocrText || "Chưa đọc được"}`;
    el.addEventListener("click", e => {
      if (didPanDrag) return;
      e.stopPropagation();
      selectBox(b.id, true);
    });
    layer.appendChild(el);
  }
}

function selectBox(id, scrollRow = false) {
  activeId = id;
  document.querySelectorAll(".bbox.active, #tableBody tr.active, .crop-row.active").forEach(el => el.classList.remove("active"));
  for (const prefix of ["bbox", "row", "crop"]) $(`${prefix}-${id}`)?.classList.add("active");
  if (scrollRow) $(`${activeTab === "results" ? "row" : "crop"}-${id}`)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  const b = lastData?.boxes?.find(box => box.id === id);
  if (!b) return;
  $("selectedRegionText").textContent = `Vùng #${id}: ${b.ocrText || "—"}${b.edited ? " · Đã sửa" : ""}`;
  const detail = $("regionDetail");
  detail.hidden = activeTab !== "results";
  detail.innerHTML = `<div class="detail-heading"><h3>Đối chiếu vùng #${id}</h3><span>${b.edited ? "Nội dung đã sửa" : "Blue-Box + 3 vùng YOLO"}</span></div>${imageHtml(b.processedPreview || b.rawCropPreview, "detail-crop", `Ảnh nguồn vùng ${id}`)}<div class="detail-regions">${regionImages(b)}</div>`;
}

function switchTab(tab) {
  activeTab = tab;
  for (const name of ["results", "crops"]) {
    const active = name === tab;
    $(`${name}Tab`).classList.toggle("active", active);
    $(`${name}Tab`).setAttribute("aria-selected", String(active));
    $(`${name}Tab`).tabIndex = active ? 0 : -1;
    $(`${name}View`).hidden = !active;
  }
  if (activeId !== null) selectBox(activeId);
}

function showRegions(value) {
  regionsVisible = value;
  $("bboxLayer").hidden = !value;
  for (const [id, active] of [["showRegionsBtn", value], ["imageOnlyBtn", !value]]) {
    $(id).classList.toggle("active", active);
    $(id).setAttribute("aria-pressed", String(active));
  }
}

function setZoom(value, clientX = null, clientY = null) {
  const image = $("pageImg"), wrap = $("canvasWrap"), stage = $("drawingStage");
  if (!image.naturalWidth) return;
  const previousRect = stage.getBoundingClientRect();
  const imageX = clientX === null ? null : (clientX - previousRect.left) / currentZoom;
  const imageY = clientY === null ? null : (clientY - previousRect.top) / currentZoom;
  currentZoom = Math.min(5, Math.max(0.05, value));
  $("stageSizer").style.width = `${Math.round(image.naturalWidth * currentZoom)}px`;
  $("stageSizer").style.height = `${Math.round(image.naturalHeight * currentZoom)}px`;
  stage.style.width = `${image.naturalWidth}px`;
  stage.style.height = `${image.naturalHeight}px`;
  stage.style.transform = `scale(${currentZoom})`;
  $("zoomText").textContent = `${Math.round(currentZoom * 100)}%`;
  if (imageX !== null) {
    const nextRect = stage.getBoundingClientRect();
    wrap.scrollLeft += nextRect.left + imageX * currentZoom - clientX;
    wrap.scrollTop += nextRect.top + imageY * currentZoom - clientY;
  }
}

function fitToScreen() {
  const image = $("pageImg"), wrap = $("canvasWrap");
  if (!image.naturalWidth || !wrap.clientWidth) return;
  setZoom(Math.min((wrap.clientWidth - 20) / image.naturalWidth, (wrap.clientHeight - 20) / image.naturalHeight));
  wrap.scrollLeft = wrap.scrollTop = 0;
}

/* ==================== UPLOAD & DRAG-AND-DROP ==================== */
async function handleFileObject(file) {
  if (!file || scanning) return;
  const ext = file.name.split(".").pop()?.toLowerCase() || "";
  if (!["pdf", "png", "jpg", "jpeg"].includes(ext)) {
    setStatus("Chỉ hỗ trợ file .pdf, .png, .jpg", "error");
    return;
  }
  setBusy(true);
  setStatus(`Đang tải lên ${file.name}…`);
  try {
    const dataBase64 = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
    const out = await requestJson("/api/upload-pdf", { filename: file.name, dataBase64 });
    const existingIdx = storeFiles.findIndex(f => f.sourceType === "upload" && f.name === out.filename);
    if (existingIdx >= 0) storeFiles.splice(existingIdx, 1);
    storeFiles.unshift({ name: out.filename, sourceType: "upload", sizeKb: out.sizeKb || 0 });
    pageIndex = 0;
    totalPages = out.totalPages || 1;
    selectDrawing("upload", out.filename);
    if ($("storesDialog")?.open) $("storesDialog").close();
  } catch (error) {
    setStatus(`Lỗi tải file: ${error.message}`, "error");
    return;
  } finally {
    setBusy(false);
    if ($("fileInput")) $("fileInput").value = "";
  }
  await runScan();
}

async function uploadFile(event) {
  const file = event.target.files?.[0];
  if (file) await handleFileObject(file);
}

/* ==================== STORES (DRAWINGS + HISTORY) ==================== */
function renderStores() {
  const storeEl = $("drawingStoreList");
  storeEl.replaceChildren();
  for (const item of storeFiles) {
    const btn = document.createElement("button");
    btn.type = "button";
    const isActive = currentPdf.sourceType === item.sourceType && currentPdf.filename === item.name;
    btn.className = `store-item${isActive ? " active" : ""}`;
    const tag = item.sourceType === "upload" ? "Đã tải lên" : "Mẫu";
    btn.innerHTML = `<b>${escapeHtml(item.name)}</b><span>${tag}${item.sizeKb ? ` · ${item.sizeKb} KB` : ""}</span>`;
    btn.addEventListener("click", () => {
      if (scanning) return;
      pageIndex = 0;
      totalPages = 1;
      selectDrawing(item.sourceType, item.name);
      $("storesDialog").close();
      runScan();
    });
    storeEl.appendChild(btn);
  }
  if (!storeFiles.length) {
    storeEl.innerHTML = '<p class="history-empty">Chưa có bản vẽ nào trong kho.</p>';
  }

  const list = $("historyList");
  list.replaceChildren();
  for (const entry of scanHistory) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "history-item";
    const time = new Date(entry.timestamp).toLocaleString("vi-VN");
    button.innerHTML = `<b>${escapeHtml(entry.filename)}</b><span>Trang ${entry.pageIndex + 1} · ${entry.boxes} vùng<br>${escapeHtml(time)}</span>`;
    button.addEventListener("click", () => {
      if (scanning) return;
      pageIndex = entry.pageIndex || 0;
      selectDrawing(entry.sourceType, entry.filename);
      if (yoloCatalog.some(m => m.name === entry.yolo)) $("yoloSelect").value = entry.yolo;
      if ([...$("ppocrSelect").options].some(o => o.value === entry.ppocr)) $("ppocrSelect").value = entry.ppocr;
      update3RegionLabel();
      $("storesDialog").close();
      runScan();
    });
    list.appendChild(button);
  }
  if (!scanHistory.length) {
    list.innerHTML = '<p class="history-empty">Chưa có lịch sử quét.</p>';
  }
}

function exportExcel() {
  if (!lastData || !window.XLSX) {
    setStatus("Không tải được thư viện xuất Excel", "error");
    return;
  }
  const rows = (lastData.boxes || []).filter(b => b.included).map(b => ({
    "Bản vẽ": currentPdf.filename,
    "Trang": pageIndex + 1,
    "Vùng": b.id,
    "Người đo": $("operator").value,
    "Linh kiện": $("partName").value,
    "Mold name": $("moldName").value,
    "Ngày đo": $("measurementDate").value,
    "Nominal": b.semantics?.nominal || b.ocrText || "",
    "Tol -": b.semantics?.lowerTol || "",
    "Tol +": b.semantics?.upperTol || "",
    "Kết quả OCR": b.ocrText || "",
    "Đã kiểm tra": b.reviewed ? "Có" : "Chưa",
    "Quan trọng": b.important ? "Có" : "Không",
    "Đã chỉnh sửa": b.edited ? "Có" : "Không"
  }));
  const workbook = XLSX.utils.book_new();
  const sheet = XLSX.utils.json_to_sheet(rows);
  sheet["!cols"] = [
    { wch: 28 }, { wch: 8 }, { wch: 8 }, { wch: 20 }, { wch: 20 }, { wch: 18 },
    { wch: 14 }, { wch: 18 }, { wch: 16 }, { wch: 16 }, { wch: 32 }, { wch: 15 }, { wch: 15 }, { wch: 16 }
  ];
  XLSX.utils.book_append_sheet(workbook, sheet, "Kết quả OCR");
  XLSX.writeFile(workbook, `${currentPdf.filename.replace(/\.[^.]+$/, "")}_OCR.xlsx`);
  setStatus(`Đã xuất ${rows.length} vùng ra Excel`, "success");
}

/* ==================== PAN, ZOOM & DRAG-DROP EVENTS ==================== */
let isPanning = false, didPanDrag = false, panStartX = 0, panStartY = 0, panScrollLeft = 0, panScrollTop = 0;
const wrap = $("canvasWrap");
wrap.addEventListener("wheel", event => {
  if (!lastData) return;
  event.preventDefault();
  setZoom(currentZoom * (event.deltaY < 0 ? 1.14 : 1 / 1.14), event.clientX, event.clientY);
}, { passive: false });

wrap.addEventListener("mousedown", event => {
  if (event.button === 1 || (event.button === 0 && event.ctrlKey)) {
    event.preventDefault();
    isPanning = true;
    didPanDrag = false;
    panStartX = event.clientX;
    panStartY = event.clientY;
    panScrollLeft = wrap.scrollLeft;
    panScrollTop = wrap.scrollTop;
    wrap.style.cursor = "grabbing";
  }
});

window.addEventListener("mousemove", event => {
  if (!isPanning) return;
  const dx = event.clientX - panStartX, dy = event.clientY - panStartY;
  if (Math.abs(dx) > 3 || Math.abs(dy) > 3) didPanDrag = true;
  wrap.scrollLeft = panScrollLeft - dx;
  wrap.scrollTop = panScrollTop - dy;
});

window.addEventListener("mouseup", () => {
  if (isPanning) {
    isPanning = false;
    wrap.style.cursor = "";
    setTimeout(() => { didPanDrag = false; }, 50);
  }
});
wrap.addEventListener("auxclick", e => { if (e.button === 1) e.preventDefault(); });

// Drag and drop anywhere on Drawing Panel or window
let dragDepth = 0;
const drawingPanel = $("drawingPanel");
for (const evtName of ["dragenter", "dragover", "dragleave", "drop"]) {
  window.addEventListener(evtName, e => {
    if (e.dataTransfer?.types?.includes("Files")) e.preventDefault();
  });
}
drawingPanel.addEventListener("dragenter", e => {
  if (!e.dataTransfer?.types?.includes("Files") || scanning) return;
  dragDepth++;
  drawingPanel.classList.add("drag-over");
  if (!$("canvasEmpty") || $("canvasEmpty").hidden) $("dropOverlay").hidden = false;
});
drawingPanel.addEventListener("dragover", e => {
  if (!e.dataTransfer?.types?.includes("Files") || scanning) return;
  e.dataTransfer.dropEffect = "copy";
});
drawingPanel.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) {
    drawingPanel.classList.remove("drag-over");
    $("dropOverlay").hidden = true;
  }
});
drawingPanel.addEventListener("drop", e => {
  dragDepth = 0;
  drawingPanel.classList.remove("drag-over");
  $("dropOverlay").hidden = true;
  const file = e.dataTransfer?.files?.[0];
  if (file) handleFileObject(file);
});

$("emptyDropCard")?.addEventListener("click", e => {
  if (e.target.closest("#emptyStoresBtn")) return;
  $("fileInput").click();
});
$("emptyDropCard")?.addEventListener("keydown", e => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    $("fileInput").click();
  }
});
$("emptyStoresBtn")?.addEventListener("click", e => {
  e.stopPropagation();
  renderStores();
  $("storesDialog").showModal();
});

window.addEventListener("resize", fitToScreen);
document.addEventListener("fullscreenchange", () => requestAnimationFrame(fitToScreen));
document.addEventListener("keydown", event => {
  if (event.key === "F5") {
    event.preventDefault();
    if (!$("modelDialog").open && !$("storesDialog").open && !$("settingsDialog").open) runScan();
  }
});

/* ==================== MENU & TOOLBAR BINDINGS ==================== */
$("uploadBtn").addEventListener("click", () => $("fileInput").click());
$("storesUploadBtn")?.addEventListener("click", () => $("fileInput").click());
$("fileInput").addEventListener("change", uploadFile);

$("modelBtn").addEventListener("click", () => {
  $("modelDialog").returnValue = "";
  $("modelDialog").showModal();
});
$("modelDialog").addEventListener("close", () => {
  update3RegionLabel();
  if ($("modelDialog").returnValue === "save" && currentPdf.filename) runScan();
});

const openStores = () => {
  renderStores();
  $("storesDialog").showModal();
};
$("storesBtn").addEventListener("click", openStores);
$("docPillBtn").addEventListener("click", openStores);
$("closeStores").addEventListener("click", () => $("storesDialog").close());

$("settingsBtn").addEventListener("click", () => {
  $("settingsDialog").returnValue = "";
  $("settingsDialog").showModal();
});

$("yoloSelect").addEventListener("change", update3RegionLabel);
$("scanBtn").addEventListener("click", runScan);
$("prevPage").addEventListener("click", () => { if (!scanning && pageIndex > 0) { pageIndex--; runScan(); } });
$("nextPage").addEventListener("click", () => { if (!scanning && pageIndex + 1 < totalPages) { pageIndex++; runScan(); } });
$("zoomOut").addEventListener("click", () => setZoom(currentZoom / 1.2));
$("zoomIn").addEventListener("click", () => setZoom(currentZoom * 1.2));
$("fitBtn").addEventListener("click", fitToScreen);
$("fullscreenBtn").addEventListener("click", async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await $("drawingPanel").requestFullscreen();
  } catch {
    setStatus("Trình duyệt không hỗ trợ toàn màn hình", "error");
  }
});
$("showRegionsBtn").addEventListener("click", () => showRegions(true));
$("imageOnlyBtn").addEventListener("click", () => showRegions(false));
$("resultsTab").addEventListener("click", () => switchTab("results"));
$("cropsTab").addEventListener("click", () => switchTab("crops"));
for (const name of ["results", "crops"]) {
  $(`${name}Tab`).addEventListener("keydown", event => {
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      const next = name === "results" ? "crops" : "results";
      switchTab(next);
      $(`${next}Tab`).focus();
    }
  });
}
$("selectAll").addEventListener("change", () => {
  for (const b of lastData?.boxes || []) {
    b.included = $("selectAll").checked;
    persistBox(b);
  }
  renderTable();
  updateReviewSummary();
});
$("exportBtn").addEventListener("click", exportExcel);

for (const id of ["operator", "partName", "moldName", "measurementDate"]) {
  $(id)?.addEventListener("change", () => {
    metadata.operator = $("operator").value;
    metadata.measurementDate = $("measurementDate").value;
    if (currentPdf.filename) {
      metadata[currentPdf.filename] = { partName: $("partName").value, moldName: $("moldName").value };
    }
    saveStorage("drawing-ocr.metadata", metadata);
  });
}

async function init() {
  hydrateIcons();
  $("operator").value = metadata.operator || "";
  $("measurementDate").value = metadata.measurementDate || "2026-10-07";
  try {
    const catalog = await requestJson("/api/models");
    yoloCatalog = (catalog.yoloOnnx || []).filter(m => m.numClasses === 1);
    if (!yoloCatalog.length) yoloCatalog = catalog.yoloOnnx || [];
    fillSelect($("yoloSelect"), yoloCatalog, catalog.defaults?.yoloModel, true);
    fillSelect($("ppocrSelect"), catalog.ppocrOnnx || [], catalog.defaults?.ppocrModel);

    storeFiles = [
      ...(catalog.uploadedPdfs || []).map(f => ({ ...f, sourceType: "upload" })),
      ...(catalog.samplePdfs || []).map(f => ({ ...f, sourceType: "sample" }))
    ];
    for (const file of storeFiles) {
      $("pdfSelect").add(new Option(file.name, `${file.sourceType}:${file.name}`));
    }
    update3RegionLabel();
  } catch (error) {
    setStatus(`Không tải được cấu hình: ${error.message}`, "error");
  }
}
init();
