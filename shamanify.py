#!/usr/bin/env python3
"""Shamanify — PyQt6. Proximity-режим (сохранение оригинальной картинки), плавные волны."""

import sys
import math
import random
import traceback

import numpy as np
from PIL import Image, ImageOps

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QProgressBar, QFileDialog, QMessageBox, QGroupBox, QFormLayout,
    QSizePolicy, QSlider
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage

try:
    import imageio
    import imageio_ffmpeg
    HAVE_IMAGEIO = True
except ImportError:
    HAVE_IMAGEIO = False


# ==================== UTIL ====================

def apply_easing(x, curve):
    x = max(0.0, min(1.0, x))
    if curve == "linear":
        return x
    if curve == "ease_in_out":
        return x * x * (3 - 2 * x)
    if curve == "ease_out_cubic":
        return 1.0 - (1.0 - x) ** 3
    if curve == "ease_in_cubic":
        return x ** 3
    if curve == "smootherstep":
        return x * x * x * (x * (x * 6 - 15) + 10)
    if curve == "overshoot":
        c = 1.70158
        x -= 1
        return 1 + (c + 1) * x * x * x + c * x * x
    return x


def split_into_tiles(src, grid_w, grid_h, out_width=None):
    if out_width is not None:
        w, h = src.size
        if out_width < w:
            src = src.resize((out_width, max(1, int(h * out_width / w))), Image.LANCZOS)
    w, h = src.size
    tw = w // grid_w
    th = h // grid_h
    if tw < 1 or th < 1:
        raise ValueError(f"Слишком мелкая сетка {grid_w}x{grid_h} для {w}x{h}")
    src = src.crop((0, 0, tw * grid_w, th * grid_h))
    tiles = []
    for gy in range(grid_h):
        for gx in range(grid_w):
            tiles.append(src.crop((gx * tw, gy * th, (gx + 1) * tw, (gy + 1) * th)))
    return tiles, tw, th


def sort_inside_tile(tile):
    arr = np.asarray(tile.convert("RGB"), dtype=np.uint8)
    flat = arr.reshape(-1, 3)
    lum = flat @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    order = np.argsort(lum)
    return Image.fromarray(flat[order].reshape(arr.shape), mode="RGB")


# ==================== FEATURES / SCORING ====================

def compute_tile_features(tile):
    arr = np.asarray(tile.convert("RGB"), dtype=np.float32)
    la = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    luma = float(la.mean())
    contrast = float(la.std())
    r = float(arr[:, :, 0].mean())
    g = float(arr[:, :, 1].mean())
    b = float(arr[:, :, 2].mean())
    rg = (r - g + 255.0) / 2.0
    yb = ((r + g) / 2.0 - b + 255.0) / 2.0
    return np.array([luma, contrast, rg, yb], dtype=np.float32)


def compute_target_features(target_path, grid_w, grid_h):
    img = Image.open(target_path).convert("RGB")
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.resize((grid_w, grid_h), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    padded = np.pad(luma, 1, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    contrast = windows.std(axis=(2, 3))
    rg = (r - g + 255.0) / 2.0
    yb = ((r + g) / 2.0 - b + 255.0) / 2.0
    return np.stack([luma.flatten(), contrast.flatten(),
                     rg.flatten(), yb.flatten()], axis=1).astype(np.float32)


def composite_scores(features, weights):
    mins = features.min(axis=0)
    maxs = features.max(axis=0)
    ranges = maxs - mins
    ranges = np.where(ranges > 1e-6, ranges, 1.0)
    normed = (features - mins) / ranges
    return normed @ np.asarray(weights, dtype=np.float32)


def optimize_placement(placement, tile_rank, cell_rank, grid_w, grid_h, alpha, n):
    """Итеративный локальный поиск для минимизации штрафа за расстояние."""
    if alpha >= 0.999:
        return np.arange(n, dtype=np.int64)
    if alpha <= 0.001:
        return placement
        
    max_dist = math.sqrt(grid_w**2 + grid_h**2)
    t_x = np.arange(n) % grid_w
    t_y = np.arange(n) // grid_w
    
    p = placement.copy()
    
    total_swaps = min(3000000, n * 100)
    batch_size = min(200000, n * 10)
    iterations = max(1, total_swaps // batch_size)
    
    for _ in range(iterations):
        c1 = np.random.randint(0, n, size=batch_size)
        c2 = np.random.randint(0, n, size=batch_size)
        
        t1 = p[c1]
        t2 = p[c2]
        
        mask = (t1 != t2) & (c1 != c2)
        c1 = c1[mask]
        c2 = c2[mask]
        t1 = t1[mask]
        t2 = t2[mask]
        
        if len(c1) == 0:
            continue
            
        f11 = np.abs(tile_rank[t1] - cell_rank[c1]) / n
        f22 = np.abs(tile_rank[t2] - cell_rank[c2]) / n
        f12 = np.abs(tile_rank[t2] - cell_rank[c1]) / n
        f21 = np.abs(tile_rank[t1] - cell_rank[c2]) / n
        
        d11 = np.sqrt((t_x[t1] - t_x[c1])**2 + (t_y[t1] - t_y[c1])**2) / max_dist
        d22 = np.sqrt((t_x[t2] - t_x[c2])**2 + (t_y[t2] - t_y[c2])**2) / max_dist
        d12 = np.sqrt((t_x[t2] - t_x[c1])**2 + (t_y[t2] - t_y[c1])**2) / max_dist
        d21 = np.sqrt((t_x[t1] - t_x[c2])**2 + (t_y[t1] - t_y[c2])**2) / max_dist
        
        old_cost = (1 - alpha) * (f11 + f22) + alpha * (d11 + d22)
        new_cost = (1 - alpha) * (f12 + f21) + alpha * (d12 + d21)
        
        improve = new_cost < old_cost - 1e-6 
        
        valid_c1 = c1[improve]
        valid_c2 = c2[improve]
        
        if len(valid_c1) == 0:
            continue
            
        visited = np.zeros(n, dtype=bool)
        for i in range(len(valid_c1)):
            v1 = valid_c1[i]
            v2 = valid_c2[i]
            if not visited[v1] and not visited[v2]:
                p[v1], p[v2] = p[v2], p[v1]
                visited[v1] = True
                visited[v2] = True
                
    return p


def compute_placement(source_path, target_path, grid_w, grid_h,
                      tile_sort, out_width, weights, proximity):
    src = Image.open(source_path).convert("RGB")
    src = ImageOps.autocontrast(src, cutoff=1)
    tiles, tw, th = split_into_tiles(src, grid_w, grid_h, out_width)
    n = grid_w * grid_h

    if tile_sort:
        tiles = [sort_inside_tile(t) for t in tiles]

    tile_feats = np.stack([compute_tile_features(t) for t in tiles], axis=0)
    tile_scores = composite_scores(tile_feats, weights) + np.random.random(n) * 1e-6
    tile_order = np.argsort(tile_scores)

    target_feats = compute_target_features(target_path, grid_w, grid_h)
    cell_scores = composite_scores(target_feats, weights) + np.random.random(n) * 1e-6
    cell_order = np.argsort(cell_scores)

    placement = np.zeros(n, dtype=np.int64)
    placement[cell_order] = tile_order
    
    if proximity > 0:
        tile_rank = np.empty(n, dtype=np.int64)
        tile_rank[tile_order] = np.arange(n)
        cell_rank = np.empty(n, dtype=np.int64)
        cell_rank[cell_order] = np.arange(n)
        
        placement = optimize_placement(placement, tile_rank, cell_rank, grid_w, grid_h, proximity / 100.0, n)

    return tiles, placement, tw, th


def compute_wave_indices(grid_w, grid_h, num_batches, mode):
    n = grid_w * grid_h
    waves = [0] * n
    cx, cy = (grid_w - 1) / 2, (grid_h - 1) / 2
    max_dist = math.sqrt(cx * cx + cy * cy) or 1.0
    for i in range(n):
        tcx, tcy = i % grid_w, i // grid_w
        if mode == "center_out":
            d = math.sqrt((tcx - cx) ** 2 + (tcy - cy) ** 2) / max_dist
            waves[i] = int(d * (num_batches - 1))
        elif mode == "edges_in":
            d = math.sqrt((tcx - cx) ** 2 + (tcy - cy) ** 2) / max_dist
            waves[i] = int((1 - d) * (num_batches - 1))
        elif mode == "top_down":
            waves[i] = int((tcy / max(1, grid_h - 1)) * (num_batches - 1))
        elif mode == "left_right":
            waves[i] = int((tcx / max(1, grid_w - 1)) * (num_batches - 1))
        else:
            waves[i] = random.randint(0, num_batches - 1)
    return waves


def render_frame(tiles, placement, waves, tw, th, grid_w, grid_h,
                 t, num_batches, overlap, easing):
    canvas = Image.new("RGB", (grid_w * tw, grid_h * th), (0, 0, 0))
    n = grid_w * grid_h
    duration = overlap / num_batches
    max_start = max(0.0, 1.0 - duration)
    denom = max(1, num_batches - 1)

    for cell_idx in range(n):
        tile_idx = int(placement[cell_idx])
        tile_wave = waves[cell_idx]

        ox = (tile_idx % grid_w) * tw
        oy = (tile_idx // grid_w) * th
        tx = (cell_idx % grid_w) * tw
        ty = (cell_idx // grid_w) * th

        tile_start = (tile_wave / denom) * max_start
        if duration > 0:
            progress = (t - tile_start) / duration
        else:
            progress = 1.0
        progress = max(0.0, min(1.0, progress))

        if progress <= 0.0:
            px, py = ox, oy
        elif progress >= 1.0:
            px, py = tx, ty
        else:
            e = apply_easing(progress, easing)
            px = int(round((1 - e) * ox + e * tx))
            py = int(round((1 - e) * oy + e * ty))

        canvas.paste(tiles[tile_idx], (px, py))
    return canvas


# ==================== QT UTILS ====================

def pil_to_qpixmap(pil_img):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    data = pil_img.tobytes("raw", "RGB")
    qimg = QImage(data, pil_img.width, pil_img.height,
                  pil_img.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def scale_fit(pil_img, max_w, max_h):
    w, h = pil_img.size
    s = min(max_w / w, max_h / h, 1.0)
    if s >= 1.0:
        return pil_img
    return pil_img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)


# ==================== WORKER ====================

class AnimationWorker(QThread):
    frame_ready = pyqtSignal(object)
    progress = pyqtSignal(int, int)
    finished_ok = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, source_path, target_path, grid_w, grid_h,
                 tile_sort, out_width, num_frames, num_batches, mode,
                 weights, overlap, easing, proximity):
        super().__init__()
        self.source_path = source_path
        self.target_path = target_path
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.tile_sort = tile_sort
        self.out_width = out_width
        self.num_frames = num_frames
        self.num_batches = num_batches
        self.mode = mode
        self.weights = weights
        self.overlap = overlap
        self.easing = easing
        self.proximity = proximity
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            tiles, placement, tw, th = compute_placement(
                self.source_path, self.target_path,
                self.grid_w, self.grid_h, self.tile_sort, self.out_width,
                self.weights, self.proximity,
            )
            waves = compute_wave_indices(
                self.grid_w, self.grid_h, self.num_batches, self.mode
            )
            frames = []
            for f in range(self.num_frames):
                if self._stop:
                    return
                t = f / max(1, self.num_frames - 1)
                frame = render_frame(tiles, placement, waves, tw, th,
                                     self.grid_w, self.grid_h, t,
                                     self.num_batches, self.overlap, self.easing)
                frames.append(frame)
                self.frame_ready.emit(frame)
                self.progress.emit(f + 1, self.num_frames)
            self.finished_ok.emit(frames)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(f"{type(e).__name__}: {e}")


# ==================== MAIN WINDOW ====================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shamanify")
        self.resize(1300, 820)

        self.source_path = None
        self.target_path = None
        self.worker = None
        self.frames = []
        self.replay_timer = None
        self.replay_idx = 0
        
        self._current_preview_qpixmap = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # ---------- LEFT ----------
        left = QWidget()
        left.setFixedWidth(420)
        left_l = QVBoxLayout(left)

        src_group = QGroupBox("Исходное изображение")
        src_l = QVBoxLayout(src_group)
        self.src_btn = QPushButton("Загрузить исходник…")
        self.src_btn.clicked.connect(self.load_source)
        self.src_prev = QLabel("не загружено")
        self.src_prev.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.src_prev.setFixedHeight(120)
        self.src_prev.setStyleSheet("background:#222;color:#888;")
        src_l.addWidget(self.src_btn)
        src_l.addWidget(self.src_prev)
        left_l.addWidget(src_group)

        tgt_group = QGroupBox("Образец (Шаман)")
        tgt_l = QVBoxLayout(tgt_group)
        self.tgt_btn = QPushButton("Загрузить образец…")
        self.tgt_btn.clicked.connect(self.load_target)
        self.tgt_prev = QLabel("не загружено")
        self.tgt_prev.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tgt_prev.setFixedHeight(120)
        self.tgt_prev.setStyleSheet("background:#222;color:#888;")
        tgt_l.addWidget(self.tgt_btn)
        tgt_l.addWidget(self.tgt_prev)
        left_l.addWidget(tgt_group)

        weight_group = QGroupBox("Матчинг и сохранение оригинальной картинки")
        wform = QFormLayout(weight_group)
        self.luma_slider = self._add_int_slider(wform, "Яркость", 100)
        self.contrast_slider = self._add_int_slider(wform, "Контраст", 30)
        self.color_slider = self._add_int_slider(wform, "Цвет", 30)
        
        # Новый слайдер
        self.proximity_slider = self._add_int_slider(wform, "Сохранение позиций", 0, max_val=100)
        self.proximity_slider.setToolTip("0 = полная пересборка (глитч), 100 = картинка не меняется")
        
        left_l.addWidget(weight_group)

        anim_group = QGroupBox("Плавность анимации")
        aform = QFormLayout(anim_group)
        self.overlap_slider = self._add_int_slider(aform, "Перекрытие волн", 3, max_val=20)
        self.overlap_slider.setToolTip("1 = дискретные волны; 3–5 = плавно; 20 = всё едет разом")
        self.easing_combo = QComboBox()
        self.easing_combo.addItems(
            ["smootherstep", "ease_out_cubic", "ease_in_out",
             "ease_in_cubic", "overshoot", "linear"]
        )
        aform.addRow("Кривая движения:", self.easing_combo)
        self.replay_fps_slider = self._add_int_slider(aform, "FPS проигр.", 30, max_val=120)
        left_l.addWidget(anim_group)

        params_group = QGroupBox("Параметры")
        form = QFormLayout(params_group)

        self.grid_w_spin = QSpinBox()
        self.grid_w_spin.setRange(32, 256)
        self.grid_w_spin.setValue(128)
        form.addRow("Ширина сетки:", self.grid_w_spin)

        self.grid_h_spin = QSpinBox()
        self.grid_h_spin.setRange(0, 256)
        self.grid_h_spin.setValue(0)
        self.grid_h_spin.setSpecialValueText("авто")
        form.addRow("Высота сетки:", self.grid_h_spin)

        self.out_w_spin = QSpinBox()
        self.out_w_spin.setRange(320, 4096)
        self.out_w_spin.setValue(960)
        self.out_w_spin.setSingleStep(160)
        form.addRow("Ширина выхода:", self.out_w_spin)

        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(10, 300)
        self.frames_spin.setValue(90)
        form.addRow("Кадров:", self.frames_spin)

        self.batches_spin = QSpinBox()
        self.batches_spin.setRange(5, 60)
        self.batches_spin.setValue(30)
        form.addRow("Волн:", self.batches_spin)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(
            ["center_out", "edges_in", "top_down", "left_right", "random"]
        )
        form.addRow("Появление:", self.mode_combo)

        self.tile_sort_cb = QCheckBox("Сортировать пиксели внутри тайлов")
        form.addRow(self.tile_sort_cb)

        self.cyber_cb = QCheckBox("⚡ Режим киберпсиха (без ограничений)")
        self.cyber_cb.toggled.connect(self._toggle_cyber)
        form.addRow(self.cyber_cb)

        left_l.addWidget(params_group)

        self.start_btn = QPushButton("Запустить")
        self.start_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self.start_btn.clicked.connect(self.start_animation)
        left_l.addWidget(self.start_btn)

        btn_row = QHBoxLayout()
        self.save_gif_btn = QPushButton("Сохранить GIF")
        self.save_gif_btn.clicked.connect(self.save_gif)
        self.save_gif_btn.setEnabled(False)
        self.save_mp4_btn = QPushButton("Сохранить MP4")
        self.save_mp4_btn.clicked.connect(self.save_mp4)
        self.save_mp4_btn.setEnabled(False)
        btn_row.addWidget(self.save_gif_btn)
        btn_row.addWidget(self.save_mp4_btn)
        left_l.addLayout(btn_row)

        self.replay_btn = QPushButton("▶ Проиграть в окне")
        self.replay_btn.clicked.connect(self.replay_animation)
        self.replay_btn.setEnabled(False)
        left_l.addWidget(self.replay_btn)

        left_l.addStretch()
        root.addWidget(left)

        # ---------- RIGHT ----------
        right = QWidget()
        right_l = QVBoxLayout(right)
        
        self.preview = QLabel("Здесь будет превью")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#111;color:#666;")
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.preview.setMinimumSize(200, 200)
        right_l.addWidget(self.preview, 1)
        
        self.progress = QProgressBar()
        right_l.addWidget(self.progress)
        self.status = QLabel("Готов")
        right_l.addWidget(self.status)
        root.addWidget(right, 1)

    def _add_int_slider(self, form, label, default, max_val=200):
        wrapper = QWidget()
        h = QHBoxLayout(wrapper)
        h.setContentsMargins(0, 0, 0, 0)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(1 if max_val <= 20 else 0, max_val)
        slider.setValue(default)
        val_label = QLabel(str(default))
        val_label.setFixedWidth(40)
        slider.valueChanged.connect(lambda v, lb=val_label: lb.setText(str(v)))
        h.addWidget(slider)
        h.addWidget(val_label)
        form.addRow(label, wrapper)
        return slider

    def _toggle_cyber(self, enabled):
        if enabled:
            QMessageBox.warning(
                self, "⚡ Режим киберпсиха",
                "Ограничения сняты. Кадры держатся в RAM. 5000 кадров на 1000x800 "
                "≈ 12 ГБ. Включай с умом."
            )
            self.grid_w_spin.setRange(1, 4096)
            self.grid_h_spin.setRange(0, 4096)
            self.out_w_spin.setRange(1, 16384)
            self.frames_spin.setRange(1, 5000)
            self.batches_spin.setRange(1, 500)
            self.overlap_slider.setRange(1, 500)
            self.replay_fps_slider.setRange(1, 1000)
        else:
            self.grid_w_spin.setRange(32, 256)
            self.grid_h_spin.setRange(0, 256)
            self.out_w_spin.setRange(320, 4096)
            self.frames_spin.setRange(10, 300)
            self.batches_spin.setRange(5, 60)
            self.overlap_slider.setRange(1, 20)
            self.replay_fps_slider.setRange(1, 120)

    def _weights_array(self):
        c = self.color_slider.value() / 100.0
        return np.array([
            self.luma_slider.value() / 100.0,
            self.contrast_slider.value() / 100.0,
            c, c,
        ], dtype=np.float32)

    def load_source(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Исходное изображение", "",
            "Изображения (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return
        self.source_path = path
        img = Image.open(path).convert("RGB")
        self.src_prev.setPixmap(pil_to_qpixmap(scale_fit(img, 400, 120)))
        self.src_prev.setText("")

    def load_target(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Образец", "",
            "Изображения (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return
        self.target_path = path
        img = Image.open(path).convert("RGB")
        self.tgt_prev.setPixmap(pil_to_qpixmap(scale_fit(img, 400, 120)))
        self.tgt_prev.setText("")

    def _resolve_grid_h(self):
        if self.grid_h_spin.value() > 0:
            return self.grid_h_spin.value()
        if not self.target_path:
            return self.grid_w_spin.value()
        with Image.open(self.target_path) as tgt:
            tw, th = tgt.size
        return max(1, round(self.grid_w_spin.value() * th / tw))

    # ---------- PREVIEW ----------

    def _set_preview(self, pil_img):
        self._current_preview_qpixmap = pil_to_qpixmap(pil_img)
        self._update_preview_pixmap()

    def _update_preview_pixmap(self):
        if self._current_preview_qpixmap is None:
            return
        w = max(50, self.preview.width())
        h = max(50, self.preview.height())
        
        scaled_pixmap = self._current_preview_qpixmap.scaled(
            w, h, 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.preview.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_preview_pixmap()

    # ---------- START / PROGRESS ----------

    def start_animation(self):
        if not self.source_path or not self.target_path:
            QMessageBox.warning(self, "Ошибка", "Загрузи исходник и образец.")
            return

        self.start_btn.setEnabled(False)
        self.save_gif_btn.setEnabled(False)
        self.save_mp4_btn.setEnabled(False)
        self.replay_btn.setEnabled(False)
        self.progress.setValue(0)
        self.frames = []
        self.status.setText("Генерация кадров…")

        self.worker = AnimationWorker(
            self.source_path, self.target_path,
            self.grid_w_spin.value(), self._resolve_grid_h(),
            self.tile_sort_cb.isChecked(),
            self.out_w_spin.value(),
            self.frames_spin.value(),
            self.batches_spin.value(),
            self.mode_combo.currentText(),
            self._weights_array(),
            float(self.overlap_slider.value()),
            self.easing_combo.currentText(),
            float(self.proximity_slider.value()),
        )
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_frame(self, pil_img):
        self._set_preview(pil_img)

    def on_progress(self, cur, total):
        self.progress.setMaximum(total)
        self.progress.setValue(cur)

    def on_finished(self, frames):
        self.frames = frames
        self.start_btn.setEnabled(True)
        self.save_gif_btn.setEnabled(True)
        self.save_mp4_btn.setEnabled(HAVE_IMAGEIO)
        self.replay_btn.setEnabled(True)
        self.status.setText(f"Готово: {len(frames)} кадров")

    def on_error(self, msg):
        self.start_btn.setEnabled(True)
        self.status.setText("Ошибка")
        QMessageBox.critical(self, "Ошибка", msg)

    # ---------- REPLAY ----------

    def replay_animation(self):
        if not self.frames:
            return
        if self.replay_timer is not None:
            self.replay_timer.stop()
        self.replay_idx = 0
        fps = max(1, self.replay_fps_slider.value())
        self.replay_timer = QTimer(self)
        self.replay_timer.timeout.connect(self._replay_tick)
        self.replay_timer.start(max(8, int(1000 / fps)))

    def _replay_tick(self):
        if self.replay_idx >= len(self.frames):
            self.replay_timer.stop()
            self.replay_timer = None
            return
        frame = self.frames[self.replay_idx]
        self._set_preview(frame)
        self.replay_idx += 1

    # ---------- SAVE ----------

    def save_gif(self):
        if not self.frames:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить GIF", "shamanify.gif", "GIF (*.gif)"
        )
        if not path:
            return
        try:
            max_dim = 640
            scaled = []
            for fr in self.frames:
                w, h = fr.size
                s = min(max_dim / w, max_dim / h, 1.0)
                if s < 1.0:
                    fr = fr.resize((int(w * s), int(h * s)), Image.LANCZOS)
                scaled.append(fr.convert("P", palette=Image.ADAPTIVE, colors=128))
            fps = max(1, self.replay_fps_slider.value())
            duration = max(20, int(1000 / fps))
            scaled[0].save(path, save_all=True, append_images=scaled[1:],
                           duration=duration, loop=0, optimize=False, disposal=2)
            self.status.setText(f"Сохранено: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка сохранения", str(e))

    def save_mp4(self):
        if not self.frames or not HAVE_IMAGEIO:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить MP4", "shamanify.mp4", "MP4 (*.mp4)"
        )
        if not path:
            return
        try:
            fps = max(1, self.replay_fps_slider.value())
            writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                        quality=8, pixelformat="yuv420p")
            for fr in self.frames:
                writer.append_data(np.asarray(fr.convert("RGB")))
            writer.close()
            self.status.setText(f"Сохранено: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка сохранения", str(e))

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        if self.replay_timer is not None:
            self.replay_timer.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()