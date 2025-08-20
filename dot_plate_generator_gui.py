# dot_plate_generator_gui.py
# 必要ライブラリ: PyQt5, PIL, numpy, trimesh, shapely, skimage, scipy, matplotlib, OpenCV (cv2)

import sys
import os
import json
import pickle
import base64
import numpy as np
from PIL import Image
import cv2
from collections import Counter
from scipy.spatial import distance
import trimesh
from trimesh.creation import box
# Qt Widgets
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QScrollArea,
    QListWidget, QListWidgetItem, QVBoxLayout, QHBoxLayout, QSlider, QSpinBox,
    QGridLayout, QDoubleSpinBox, QToolButton, QDialog, QGroupBox, QFrame,
    QSizePolicy, QToolTip, QMainWindow, QColorDialog, QCheckBox, QComboBox,
    QMenu, QAction, QMenuBar, QRubberBand, QAbstractItemView, QDockWidget,
    QDialogButtonBox, QTextBrowser
)
# 以下のウィジェットを追加インポート（APIキーダイアログ・メッセージボックス用）
from PyQt5.QtWidgets import QMessageBox, QInputDialog, QLineEdit
from PyQt5.QtCore import Qt, QSize, QTimer, QPoint, QSettings, QEvent, QRect
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QPen, QCursor, QIcon, QMouseEvent
from shapely.geometry import Polygon
from skimage import measure
from scipy.ndimage import binary_fill_holes
from io import BytesIO
import threading
import openai  # OpenAI API for AIブラシ機能
import ast
import time
import tempfile
# for line-art conversion (using numpy channel swap; no OpenCV required)
from skimage.morphology import skeletonize
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops
# Embedded line-art conversion function (copied from coloring_book)
def build_preview_and_svg(img, canny_lo=40, canny_hi=120,
                          close_iter=3, dilate_iter=2,
                          min_area_pct=0.0005, eps_ratio=0.0015):
    """
    Generate line-art preview (BGR), simplified contours, and region mask from BGR image.
    """
    # Grayscale + noise reduction
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 50, 50)
    gray = cv2.equalizeHist(gray)
    # Canny edge detection
    edges = cv2.Canny(gray, int(canny_lo), int(canny_hi))
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=int(close_iter))
    dil = cv2.dilate(closed, kernel, iterations=int(dilate_iter))
    # Skeletonization
    skel = skeletonize((dil > 0).astype(np.uint8)).astype(np.uint8) * 255
    # Invert + threshold
    inv = cv2.bitwise_not(skel)
    _, thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Distance transform + watershed
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)
    local_max = peak_local_max(dist, min_distance=6, labels=thresh.astype(bool))
    markers = np.zeros_like(dist, dtype=int)
    for i, (y, x) in enumerate(local_max, start=1):
        markers[y, x] = i
    labels_ws = watershed(-dist, markers, mask=thresh.astype(bool))
    # Filter small regions
    min_area = (img.shape[0] * img.shape[1]) * float(min_area_pct)
    mask_regions = np.zeros_like(labels_ws, dtype=np.uint8)
    for r in regionprops(labels_ws):
        if r.area >= min_area:
            mask_regions[labels_ws == r.label] = 255
    # Simplify contours
    contours, _ = cv2.findContours(mask_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    def simplify_contour(cnt):
        peri = cv2.arcLength(cnt, True)
        return cv2.approxPolyDP(cnt, float(eps_ratio) * peri, True)
    simplified = [simplify_contour(c) for c in contours]
    # Build preview image with mean region colors
    preview = np.ones_like(img) * 255
    nlabels, labels2 = cv2.connectedComponents(mask_regions)
    for label_id in range(1, nlabels):
        mask = (labels2 == label_id)
        if not np.any(mask):
            continue
        mean_color = img[mask].mean(axis=0)
        preview[mask] = mean_color
    # Overlay edges
    edges_for_overlay = cv2.dilate(skel, kernel, iterations=1)
    preview[edges_for_overlay > 0] = (0, 0, 0)
    return preview, simplified, mask_regions

# Vedoをインポート (VTKベースの3D可視化ライブラリ)
# Matplotlibを常に使用するように変更
import matplotlib.pyplot as plt
VEDO_AVAILABLE = False

# try:
#     import vedo
#     VEDO_AVAILABLE = True
# except ImportError:
#     print("vedo library not available, please install with: pip install vedo")
#     import matplotlib.pyplot as plt
#     VEDO_AVAILABLE = False

# -------------------------------
# 補助関数
# -------------------------------
def normalize_colors(pixels, step):
    """単純な量子化による減色"""
    return (pixels // step) * step

def map_to_closest_color(pixel, palette):
    """ユークリッド距離で最も近い色を選択"""
    return min(palette, key=lambda c: distance.euclidean(pixel, c))

def get_median_cut_palette(pixels, num_colors):
    """メディアンカット法でカラーパレットを生成"""
    if len(pixels) == 0:
        return np.array([], dtype=np.uint8)
    
    # RGB値をfloatに変換してコピー
    pixels_copy = pixels.copy().astype(np.float64)
    
    # 各カラーチャンネルの範囲
    ranges = np.max(pixels_copy, axis=0) - np.min(pixels_copy, axis=0)
    
    # 最大範囲を持つチャンネル
    channel = np.argmax(ranges)
    
    # 色空間を分割
    def split_colors(pixels_subset, colors_left, result_palette):
        if colors_left <= 1 or len(pixels_subset) == 0:
            # このグループの代表色として平均値を計算
            if len(pixels_subset) > 0:
                avg_color = np.mean(pixels_subset, axis=0).astype(np.uint8)
                result_palette.append(avg_color)
            return
        
        # 各チャンネルの範囲
        ranges = np.max(pixels_subset, axis=0) - np.min(pixels_subset, axis=0)
        
        # 最大範囲を持つチャンネル
        channel = np.argmax(ranges)
        
        # そのチャンネルでソート
        sorted_pixels = pixels_subset[pixels_subset[:, channel].argsort()]
        
        # 中央で分割
        median_idx = len(sorted_pixels) // 2
        
        # 再帰的に分割
        split_colors(sorted_pixels[:median_idx], colors_left // 2, result_palette)
        split_colors(sorted_pixels[median_idx:], colors_left - colors_left // 2, result_palette)
    
    # パレット生成
    palette = []
    split_colors(pixels_copy, num_colors, palette)
    
    return np.array(palette, dtype=np.uint8)

def get_kmeans_palette(pixels, num_colors):
    """K-means法でカラーパレットを生成"""
    from sklearn.cluster import KMeans
    import warnings
    
    # 警告を無視（K-meansの収束警告など）
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # 入力データが少なすぎる場合はnum_colorsを調整
        n_colors = min(num_colors, len(pixels))
        if n_colors == 0:
            return np.array([], dtype=np.uint8)
            
        # K-means実行
        kmeans = KMeans(n_clusters=n_colors, random_state=0, n_init=10)
        kmeans.fit(pixels)
        
        # クラスタ中心がパレット色
        palette = kmeans.cluster_centers_.astype(np.uint8)
        
        return palette

def get_toon_palette(pixels, num_colors):
    """トゥーンアニメ風のパレットを生成する
    
    以下の特徴を持つ色パレットを生成：
    1. 彩度が高く、明確な色を優先
    2. 同系色での階調が少なく、はっきりとした色の差を作る
    3. ベースカラー、シャドウ、ハイライトの3トーン構成
    """
    from skimage import color
    
    # RGBからHSVに変換して色相、彩度、明度を分析
    hsv_pixels = color.rgb2hsv(pixels.reshape(-1, 1, 3))
    hsv_pixels = hsv_pixels.reshape(-1, 3)
    
    # 彩度と明度に基づいて色をグループ化
    # 高彩度領域を優先して選択
    high_sat_mask = hsv_pixels[:, 1] > 0.4  # 彩度が高い色
    high_sat_pixels = hsv_pixels[high_sat_mask]
    
    # 色相に基づいて主要な色を特定
    n_hue_bins = max(3, num_colors // 3)  # 少なくとも3つの色相ビン
    hist, bin_edges = np.histogram(hsv_pixels[:, 0], bins=n_hue_bins)
    
    # 最も頻度の高い色相ビンを特定
    sorted_bins = np.argsort(-hist)
    
    # 主要な色相ごとに3トーン（ベース、シャドウ、ハイライト）を選定
    palette = []
    used_hues = set()
    
    # 主要な色相から色を選定
    for bin_idx in sorted_bins:
        if len(palette) >= num_colors:
            break
            
        # このビンの色相範囲
        h_min = bin_edges[bin_idx]
        h_max = bin_edges[bin_idx + 1]
        h_center = (h_min + h_max) / 2
        
        # 既に使用した色相と近すぎる場合はスキップ
        if any(abs(h_center - h) < 0.05 for h in used_hues):
            continue
            
        # この色相範囲内のピクセル
        hue_mask = (hsv_pixels[:, 0] >= h_min) & (hsv_pixels[:, 0] < h_max)
        bin_pixels = hsv_pixels[hue_mask]
        
        if len(bin_pixels) == 0:
            continue
            
        # 彩度で上位の色を取得
        sorted_sat_idx = np.argsort(-bin_pixels[:, 1])
        
        # 選択した色相でベース、シャドウ、ハイライトの3トーンを作成
        if len(sorted_sat_idx) > 0:
            base_hsv = bin_pixels[sorted_sat_idx[0]].copy()
            base_hsv[1] = min(1.0, base_hsv[1] + 0.2)  # 彩度を少し上げる
            base_hsv[2] = 0.6  # 中間の明度
            
            shadow_hsv = base_hsv.copy()
            shadow_hsv[2] = 0.3  # 暗め
            
            highlight_hsv = base_hsv.copy()
            highlight_hsv[2] = 0.9  # 明るめ
            
            # HSVからRGBに戻す
            base_rgb = color.hsv2rgb(base_hsv.reshape(1, 1, 3)).reshape(3)
            shadow_rgb = color.hsv2rgb(shadow_hsv.reshape(1, 1, 3)).reshape(3)
            highlight_rgb = color.hsv2rgb(highlight_hsv.reshape(1, 1, 3)).reshape(3)
            
            # パレットに追加
            palette.append(tuple((base_rgb * 255).astype(np.uint8)))
            if len(palette) < num_colors:
                palette.append(tuple((shadow_rgb * 255).astype(np.uint8)))
            if len(palette) < num_colors:
                palette.append(tuple((highlight_rgb * 255).astype(np.uint8)))
                
            used_hues.add(h_center)
    
    # 黒と白を追加（トゥーンアニメには必須）
    if len(palette) < num_colors:
        palette.append((0, 0, 0))  # 黒
    if len(palette) < num_colors:
        palette.append((255, 255, 255))  # 白
        
    # グレースケール階調を追加して残りを埋める
    remaining = num_colors - len(palette)
    if remaining > 0:
        gray_step = 240 // (remaining + 1)
        for i in range(1, remaining + 1):
            gray_val = i * gray_step
            palette.append((gray_val, gray_val, gray_val))
    
    return palette

def get_octree_palette(pixels, num_colors):
    """オクトツリー量子化でカラーパレットを生成"""
    # 安全な実装のためのシンプルなアプローチ
    try:
        # PIL ImageQuantを使用
        from PIL import Image
        
        # ピクセルデータをIm​age形式に変換
        # ピクセル形状問題を修正
        if len(pixels) == 0:
            return np.array([], dtype=np.uint8)
            
        # 入力が2次元配列でない場合を処理
        if len(pixels.shape) == 1:
            # 1次元配列の場合、3列の2次元配列に変形
            pixels_2d = pixels.reshape(-1, 3)
        elif len(pixels.shape) > 2:
            # 3次元以上の場合、平坦化して2次元に
            pixels_2d = pixels.reshape(-1, 3)
        else:
            # 既に2次元の場合はそのまま
            pixels_2d = pixels
            
        # 一時的なカラー画像を作成
        img_size = int(np.ceil(np.sqrt(len(pixels_2d))))
        temp_img = Image.new('RGB', (img_size, img_size), (0, 0, 0))
        
        # ピクセルデータを画像に設定
        for i, (r, g, b) in enumerate(pixels_2d):
            if i >= img_size * img_size:
                break
            x = i % img_size
            y = i // img_size
            temp_img.putpixel((x, y), (int(r), int(g), int(b)))
        
        # Octree量子化（method=2）を実行
        quantized = temp_img.quantize(colors=min(num_colors, 256), method=2)
        
        # パレット画像に変換
        palette_img = quantized.convert('RGB')
        
        # パレットカラー抽出
        colors = palette_img.getcolors(maxcolors=num_colors*2)
        
        if not colors:
            # getcolorsが失敗した場合、単純な減色にフォールバック
            # ここはmedian cut法を使用
            return get_median_cut_palette(pixels, num_colors)
            
        # パレットを構築
        palette = []
        for count, color in colors:
            palette.append(color)
            
        # NumPy配列に変換
        palette_array = np.array(palette, dtype=np.uint8)
        
        # 色数が少なすぎる場合の対応
        if len(palette_array) < num_colors:
            # 足りない色は元の画像からランダムサンプリング
            missing = num_colors - len(palette_array)
            indices = np.random.choice(len(pixels_2d), size=missing, replace=False)
            additional_colors = pixels_2d[indices]
            palette_array = np.vstack([palette_array, additional_colors])
        
        # 必要数を超えた場合は切り詰め
        return palette_array[:num_colors]
        
    except Exception as e:
        # エラーが発生した場合はMedian Cut法にフォールバック
        print(f"オクトツリー法でエラーが発生したため、Median Cut法を使用します: {str(e)}")
        return get_median_cut_palette(pixels, num_colors)
 
def floyd_steinberg_dither_pil(img, palette):
    """
    PIL.Image と固定パレットによる Floyd-Steinberg ディザリング
    img     : PIL.Image (RGB)
    palette : list of (r,g,b) tuples
    戻り値  : PIL.Image (RGB)
    """
    arr = np.array(img, dtype=np.float32)
    h, w, _ = arr.shape
    out = np.zeros_like(arr)
    pal = np.array(palette, dtype=np.float32)
    for y in range(h):
        for x in range(w):
            old = arr[y, x]
            # 最近色検索
            diffs = pal - old[None, :]
            idx = np.argmin(np.sum(diffs * diffs, axis=1))
            new = pal[idx]
            out[y, x] = new
            err = old - new
            # 誤差拡散
            if x+1 < w:      arr[y,   x+1] += err * (7/16)
            if y+1 < h and x>0: arr[y+1, x-1] += err * (3/16)
            if y+1 < h:      arr[y+1, x  ] += err * (5/16)
            if y+1 < h and x+1<w: arr[y+1, x+1] += err * (1/16)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode='RGB')

FIXED_PALETTE = []  # グローバル固定パレット格納
def generate_preview_image(image_path, grid_size, color_step, top_color_limit, zoom_factor=10, 
                       custom_pixels=None, highlight_pos=None, hover_pos=None, color_algo="simple", highlight_color=None):
    """
    プレビュー画像を生成する関数
    
    Args:
        image_path: 元画像のパス
        grid_size: グリッドサイズ（ドット解像度）
        color_step: 色の量子化ステップ（simpleアルゴリズム用）
        top_color_limit: 使用する上位色数
        zoom_factor: 表示倍率
        custom_pixels: カスタムピクセルデータ（編集済みの場合）
        highlight_pos: ハイライトする位置
        hover_pos: ホバー中の位置
        color_algo: 減色アルゴリズム ("simple", "median_cut", "kmeans", "octree")
        highlight_color: ハイライトする色 (r, g, b)形式
    """
    # グリッド幅と高さを決定
    if custom_pixels is not None:
        # カスタムピクセルデータをそのまま使用
        if not isinstance(custom_pixels, np.ndarray) or custom_pixels.ndim != 3 or custom_pixels.shape[2] != 3:
            raise ValueError("custom_pixels must be a 3D numpy array with shape (height, width, 3)")
        pixels_array = custom_pixels
        grid_h, grid_w = pixels_array.shape[:2]
    else:
        # 画像または MRPAF からピクセルデータを生成
        ext = os.path.splitext(image_path)[1].lower()
        if ext == ".mrpaf":
            img = load_mrpaf(image_path).convert("RGB")
        else:
            img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size
        # grid_size を幅とみなし、高さをアスペクト比から算出
        grid_w = grid_size
        grid_h = int(round(grid_w * orig_h / orig_w)) if orig_w > 0 else grid_w
        grid_h = max(1, grid_h)
        img_resized = img.resize((grid_w, grid_h), resample=Image.NEAREST)
        pixels = np.array(img_resized).reshape(-1, 3)
        
        # 選択されたアルゴリズムで減色処理
        if color_algo == "simple":
            # 単純な量子化アルゴリズム（従来のもの）
            pixels_normalized = normalize_colors(pixels, color_step)
            colors = [tuple(c) for c in pixels_normalized]
            color_counts = Counter(colors)
            top_colors = [c for c, _ in color_counts.most_common(top_color_limit)]
            pixels_rounded = [map_to_closest_color(c, top_colors) for c in colors]
            
        elif color_algo == "median_cut":
            # メディアンカット法
            palette = get_median_cut_palette(pixels, top_color_limit)
            pixels_rounded = [map_to_closest_color(c, palette) for c in pixels]
            
        elif color_algo == "kmeans":
            # K-means法
            try:
                palette = get_kmeans_palette(pixels, top_color_limit)
                pixels_rounded = [map_to_closest_color(c, palette) for c in pixels]
            except ImportError:
                # scikit-learnがインストールされていない場合は単純アルゴリズムにフォールバック
                print("K-means減色にはscikit-learnが必要です。単純アルゴリズムを使用します。")
                pixels_normalized = normalize_colors(pixels, color_step)
                colors = [tuple(c) for c in pixels_normalized]
                color_counts = Counter(colors)
                top_colors = [c for c, _ in color_counts.most_common(top_color_limit)]
                pixels_rounded = [map_to_closest_color(c, top_colors) for c in colors]
                
        elif color_algo == "octree":
            # オクトツリー法
            palette = get_octree_palette(pixels, top_color_limit)
            pixels_rounded = [map_to_closest_color(c, palette) for c in pixels]
        elif color_algo == "fixed_palette":
            # 固定パレット + Floyd-Steinberg ディザリング
            global FIXED_PALETTE
            if not FIXED_PALETTE:
                # パレット未設定なら通常量子化にフォールバック
                pixels_normalized = normalize_colors(pixels, color_step)
                colors = [tuple(c) for c in pixels_normalized]
                color_counts = Counter(colors)
                top_colors = [c for c, _ in color_counts.most_common(top_color_limit)]
                pixels_rounded = [map_to_closest_color(c, top_colors) for c in colors]
            else:
                # ディザリング適用
                dithered = floyd_steinberg_dither_pil(img_resized, FIXED_PALETTE)
                flat = np.array(dithered).reshape(-1, 3)
                pixels_rounded = [tuple(c) for c in flat]
        elif color_algo == "none":
            # 減色なし - 元の色をそのまま使用
            pixels_rounded = pixels.tolist()  # NumPy配列をリストに変換
        else:
            # デフォルトは単純アルゴリズム
            pixels_normalized = normalize_colors(pixels, color_step)
            colors = [tuple(c) for c in pixels_normalized]
            color_counts = Counter(colors)
            top_colors = [c for c, _ in color_counts.most_common(top_color_limit)]
            pixels_rounded = [map_to_closest_color(c, top_colors) for c in colors]
        
        # 適切な形状のnumpy配列に変換
        pixels_array = np.array(pixels_rounded, dtype=np.uint8).reshape((grid_h, grid_w, 3))
    
    # 透過色（黒=0,0,0）を特別処理
    # RGBAモードで新しい画像を作成してアルファチャンネルを追加
    img_rgba = np.zeros((pixels_array.shape[0], pixels_array.shape[1], 4), dtype=np.uint8)
    img_rgba[:, :, :3] = pixels_array  # RGB値をコピー
    
    # 黒色（0,0,0）のピクセルを透明に設定
    black_mask = (pixels_array[:, :, 0] == 0) & (pixels_array[:, :, 1] == 0) & (pixels_array[:, :, 2] == 0)
    img_rgba[black_mask, 3] = 0  # 透明に設定
    img_rgba[~black_mask, 3] = 255  # 非透明に設定
    
    # RGBA画像を作成
    img_preview = Image.fromarray(img_rgba, mode="RGBA")
    
    # 透明部分が見えるように市松模様の背景を作成
    from PIL import ImageDraw
    width_px = grid_w * zoom_factor
    height_px = grid_h * zoom_factor
    checkerboard = Image.new('RGBA', (width_px, height_px), (255, 255, 255, 255))
    pattern = Image.new('RGBA', (zoom_factor * 2, zoom_factor * 2), (255, 255, 255, 0))
    draw = ImageDraw.Draw(pattern)
    draw.rectangle((0, 0, zoom_factor, zoom_factor), fill=(200, 200, 200, 255))
    draw.rectangle((zoom_factor, zoom_factor, zoom_factor * 2, zoom_factor * 2), fill=(200, 200, 200, 255))
    
    # 市松模様パターンを繰り返し配置
    for y in range(0, height_px, zoom_factor * 2):
        for x in range(0, width_px, zoom_factor * 2):
            checkerboard.paste(pattern, (x, y), pattern)
    
    # 拡大したプレビュー画像
    img_preview = img_preview.resize((width_px, height_px), resample=Image.NEAREST)
    
    # 市松模様の背景と合成
    result = Image.alpha_composite(checkerboard, img_preview)
    
    # 共通の枠線描画関数
    def draw_grid_highlight(grid_pos, color, width_factor=10):
        grid_x, grid_y = grid_pos
        # 有効なグリッド位置かチェック
        if 0 <= grid_x < grid_w and 0 <= grid_y < grid_h:
            draw = ImageDraw.Draw(result)
            # ドットの周りに枠線を描画
            x0 = grid_x * zoom_factor
            y0 = grid_y * zoom_factor
            x1 = x0 + zoom_factor - 1
            y1 = y0 + zoom_factor - 1
            
            # 枠線の太さを計算
            line_width = max(1, zoom_factor // width_factor)
            
            # 四角形の枠線を描画
            draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)
    
    # ホバー中のドットを薄いハイライト表示
    if hover_pos is not None:
        draw_grid_highlight(hover_pos, (0, 180, 255, 220), width_factor=15)  # 青色の薄い枠線
    
    # 選択されたドットを強調ハイライト表示
    if highlight_pos is not None:
        draw_grid_highlight(highlight_pos, (255, 0, 0, 255), width_factor=10)  # 赤色の枠線
    
    # 特定の色に近いドットをすべてハイライト（ユークリッド距離による近似）
    if highlight_color is not None:
        r, g, b = highlight_color
        # ハイライトに使用する距離の閾値（0-255の色空間）
        threshold = 30
        for y in range(grid_h):
            for x in range(grid_w):
                pixel_color = tuple(pixels_array[y, x])
                # 色空間で近い色を検出
                if distance.euclidean(pixel_color, (r, g, b)) <= threshold:
                    draw_grid_highlight((x, y), (255, 0, 0, 255), width_factor=15)
    
    return result

# -------------------------------
# モデル生成関数
# -------------------------------
def generate_html_report(self, stl_path, mesh):
    """STL情報とアプリの情報をHTMLレポートとして保存する"""
    try:
        # HTMLファイルパスを取得（STLと同じ名前＋.html）
        html_path = f"{os.path.splitext(stl_path)[0]}.html"
        
        # パラメータ値を取得
        params = {key: spin.value() for key, spin in self.controls.items()}
        
        # オリジナル画像と減色プレビュー画像のパス
        timestamp = int(time.time())
        original_img_path = f"{os.path.splitext(stl_path)[0]}_original_{timestamp}.png"
        preview_img_path = f"{os.path.splitext(stl_path)[0]}_preview_{timestamp}.png"
        stl_preview_img_path = f"{os.path.splitext(stl_path)[0]}_stl_preview_{timestamp}.png"
        
        # 画像を保存
        if self.image_path:
            orig_img = Image.open(self.image_path)
            orig_img.save(original_img_path)
        
        # プレビュー画像を保存
        if self.preview_pixmap:
            self.preview_pixmap.save(preview_img_path)
        
        # STLプレビュー画像を生成・保存
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # トリメッシュのメッシュをMatplotlibで描画
        vertices = mesh.vertices
        faces = mesh.faces
        
        # メッシュをプロット
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                        triangles=faces, color='lightgray', alpha=0.8, shade=True)
        
        # 画軸の設定
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # カメラアングルを等角投影に近づける
        ax.view_init(elev=30, azim=45)
        
        # 軸を均等にして歪みを防ぐ
        plt.tight_layout()
        plt.savefig(stl_preview_img_path)
        plt.close()
        
        # 色の使用率とボリュームを計算
        color_stats = self.get_color_statistics(mesh)
        
        # 色テーブルHTMLを生成
        color_table_html = self.generate_color_table_html(color_stats)
        
        # HTMLレポートを生成
        html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ドットプレート生成レポート</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
        .section {{ flex: 1; min-width: 300px; background: #f9f9f9; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        .color-swatch {{ width: 24px; height: 24px; display: inline-block; border: 1px solid #ccc; }}
        img {{ max-width: 100%; height: auto; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .image-container {{ display: flex; justify-content: space-between; gap: 20px; flex-wrap: wrap; }}
        .image-box {{ flex: 1; min-width: 300px; }}
        footer {{ margin-top: 30px; text-align: center; font-size: 0.8em; color: #888; }}
    </style>
</head>
<body>
    <h1>ドットプレート生成レポート</h1>
    <div class="container">
        <div class="section">
            <h2>プロジェクト情報</h2>
            <table>
                <tr><th>項目</th><th>値</th></tr>
                <tr><td>元画像</td><td>{os.path.basename(self.image_path) if self.image_path else 'なし'}</td></tr>
                <tr><td>生成日時</td><td>{time.strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
                <tr><td>グリッドサイズ</td><td>{self.controls['grid_size'].value()} x {self.controls['grid_size'].value()}</td></tr>
                <tr><td>ドットサイズ</td><td>{self.controls['dot_size'].value()} mm</td></tr>
                <tr><td>使用色数</td><td>{len(color_stats)}</td></tr>
                <tr><td>減色アルゴリズム</td><td>{self.color_algo_combo.currentText()}</td></tr>
                <tr><td>STLファイル名</td><td>{os.path.basename(stl_path)}</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>パラメータ設定</h2>
            <table>
                <tr><th>パラメータ</th><th>値</th></tr>
                {''.join([f"<tr><td>{key}</td><td>{value}</td></tr>" for key, value in params.items()])}
                <tr><td>Wall Color</td><td style="display:flex;align-items:center;"><div class="color-swatch" style="background-color:rgb{tuple(self.wall_color) if isinstance(self.wall_color, tuple) else self.wall_color.getRgb()[:3]};"></div>&nbsp;RGB{tuple(self.wall_color) if isinstance(self.wall_color, tuple) else self.wall_color.getRgb()[:3]}</td></tr>
                <tr><td>同色ドット壁省略</td><td>{'オン' if self.merge_walls_checkbox.isChecked() else 'オフ'}</td></tr>
            </table>
        </div>
    </div>
    
    <div class="section">
        <h2>色情報</h2>
        {color_table_html}
    </div>
    
    <h2>プレビュー</h2>
    <div class="image-container">
        <div class="image-box">
            <h3>オリジナル画像</h3>
            <img src="{os.path.basename(original_img_path)}" alt="オリジナル画像">
        </div>
        <div class="image-box">
            <h3>減色済みプレビュー</h3>
            <img src="{os.path.basename(preview_img_path)}" alt="プレビュー画像">
        </div>
        <div class="image-box">
            <h3>3Dモデルプレビュー</h3>
            <img src="{os.path.basename(stl_preview_img_path)}" alt="STLプレビュー">
        </div>
    </div>
    
    <footer>
        <p>Generated by Dot Plate Generator • {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </footer>
</body>
</html>
"""
        
        # HTMLファイルに保存
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"HTMLレポートを保存しました: {html_path}")
        return html_path
        
    except Exception as e:
        print(f"HTMLレポート生成中にエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_color_statistics(self, mesh):
    """メッシュ内の色の統計情報を取得する"""
    try:
        # 現在のピクセルデータを取得
        if not hasattr(self, 'pixels_rounded_np') or self.pixels_rounded_np is None:
            return []
        
        # 各色の出現回数をカウント
        color_counts = Counter()
        grid_size = self.pixels_rounded_np.shape[0]
        
        for y in range(grid_size):
            for x in range(grid_size):
                pixel_color = tuple(self.pixels_rounded_np[y, x])
                # 黒色（透明）をスキップ
                if pixel_color != (0, 0, 0):
                    color_counts[pixel_color] += 1
        
        # 各色のボリュームを計算
        dot_size = self.controls['dot_size'].value()
        wall_height = self.controls['wall_height'].value()
        base_height = self.controls['base_height'].value()
        
        # 色の統計情報を作成
        color_stats = []
        total_dots = sum(color_counts.values())
        
        for color, count in color_counts.items():
            # 1ドットあたりのボリュームを計算 (mm^3) - 簡易版
            dot_volume = dot_size * dot_size * (base_height + wall_height)
            color_volume = count * dot_volume
            
            # ボリューム百分率
            volume_percent = (count / total_dots) * 100 if total_dots > 0 else 0
            
            # 統計情報を追加
            color_stats.append({
                'color': color,
                'count': count,
                'percentage': (count / total_dots) * 100 if total_dots > 0 else 0,
                'volume': color_volume,
                'volume_percent': volume_percent
            })
        
        # 使用頻度順にソート
        color_stats.sort(key=lambda x: x['count'], reverse=True)
        
        return color_stats
        
    except Exception as e:
        print(f"色統計情報の取得中にエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def generate_color_table_html(self, color_stats):
    """色の統計情報からHTMLテーブルを生成する"""
    if not color_stats:
        return "<p>色情報が利用できません</p>"
    
    table_html = """
    <table>
        <tr>
            <th>色</th>
            <th>RGB値</th>
            <th>ドット数</th>
            <th>使用率</th>
            <th>体積 (mm³)</th>
            <th>体積比率</th>
        </tr>
    """
    
    for stat in color_stats:
        color = stat['color']
        color_rgb = f"rgb{color}"
        count = stat['count']
        percentage = f"{stat['percentage']:.1f}%"
        volume = f"{stat['volume']:.1f}"
        volume_percent = f"{stat['volume_percent']:.1f}%"
        
        table_html += f"""
        <tr>
            <td><div class="color-swatch" style="background-color:{color_rgb};"></div></td>
            <td>{color}</td>
            <td>{count}</td>
            <td>{percentage}</td>
            <td>{volume}</td>
            <td>{volume_percent}</td>
        </tr>
        """
    
    table_html += """
    </table>
    """
    
    return table_html
   
# -------------------------------
# MRPAF loader and helper functions
# -------------------------------
def decode_rle(rle_string, width, height):
    rows = rle_string.split("|")
    arr = []
    for row_str in rows:
        row = []
        for run in row_str.split(","):
            if not run:
                continue
            val, cnt = run.split(":")
            row.extend([int(val)] * int(cnt))
        arr.append(row)
    if len(arr) < height:
        arr += [[0] * width for _ in range(height - len(arr))]
    arr = [r[:width] + [0] * max(0, width - len(r)) for r in arr]
    return arr[:height]

def load_mrpaf(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    canvas_info = data.get("canvas", {})
    base_w = int(canvas_info.get("baseWidth", 0))
    base_h = int(canvas_info.get("baseHeight", 0))
    palette = {}
    for entry in data.get("palette", []):
        pid = entry.get("id")
        hexstr = entry.get("hex", "#00000000").lstrip("#")
        if len(hexstr) == 8:
            r = int(hexstr[0:2], 16)
            g = int(hexstr[2:4], 16)
            b = int(hexstr[4:6], 16)
            a = int(hexstr[6:8], 16)
        elif len(hexstr) == 6:
            r = int(hexstr[0:2], 16)
            g = int(hexstr[2:4], 16)
            b = int(hexstr[4:6], 16)
            a = 255
        else:
            r = g = b = a = 0
        palette[pid] = (r, g, b, a)
    base_img = Image.new("RGBA", (base_w, base_h), (0, 0, 0, 0))
    for layer in sorted(data.get("layers", []), key=lambda l: l.get("id", 0)):
        res = layer.get("resolution", {})
        w_hi = int(res.get("pixelArraySize", {}).get("width", 0))
        h_hi = int(res.get("pixelArraySize", {}).get("height", 0))
        scale = float(res.get("scale", 1.0))
        pix = layer.get("pixels")
        if not pix or (pix.get("encoding") is None and pix.get("format") is None):
            continue
        fmt_field = pix.get("encoding") or pix.get("format")
        enc = fmt_field.lower() if isinstance(fmt_field, str) else fmt_field
        if enc == "array":
            flat = pix.get("data", []) or []
            arr = []
            for i in range(h_hi):
                row = []
                for j in range(w_hi):
                    idx = i * w_hi + j
                    val = flat[idx] if idx < len(flat) else None
                    row.append(val if isinstance(val, int) else 0)
                arr.append(row)
        elif enc == "rle":
            arr = decode_rle(pix.get("data", ""), w_hi, h_hi)
        elif enc == "sparse":
            dims = pix.get("dimensions", {"width": w_hi, "height": h_hi})
            w_hi = int(dims.get("width", w_hi))
            h_hi = int(dims.get("height", h_hi))
            default = pix.get("defaultValue", 0)
            arr = [[default] * w_hi for _ in range(h_hi)]
            for item in pix.get("data", []):
                x = item.get("x", 0)
                y = item.get("y", 0)
                col = item.get("color", default)
                if 0 <= x < w_hi and 0 <= y < h_hi:
                    arr[y][x] = col
        else:
            raise NotImplementedError(f"Unsupported encoding: {enc}")
        img_hi = Image.new("RGBA", (w_hi, h_hi))
        px_hi = img_hi.load()
        for yy in range(h_hi):
            for xx in range(w_hi):
                cid = arr[yy][xx] if yy < len(arr) and xx < len(arr[yy]) else 0
                px_hi[xx, yy] = palette.get(cid, (0, 0, 0, 0))
        if scale != 1.0 and scale > 0:
            w_lo = int(round(w_hi / scale))
            h_lo = int(round(h_hi / scale))
            img_lo = img_hi.resize((w_lo, h_lo), resample=Image.NEAREST)
        else:
            img_lo = img_hi
        place = layer.get("placement", {})
        x0 = place.get("x", 0)
        y0 = place.get("y", 0)
        base_img.alpha_composite(img_lo, dest=(int(x0), int(y0)))
    return base_img

def generate_dot_plate_stl(image_path, output_path, grid_size, dot_size,
                           wall_thickness, wall_height, base_height,
                           color_step, top_color_limit, out_thickness=0.1, 
                           wall_color=(255, 255, 255), # 壁の色（デフォルトは白）
                           merge_same_color=False,     # 同じ色のドット間の内壁を省略するオプション
                           return_colors=False):
    # サポート: MRPAF ファイル読み込み
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".mrpaf":
        img = load_mrpaf(image_path).convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((grid_size, grid_size), resample=Image.NEAREST)
    pixels = np.array(img_resized).reshape(-1, 3)
    pixels_normalized = normalize_colors(pixels, color_step)
    colors = [tuple(c) for c in pixels_normalized]
    color_counts = Counter(colors)
    top_colors = [c for c, _ in color_counts.most_common(top_color_limit)]
    pixels_rounded = [map_to_closest_color(c, top_colors) for c in colors]
    pixels_rounded_np = np.array(pixels_rounded, dtype=np.uint8).reshape((grid_size, grid_size, 3))
    # 黒色（0,0,0）を透過色として扱い、マスクから除外する
    # Mask out transparent pixels (black) to discard them from STL
    mask = np.array([[tuple(px) != (0, 0, 0) for px in row] for row in pixels_rounded_np], dtype=np.uint8)
    # Do not fill interior holes here; preserve transparent areas
    
    base_blocks = []
    wall_blocks = []
    
    # 色情報とジオメトリの対応を保存
    color_mapping = {}
    
    for y in range(grid_size):
        for x in range(grid_size):
            if mask[y, x]:
                # 現在のピクセルの色を取得
                pixel_color = tuple(pixels_rounded_np[y, x])
                
                # 隣接ドットの確認（表示に利用するが壁の生成には直接影響させない）
                has_left = x > 0 and mask[y, x-1]
                has_right = x < grid_size - 1 and mask[y, x+1]
                has_top = y > 0 and mask[y-1, x]
                has_bottom = y < grid_size - 1 and mask[y+1, x]
                
                # 外周条件の確認（これは壁の生成に使用）
                if merge_same_color:
                    # 同じ色のドット間には壁を作らない場合の条件
                    is_left_edge = (x == 0 or not mask[y, x-1] or 
                                   (mask[y, x-1] and tuple(pixels_rounded_np[y, x-1]) != pixel_color))
                    is_right_edge = (x == grid_size - 1 or not mask[y, x+1] or 
                                    (mask[y, x+1] and tuple(pixels_rounded_np[y, x+1]) != pixel_color))
                    is_top_edge = (y == 0 or not mask[y-1, x] or 
                                  (mask[y-1, x] and tuple(pixels_rounded_np[y-1, x]) != pixel_color))
                    is_bottom_edge = (y == grid_size - 1 or not mask[y+1, x] or 
                                     (mask[y+1, x] and tuple(pixels_rounded_np[y+1, x]) != pixel_color))
                else:
                    # 従来通り、隣接するドットとの間に常に壁を作る
                    is_left_edge = x == 0 or not mask[y, x-1]
                    is_right_edge = x == grid_size - 1 or not mask[y, x+1]
                    is_top_edge = y == 0 or not mask[y-1, x]
                    is_bottom_edge = y == grid_size - 1 or not mask[y+1, x]
                
                # ベースと外周壁の外側拡張: プレート外周のみ適用
                extend_left = out_thickness if x == 0 else 0
                extend_right = out_thickness if x == grid_size - 1 else 0
                extend_top = out_thickness if y == 0 else 0
                extend_bottom = out_thickness if y == grid_size - 1 else 0
                
                # 基準座標を設定（拡張なしの場合）
                x0 = x * dot_size
                y0 = (grid_size - 1 - y) * dot_size
                
                # 各方向の拡張を考慮した座標と大きさの調整
                base_width = dot_size + extend_left + extend_right
                base_depth = dot_size + extend_top + extend_bottom
                
                # ベースブロックを適切な大きさで作成
                block = box(extents=[base_width, base_depth, base_height])
                
                # 位置の調整（中心座標に移動）
                x_center = x0 - extend_left + base_width / 2
                y_center = y0 - extend_top + base_depth / 2
                block.apply_translation([x_center, y_center, base_height / 2])
                
                # 色情報を追加
                color_mapping[len(base_blocks)] = {
                    'type': 'base', 
                    'color': pixel_color, 
                    'position': [x, y]
                }
                
                base_blocks.append(block)
                # 壁の生成
                # 右方向
                if x == grid_size - 1:
                    # 右外周壁
                    thickness = wall_thickness + out_thickness
                    w = box(extents=[thickness, dot_size, wall_height])
                    pos_x = x0 + dot_size + thickness / 2
                    pos_y = y0 + dot_size / 2
                    w.apply_translation([pos_x, pos_y, base_height + wall_height / 2])
                    wall_blocks.append(w)
                else:
                    neighbor_mask = mask[y, x+1]
                    neighbor_color = tuple(pixels_rounded_np[y, x+1]) if neighbor_mask else None
                    if (not neighbor_mask) or (not merge_same_color) or (merge_same_color and neighbor_color != pixel_color):
                        # 内部右壁
                        thickness = wall_thickness
                        w = box(extents=[thickness, dot_size, wall_height])
                        pos_x = x0 + dot_size + thickness / 2
                        pos_y = y0 + dot_size / 2
                        w.apply_translation([pos_x, pos_y, base_height + wall_height / 2])
                        wall_blocks.append(w)
                # 下方向
                if y == grid_size - 1:
                    # 下外周壁
                    thickness = wall_thickness + out_thickness
                    w = box(extents=[dot_size, thickness, wall_height])
                    pos_x = x0 + dot_size / 2
                    pos_y = y0 - thickness / 2
                    w.apply_translation([pos_x, pos_y, base_height + wall_height / 2])
                    wall_blocks.append(w)
                else:
                    neighbor_mask = mask[y+1, x]
                    neighbor_color = tuple(pixels_rounded_np[y+1, x]) if neighbor_mask else None
                    if (not neighbor_mask) or (not merge_same_color) or (merge_same_color and neighbor_color != pixel_color):
                        # 内部下壁
                        thickness = wall_thickness
                        w = box(extents=[dot_size, thickness, wall_height])
                        pos_x = x0 + dot_size / 2
                        pos_y = y0 - thickness / 2
                        w.apply_translation([pos_x, pos_y, base_height + wall_height / 2])
                        wall_blocks.append(w)
                # 左方向
                if x == 0:
                    # 左外周壁
                    thickness = wall_thickness + out_thickness
                    w = box(extents=[thickness, dot_size, wall_height])
                    pos_x = x0 - thickness / 2
                    pos_y = y0 + dot_size / 2
                    w.apply_translation([pos_x, pos_y, base_height + wall_height / 2])
                    wall_blocks.append(w)
                else:
                    neighbor_mask = mask[y, x-1]
                    neighbor_color = tuple(pixels_rounded_np[y, x-1]) if neighbor_mask else None
                    if (not neighbor_mask) or (not merge_same_color) or (merge_same_color and neighbor_color != pixel_color):
                        # 内部左壁
                        thickness = wall_thickness
                        w = box(extents=[thickness, dot_size, wall_height])
                        pos_x = x0 - thickness / 2
                        pos_y = y0 + dot_size / 2
                        w.apply_translation([pos_x, pos_y, base_height + wall_height / 2])
                        wall_blocks.append(w)
                # 上方向
                if y == 0:
                    # 上外周壁
                    thickness = wall_thickness + out_thickness
                    w = box(extents=[dot_size, thickness, wall_height])
                    pos_x = x0 + dot_size / 2
                    pos_y = y0 + dot_size + thickness / 2
                    w.apply_translation([pos_x, pos_y, base_height + wall_height / 2])
                    wall_blocks.append(w)
                else:
                    neighbor_mask = mask[y-1, x]
                    neighbor_color = tuple(pixels_rounded_np[y-1, x]) if neighbor_mask else None
                    if (not neighbor_mask) or (not merge_same_color) or (merge_same_color and neighbor_color != pixel_color):
                        # 内部上壁
                        thickness = wall_thickness
                        w = box(extents=[dot_size, thickness, wall_height])
                        pos_x = x0 + dot_size / 2
                        pos_y = y0 + dot_size + thickness / 2
                        w.apply_translation([pos_x, pos_y, base_height + wall_height / 2])
                        wall_blocks.append(w)
    
    # メッシュを作成
    mesh = trimesh.util.concatenate(base_blocks + wall_blocks)
    
    # 色情報を設定
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'face_colors'):
        # デフォルト色（指定した壁の色）
        r, g, b = wall_color
        wall_color_array = np.array([r, g, b, 255], dtype=np.uint8)
        mesh.visual.face_colors = np.ones((len(mesh.faces), 4), dtype=np.uint8) * wall_color_array
        
        # 各面がどのオブジェクトに属するかをマッピング
        face_index = 0
        
        # ベースブロックの色を設定
        for i, block in enumerate(base_blocks):
            if i in color_mapping:
                color_info = color_mapping[i]
                r, g, b = color_info['color']
                color = np.array([r, g, b, 255], dtype=np.uint8)
                
                # このブロックの面数
                num_faces = len(block.faces)
                
                # 該当する面すべてに色を設定
                mesh.visual.face_colors[face_index:face_index + num_faces] = color
                
                # 次のブロックの最初の面インデックス
                face_index += num_faces
        
        # 壁ブロックは指定色
        # face_indexは既にベースブロックの終了位置に設定されているので、追加の処理は不要
    
    # STLファイルに保存
    mesh.export(output_path)
    
    # 色情報を返すかどうか
    if return_colors:
        return mesh, pixels_rounded_np
    return mesh

def generate_layered_stl(pixels_rounded_np, output_path, grid_size, dot_size, base_height, wall_thickness, wall_height, layer_heights, layer_order):
    """Generate STL with per-color layer heights."""
    # Determine actual grid dimensions from pixel array (height x width)
    grid_h, grid_w = pixels_rounded_np.shape[:2]
    # Create base blocks for non-transparent pixels (exclude transparent color)
    blocks = []
    cumulative_z = base_height
    transparent_color = (0, 0, 0)
    # Base layer: for each pixel not transparent, add a block of base_height
    for y in range(grid_h):
        for x in range(grid_w):
            if tuple(pixels_rounded_np[y, x]) != transparent_color:
                x0 = x * dot_size
                y0 = (grid_h - 1 - y) * dot_size
                base_block = box(extents=[dot_size, dot_size, base_height])
                base_block.apply_translation([
                    x0 + dot_size / 2,
                    y0 + dot_size / 2,
                    base_height / 2
                ])
                blocks.append(base_block)
    # Prepare color layers in specified order
    colors = [c for c in layer_order if c in layer_heights]
    # Process each color layer
    for idx, color in enumerate(colors):
        h = layer_heights[color]
        if h <= 0:
            cumulative_z += h
            continue
        z0 = cumulative_z
        # Support region: fill under higher layers
        support_colors = colors[idx:]
        # Support region mask under this and higher layers
        mask_support = np.zeros((grid_h, grid_w), dtype=bool)
        for sc in support_colors:
            sc_arr = np.array(sc, dtype=np.uint8)
            mask_support |= np.all(pixels_rounded_np == sc_arr, axis=2)
        # Add support blocks
        for y, x in np.argwhere(mask_support):
            x0 = x * dot_size
            y0 = (grid_h - 1 - y) * dot_size
            block = box(extents=[dot_size, dot_size, h])
            block.apply_translation([x0 + dot_size/2, y0 + dot_size/2, z0 + h/2])
            blocks.append(block)
        # Add perimeter walls for this layer's actual color region (height = wall_height)
        color_arr = np.array(color, dtype=np.uint8)
        mask_color = np.all(pixels_rounded_np == color_arr, axis=2)
        wt = wall_thickness
        for y, x in np.argwhere(mask_color):
            x0 = x * dot_size
            y0 = (grid_h - 1 - y) * dot_size
            y_center = y0 + dot_size / 2
            # Left wall
            if x == 0 or not mask_color[y, x-1]:
                w = box(extents=[wt, dot_size, wall_height])
                w.apply_translation([x0 - wt/2, y_center, z0 + wall_height/2])
                blocks.append(w)
            # Right wall
            if x == grid_w-1 or not mask_color[y, x+1]:
                w = box(extents=[wt, dot_size, wall_height])
                w.apply_translation([x0 + dot_size + wt/2, y_center, z0 + wall_height/2])
                blocks.append(w)
            # Top wall (positive Y direction)
            if y == 0 or not mask_color[y-1, x]:
                w = box(extents=[dot_size, wt, wall_height])
                w.apply_translation([x0 + dot_size/2, y0 + dot_size + wt/2, z0 + wall_height/2])
                blocks.append(w)
            # Bottom wall (negative Y direction)
            if y == grid_h-1 or not mask_color[y+1, x]:
                w = box(extents=[dot_size, wt, wall_height])
                w.apply_translation([x0 + dot_size/2, y0 - wt/2, z0 + wall_height/2])
                blocks.append(w)
        cumulative_z += h
    # Concatenate all blocks and export
    mesh = trimesh.util.concatenate(blocks)
    mesh.export(output_path)
    return mesh

def generate_checkerboard_stl(grid_size, dot_size, base_height,
                              wall_thickness, wall_height, mask=None):
    """
    改良版市松模様パターンのSTL生成（6段階高さ）
    
    隣接する8方向（縦横斜め）のドットの高さ重複を大幅削減し、
    斜め方向への色移りを防止する。
    
    Args:
        grid_size: 1辺あたりのマス数
        dot_size: 各マスのサイズ(mm)
        base_height: ベースプレート厚み(mm)
        wall_thickness: 側壁の厚み(mm)
        wall_height: 凸凹の高さ(mm)
        mask: 2D boolean配列。False はモデル除去。
    
    Returns:
        trimesh.Trimesh: 生成されたメッシュ
    """
    import trimesh
    from trimesh.creation import box
    import numpy as np

    cells = []
    
    # 6段階の高さレベルを定義（実用的な最適解）
    height_levels = [
        -wall_height,        # レベル0: 最も深い凹
        -wall_height * 2/3,  # レベル1: 深い凹  
        -wall_height * 1/3,  # レベル2: 浅い凹
        +wall_height * 1/3,  # レベル3: 浅い凸
        +wall_height * 2/3,  # レベル4: 高い凸
        +wall_height         # レベル5: 最も高い凸
    ]
    
    def get_height_level(i, j):
        """
        座標(i,j)に対応する高さレベル（0-5）を取得
        隣接する8方向の高さ重複を最小化するよう配置
        """
        # 6x6パターンマトリックス（隣接8方向の重複を大幅削減）
        pattern_matrix = [
            [0, 5, 2, 4, 1, 3],
            [3, 1, 4, 2, 5, 0],
            [1, 4, 0, 5, 3, 2],
            [4, 2, 5, 1, 0, 3],
            [2, 0, 3, 4, 1, 5],
            [5, 3, 1, 0, 2, 4]
        ]
        
        pattern_x = i % 6
        pattern_y = j % 6
        return pattern_matrix[pattern_y][pattern_x]
    
    # ベースセルの生成（mask指定で各セルごとに生成）
    for i in range(grid_size):
        for j in range(grid_size):
            if mask is not None and not mask[j, i]:
                continue
                
            x0 = i * dot_size
            y0 = j * dot_size
            base_cube = box(extents=(dot_size, dot_size, base_height))
            base_cube.apply_translation((x0 + dot_size/2,
                                       y0 + dot_size/2,
                                       base_height/2))
            cells.append(base_cube)
    
    # 輪郭検知: 側壁の追加
    for i in range(grid_size):
        for j in range(grid_size):
            if mask is not None and not mask[j, i]:
                continue
                
            x0 = i * dot_size
            y0 = j * dot_size
            
            for dx, dy, orient in [(-1, 0, 'L'), (1, 0, 'R'), (0, -1, 'B'), (0, 1, 'T')]:
                ni, nj = i + dx, j + dy
                neighbor = False
                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    neighbor = mask[nj, ni] if mask is not None else True
                    
                if neighbor:
                    continue
                    
                # 壁ボックス作成
                if orient in ('L', 'R'):
                    w = box(extents=(wall_thickness, dot_size, base_height))
                    cx = (x0 - wall_thickness/2) if orient == 'L' else (x0 + dot_size + wall_thickness/2)
                    cy = y0 + dot_size/2
                else:
                    w = box(extents=(dot_size, wall_thickness, base_height))
                    cx = x0 + dot_size/2
                    cy = (y0 - wall_thickness/2) if orient == 'B' else (y0 + dot_size + wall_thickness/2)
                    
                w.apply_translation((cx, cy, base_height/2))
                cells.append(w)
    
    # 6段階凸凹パターン
    for i in range(grid_size):
        for j in range(grid_size):
            if mask is not None and not mask[j, i]:
                continue
                
            x0 = i * dot_size
            y0 = j * dot_size
            
            # この位置の高さレベルを取得
            level = get_height_level(i, j)
            height_offset = height_levels[level]
            
            # 凸凹ブロック作成
            h = abs(height_offset)
            if height_offset > 0:
                # 凸（上に突出）
                zc = base_height + h/2
            else:
                # 凹（下に凹む）
                zc = base_height - h/2
                
            cube = box(extents=(dot_size, dot_size, h))
            cube.apply_translation((x0 + dot_size/2,
                                  y0 + dot_size/2,
                                  zc))
            cells.append(cube)
    
    return trimesh.util.concatenate(cells) if cells else None

def generate_layer_stack_stl(pixels_rounded_np, output_base_path, grid_size, dot_size, 
                            wall_thickness, wall_height, base_height, out_thickness,
                            layer_color_order, layer_heights):
    """
    レイヤースタックモード用のSTL生成
    各レイヤーを個別のSTLファイルとして出力
    
    レイヤー構造：
    - 全レイヤーのベース高さ = base_height（統一）
    - レイヤー1: ビル高さ = wall_height
    - レイヤー2: ビル高さ = wall_height - base_height × 1  
    - レイヤーn: ビル高さ = wall_height - base_height × (n-1)
    - 重ね合わせ後の最終ビル高さ = base_height + wall_height（統一）
    - 下位レイヤーのビル部分は上位レイヤーで貫通穴として処理
    - CSGを使わず、セル単位での構築による安定した処理
    """
    import trimesh
    from trimesh.creation import box
    import numpy as np
    import os
    
    if len(layer_color_order) == 0:
        return []
    
    generated_meshes = []
    
    # 各レイヤーのマスクを事前計算
    layer_masks = {}
    for layer_idx, color in enumerate(layer_color_order):
        color_arr = np.array(color, dtype=np.uint8)
        layer_masks[layer_idx] = np.all(pixels_rounded_np == color_arr, axis=2)
    
    # 各レイヤーを処理
    for layer_idx, color in enumerate(layer_color_order):
        layer_num = layer_idx + 1  # レイヤー番号は1から開始
        
        print(f"\nレイヤー{layer_num}処理開始 - 色: RGB{color}")
        
        # レイヤーのベース高さとビル高さを計算
        layer_base_height = base_height  # 全レイヤー共通
        layer_building_height = max(wall_height - base_height * (layer_num - 1), wall_height / 3)
        
        print(f"ベース高さ: {layer_base_height}")
        print(f"ビル高さ: {layer_building_height}")
        
        # 上位レイヤーの穴位置を計算（このレイヤーより上位の色）
        upper_layer_holes = set()
        current_layer_positions = set()
        non_building_positions = set()
        
        for y in range(grid_size):
            for x in range(grid_size):
                dot_center_x = x * dot_size + dot_size / 2
                dot_center_y = (grid_size - 1 - y) * dot_size + dot_size / 2
                
                # このレイヤーのビル位置かチェック
                if layer_masks[layer_idx][y, x]:
                    current_layer_positions.add((dot_center_x, dot_center_y))
                    continue
                
                # 上位レイヤー（layer_idx+1以降）のビル位置かチェック
                is_upper_layer = False
                for upper_idx in range(layer_idx + 1, len(layer_color_order)):
                    if layer_masks[upper_idx][y, x]:
                        upper_layer_holes.add((dot_center_x, dot_center_y))
                        is_upper_layer = True
                        break
                
                # どのレイヤーのビルでもない場合は空洞化対象
                if not is_upper_layer:
                    non_building_positions.add((dot_center_x, dot_center_y))
        
        print(f"  上位レイヤー穴: {len(upper_layer_holes)}個")
        print(f"  ビル建設位置: {len(current_layer_positions)}個")  
        print(f"  空洞化位置: {len(non_building_positions)}個")
        
        # 全体サイズ計算
        total_size = grid_size * dot_size + 2 * out_thickness
        
        # ベースプレートをグリッド単位で構築（穴と空洞を除く）
        layer_blocks = []
        
        # 外周部分のベースプレート
        # 左側
        left_block = box(extents=[out_thickness, total_size, layer_base_height])
        left_block.apply_translation([-out_thickness/2, (total_size)/2 - out_thickness, layer_base_height/2])
        layer_blocks.append(left_block)
        
        # 右側  
        right_block = box(extents=[out_thickness, total_size, layer_base_height])
        right_block.apply_translation([grid_size * dot_size + out_thickness/2, (total_size)/2 - out_thickness, layer_base_height/2])
        layer_blocks.append(right_block)
        
        # 上側
        top_block = box(extents=[grid_size * dot_size, out_thickness, layer_base_height])
        top_block.apply_translation([(grid_size * dot_size)/2, -out_thickness/2, layer_base_height/2])
        layer_blocks.append(top_block)
        
        # 下側
        bottom_block = box(extents=[grid_size * dot_size, out_thickness, layer_base_height])
        bottom_block.apply_translation([(grid_size * dot_size)/2, grid_size * dot_size + out_thickness/2, layer_base_height/2])
        layer_blocks.append(bottom_block)
        
        # グリッド内のベースプレート（必要な部分のみ）
        for y in range(grid_size):
            for x in range(grid_size):
                dot_center_x = x * dot_size + dot_size / 2
                dot_center_y = (grid_size - 1 - y) * dot_size + dot_size / 2
                
                # 上位レイヤーの穴でも空洞化位置でもない場合のみベースプレートを作成
                if (dot_center_x, dot_center_y) not in upper_layer_holes and (dot_center_x, dot_center_y) not in non_building_positions:
                    base_cell = box(extents=[dot_size, dot_size, layer_base_height])
                    base_cell.apply_translation([dot_center_x, dot_center_y, layer_base_height/2])
                    layer_blocks.append(base_cell)
        
        # ビル建設（このレイヤーの色の位置のみ）
        for y in range(grid_size):
            for x in range(grid_size):
                if layer_masks[layer_idx][y, x]:
                    # ドットの中心座標計算
                    dot_center_x = x * dot_size + dot_size / 2
                    dot_center_y = (grid_size - 1 - y) * dot_size + dot_size / 2
                    
                    # ビル部分を (dot_size - wall_thickness) × (dot_size - wall_thickness) × layer_building_height で作成
                    building_size = dot_size - wall_thickness
                    building_block = box(extents=[building_size, building_size, layer_building_height])
                    building_z = layer_base_height + layer_building_height / 2
                    building_block.apply_translation([dot_center_x, dot_center_y, building_z])
                    layer_blocks.append(building_block)
        
        # レイヤーメッシュを統合してファイル出力
        if layer_blocks:
            try:
                layer_mesh = trimesh.util.concatenate(layer_blocks)
                layer_filename = f"{output_base_path}_layer_{layer_num:02d}_{color[0]:03d}_{color[1]:03d}_{color[2]:03d}.stl"
                layer_mesh.export(layer_filename)
                generated_meshes.append(layer_mesh)
                print(f"レイヤー {layer_num} (色: RGB{color}) を {layer_filename} に出力しました")
                print(f"頂点数: {len(layer_mesh.vertices)}, 面数: {len(layer_mesh.faces)}")
                print(f"バウンディングボックス: {layer_mesh.bounds}")
            except Exception as e:
                print(f"レイヤー {layer_num} のメッシュ生成エラー: {str(e)}")
                continue
    
    return generated_meshes

def generate_plastic_model_stl(pixels_rounded_np, output_base_path, grid_size, dot_size, 
                               wall_thickness, wall_height, base_height, out_thickness,
                               layer_color_order, connection_thickness=0.1, sprue_width=0.3):
    """
    プラモデル組み立て式モード用のSTL生成
    各色レイヤーを薄皮で連結した組み立て式パーツとして出力
    """
    import trimesh
    from trimesh.creation import box, cylinder
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    from scipy.sparse.csgraph import minimum_spanning_tree
    
    if len(layer_color_order) == 0:
        return []
    
    generated_meshes = []
    
    def create_connection_bridges(positions, grid_size, dot_size):
        """最小全域木アルゴリズムで同色ドット間を効率的に連結"""
        if len(positions) <= 1:
            return []
        
        bridges = []
        
        # 座標を実際の物理位置に変換
        physical_positions = []
        for x, y in positions:
            phys_x = x * dot_size + dot_size / 2
            phys_y = (grid_size - 1 - y) * dot_size + dot_size / 2
            physical_positions.append([phys_x, phys_y])
        
        physical_positions = np.array(physical_positions)
        
        # 最小全域木で連結パスを計算
        distances = pdist(physical_positions)
        dist_matrix = squareform(distances)
        mst = minimum_spanning_tree(dist_matrix)
        mst_array = mst.toarray()
        
        # MST のエッジからブリッジを生成
        for i in range(len(physical_positions)):
            for j in range(i + 1, len(physical_positions)):
                if mst_array[i, j] > 0:  # エッジが存在
                    pos1 = physical_positions[i]
                    pos2 = physical_positions[j]
                    
                    # 2点間のブリッジを作成
                    bridge = create_bridge_between_points(pos1, pos2, connection_thickness, sprue_width)
                    if bridge:
                        bridges.append(bridge)
        
        return bridges
    
    def create_bridge_between_points(pos1, pos2, thickness, width):
        """2点間の薄皮ブリッジを作成"""
        vec = pos2 - pos1
        length = np.linalg.norm(vec)
        
        if length < 0.01:
            return None
        
        # ブリッジの中心位置と向き
        center = (pos1 + pos2) / 2
        center_3d = [center[0], center[1], base_height + thickness / 2]
        
        # 回転角度計算
        angle = np.arctan2(vec[1], vec[0])
        
        # ブリッジボックス作成
        bridge = box(extents=[length, width, thickness])
        bridge.apply_translation(center_3d)
        
        # Z軸周りの回転
        rotation_matrix = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        bridge.apply_transform(rotation_matrix)
        
        return bridge
    
    def create_sprue_system(positions, grid_size, dot_size):
        """ランナーシステム（取り外し可能な支持構造）を作成"""
        if len(positions) == 0:
            return []
        
        sprue_blocks = []
        
        # メインランナー（外周に配置）
        main_runner_y = grid_size * dot_size + out_thickness * 2
        main_runner = box(extents=[grid_size * dot_size, sprue_width, connection_thickness])
        main_runner.apply_translation([
            grid_size * dot_size / 2,
            main_runner_y,
            base_height + connection_thickness / 2
        ])
        sprue_blocks.append(main_runner)
        
        # 各ドットからメインランナーへの接続
        for x, y in positions:
            dot_x = x * dot_size + dot_size / 2
            dot_y = (grid_size - 1 - y) * dot_size + dot_size / 2
            
            # ドットからメインランナーまでの垂直接続
            connection_length = main_runner_y - dot_y - dot_size / 2
            
            if connection_length > 0:
                connector = box(extents=[sprue_width, connection_length, connection_thickness])
                connector_y = dot_y + dot_size / 2 + connection_length / 2
                connector.apply_translation([
                    dot_x,
                    connector_y,
                    base_height + connection_thickness / 2
                ])
                sprue_blocks.append(connector)
        
        return sprue_blocks
    
    def add_assembly_pins(blocks, color, grid_size):
        """組み立て用のピン・穴システムを追加"""
        pin_blocks = []
        
        # 4隅にピン配置
        pin_radius = 0.5
        pin_height = wall_height / 2
        corner_positions = [
            (-out_thickness / 2, -out_thickness / 2),
            (grid_size * dot_size + out_thickness / 2, -out_thickness / 2),
            (-out_thickness / 2, grid_size * dot_size + out_thickness / 2),
            (grid_size * dot_size + out_thickness / 2, grid_size * dot_size + out_thickness / 2)
        ]
        
        color_index = layer_color_order.index(color) if color in layer_color_order else 0
        
        for i, (px, py) in enumerate(corner_positions):
            if color_index == 0:  # 最下層にはピン
                pin = cylinder(radius=pin_radius, height=pin_height)
                pin.apply_translation([px, py, base_height + pin_height / 2])
                pin_blocks.append(pin)
        
        return pin_blocks
    
    # 各色レイヤーを処理
    for color in layer_color_order:
        # この色のドット位置を収集
        color_arr = np.array(color, dtype=np.uint8)
        color_mask = np.all(pixels_rounded_np == color_arr, axis=2)
        
        positions = []
        for y in range(grid_size):
            for x in range(grid_size):
                if color_mask[y, x]:
                    positions.append((x, y))
        
        if not positions:
            continue
        
        layer_blocks = []
        
        # 各ドットのビル構造を作成
        for x, y in positions:
            # メインビル
            building_block = box(extents=[dot_size, dot_size, wall_height])
            building_x = x * dot_size + dot_size / 2
            building_y = (grid_size - 1 - y) * dot_size + dot_size / 2
            building_z = base_height + wall_height / 2
            building_block.apply_translation([building_x, building_y, building_z])
            layer_blocks.append(building_block)
            
            # ビルの外周壁
            for dx, dy, wall_type in [(-1, 0, 'left'), (1, 0, 'right'), (0, -1, 'bottom'), (0, 1, 'top')]:
                nx, ny = x + dx, y + dy
                
                need_wall = True
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    if color_mask[ny, nx]:
                        need_wall = False
                
                if need_wall:
                    if wall_type in ('left', 'right'):
                        wall_block = box(extents=[wall_thickness, dot_size, wall_height])
                        wall_x = building_x + (dot_size/2 + wall_thickness/2) * (1 if wall_type == 'right' else -1)
                        wall_y = building_y
                    else:
                        wall_block = box(extents=[dot_size, wall_thickness, wall_height])
                        wall_x = building_x
                        wall_y = building_y + (dot_size/2 + wall_thickness/2) * (1 if wall_type == 'top' else -1)
                    
                    wall_block.apply_translation([wall_x, wall_y, building_z])
                    layer_blocks.append(wall_block)
        
        # スプルーシステム（ランナー）を追加（最小限のみ）
        if len(positions) > 2:  # ドット数が少ない場合はランナーも省略
            sprue_blocks = create_sprue_system(positions, grid_size, dot_size)
            layer_blocks.extend(sprue_blocks)
        
        # レイヤーメッシュを統合
        if layer_blocks:
            try:
                layer_mesh = trimesh.util.concatenate(layer_blocks)
                layer_filename = f"{output_base_path}_plastic_{color[0]:03d}_{color[1]:03d}_{color[2]:03d}.stl"
                layer_mesh.export(layer_filename)
                generated_meshes.append(layer_mesh)
            except Exception as e:
                print(f"レイヤー RGB{color} のメッシュ生成エラー: {str(e)}")
                continue
    
    # 組み立て説明書用HTMLファイル生成
    generate_assembly_instructions(output_base_path, layer_color_order, grid_size, dot_size)
    
    return generated_meshes

def generate_assembly_instructions(output_base_path, layer_color_order, grid_size, dot_size):
    """組み立て説明書用のHTMLファイルを生成"""
    
    html_content = f'''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>プラモデル組み立て説明書</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .step {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .color-swatch {{ width: 20px; height: 20px; display: inline-block; border: 1px solid #ccc; margin-right: 10px; }}
        .warning {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>🔧 プラモデル組み立て説明書</h1>
    
    <div class="warning">
        <strong>⚠️ 注意事項</strong>
        <ul>
            <li>各パーツは薄いランナー（スプルー）で連結されています</li>
            <li>組み立て前にニッパーでランナーを切り離してください</li>
            <li>塗装は組み立て前に各色ごとに行うことを推奨します</li>
        </ul>
    </div>
    
    <h2>📦 パーツリスト</h2>
    <table border="1" style="border-collapse: collapse; width: 100%;">
        <tr><th>色</th><th>ファイル名</th><th>塗装色</th></tr>'''
    
    for i, color in enumerate(layer_color_order):
        r, g, b = color
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        filename = f"{output_base_path}_plastic_{r:03d}_{g:03d}_{b:03d}.stl"
        
        html_content += f'''
        <tr>
            <td><div class="color-swatch" style="background-color: {hex_color};"></div></td>
            <td>{filename}</td>
            <td>RGB({r}, {g}, {b})</td>
        </tr>'''
    
    html_content += f'''
    </table>
    
    <h2>🔨 組み立て手順</h2>
    <div class="step">
        <h3>ステップ1: パーツの準備</h3>
        <ol>
            <li>各STLファイルを3Dプリントします</li>
            <li>ニッパーでランナーを切り離します</li>
            <li>切り口をやすりで滑らかに仕上げます</li>
        </ol>
    </div>
    
    <div class="step">
        <h3>ステップ2: 塗装</h3>
        <ol>
            <li>各色グループごとに塗装を行います</li>
            <li>同色のパーツをまとめて塗装できるため効率的です</li>
        </ol>
    </div>
    
    <div class="step">
        <h3>ステップ3: 組み立て</h3>
        <ol>
            <li>ベースから順番に重ねていきます</li>
            <li>各コーナーのピン穴に合わせて位置を調整します</li>
            <li>必要に応じて接着剤で固定します</li>
        </ol>
    </div>
    
    <h2>📐 仕様情報</h2>
    <ul>
        <li>グリッドサイズ: {grid_size}×{grid_size}</li>
        <li>ドットサイズ: {dot_size}mm</li>
        <li>完成サイズ: 約{grid_size * dot_size}×{grid_size * dot_size}mm</li>
    </ul>
</body>
</html>'''
    
    instructions_path = f"{output_base_path}_assembly_instructions.html"
    with open(instructions_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def find_connected_components(grid_positions, grid_size):
    """
    グリッド座標から連結成分（島）を抽出する
    
    Args:
        grid_positions: グリッド座標のリスト [(x, y), ...]
        grid_size: グリッドサイズ
    
    Returns:
        List[List[Tuple[int, int]]]: 各島のグリッド座標リスト
    """
    from collections import deque
    
    if not grid_positions:
        return []
    
    # グリッド座標をセットに変換（高速検索用）
    position_set = set(grid_positions)
    visited = set()
    islands = []
    
    # 4方向の隣接チェック（上下左右）
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    for x, y in grid_positions:
        if (x, y) not in visited:
            # 新しい島を発見、幅優先探索で連結成分を抽出
            island = []
            queue = deque([(x, y)])
            visited.add((x, y))
            
            while queue:
                curr_x, curr_y = queue.popleft()
                island.append((curr_x, curr_y))
                
                # 4方向の隣接セルをチェック
                for dx, dy in directions:
                    next_x, next_y = curr_x + dx, curr_y + dy
                    
                    # グリッド範囲内かつ未訪問かつ同色ドットの場合
                    if (0 <= next_x < grid_size and 
                        0 <= next_y < grid_size and
                        (next_x, next_y) not in visited and
                        (next_x, next_y) in position_set):
                        
                        visited.add((next_x, next_y))
                        queue.append((next_x, next_y))
            
            islands.append(island)
    
    return islands

def extract_base_candidates_around_dots(grid_positions, out_thickness_cells):
    """
    各ドットの周辺（out_thickness分拡張）にベース配置候補を抽出
    
    Args:
        grid_positions: ドットのグリッド座標リスト [(x, y), ...]
        out_thickness_cells: out_thicknessをグリッド単位に変換した値
    
    Returns:
        Set[Tuple[int, int]]: ベース配置候補のグリッド座標セット
    """
    base_candidates = set()
    
    # 各ドットの周辺にベース配置候補を追加
    for x, y in grid_positions:
        # ドット周辺の拡張範囲を計算（グリッド単位）
        for bx in range(x - out_thickness_cells, x + out_thickness_cells + 1):
            for by in range(y - out_thickness_cells, y + out_thickness_cells + 1):
                base_candidates.add((bx, by))
    
    return base_candidates

def generate_connection_paths_between_islands(islands):
    """
    すべての島を一体化するための連結パスを生成
    マンハッタン距離（水平→垂直）でパスを作成
    
    Args:
        islands: 各島のグリッド座標リスト [[(x, y), ...], ...]
    
    Returns:
        Set[Tuple[int, int]]: 連結パスのグリッド座標セット
    """
    if len(islands) <= 1:
        return set()  # 島が1つ以下の場合は連結不要
    
    connection_paths = set()
    
    # 各島の代表点を選択（左下座標を使用）：島底面をカバーするため下端を基準に連結
    representatives = []
    for island in islands:
        # 左側（最小x）のセルを取得
        rep_x = min(pos[0] for pos in island)
        # 左側列における最下部（最大y）を代表点とする
        ys_at_x = [pos[1] for pos in island if pos[0] == rep_x]
        rep_y = max(ys_at_x)
        representatives.append((rep_x, rep_y))
    
    print(f"    島の代表点: {representatives}")
    
    # 隣接する島同士を順次接続（チェーン状に接続）
    for i in range(len(representatives) - 1):
        start_x, start_y = representatives[i]
        end_x, end_y = representatives[i + 1]
        
        # マンハッタン距離でパスを生成（水平→垂直）
        # 1. 水平移動（start_x → end_x）
        if start_x <= end_x:
            for x in range(start_x, end_x + 1):
                connection_paths.add((x, start_y))
        else:
            for x in range(end_x, start_x + 1):
                connection_paths.add((x, start_y))
        
        # 2. 垂直移動（start_y → end_y）
        if start_y <= end_y:
            for y in range(start_y, end_y + 1):
                connection_paths.add((end_x, y))
        else:
            for y in range(end_y, start_y + 1):
                connection_paths.add((end_x, y))
        
        print(f"    島{i+1} → 島{i+2}: ({start_x},{start_y}) → ({end_x},{end_y})")
    
    return connection_paths

def filter_base_positions_by_output_history(base_candidates, already_output_grid_positions):
    """
    ベース配置候補から、これまでに出力済みの座標を除外
    
    Args:
        base_candidates: ベース配置候補のグリッド座標セット
        already_output_grid_positions: これまでに出力済みのグリッド座標セット
    
    Returns:
        Set[Tuple[int, int]]: フィルタ後のベース配置座標セット
    """
    # 【重要】ベース配置判定ロジック:
    # - これまでに出力済み（手前レイヤーでビル/ベース済み）の場所のみNG
    # - 今後の上位レイヤーでビルが立つ予定の座標は配置OK
    filtered_positions = set()
    
    for grid_pos in base_candidates:
        if grid_pos not in already_output_grid_positions:
            # まだ出力済みでない座標 → ベース配置OK
            filtered_positions.add(grid_pos)
        # else: 既に出力済みの座標 → ベース配置NG
    
    return filtered_positions

def convert_grid_to_world_coordinates(grid_positions, dot_size, grid_size):
    """
    グリッド座標をワールド座標（物理座標）に変換
    
    Args:
        grid_positions: グリッド座標のセット/リスト
        dot_size: ドットサイズ（mm）
        grid_size: グリッドサイズ
    
    Returns:
        Set[Tuple[float, float]]: ワールド座標のセット
    """
    world_positions = set()
    
    for x, y in grid_positions:
        # グリッド座標をワールド座標に変換
        world_x = x * dot_size + dot_size / 2
        world_y = (grid_size - 1 - y) * dot_size + dot_size / 2
        world_positions.add((world_x, world_y))
    
    return world_positions

def convert_world_to_grid_coordinates(world_positions, dot_size, grid_size):
    """
    ワールド座標をグリッド座標に変換
    
    Args:
        world_positions: ワールド座標のセット/リスト
        dot_size: ドットサイズ（mm）
        grid_size: グリッドサイズ
    
    Returns:
        Set[Tuple[int, int]]: グリッド座標のセット
    """
    grid_positions = set()
    
    for world_x, world_y in world_positions:
        # ワールド座標をグリッド座標に変換
        x = int(round((world_x - dot_size / 2) / dot_size))
        y = grid_size - 1 - int(round((world_y - dot_size / 2) / dot_size))
        grid_positions.add((x, y))
    
    return grid_positions

def generate_color_separated_layers_stl(pixels_rounded_np, output_base_path, grid_size, dot_size, 
                                       wall_thickness, wall_height, base_height, out_thickness,
                                       layer_color_order):
    """
    色別レイヤー分離出力モード用のSTL生成
    各色ごとに分離したSTLファイルを出力。
    各色レイヤーは「同色ドットでできた3Dビル」と「物理的に一体化する階段/直線ベース」で構成。
    
    【重要な仕様】
    1. ベース配置条件: これまでに出力済みの座標のみNG、今後の上位レイヤー予定座標は配置OK
    2. 島の一体化: すべての同色ドット島を物理的に連結
    3. グリッド単位管理: 座標はすべてグリッドインデックスで一意管理
    """
    import trimesh
    from trimesh.creation import box
    import numpy as np
    import os
    
    if len(layer_color_order) == 0:
        return []
    
    generated_meshes = []
    
    # out_thicknessをグリッド単位に変換
    out_thickness_cells = max(1, int(round(out_thickness / dot_size)))
    print(f"Out thickness: {out_thickness}mm → {out_thickness_cells}グリッド")
    
    # 各色のグリッド座標を事前計算
    color_grid_positions = {}
    for color in layer_color_order:
        color_arr = np.array(color, dtype=np.uint8)
        mask = np.all(pixels_rounded_np == color_arr, axis=2)
        y_indices, x_indices = np.where(mask)
        
        if len(x_indices) > 0:
            # グリッド座標として保存 (x, y)
            color_grid_positions[color] = list(zip(x_indices, y_indices))
    
    # これまでに出力済みのグリッド座標を追跡（ビル + ベース両方）
    already_output_grid_positions = set()
    
    for layer_idx, color in enumerate(layer_color_order):
        layer_num = layer_idx + 1
        print(f"\n=== 色別レイヤー {layer_num} 処理開始 - RGB{color} ===")
        
        if color not in color_grid_positions or len(color_grid_positions[color]) == 0:
            print(f"  色 RGB{color} のピクセルが見つかりません。スキップします。")
            continue
        
        grid_positions = color_grid_positions[color]
        print(f"  ドット位置: {len(grid_positions)}個 {grid_positions[:5]}{'...' if len(grid_positions) > 5 else ''}")
        
        # === 1. 島（連結成分）の抽出 ===
        islands = find_connected_components(grid_positions, grid_size)
        print(f"  検出された島数: {len(islands)}個")
        for i, island in enumerate(islands):
            print(f"    島{i+1}: {len(island)}ドット {island[:3]}{'...' if len(island) > 3 else ''}")
        
        # === 2. ベース配置候補の抽出 ===
        base_candidates = extract_base_candidates_around_dots(grid_positions, out_thickness_cells)
        print(f"  ベース配置候補: {len(base_candidates)}箇所")
        
        # === 3. 島間連結パスの生成 ===
        connection_paths = generate_connection_paths_between_islands(islands)
        print(f"  島間連結パス: {len(connection_paths)}箇所")
        
        # === 4. ベース配置位置の決定（出力済み座標を除外） ===
        all_base_candidates = base_candidates | connection_paths
        final_base_positions = filter_base_positions_by_output_history(
            all_base_candidates, already_output_grid_positions)
        print(f"  最終ベース配置: {len(final_base_positions)}箇所（除外: {len(all_base_candidates) - len(final_base_positions)}箇所）")
        
        # === 5. STLメッシュ構築 ===
        layer_blocks = []
        
        # 5-1. ビル（3Dブロック）を生成
        current_building_grid_positions = set(grid_positions)
        building_world_positions = convert_grid_to_world_coordinates(
            current_building_grid_positions, dot_size, grid_size)
        
        for world_x, world_y in building_world_positions:
            building = box(extents=[dot_size - wall_thickness, dot_size - wall_thickness, wall_height])
            # ビルを base_height 分下げる: Z 位置は壁高さの半分のみ
            building.apply_translation([world_x, world_y, wall_height / 2])
            layer_blocks.append(building)
        # 5-1b. 斜め隣接するドット間をコネクトする小ビルを配置
        connector_size = dot_size / 5.0
        connector_height = base_height
        # 対角方向の隣接チェック (右上・右下)
        positions_set = set(grid_positions)
        diag_pairs = set()
        for x, y in grid_positions:
            for dx, dy in ((1, 1), (1, -1)):
                nx, ny = x + dx, y + dy
                if (nx, ny) in positions_set:
                    # 一意にペア化
                    pair = tuple(sorted(((x, y), (nx, ny))))
                    diag_pairs.add(pair)
        # コネクタ生成
        for (x1, y1), (x2, y2) in diag_pairs:
            # グリッド→ワールド座標変換
            wx1 = x1 * dot_size + dot_size / 2.0
            wy1 = (grid_size - 1 - y1) * dot_size + dot_size / 2.0
            wx2 = x2 * dot_size + dot_size / 2.0
            wy2 = (grid_size - 1 - y2) * dot_size + dot_size / 2.0
            cx = (wx1 + wx2) / 2.0
            cy = (wy1 + wy2) / 2.0
            conn = box(extents=[connector_size, connector_size, connector_height])
            # 45度回転して菱形(ダイヤモンド)にする
            try:
                from trimesh.transformations import rotation_matrix
                angle = np.deg2rad(45)
                R = rotation_matrix(angle, [0, 0, 1])
                conn.apply_transform(R)
            except Exception:
                pass
            conn.apply_translation([cx, cy, connector_height / 2.0])
            layer_blocks.append(conn)
        
        # 5-2. ベース（底面プレート）を生成
        base_world_positions = convert_grid_to_world_coordinates(
            final_base_positions, dot_size, grid_size)
        
        for world_x, world_y in base_world_positions:
            base_block = box(extents=[dot_size, dot_size, base_height])
            base_block.apply_translation([world_x, world_y, base_height / 2])
            layer_blocks.append(base_block)
        
        print(f"  ビル: {len(building_world_positions)}個, ベース: {len(base_world_positions)}個, 連結パス: {len(connection_paths)}箇所")
        
        # === 6. STLファイル出力 ===
        try:
            if layer_blocks:
                layer_mesh = trimesh.util.concatenate(layer_blocks)
                
                # STLファイルとして保存
                layer_filename = f"{output_base_path}_color_{layer_num:02d}_RGB{color[0]:03d}_{color[1]:03d}_{color[2]:03d}.stl"
                layer_mesh.export(layer_filename)
                print(f"  出力: {layer_filename}")
                
                generated_meshes.append(layer_mesh)
                
                # === 7. 出力済み座標の更新 ===
                # ビル + ベース両方の座標を出力済みとして記録
                already_output_grid_positions.update(current_building_grid_positions)
                already_output_grid_positions.update(final_base_positions)
                print(f"  出力済み座標更新: +{len(current_building_grid_positions) + len(final_base_positions)}箇所 (累計: {len(already_output_grid_positions)}箇所)")
            else:
                print(f"  色 RGB{color} のメッシュブロックが作成されませんでした。")
        except Exception as e:
            print(f"  色 RGB{color} のメッシュ生成エラー: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # === トッププレート生成 ===
    # プレビュー画像の透過部分（色未指定セル）だけを残して抜き出すプレート
    try:
        print(f"\n=== トッププレート生成 ===")
        # 全グリッド座標から、色付きドット座標を除外して透明部分を取得
        all_cells = {(x, y) for x in range(grid_size) for y in range(grid_size)}
        colored_cells = set()
        for pts in color_grid_positions.values():
            colored_cells.update(pts)
        transparent_cells = all_cells - colored_cells
        # プレート厚みはビルの高さ（wall_height）を利用
        plate_thickness = wall_height
        plate_blocks = []
        # 透明セルごとに薄板を配置
        # 座標変換にまとめて利用
        if transparent_cells:
            # 透明セルごとに薄板を配置
            worlds = convert_grid_to_world_coordinates(transparent_cells, dot_size, grid_size)
            for wx, wy in worlds:
                blk = box(extents=[dot_size, dot_size, plate_thickness])
                # Z位置は既存レイヤーの最上部に配置
                blk.apply_translation([wx, wy, base_height + wall_height + plate_thickness / 2])
                plate_blocks.append(blk)
            # === トッププレート外枠生成（上下左右） ===
            # プレビュー画像（有色セル）を中心に均等に枠を配置
            # 有色セルのグリッド位置を集約
            if color_grid_positions:
                colored_cells = set().union(*color_grid_positions.values())
            else:
                colored_cells = set()
            if colored_cells:
                # 有色セルをワールド座標に変換
                colored_worlds = convert_grid_to_world_coordinates(colored_cells, dot_size, grid_size)
                xs_c = [wx for wx, wy in colored_worlds]
                ys_c = [wy for wx, wy in colored_worlds]
                # プレビュー画像のバウンディングボックス
                half_dot = dot_size / 2
                min_px = min(xs_c) - half_dot
                max_px = max(xs_c) + half_dot
                min_py = min(ys_c) - half_dot
                max_py = max(ys_c) + half_dot
                preview_w = max_px - min_px
                preview_h = max_py - min_py
                # 枠内側領域サイズ（フレーム内壁までのスペース）
                extra_space = 2 * dot_size + 1.0
                inner_w = grid_size * dot_size + 2 * out_thickness + extra_space
                inner_h = inner_w
                # プレビュー中心
                center_px = (min_px + max_px) / 2
                center_py = (min_py + max_py) / 2
                # 枠領域バウンディング
                min_fx = center_px - inner_w / 2
                max_fx = center_px + inner_w / 2
                min_fy = center_py - inner_h / 2
                max_fy = center_py + inner_h / 2
                # Z位置
                zc = base_height + wall_height + plate_thickness / 2
                # 下枠: x全域, y=[min_fy, min_py]
                bottom_h = min_py - min_fy
                if bottom_h > 0:
                    blk = box(extents=[inner_w, bottom_h, plate_thickness])
                    blk.apply_translation([center_px, min_fy + bottom_h/2, zc])
                    plate_blocks.append(blk)
                # 上枠: x全域, y=[max_py, max_fy]
                top_h = max_fy - max_py
                if top_h > 0:
                    blk = box(extents=[inner_w, top_h, plate_thickness])
                    blk.apply_translation([center_px, max_py + top_h/2, zc])
                    plate_blocks.append(blk)
                # 左枠: x=[min_fx, min_px], y全域プレビュー高さ
                left_w = min_px - min_fx
                if left_w > 0 and preview_h > 0:
                    blk = box(extents=[left_w, preview_h, plate_thickness])
                    blk.apply_translation([min_fx + left_w/2, center_py, zc])
                    plate_blocks.append(blk)
                # 右枠: x=[max_px, max_fx], y全域プレビュー高さ
                right_w = max_fx - max_px
                if right_w > 0 and preview_h > 0:
                    blk = box(extents=[right_w, preview_h, plate_thickness])
                    blk.apply_translation([max_px + right_w/2, center_py, zc])
                    plate_blocks.append(blk)
        if plate_blocks:
            top_plate = trimesh.util.concatenate(plate_blocks)
            top_filename = f"{output_base_path}_top_plate.stl"
            top_plate.export(top_filename)
            print(f"  トッププレート出力: {top_filename}")
            generated_meshes.append(top_plate)
        else:
            print("  トッププレート生成対象の透明セルがありません。スキップします。")
    except Exception:
        print("  トッププレート生成中にエラーが発生しました。スキップします。")
        import traceback; traceback.print_exc()
    # レジン固め用フレーム生成
    print(f"\n=== レジン固め用フレーム生成 ===")
    frame_mesh = generate_resin_frame_stl(
        output_base_path, grid_size, dot_size, base_height, wall_height, out_thickness, 
        layer_count=len(layer_color_order)
    )
    if frame_mesh:
        generated_meshes.append(frame_mesh)
    
    # 色別分離用HTMLレポート生成
    generate_color_separation_report(output_base_path, layer_color_order, grid_size, dot_size)
    
    return generated_meshes

def generate_resin_frame_stl(output_base_path, grid_size, dot_size, base_height, wall_height, 
                            out_thickness, layer_count, frame_wall_thickness=1.0, frame_height_margin=5.0):
    """
    レジン固め用フレーム（天井だけが空いた箱型）を生成
    
    Args:
        output_base_path: 出力ファイルのベースパス
        grid_size: グリッドサイズ
        dot_size: ドットサイズ（mm）
        base_height: ベース高さ（mm）
        wall_height: ビル高さ（mm）
        out_thickness: 外側厚み（mm）
        layer_count: レイヤー数（フレーム高さ計算用）
        frame_wall_thickness: フレーム壁厚（mm）
        frame_height_margin: フレーム高さマージン（mm）
    
    Returns:
        trimesh.Trimesh: フレームのメッシュオブジェクト
    """
    import trimesh
    from trimesh.creation import box
    import numpy as np
    
    # 全体サイズ計算
    total_width = grid_size * dot_size + 2 * out_thickness
    total_depth = grid_size * dot_size + 2 * out_thickness
    total_layer_height = base_height + wall_height
    
    # 刻み間隔の計算（BaseHeight + WallHeight + 2mm）
    notch_interval = total_layer_height + 2.0
    
    # フレーム高さ計算（レイヤー数に基づいて動的計算）
    frame_height = layer_count * notch_interval + frame_height_margin
    
    # 内側寸法（ベースプレートも含めてドット2つ分+1mm余裕を追加）
    extra_space = 2 * dot_size + 1.0  # ドット2つ分 + 1mm余裕
    inner_width = total_width + extra_space
    inner_depth = total_depth + extra_space
    inner_height = frame_height
    
    # 外側寸法（フレーム壁厚を追加）
    outer_width = inner_width + 2 * frame_wall_thickness
    outer_depth = inner_depth + 2 * frame_wall_thickness
    outer_height = inner_height
    
    print(f"  フレーム寸法: 外側 {outer_width:.1f}x{outer_depth:.1f}x{outer_height:.1f}mm")
    print(f"  内側収納: {inner_width:.1f}x{inner_depth:.1f}x{inner_height:.1f}mm")
    print(f"  刻み間隔: {notch_interval:.1f}mm")
    
    frame_blocks = []
    
    # === 1. 基本フレーム構造（底面と壁を分離）を作成 ===
    
    # 底面プレートと壁を分離して生成
    frame_meshes = []
    
    # 1-1. 底面プレート（別STLファイルとして出力）
    bottom_plate = box(extents=[outer_width, outer_depth, frame_wall_thickness])
    bottom_plate.apply_translation([outer_width/2 - frame_wall_thickness, 
                                   outer_depth/2 - frame_wall_thickness, 
                                   frame_wall_thickness/2])
    
    # 底面プレートを別ファイルとして保存
    bottom_filename = f"{output_base_path}_resin_frame_bottom.stl"
    bottom_plate.export(bottom_filename)
    print(f"  レジンフレーム底面出力: {bottom_filename}")
    frame_meshes.append(bottom_plate)
    
    # 1-2. 壁部分（4面）をframe_blocksに追加
    wall_blocks = []
    
    # 左壁
    left_wall = box(extents=[frame_wall_thickness, outer_depth, outer_height])
    left_wall.apply_translation([frame_wall_thickness/2 - frame_wall_thickness, 
                                outer_depth/2 - frame_wall_thickness, 
                                outer_height/2])
    wall_blocks.append(left_wall)
    
    # 右壁
    right_wall = box(extents=[frame_wall_thickness, outer_depth, outer_height])
    right_wall.apply_translation([outer_width - frame_wall_thickness/2 - frame_wall_thickness, 
                                 outer_depth/2 - frame_wall_thickness, 
                                 outer_height/2])
    wall_blocks.append(right_wall)
    
    # 奥壁
    back_wall = box(extents=[inner_width, frame_wall_thickness, outer_height])
    back_wall.apply_translation([inner_width/2, 
                                outer_depth - frame_wall_thickness/2 - frame_wall_thickness, 
                                outer_height/2])
    wall_blocks.append(back_wall)
    
    # 手前壁
    front_wall = box(extents=[inner_width, frame_wall_thickness, outer_height])
    front_wall.apply_translation([inner_width/2, 
                                 frame_wall_thickness/2 - frame_wall_thickness, 
                                 outer_height/2])
    wall_blocks.append(front_wall)
    
    frame_blocks.extend(wall_blocks)
    
    # === 2. 内壁の水平ガイドライン（カップラーメン風の線）を作成 ===
    
    line_height = 0.8  # ライン高さ（mm）
    line_depth = 0.4   # ライン深さ（mm）
    
    # ガイドラインの数を計算（各レイヤーの配置位置に）
    num_guidelines = layer_count
    
    print(f"  レイヤーガイドライン数: {num_guidelines}個")
    
    # 各レイヤーの配置高さにガイドライン（線状の突起）を作成
    groove_blocks = []
    for i in range(1, num_guidelines + 1):
        guideline_height = i * notch_interval
        
        if guideline_height <= frame_height - line_height/2:
            # 左壁の線
            left_line = box(extents=[line_depth, inner_depth - 0.5, line_height])
            left_line.apply_translation([frame_wall_thickness - line_depth/2 - frame_wall_thickness, 
                                       inner_depth/2, 
                                       guideline_height])
            groove_blocks.append(left_line)
            
            # 右壁の線
            right_line = box(extents=[line_depth, inner_depth - 0.5, line_height])
            right_line.apply_translation([outer_width - frame_wall_thickness + line_depth/2 - frame_wall_thickness, 
                                        inner_depth/2, 
                                        guideline_height])
            groove_blocks.append(right_line)
            
            # 奥壁の線
            back_line = box(extents=[inner_width - 0.5, line_depth, line_height])
            back_line.apply_translation([inner_width/2, 
                                       outer_depth - frame_wall_thickness + line_depth/2 - frame_wall_thickness, 
                                       guideline_height])
            groove_blocks.append(back_line)
            
            # 手前壁の線
            front_line = box(extents=[inner_width - 0.5, line_depth, line_height])
            front_line.apply_translation([inner_width/2, 
                                        frame_wall_thickness - line_depth/2 - frame_wall_thickness, 
                                        guideline_height])
            groove_blocks.append(front_line)
            
            print(f"    レイヤー{i}ガイドライン: 高さ {guideline_height:.1f}mm")
    
    # === 3. 壁部分とガイドラインを結合して出力 ===
    
    # 壁部分を結合
    walls_mesh = trimesh.util.concatenate(frame_blocks)
    
    # ガイドライン突起を追加
    if groove_blocks:
        groove_mesh = trimesh.util.concatenate(groove_blocks)
        walls_mesh = trimesh.util.concatenate([walls_mesh, groove_mesh])
    
    # 壁部分を別ファイルとして保存
    walls_filename = f"{output_base_path}_resin_frame_walls.stl"
    walls_mesh.export(walls_filename)
    print(f"  レジンフレーム壁面出力: {walls_filename}")
    frame_meshes.append(walls_mesh)
    
    # === 4. 最終フレーム出力（参考用統合版） ===
    try:
        # 統合版も出力（参考用）
        combined_mesh = trimesh.util.concatenate([bottom_plate, walls_mesh])
        combined_filename = f"{output_base_path}_resin_frame_combined.stl"
        combined_mesh.export(combined_filename)
        print(f"  レジンフレーム統合版: {combined_filename}")
        
        return combined_mesh
        
    except Exception as e:
        print(f"  レジンフレーム生成エラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_color_separation_report(output_base_path, layer_color_order, grid_size, dot_size):
    """色別レイヤー分離用のHTMLレポートを生成"""
    import time
    import os
    
    html_content = f'''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>色別レイヤー分離レポート</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .layer-info {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
        .color-sample {{ 
            display: inline-block; 
            width: 30px; 
            height: 30px; 
            border: 1px solid #000; 
            margin-right: 10px; 
            vertical-align: middle;
        }}
        .info-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        .info-table th, .info-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .info-table th {{ background-color: #f2f2f2; }}
        .instructions {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>色別レイヤー分離STLファイル</h1>
        <p>生成日時: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    <div class="instructions">
        <h2>使用方法</h2>
        <p>各色のSTLファイルを3Dプリンターで異なる色の材料で印刷し、重ね合わせることで元の絵を再現できます。</p>
        <ul>
            <li>各レイヤーは物理的に独立しており、個別に印刷可能です</li>
            <li>同色の「島」部分は階段/直線ベースで接続されており、一体成形されます</li>
            <li>レイヤー番号が小さいほど手前（上位）に配置されます</li>
            <li><strong>レジンフレーム:</strong> 全レイヤーを固定するコの字型フレームも生成されます</li>
        </ul>
    </div>
    
    <div class="instructions" style="background-color: #e8f5e8;">
        <h2>レジン作品制作手順</h2>
        <ol>
            <li>レジンフレーム（{os.path.basename(output_base_path)}_resin_frame.stl）を3D印刷</li>
            <li>各色レイヤーを順番に配置（内壁の刻みがガイドライン）</li>
            <li>透明レジンを流し込んで固化</li>
            <li>フレームから取り出して完成</li>
        </ol>
    </div>
    
    <table class="info-table">
        <tr><th>設定項目</th><th>値</th></tr>
        <tr><td>グリッドサイズ</td><td>{grid_size} x {grid_size}</td></tr>
        <tr><td>ドットサイズ</td><td>{dot_size:.2f} mm</td></tr>
        <tr><td>総レイヤー数</td><td>{len(layer_color_order)}</td></tr>
        <tr><td>レジンフレーム</td><td>{os.path.basename(output_base_path)}_resin_frame.stl</td></tr>
        <tr><td>フレーム用途</td><td>レジン固化用コの字型容器</td></tr>
    </table>
    
    <h2>レイヤー詳細</h2>'''
    
    for i, color in enumerate(layer_color_order):
        layer_num = i + 1
        color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        filename = f"{os.path.basename(output_base_path)}_color_{layer_num:02d}_RGB{color[0]:03d}_{color[1]:03d}_{color[2]:03d}.stl"
        
        html_content += f'''
    <div class="layer-info">
        <h3>レイヤー {layer_num}</h3>
        <div class="color-sample" style="background-color: {color_hex};"></div>
        <strong>色:</strong> RGB({color[0]}, {color[1]}, {color[2]})
        <br><strong>ファイル:</strong> {filename}
    </div>'''
    
    html_content += '''
</body>
</html>'''
    
    report_path = f"{output_base_path}_color_separation_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"色別分離レポート生成: {report_path}")

# -------------------------------
# ヘルプダイアログクラス
# -------------------------------
class ParameterHelpDialog(QDialog):
    def __init__(self, parameter_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{parameter_name} についての説明")
        self.setMinimumSize(400, 200)
        
        layout = QVBoxLayout()
        
        descriptions = {
            "Grid Size": "ドット絵変換後のグリッド解像度です。\n値が大きいほど詳細なドットパターンになりますが、STLファイルのサイズも大きくなります。",
            "Dot Size": "1ドットの物理サイズ（mm）です。\n大きな値にするとプレート全体のサイズが大きくなります。",
            "Wall Thickness": "凹みを囲う壁の太さ（mm）です。\n値が小さすぎると壁が壊れやすくなる可能性があります。",
            "Wall Height": "凹みを囲う壁の高さ（mm）です。\n壁が高いほど深い凹みになります。",
            "Base Height": "プレート自体の厚さ（mm）です。\n薄すぎると脆くなる可能性があります。",
            "Out Thickness": "ベースと壁の外周を外側に拡張する幅（mm）です。\n外側の輪郭部分のみを拡張し、内側の壁には影響しません。",
            "Color Step": "色のステップ単位正規化（似た色を統一）を行うときの単位です。\n値が大きいほど使用される色数が減ります。",
            "Top Colors": "使用する上位色数制限です。\n色数を制限することでパターンをシンプルにできます。"
        }
        
        description = descriptions.get(parameter_name, "説明が見つかりません。")
        
        text_label = QLabel(description)
        text_label.setWordWrap(True)
        
        close_button = QPushButton("閉じる")
        close_button.clicked.connect(self.accept)
        
        layout.addWidget(text_label)
        layout.addWidget(close_button)
        
        self.setLayout(layout)
        
        # end of ParameterHelpDialog
        
# -------------------------------
# HTMLヘルプダイアログ
# -------------------------------
class HtmlHelpDialog(QDialog):
    """HTML形式のヘルプを表示するダイアログ"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ヘルプ")
        self.setMinimumSize(600, 400)
        # レイアウト
        layout = QVBoxLayout(self)
        # HTMLビューア
        browser = QTextBrowser()
        help_path = os.path.join(os.path.dirname(__file__), "help.html")
        try:
            with open(help_path, "r", encoding="utf-8") as f:
                html = f.read()
            browser.setHtml(html)
        except Exception:
            browser.setText("ヘルプファイル(help.html)が見つかりません。")
        layout.addWidget(browser)
        # 閉じるボタン
        btn = QPushButton("閉じる")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


# -------------------------------
# GUI クラス
# -------------------------------
class PanelManagerDialog(QDialog):
    """各ドッキングパネルの表示/非表示を管理するダイアログ"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("パネル管理")
        self.setMinimumSize(250, 120)
        layout = QVBoxLayout(self)
        # 管理対象のドックウィジェット
        docks = [
            ("ファイル操作", parent.file_dock),
            ("パラメータ設定", parent.param_dock),
            ("レイヤー設定", parent.layer_dock),
            ("STL プレビュー", parent.stl_dock),
        ]
        for name, dock in docks:
            cb = QCheckBox(name)
            cb.setChecked(dock.isVisible())
            cb.toggled.connect(dock.setVisible)
            layout.addWidget(cb)
        # 閉じるボタン
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
class DotPlateApp(QMainWindow):
    # メニューバー作成とプロジェクトファイル操作
    def create_menu_bar(self):
        """アプリケーションのメニューバーを作成する"""
        menubar = self.menuBar()
        
        # ファイルメニュー
        file_menu = menubar.addMenu('ファイル')
        
        # 画像を開く
        open_img_action = QAction('画像を開く', self)
        open_img_action.triggered.connect(self.select_image)
        file_menu.addAction(open_img_action)
        
        # プロジェクトを開く
        open_project_action = QAction('プロジェクトを開く', self)
        open_project_action.triggered.connect(self.load_project)
        file_menu.addAction(open_project_action)
        
        file_menu.addSeparator()
        
        # プロジェクトを保存
        save_project_action = QAction('プロジェクトを保存', self)
        save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(save_project_action)
        
        # STLエクスポート
        export_stl_action = QAction('STLファイルを出力', self)
        export_stl_action.triggered.connect(self.export_stl)
        file_menu.addAction(export_stl_action)
        
        file_menu.addSeparator()
        
        # 終了
        exit_action = QAction('終了', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 編集メニュー
        edit_menu = menubar.addMenu('編集')
        
        # 元に戻す
        undo_action = QAction('元に戻す', self)
        undo_action.triggered.connect(self.undo_edit)
        edit_menu.addAction(undo_action)
        
        # やり直し
        redo_action = QAction('やり直し', self)
        redo_action.triggered.connect(self.redo_edit)
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        # プレビューをクリア
        clear_action = QAction('プレビューをクリア', self)
        clear_action.triggered.connect(self.clear_preview_for_scratch)
        edit_menu.addAction(clear_action)
        # 画像のトリム（余白自動切り抜き）
        edit_menu.addSeparator()
        trim_action = QAction('画像をトリム', self)
        trim_action.triggered.connect(self.trim_image)
        edit_menu.addAction(trim_action)
        # 設定メニュー: APIキー設定
        settings_menu = menubar.addMenu('設定')
        api_key_action = QAction('APIキー設定', self)
        api_key_action.triggered.connect(self.show_api_key_dialog)
        settings_menu.addAction(api_key_action)
        # ユーザー塗料パレット設定
        palette_action = QAction('パレット設定', self)
        palette_action.triggered.connect(self.show_palette_settings_dialog)
        settings_menu.addAction(palette_action)
        # ヘルプメニュー: HTML形式のヘルプを表示
        # ヘルプメニュー: HTML形式のヘルプを表示
        help_menu = menubar.addMenu('ヘルプ')
        help_action = QAction('ヘルプ', self)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)
    
    def show_api_key_dialog(self):
        """OpenAI APIキーを設定するダイアログを表示"""
        # 入力ダイアログでAPIキーを取得
        key, ok = QInputDialog.getText(self, "APIキー設定", "OpenAI APIキーを入力してください:", QLineEdit.Normal, getattr(self, 'openai_api_key', ''))
        if ok and key:
            # 設定の永続化
            settings = QSettings("DotPlateGenerator", "DotPlateApp")
            settings.setValue("openai_api_key", key)
            # APIキーを適用
            self.openai_api_key = key
            openai.api_key = key
            self.statusBar().showMessage("APIキーを保存しました")
    
    def show_help(self):
        """HTMLヘルプを表示するダイアログを起動"""
        dlg = HtmlHelpDialog(self)
        dlg.exec_()
    
    def get_nearest_palette_color(self, color):
        """登録パレットから最も近い色を返す (R,G,B tuple)"""
        if not hasattr(self, 'palette_colors') or not self.palette_colors:
            return None
        # Cast to Python ints to avoid numpy uint8 wrap-around
        try:
            r, g, b = map(int, color)
        except Exception:
            r = int(color[0]); g = int(color[1]); b = int(color[2])
        # Euclidean squared distance
        best = min(self.palette_colors, key=lambda c: (r-c[0])**2 + (g-c[1])**2 + (b-c[2])**2)
        return best
    
    def get_palette_mix(self, color, max_denominator=3):
        """Return list of palette colors to mix to approximate target color."""
        # color: tuple (r,g,b)
        # palette_colors: list of tuples
        if not hasattr(self, 'palette_colors') or not self.palette_colors:
            return []
        # If exact palette color exists, use it directly
        if color in self.palette_colors:
            return [color]
        # Ensure we work with Python ints to avoid uint8 overflow during arithmetic
        try:
            r, g, b = map(int, color)
        except Exception:
            # Fallback: unpack and cast
            r = int(color[0]); g = int(color[1]); b = int(color[2])
        # Single color error
        best_error = float('inf')
        best_mix = []
        best_single = None
        # one color
        for c in self.palette_colors:
            err = (r-c[0])**2 + (g-c[1])**2 + (b-c[2])**2
            if err < best_error:
                best_error = err
                best_single = c
                best_mix = [c]
        # two-color mix
        n = len(self.palette_colors)
        for i in range(n):
            ci = self.palette_colors[i]
            for j in range(i+1, n):
                cj = self.palette_colors[j]
                dr = ci[0] - cj[0]
                dg = ci[1] - cj[1]
                db = ci[2] - cj[2]
                denom = dr*dr + dg*dg + db*db
                if denom == 0:
                    continue
                # optimal alpha for ci
                alpha = ((r-cj[0])*dr + (g-cj[1])*dg + (b-cj[2])*db) / denom
                alpha = max(0.0, min(1.0, alpha))
                # mix color
                mr = alpha*ci[0] + (1-alpha)*cj[0]
                mg = alpha*ci[1] + (1-alpha)*cj[1]
                mb = alpha*ci[2] + (1-alpha)*cj[2]
                err = (r-mr)**2 + (g-mg)**2 + (b-mb)**2
                if err < best_error:
                    best_error = err
                    # determine integer ratio p:q
                    best_p, best_q, best_pair = 0, 0, (ci, cj)
                    best_rel = float('inf')
                    for d in range(2, max_denominator+1):
                        p = int(round(alpha * d))
                        q = d - p
                        if p <= 0 or q <= 0:
                            continue
                        rel = abs((p/d) - alpha)
                        if rel < best_rel:
                            best_rel = rel
                            best_p, best_q = p, q
                    if best_p > 0 and best_q > 0:
                        best_mix = [ci] * best_p + [cj] * best_q
                    else:
                        best_mix = [ci]
        return best_mix
    
    def show_palette_settings_dialog(self):
        """ユーザー塗料パレット設定ダイアログを表示"""
        dialog = QDialog(self)
        dialog.setWindowTitle("パレット設定")
        dialog.resize(300, 400)
        layout = QVBoxLayout(dialog)
        # パレットリスト
        palette_list = QListWidget()
        palette_list.setSelectionMode(QAbstractItemView.SingleSelection)
        # 登録済み色を表示
        for color in self.palette_colors:
            item = QListWidgetItem(f"RGB{color}")
            pix = QPixmap(20, 20)
            pix.fill(QColor(*color))
            item.setIcon(QIcon(pix))
            item.setData(Qt.UserRole, color)
            palette_list.addItem(item)
        # 追加・削除ボタン
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("＋")
        remove_btn = QPushButton("－")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(remove_btn)
        # ボタン動作
        def add_color():
            col = QColorDialog.getColor(parent=dialog)
            if col.isValid():
                tup = (col.red(), col.green(), col.blue())
                if tup not in self.palette_colors:
                    self.palette_colors.append(tup)
                    new_item = QListWidgetItem(f"RGB{tup}")
                    pix2 = QPixmap(20, 20)
                    pix2.fill(col)
                    new_item.setIcon(QIcon(pix2))
                    new_item.setData(Qt.UserRole, tup)
                    palette_list.addItem(new_item)
                    # 設定保存
                    settings = QSettings("DotPlateGenerator", "DotPlateApp")
                    hexs = [f"#{c[0]:02X}{c[1]:02X}{c[2]:02X}" for c in self.palette_colors]
                    settings.setValue("palette_colors", hexs)
        def remove_color():
            row = palette_list.currentRow()
            if row >= 0:
                item = palette_list.takeItem(row)
                color = item.data(Qt.UserRole)
                if color in self.palette_colors:
                    self.palette_colors.remove(color)
                # 設定保存
                settings = QSettings("DotPlateGenerator", "DotPlateApp")
                hexs = [f"#{c[0]:02X}{c[1]:02X}{c[2]:02X}" for c in self.palette_colors]
                settings.setValue("palette_colors", hexs)
        add_btn.clicked.connect(add_color)
        remove_btn.clicked.connect(remove_color)
        # レイアウト組み立て
        layout.addWidget(palette_list)
        layout.addLayout(btn_layout)
        dialog.exec_()
    
    def trim_image(self):
        """トリム範囲選択モードを開始する"""
        if not hasattr(self, 'image_path') or not self.image_path:
            QMessageBox.warning(self, "トリムエラー", "画像が開かれていません。")
            return
        # マニュアルトリム選択モードを有効化
        self.trim_selecting = True
        # ラバーバンドをリセットして非表示
        self.rubber_band.hide()
        # クロスカーソルに変更
        self.original_image_label.setCursor(Qt.CrossCursor)
        self.statusBar().showMessage("ドラッグでトリミング範囲を選択してください")
    
    def eventFilter(self, obj, event):
        """Original image label 用のトリム操作をキャプチャする"""
        # プレビュー上でのコピー/ペースト矩形選択
        # コピー/ペースト用の矩形選択エリアをキャプチャ（preview_label またはそのビューポート）
        if obj is getattr(self, 'preview_label', None) or (hasattr(self, 'preview_scroll') and obj is self.preview_scroll.viewport()):
            # コピー選択モード
            if getattr(self, 'is_copy_mode', False):
                if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                    # 開始点を保存してラバーバンド表示
                    self._copy_origin = event.pos()
                    self.preview_rubber_band.setGeometry(QRect(self._copy_origin, QSize()))
                    self.preview_rubber_band.show()
                    return True
                elif event.type() == QEvent.MouseMove and self.preview_rubber_band.isVisible():
                    # 選択範囲を更新
                    rect = QRect(self._copy_origin, event.pos()).normalized()
                    self.preview_rubber_band.setGeometry(rect)
                    return True
                elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton and self.preview_rubber_band.isVisible():
                    # 選択確定
                    rect = self.preview_rubber_band.geometry()
                    self.preview_rubber_band.hide()
                    # 開始・終了グリッド座標
                    p0 = self.preview_label.get_grid_position(rect.topLeft())
                    p1 = self.preview_label.get_grid_position(rect.bottomRight())
                    if p0 and p1:
                        x0, y0 = p0; x1, y1 = p1
                        x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1))
                        # コピーバッファ格納
                        self.copy_buffer = self.pixels_rounded_np[y0:y1+1, x0:x1+1].copy()
                        self.statusBar().showMessage(f"コピー: ({x0},{y0}) サイズ {x1-x0+1}x{y1-y0+1}")
                    else:
                        QMessageBox.warning(self, "コピーエラー", "範囲選択が無効です。")
                    # モード解除
                    self.is_copy_mode = False
                    self.copy_btn.setChecked(False)
                    return True
            # ペーストモード
            if getattr(self, 'is_paste_mode', False):
                if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                    # ペースト位置取得
                    gp = self.preview_label.get_grid_position(event.pos())
                    if gp and getattr(self, 'copy_buffer', None) is not None:
                        x, y = gp
                        h, w = self.copy_buffer.shape[:2]
                        H, W = self.pixels_rounded_np.shape[:2]
                        # 範囲クリップ
                        h2 = min(h, H - y)
                        w2 = min(w, W - x)
                        if h2 > 0 and w2 > 0:
                            self.save_edit_history()
                            self.pixels_rounded_np[y:y+h2, x:x+w2] = self.copy_buffer[:h2, :w2]
                            self.update_preview(custom_pixels=self.pixels_rounded_np)
                            self.statusBar().showMessage(f"ペースト: ({x},{y})")
                    else:
                        QMessageBox.warning(self, "ペーストエラー", "ペースト範囲が不正です。")
                    # モード解除
                    self.is_paste_mode = False
                    self.paste_btn.setChecked(False)
                    return True
        # ここまでプレビューイベント処理
        # トリム範囲選択モードをキャプチャ
        if obj is getattr(self, 'original_image_label', None) and getattr(self, 'trim_selecting', False):
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._trim_origin = event.pos()
                self.rubber_band.setGeometry(QRect(self._trim_origin, QSize()))
                self.rubber_band.show()
                return True
            elif event.type() == QEvent.MouseMove and self.rubber_band.isVisible():
                # 自由比率でトリム選択。Shiftキー押下時のみ1:1固定
                current_pos = event.pos()
                dx = current_pos.x() - self._trim_origin.x()
                dy = current_pos.y() - self._trim_origin.y()
                # Shiftで正方形固定
                if isinstance(event, QMouseEvent) and (event.modifiers() & Qt.ShiftModifier):
                    side = min(abs(dx), abs(dy))
                    dx = side if dx >= 0 else -side
                    dy = side if dy >= 0 else -side
                rect = QRect(self._trim_origin, QSize(dx, dy)).normalized()
                self.rubber_band.setGeometry(rect)
                return True
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self.rubber_band.hide()
                rect = self.rubber_band.geometry()
                # トリム選択完了
                self.trim_selecting = False
                self.original_image_label.setCursor(Qt.ArrowCursor)
                self.statusBar().clearMessage()
                self.crop_to_rect(rect)
                return True
        return super().eventFilter(obj, event)

    def crop_to_rect(self, rect):
        """選択矩形に基づき元画像をクロップし、プレビューを更新する"""
        try:
            img_full = Image.open(self.image_path)
            full_w, full_h = img_full.size
            # 表示画像上の選択矩形を元画像座標にマッピング
            pixmap = self.original_image_label.pixmap()
            if pixmap is None:
                raise ValueError("表示中の画像がありません。")
            disp_w, disp_h = pixmap.width(), pixmap.height()
            # ラベル上での画像表示位置（中央寄せ）によるオフセット
            label_w = self.original_image_label.width()
            label_h = self.original_image_label.height()
            offset_x = max((label_w - disp_w) / 2, 0)
            offset_y = max((label_h - disp_h) / 2, 0)
            # 選択矩形を画像描画領域内に変換
            x_pix = rect.x() - offset_x
            y_pix = rect.y() - offset_y
            # 幅・高さ
            w_pix = rect.width()
            h_pix = rect.height()
            # 画像領域外へのはみ出しを防ぐ
            x_pix = min(max(x_pix, 0), disp_w)
            y_pix = min(max(y_pix, 0), disp_h)
            end_x = min(x_pix + w_pix, disp_w)
            end_y = min(y_pix + h_pix, disp_h)
            # 元画像へのスケーリング
            # Map display selection to original image coordinates using nearest rounding
            scale_x = full_w / disp_w
            scale_y = full_h / disp_h
            x0 = int(round(x_pix * scale_x))
            y0 = int(round(y_pix * scale_y))
            x1 = int(round(end_x * scale_x))
            y1 = int(round(end_y * scale_y))
            # 画像境界内にクランプ
            x0 = max(0, min(x0, full_w))
            y0 = max(0, min(y0, full_h))
            x1 = max(0, min(x1, full_w))
            y1 = max(0, min(y1, full_h))
            # 有効な範囲かチェック
            if x1 <= x0 or y1 <= y0:
                QMessageBox.warning(self, "トリムエラー", "選択範囲が小さすぎます。")
                return
            cropped = img_full.crop((x0, y0, x1, y1))
            suffix = os.path.splitext(self.image_path)[1] or '.png'
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            cropped.save(tmp.name)
            self.image_path = tmp.name
            self.input_label.setText(self.image_path)
            # トリミング後の幅に合わせてGrid Sizeを更新
            try:
                if hasattr(self, 'controls') and 'Grid Size' in self.controls:
                    w_crop, h_crop = cropped.size
                    # 幅を優先して設定（非正方形時はアスペクト比をプレビューで維持）
                    self.controls['Grid Size'].setValue(int(w_crop))
            except Exception:
                pass
            # カスタムピクセルデータをクリアして、新画像でプレビュー再生成
            # （既存のピクセルデータを破棄し、トリム後の画像で再生成）
            self.pixels_rounded_np = None
            # プレビューを更新（custom_pixels=Noneを明示して新規生成）
            self.update_preview(custom_pixels=None)
            QMessageBox.information(self, "トリム完了", "選択範囲で画像をトリムしました。")
        except Exception as e:
            QMessageBox.critical(self, "トリムエラー", f"トリミング処理中にエラーが発生しました: {e}")

    def save_project(self):
        """プロジェクトをファイルに保存する"""
        if not hasattr(self, 'pixels_rounded_np') or self.pixels_rounded_np is None:
            self.statusBar().showMessage("保存するプロジェクトデータがありません")
            return
            
        # 保存先ファイル名を取得
        file_path, _ = QFileDialog.getSaveFileName(
            self, "プロジェクトを保存", "", "ドットプレートプロジェクト (*.dpp)")
        
        if not file_path:
            return
            
        # ファイル拡張子を確認して追加
        if not file_path.endswith('.dpp'):
            file_path += '.dpp'
            
        try:
            # 保存するデータを収集
            project_data = {
                'version': '1.0',
                'image_path': self.image_path,
                'current_grid_size': self.current_grid_size,
                'current_color_algo': self.current_color_algo,
                'zoom_factor': self.zoom_factor,
            }
            
            # パラメータを保存
            if hasattr(self, 'controls'):
                parameter_values = {}
                for key, spin in self.controls.items():
                    try:
                        # 数値型に変換して格納
                        value = spin.value()
                        if isinstance(value, (int, float)):
                            parameter_values[key] = value
                        else:
                            # QVariantなど特殊な型の場合は文字列に変換
                            parameter_values[key] = float(value)
                    except Exception as e:
                        print(f"パラメータ '{key}' の保存エラー: {str(e)}")
                project_data['parameters'] = parameter_values
            
            # ピクセルデータをBase64エンコードして保存
            if hasattr(self, 'pixels_rounded_np') and self.pixels_rounded_np is not None:
                pixel_data_binary = pickle.dumps(self.pixels_rounded_np)
                pixel_data_b64 = base64.b64encode(pixel_data_binary).decode('utf-8')
                project_data['pixels_data'] = pixel_data_b64
            
            # 壁の色を保存（QColorをRGB値のリストに変換）
            if hasattr(self, 'wall_color'):
                if isinstance(self.wall_color, QColor):
                    project_data['wall_color'] = [self.wall_color.red(), 
                                                 self.wall_color.green(), 
                                                 self.wall_color.blue()]
                else:
                    # 既にタプルやリストの場合
                    project_data['wall_color'] = list(self.wall_color)
            
            # 編集履歴（最新の状態のみ）を保存
            if hasattr(self, 'edit_history') and len(self.edit_history) > 0:
                latest_history_binary = pickle.dumps(self.edit_history[-1])
                latest_history_b64 = base64.b64encode(latest_history_binary).decode('utf-8')
                project_data['latest_history'] = latest_history_b64
            
            # レイヤー設定（色ごとの高さと順序）を保存
            if hasattr(self, 'layer_color_order') and hasattr(self, 'layer_heights'):
                layers = []
                for color in self.layer_color_order:
                    # color is a tuple (r,g,b)
                    h = self.layer_heights.get(color, 0.0)
                    layers.append({ 'color': [int(color[0]), int(color[1]), int(color[2])], 'height': float(h) })
                project_data['layers'] = layers
            
            # ファイルに書き込み
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(project_data, f, ensure_ascii=False, indent=2)
                
            self.statusBar().showMessage(f"プロジェクトを保存しました: {file_path}")
            
        except Exception as e:
            self.statusBar().showMessage(f"プロジェクト保存エラー: {str(e)}")
            print(f"プロジェクト保存エラー: {str(e)}")
    
    def load_project(self):
        """プロジェクトファイルを読み込む"""
        # ファイルを選択
        file_path, _ = QFileDialog.getOpenFileName(
            self, "プロジェクトを開く", "", "ドットプレートプロジェクト (*.dpp)")
        
        if not file_path:
            return
            
        try:
            # ファイルからプロジェクトデータを読み込み
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                try:
                    project_data = json.loads(file_content)
                except json.JSONDecodeError as je:
                    print(f"JSONデコードエラー: {je}")
                    print(f"問題のある行付近: {file_content[max(0, je.pos-50):min(len(file_content), je.pos+50)]}")
                    raise
            
            # バージョンチェック
            version = project_data.get('version', '1.0')
            
            # 画像パスを設定
            image_path = project_data.get('image_path')
            if image_path and os.path.exists(image_path):
                self.image_path = image_path
                self.input_label.setText(image_path)
            else:
                self.statusBar().showMessage("元の画像ファイルが見つかりません。プロジェクトデータのみ復元します。")
            
            # 各種パラメータを復元
            if 'parameters' in project_data and hasattr(self, 'controls'):
                for key, value in project_data['parameters'].items():
                    if key in self.controls:
                        try:
                            # 確実に数値型に変換
                            numeric_value = float(value)
                            if key in ["Grid Size", "Top Colors", "Color Step"]:
                                # 整数値が必要なパラメータ
                                numeric_value = int(numeric_value)
                            self.controls[key].setValue(numeric_value)
                        except (ValueError, TypeError) as e:
                            print(f"パラメータ '{key}' の値 '{value}' を変換できませんでした: {str(e)}")
            
            # グリッドサイズを復元
            if 'current_grid_size' in project_data:
                self.current_grid_size = project_data['current_grid_size']
            
            # 色アルゴリズムを復元
            if 'current_color_algo' in project_data:
                self.current_color_algo = project_data['current_color_algo']
                # コンボボックスも更新
                if hasattr(self, 'color_algo_combo'):
                    algo_index = 0  # デフォルトはsimple
                    if self.current_color_algo == "median_cut":
                        algo_index = 1
                    elif self.current_color_algo == "kmeans":
                        algo_index = 2
                    elif self.current_color_algo == "octree":
                        algo_index = 3
                    elif self.current_color_algo == "toon":
                        algo_index = 4
                    self.color_algo_combo.setCurrentIndex(algo_index)
            
            # ズーム係数を復元
            if 'zoom_factor' in project_data:
                self.zoom_factor = project_data['zoom_factor']
                if hasattr(self, 'zoom_slider'):
                    self.zoom_slider.setValue(self.zoom_factor)
            
            # ピクセルデータを復元
            if 'pixels_data' in project_data:
                pixels_b64 = project_data['pixels_data']
                pixels_binary = base64.b64decode(pixels_b64)
                self.pixels_rounded_np = pickle.loads(pixels_binary)
            
            # 壁の色を復元
            if 'wall_color' in project_data:
                wall_color_data = project_data['wall_color']
                # リストかタプルの場合
                if isinstance(wall_color_data, (list, tuple)) and len(wall_color_data) >= 3:
                    r, g, b = wall_color_data[0], wall_color_data[1], wall_color_data[2]
                    self.wall_color = (r, g, b)
                    if hasattr(self, 'wall_color_btn'):
                        self.set_button_color(self.wall_color_btn, QColor(r, g, b))
            
            # legacy: 同色マージオプションをSTL出力モードへマッピング
            if 'merge_same_color' in project_data:
                # True -> ドットプレート (同色内壁省略)
                self.stl_mode = 1 if project_data['merge_same_color'] else 0
                self.stl_mode_combo.setCurrentIndex(self.stl_mode)
            # レイヤー設定を復元 (色ごとの高さと順序)
            if 'layers' in project_data:
                layers = project_data.get('layers', [])
                # initialize containers
                self.layer_color_order = []
                self.layer_heights = {}
                for entry in layers:
                    col = entry.get('color')
                    h = entry.get('height', 0.0)
                    if isinstance(col, (list, tuple)) and len(col) >= 3:
                        color = (int(col[0]), int(col[1]), int(col[2]))
                        self.layer_color_order.append(color)
                        self.layer_heights[color] = float(h)
                # プロジェクトにレイヤーデータがあれば色レイヤーモードを選択
                self.stl_mode = 2
                self.stl_mode_combo.setCurrentIndex(2)
            
            # 編集履歴を初期化
            self.edit_history = []
            self.history_position = 0
            
            # 最新の履歴状態を復元
            if 'latest_history' in project_data:
                latest_history_b64 = project_data['latest_history']
                latest_history_binary = base64.b64decode(latest_history_b64)
                latest_history = pickle.loads(latest_history_binary)
                self.edit_history.append(latest_history)
                self.history_position = 0
            elif hasattr(self, 'pixels_rounded_np') and self.pixels_rounded_np is not None:
                # 履歴がない場合は現在の状態を追加
                self.edit_history.append(self.pixels_rounded_np.copy())
                self.history_position = 0
            
            # プレビューを更新
            self.update_preview(custom_pixels=self.pixels_rounded_np)
            
            self.statusBar().showMessage(f"プロジェクトを読み込みました: {file_path}")
            
        except Exception as e:
            self.statusBar().showMessage(f"プロジェクト読み込みエラー: {str(e)}")
            print(f"プロジェクト読み込みエラー: {str(e)}")
    
    # 色ハイライトと置換のための新しいメソッド
    def on_color_cell_clicked(self, url):
        """色セルがクリックされたときのハンドラー"""
        if url.startswith('color://'):
            # URL形式： color://r,g,b
            color_part = url.split('://')[-1]
            try:
                r, g, b = map(int, color_part.split(','))
                target_color = (r, g, b)
                
                # この色のドットをハイライト表示
                self.highlight_dots_with_color(target_color)
                
                # 色置換ダイアログを表示
                self.show_replace_color_dialog(target_color)
                
            except ValueError as e:
                print(f"色パース中のエラー: {str(e)}")
    
    def highlight_dots_with_color(self, target_color):
        """指定された色を持つすべてのドットをハイライト表示する"""
        if self.pixels_rounded_np is None:
            return
            
        # ハイライトする色を保存
        self.highlighted_color = target_color
        # プレビューを更新して該当色をハイライト表示
        self.update_preview(custom_pixels=self.pixels_rounded_np, highlight_color=target_color)
    
    def clear_color_highlight(self):
        """色のハイライト表示をクリアする"""
        if hasattr(self, 'highlighted_color'):
            delattr(self, 'highlighted_color')
        
        # 通常のプレビュー表示に戻す
        self.update_preview()
    
    def show_replace_color_dialog(self, target_color):
        """色置換のためのダイアログを表示"""
        if self.pixels_rounded_np is None:
            return
            
        r, g, b = target_color
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        
        # ダイアログ作成
        dialog = QDialog(self)
        dialog.setWindowTitle("色置換")
        dialog.setMinimumWidth(300)
        
        # レイアウト
        layout = QVBoxLayout(dialog)
        
        # 情報ラベル
        info_label = QLabel(f"選択した色: RGB({r}, {g}, {b}) {hex_color}")
        layout.addWidget(info_label)
        
        # 色表示
        color_preview = QLabel()
        color_preview.setFixedSize(40, 40)
        color_preview.setStyleSheet(f"background-color: {hex_color}; border: 1px solid #aaa;")
        layout.addWidget(color_preview)
        
        # 新しい色選択ボタン
        color_btn = QPushButton("新しい色を選択")
        layout.addWidget(color_btn)
        
        # 透明に変更ボタン
        transparent_btn = QPushButton("透明に変更")
        layout.addWidget(transparent_btn)
        
        # ボタン
        btn_layout = QHBoxLayout()
        cancel_btn = QPushButton("キャンセル")
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        # シグナル接続
        def on_select_new_color():
            new_color = QColorDialog.getColor(QColor(r, g, b), dialog, "新しい色を選択")
            if new_color.isValid():
                # 色を置換
                self.replace_all_same_color(target_color, (new_color.red(), new_color.green(), new_color.blue()))
                dialog.accept()
                # ハイライトを解除
                self.clear_color_highlight()
        
        def on_set_transparent():
            # 透明色に置換（ユーザー設定の透過色として扱う）
            tc = self.transparent_color
            self.replace_all_same_color(target_color, (tc.red(), tc.green(), tc.blue()))
            dialog.accept()
            # ハイライトを解除
            self.clear_color_highlight()
        
        # キャンセル時もハイライトを解除
        def on_cancel():
            dialog.reject()
            self.clear_color_highlight()
            
        color_btn.clicked.connect(on_select_new_color)
        transparent_btn.clicked.connect(on_set_transparent)
        cancel_btn.clicked.connect(on_cancel)
        
        # ダイアログが閉じられたときにもハイライトを解除（×ボタンなどの場合）
        dialog.finished.connect(self.clear_color_highlight)
        
        # ダイアログを表示
        dialog.exec_()
    
    def replace_all_same_color(self, target_color, new_color):
        """同じ色を持つすべてのドットを一括置換する"""
        if self.pixels_rounded_np is None:
            return
            
        # 編集履歴を保存
        self.save_edit_history()
        
        # ピクセルデータのコピーを作成
        modified_pixels = self.pixels_rounded_np.copy()
        
        # 指定された色と一致するピクセルを検索して置換
        r, g, b = target_color
        target_mask = (modified_pixels[:, :, 0] == r) & (modified_pixels[:, :, 1] == g) & (modified_pixels[:, :, 2] == b)
        
        # 新しい色で置換
        nr, ng, nb = new_color
        modified_pixels[target_mask, 0] = nr
        modified_pixels[target_mask, 1] = ng
        modified_pixels[target_mask, 2] = nb
        
        # 変更を適用
        self.pixels_rounded_np = modified_pixels
        
        # プレビューを更新
        self.update_preview(custom_pixels=modified_pixels)
        
        # STLプレビューも更新（現在のパラメータを使用）
        try:
            # 一時的に編集済みピクセルを画像ファイルとして保存
            from PIL import Image
            import os
            # 保存先パス生成
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img_file:
                img_path = tmp_img_file.name
            # カスタムピクセル画像を保存
            img = Image.fromarray(modified_pixels, mode='RGB')
            img.save(img_path)
            # パラメータ取得
            params = {key: spin.value() for key, spin in self.controls.items()}
            # 各パラメータ
            grid_size = int(params.get("Grid Size", 0))
            dot_size = float(params.get("Dot Size", 0.0))
            wall_thickness = float(params.get("Wall Thickness", 0.0))
            wall_height = float(params.get("Wall Height", 0.0))
            base_height = float(params.get("Base Height", 0.0))
            color_step = int(params.get("Color Step", 1))
            top_color_limit = int(params.get("Top Colors", 0))
            out_thickness = float(params.get("Out Thickness", 0.0))
            # 壁色とマージオプション (同色内壁省略はSTL出力モードで切り替え)
            if hasattr(self, 'wall_color') and isinstance(self.wall_color, QColor):
                wall_clr = (self.wall_color.red(), self.wall_color.green(), self.wall_color.blue())
            else:
                wall_clr = getattr(self, 'wall_color', (255, 255, 255))
            # STL出力モード(stl_mode==1)で同色間の内壁を省略
            merge_same = (getattr(self, 'stl_mode', 0) == 1)
            # 一時的なSTL出力ファイル
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_stl_file:
                stl_path = tmp_stl_file.name
            # STL生成 (カラーステップ済み入力画像を使用)
            mesh = generate_dot_plate_stl(
                img_path,
                stl_path,
                grid_size,
                dot_size,
                wall_thickness,
                wall_height,
                base_height,
                color_step,
                top_color_limit,
                out_thickness,
                wall_color=wall_clr,
                merge_same_color=merge_same,
                return_colors=True
            )
            # プレビュー更新
            # generate_dot_plate_stlは (mesh, colors) を返す場合がある
            preview_mesh = mesh[0] if isinstance(mesh, tuple) else mesh
            self.show_stl_preview(preview_mesh)
        except Exception as e:
            print(f"STLプレビュー更新エラー: {str(e)}")
        finally:
            # 一時ファイル削除
            try:
                os.unlink(img_path)
            except:
                pass
            try:
                os.unlink(stl_path)
            except:
                pass
        
        # ステータスバー更新
        count = np.sum(target_mask)
        if count > 0:
            self.statusBar().showMessage(f"{count}個のドットの色をRGB({r}, {g}, {b})からRGB({nr}, {ng}, {nb})に変更しました")
        else:
            self.statusBar().showMessage("該当する色のドットは見つかりませんでした")
    def __init__(self):
        super().__init__()
        # パレット設定の読み込み (ユーザー登録塗料色)
        settings = QSettings("DotPlateGenerator", "DotPlateApp")
        saved_palette = settings.value("palette_colors", []) or []
        # QSettings returns str or list
        if isinstance(saved_palette, str):
            saved_palette = [saved_palette]
        self.palette_colors = []  # 登録塗料色リスト [(r,g,b), ...]
        for h in saved_palette:
            try:
                c = QColor(h)
                if c.isValid():
                    self.palette_colors.append((c.red(), c.green(), c.blue()))
            except:
                pass
        # グローバル固定パレットを更新
        global FIXED_PALETTE
        FIXED_PALETTE = self.palette_colors.copy()
        # Initialize layer settings defaults
        self.layer_heights = {}
        self.layer_color_order = []
        self.setWindowTitle("Dot Plate Generator")
        self.setMinimumSize(1200, 700)
        
        # ステータスバーを初期化
        self.statusBar().showMessage("準備完了")
        
        # メニューバーを作成
        self.create_menu_bar()
        # 永続化されたAPIキーをQtの設定から読み込む
        settings = QSettings("DotPlateGenerator", "DotPlateApp")
        saved_key = settings.value("openai_api_key", "")
        api_key_env = os.getenv("OPENAI_API_KEY")
        if saved_key and not api_key_env:
            self.openai_api_key = saved_key
            openai.api_key = saved_key
            self.statusBar().showMessage("保存済みのAPIキーを読み込みました")
        # 環境変数からOpenAI APIキーを読み込む（優先）
        if api_key_env:
            self.openai_api_key = api_key_env
            openai.api_key = api_key_env
            self.statusBar().showMessage("APIキーを環境変数から設定されました")
        # AIブラシ用の初期設定
        self.ai_brush_mode = False
        self.ai_highlight_pixels = []
        self.ai_brush_target_color = None  # AIブラシで対象とする色
        
        # メインウィジェットとレイアウト（3カラム構成）
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # カラム1：ファイル操作、オリジナル画像表示、パラメータ設定
        column1_panel = QWidget()
        column1_layout = QVBoxLayout(column1_panel)
        
        # ファイル操作グループ
        self.file_group = QGroupBox("ファイル操作")
        file_layout = QVBoxLayout()
        
        self.input_label = QLabel("画像が選択されていません")
        self.input_label.setWordWrap(True)
        
        # 画像選択ボタンのみを表示（STLエクスポートはパラメータ設定に移動）
        file_btn_layout = QHBoxLayout()
        # 画像選択ボタン
        self.select_button = QPushButton("画像を選択")
        self.select_button.clicked.connect(self.select_image)
        file_btn_layout.addWidget(self.select_button)
        # 画像トリムボタン
        self.trim_button = QPushButton("画像をトリム")
        self.trim_button.setToolTip("画像をトリム（余白自動切り抜き）")
        self.trim_button.clicked.connect(self.trim_image)
        file_btn_layout.addWidget(self.trim_button)
        
        file_layout.addWidget(self.input_label)
        file_layout.addLayout(file_btn_layout)
        
        self.file_group.setLayout(file_layout)
        column1_layout.addWidget(self.file_group)
        
        # オリジナル画像表示エリア
        self.original_group = QGroupBox("オリジナル画像")
        original_layout = QVBoxLayout()
        # スクロール表示
        self.original_scroll = QScrollArea()
        self.original_scroll.setWidgetResizable(True)
        self.original_scroll.setMinimumHeight(250)
        self.original_image_label = QLabel("オリジナル画像が表示されます")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.original_scroll.setWidget(self.original_image_label)
        # Manual trim selection support
        self.trim_selecting = False
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.original_image_label)
        self.original_image_label.installEventFilter(self)
        original_layout.addWidget(self.original_scroll)
        # ズームコントロール（スライダー）
        orig_zoom_layout = QHBoxLayout()
        orig_zoom_label = QLabel("拡大縮小:")
        self.original_zoom_slider = QSlider(Qt.Horizontal)
        self.original_zoom_slider.setMinimum(1)
        self.original_zoom_slider.setMaximum(40)
        self.original_zoom_slider.setValue(10)
        self.original_zoom_slider.setToolTip("オリジナル画像の拡大縮小（スライダーで調整）")
        self.original_zoom_slider.valueChanged.connect(self.on_original_zoom_changed)
        orig_zoom_layout.addWidget(orig_zoom_label)
        orig_zoom_layout.addWidget(self.original_zoom_slider)
        original_layout.addLayout(orig_zoom_layout)
        self.original_group.setLayout(original_layout)
        column1_layout.addWidget(self.original_group)
        
        # パラメータ設定グループ（スクロール対応）
        self.param_group = QGroupBox("パラメータ設定")
        param_scroll = QScrollArea()
        param_scroll.setWidgetResizable(True)
        param_scroll_content = QWidget()
        param_layout = QVBoxLayout(param_scroll_content)
        
        # 減色アルゴリズム選択
        color_algo_layout = QHBoxLayout()
        color_algo_label = QLabel("減色アルゴリズム:")
        self.color_algo_combo = QComboBox()
        self.color_algo_combo.addItems([
            "単純量子化 (Simple)", 
            "メディアンカット法 (Median Cut)", 
            "K-means法 (K-means)", 
            "オクトツリー法 (Octree)",
            "トゥーンアニメ風 (Toon)",
            "固定パレット (Fixed Palette)",
            "減色なし (No Quantization)"
        ])
        self.color_algo_combo.setToolTip(
            "減色アルゴリズムの選択:\n"
            "・単純量子化: 最も高速で簡単なアルゴリズム\n"
            "・メディアンカット法: 色空間を分割し、各領域の代表色を使用\n"
            "・K-means法: 機械学習ベースの色のクラスタリング\n"
            "・オクトツリー法: 色空間の階層的分割による高品質な減色\n"
            "・トゥーンアニメ風: 鮮やかな色とはっきりした色の差を持つアニメ風の配色\n"
            "・固定パレット: Floyd–Steinberg ディザリングで指定パレットに減色（色ムラ軽減）\n"
            "・減色なし: 元画像の色をそのまま使用（高品質、多色数）"
        )
        self.color_algo_combo.currentIndexChanged.connect(self.on_color_algo_changed)
        
        color_algo_layout.addWidget(color_algo_label)
        color_algo_layout.addWidget(self.color_algo_combo)
        
        # 壁の色設定
        wall_color_layout = QHBoxLayout()
        wall_color_label = QLabel("壁の色:")
        self.wall_color_button = QPushButton()
        self.wall_color_button.setFixedSize(30, 30)
        self.wall_color = QColor(255, 255, 255)  # デフォルトは白
        self.set_button_color(self.wall_color_button, self.wall_color)
        self.wall_color_button.clicked.connect(self.select_wall_color)
        
        wall_color_layout.addWidget(wall_color_label)
        wall_color_layout.addWidget(self.wall_color_button)
        wall_color_layout.addStretch()
        # 透過色設定
        transparent_color_layout = QHBoxLayout()
        transparent_color_label = QLabel("透過色:")
        self.transparent_color_button = QPushButton()
        self.transparent_color_button.setFixedSize(30, 30)
        # デフォルトの透過色は黒
        self.transparent_color = QColor(0, 0, 0)
        self.set_button_color(self.transparent_color_button, self.transparent_color)
        self.transparent_color_button.clicked.connect(self.select_transparent_color)
        transparent_color_layout.addWidget(transparent_color_label)
        transparent_color_layout.addWidget(self.transparent_color_button)
        transparent_color_layout.addStretch()
        
        
        # ペイントツール用の変数
        self.current_paint_color = QColor(255, 0, 0)  # デフォルト色：赤
        self.is_paint_mode = True      # ペイントモード（True）または選択モード（False）
        self.is_bucket_mode = False    # 塗りつぶしモード
        self.brush_size = 1            # デフォルトのブラシサイズ
        
        # 減色アルゴリズム用変数
        self.current_color_algo = "simple"  # デフォルトアルゴリズム
        
        # クリック可能なカスタムラベルの定義
        from PyQt5.QtCore import pyqtSignal
        
        class ClickableLabel(QLabel):
            clicked = pyqtSignal(int, int)  # x, y座標を返すシグナル
            hover = pyqtSignal(int, int)    # ホバー時のx, y座標を返すシグナル
            dragPaint = pyqtSignal(int, int)  # ドラッグ中のペイント用シグナル
            mouseWheel = pyqtSignal(int)      # マウスホイール用シグナル（ズーム用）
            
            def __init__(self, text):
                super().__init__(text)
                self.pixmap_size = None
                self.grid_size = None
                self.zoom_factor = None
                self.last_clicked_pos = None  # 最後にクリックされたグリッド位置を保存
                self.hover_grid_pos = None    # ホバー中のグリッド位置
                self.setMouseTracking(True)   # マウスの移動を追跡
                self.is_dragging = False      # ドラッグ状態の追跡
                self.setFocusPolicy(Qt.StrongFocus)  # キーボードフォーカスを受け取れるように
            
            def get_grid_position(self, pos):
                """マウス位置からグリッド位置を計算する共通関数"""
                if not self.pixmap() or not self.pixmap_size or not self.grid_size or not self.zoom_factor:
                    return None
                    
                label_width = self.width()
                label_height = self.height()
                pixmap_width, pixmap_height = self.pixmap_size
                
                # ラベルとピクセル座標の比率を計算
                if label_width <= 0 or label_height <= 0:
                    return None
                    
                # ラベルとピクセルマップのサイズ比を計算
                scale_x = pixmap_width / label_width
                scale_y = pixmap_height / label_height
                
                # ピクセル座標に変換
                pixel_x = int(pos.x() * scale_x)
                pixel_y = int(pos.y() * scale_y)
                
                # グリッド座標に変換（ズームを考慮）
                grid_x = pixel_x // self.zoom_factor
                grid_y = pixel_y // self.zoom_factor
                # グリッド幅と高さを取得
                grid_w = self.grid_size
                # 行の数はピクスマップ高さ÷ズーム倍率
                grid_h = self.pixmap_size[1] // self.zoom_factor if self.zoom_factor else 0
                # グリッド範囲内かチェック
                if 0 <= grid_x < grid_w and 0 <= grid_y < grid_h:
                    return (grid_x, grid_y)
                return None
            
            def mouseMoveEvent(self, event):
                """マウス移動時のイベントハンドラ - ホバー効果とドラッグ時のペイント"""
                grid_pos = self.get_grid_position(event.pos())
                if grid_pos:
                    # ホバー位置の更新
                    if grid_pos != self.hover_grid_pos:
                        self.hover_grid_pos = grid_pos
                        self.hover.emit(grid_pos[0], grid_pos[1])
                        QToolTip.showText(event.globalPos(), f"位置: [{grid_pos[0]}, {grid_pos[1]}]", self)
                    
                    # ドラッグ中の場合は、ペイントシグナルを発信
                    if self.is_dragging and event.buttons() & Qt.LeftButton:
                        self.dragPaint.emit(grid_pos[0], grid_pos[1])
                
                super().mouseMoveEvent(event)
            
            def mousePressEvent(self, event):
                """マウスクリック時のイベントハンドラ"""
                if event.button() == Qt.LeftButton:
                    self.is_dragging = True
                    grid_pos = self.get_grid_position(event.pos())
                    if grid_pos:
                        grid_x, grid_y = grid_pos
                        # デバッグ出力
                        print(f"Label Size: {self.width()}x{self.height()}")
                        print(f"Pixmap Size: {self.pixmap_size}")
                        print(f"Click Position: {event.pos().x()}, {event.pos().y()}")
                        print(f"Grid Position: {grid_x}, {grid_y}")
                        
                        # 最後にクリックした位置を保存
                        self.last_clicked_pos = grid_pos
                        # フォーカスを取得してキーボード操作を受け付ける
                        self.setFocus()
                        # クリックがグリッド内の有効な位置にある場合にシグナルを発信
                        self.clicked.emit(grid_x, grid_y)
            
            def mouseReleaseEvent(self, event):
                """マウスリリース時のイベントハンドラ"""
                if event.button() == Qt.LeftButton:
                    self.is_dragging = False
                super().mouseReleaseEvent(event)
                
            def wheelEvent(self, event):
                """マウスホイール時のイベントハンドラ - ズームイン/アウト用"""
                delta = event.angleDelta().y()
                zoom_change = 1 if delta > 0 else -1
                self.mouseWheel.emit(zoom_change)
                event.accept()
            
            def keyPressEvent(self, event):
                """Spaceでペイント、Shift+矢印キーでカーソル移動"""
                # Spaceキーで現在のカーソル位置にペイント
                if event.key() == Qt.Key_Space:
                    if self.last_clicked_pos is not None:
                        # クリック相当の動作を発火
                        self.clicked.emit(self.last_clicked_pos[0], self.last_clicked_pos[1])
                    event.accept()
                    return
                # Shift + 矢印キーでカーソル移動
                if event.modifiers() & Qt.ShiftModifier:
                    if self.last_clicked_pos is None or self.grid_size is None:
                        return
                    x, y = self.last_clicked_pos
                    dx = dy = 0
                    if event.key() == Qt.Key_Up:
                        dy = -1
                    elif event.key() == Qt.Key_Down:
                        dy = 1
                    elif event.key() == Qt.Key_Left:
                        dx = -1
                    elif event.key() == Qt.Key_Right:
                        dx = 1
                    else:
                        super().keyPressEvent(event)
                        return
                    # 範囲内にクランプ (幅は grid_size、行数は pixmap_height/zoom)
                    new_x = max(0, min(self.grid_size - 1, x + dx))
                    # Y方向は実際のグリッド行数で制限
                    grid_h = (self.pixmap_size[1] // self.zoom_factor) if self.zoom_factor else self.grid_size
                    new_y = max(0, min(grid_h - 1, y + dy))
                    self.last_clicked_pos = (new_x, new_y)
                    # プレビュー更新（ハイライト表示）
                    try:
                        self.window().update_preview()
                    except Exception:
                        pass
                    event.accept()
                    return
                super().keyPressEvent(event)
                
        # パラメータのグリッドレイアウト
        self.param_grid = QGridLayout()
        self.controls = {}
        self.sliders = {}
        
        # パラメータ定義
        parameters = [
            ("Grid Size", 32, 8, 1024),  # 1024x1024までの大きな画像に対応
            ("Dot Size", 2.0, 0.2, 5.0),
            ("Wall Thickness", 0.2, 0.0, 5.0),
            ("Wall Height", 0.4, 0.0, 5.0),
            ("Base Height", 2.0, 0.0, 5.0),
            ("Out Thickness", 0.0, 0.0, 5.0),
            ("Color Step", 8, 1, 64),
            ("Top Colors", 36, 1, 64)
        ]
        
        for i, (label, default, minv, maxv) in enumerate(parameters):
            # パラメータラベルと説明ボタン
            param_label_layout = QHBoxLayout()
            label_widget = QLabel(label)
            
            help_button = QToolButton()
            help_button.setText("?")
            help_button.setToolTip(f"{label}についての説明を表示")
            help_button.clicked.connect(lambda checked, label=label: self.show_parameter_help(label))
            
            param_label_layout.addWidget(label_widget)
            param_label_layout.addWidget(help_button)
            
            # スピンボックス
            is_int = isinstance(default, int)
            spin = QSpinBox() if is_int else QDoubleSpinBox()
            spin.setMinimum(minv)
            spin.setMaximum(maxv)
            spin.setValue(default)
            
            if not is_int:
                spin.setSingleStep(0.1)
                spin.setDecimals(2)
            
            # スライダー
            slider = QSlider(Qt.Horizontal)
            # 整数の場合はそのまま、小数の場合は100倍して扱う
            slider_factor = 1 if is_int else 100
            slider.setMinimum(int(minv * slider_factor))
            slider.setMaximum(int(maxv * slider_factor))
            slider.setValue(int(default * slider_factor))
            
            # 値の連動
            def make_spin_changed(label, slider, is_int, slider_factor):
                def spin_changed():
                    value = self.controls[label].value()
                    self.sliders[label].setValue(int(value * slider_factor))
                    # Clear pixel data when changing parameters that affect color quantization
                    if label in ("Grid Size", "Color Step", "Top Colors") and hasattr(self, "pixels_rounded_np"):
                        self.pixels_rounded_np = None
                    self.update_preview()
                return spin_changed
            
            def make_slider_changed(label, is_int, slider_factor):
                def slider_changed(value):
                    if is_int:
                        self.controls[label].setValue(int(value / slider_factor))
                    else:
                        self.controls[label].setValue(value / slider_factor)
                    # Clear pixel data when changing parameters that affect color quantization
                    if label in ("Grid Size", "Color Step", "Top Colors") and hasattr(self, "pixels_rounded_np"):
                        self.pixels_rounded_np = None
                    self.update_preview()
                return slider_changed
            
            spin.valueChanged.connect(make_spin_changed(label, slider, is_int, slider_factor))
            slider.valueChanged.connect(make_slider_changed(label, is_int, slider_factor))
            
            # グリッドに追加
            self.param_grid.addLayout(param_label_layout, i, 0)
            self.param_grid.addWidget(spin, i, 1)
            self.param_grid.addWidget(slider, i, 2)
            
            self.controls[label] = spin
            self.sliders[label] = slider
        
        # パラメータ設定にSTLエクスポートボタンを追加
        self.param_export_button = QPushButton("STLをエクスポート")
        self.param_export_button.clicked.connect(self.export_stl)
        param_layout.addWidget(self.param_export_button)
        # プレビュー画像保存ボタン
        self.param_export_image_button = QPushButton("プレビュー画像を保存")
        self.param_export_image_button.setToolTip(
            "透過背景（市松模様）込みで、STLと同じサイズのプレビュー画像をPNG保存します"
        )
        self.param_export_image_button.clicked.connect(self.export_preview_image)
        # MRPAF形式でピクセルデータを保存
        self.param_export_mrpaf_button = QPushButton("MRPAFをエクスポート")
        self.param_export_mrpaf_button.setToolTip("プレビュー中のピクセルデータをMRPAF形式で保存します")
        self.param_export_mrpaf_button.clicked.connect(self.export_mrpaf)
        param_layout.addWidget(self.param_export_mrpaf_button)
        param_layout.addWidget(self.param_export_image_button)
        # SVGエクスポートボタン
        self.param_export_svg_button = QPushButton("SVGをエクスポート")
        self.param_export_svg_button.setToolTip("Fusion360用SVGをエクスポートします")
        self.param_export_svg_button.clicked.connect(self.export_svg)
        param_layout.addWidget(self.param_export_svg_button)

        # レイアウトに追加
        param_layout.addLayout(color_algo_layout)
        param_layout.addLayout(wall_color_layout)
        param_layout.addLayout(transparent_color_layout)
        # 線画変換オプション
        lineart_layout = QHBoxLayout()
        self.lineart_checkbox = QCheckBox("線画変換")
        self.lineart_checkbox.setToolTip("線画プレビューを切り替えます")
        self.lineart_checkbox.stateChanged.connect(self.on_lineart_toggled)
        lineart_layout.addWidget(self.lineart_checkbox)
        lineart_layout.addStretch()
        param_layout.addLayout(lineart_layout)
        # 「同色内壁省略」オプションはSTL出力モードで切り替えます
        # STL出力モード選択 (ドットプレート or 市松模様)
        mode_layout = QHBoxLayout()
        mode_label = QLabel("STL出力モード:")
        mode_label.setToolTip("出力するSTLの種類を選択")
        self.stl_mode_combo = QComboBox()
        # STL出力モード: 0=ドットプレート, 1=ドットプレート (同色内壁省略), 2=チェックボード (市松模様), 3=色レイヤーモード, 4=レイヤースタックモード, 5=プラモデル組み立て式モード, 6=色別レイヤー分離出力モード, 7=ハイブリッドモード
        self.stl_mode_combo.addItems([
            "ドットプレート",
            "ドットプレート (同色内壁省略)",
            "チェックボード (市松模様)",
            "色レイヤーモード",
            "レイヤースタックモード",
            "プラモデル組み立て式モード",
            "色別レイヤー分離出力モード",
            "ハイブリッドモード"  # 7: 同色内壁省略 + 色レイヤー化ハイブリッド
        ])
        self.stl_mode_combo.setToolTip("STL出力モードを選択")
        # 選択値を保持
        self.stl_mode = 0
        self.stl_mode_combo.currentIndexChanged.connect(
            lambda idx: setattr(self, 'stl_mode', idx)
        )
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.stl_mode_combo)
        param_layout.addLayout(mode_layout)
        param_layout.addLayout(self.param_grid)
        param_layout.addStretch()  # 下部に余白を追加
        
        # スクロールエリアの設定
        param_scroll.setWidget(param_scroll_content)
        param_scroll.setMinimumHeight(250)  # 最小の高さを設定
        
        param_group_layout = QVBoxLayout()
        param_group_layout.addWidget(param_scroll)
        self.param_group.setLayout(param_group_layout)
        # 別Dock化のためここにはパラメータ設定を追加しません
        # レイヤー設定パネル
        self.layer_group = QGroupBox("レイヤー設定")
        # レイヤー設定リスト (ドラッグで順序変更可能)
        self.layer_scroll = QScrollArea()
        self.layer_scroll.setWidgetResizable(True)
        self.layer_list = QListWidget()
        # 内部移動を有効にし、アイテムをドラッグで並び替え
        self.layer_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.layer_list.setDefaultDropAction(Qt.MoveAction)
        # 複数選択とドラッグを有効化（Shift+クリックで範囲選択）
        self.layer_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.layer_list.setDragEnabled(True)
        self.layer_list.setAcceptDrops(True)
        self.layer_list.setDropIndicatorShown(True)
        # 順序変更時にカラー順序を更新
        self.layer_list.model().rowsMoved.connect(self.on_layer_reordered)
        self.layer_scroll.setWidget(self.layer_list)
        # レイアウト設定
        layer_group_layout = QVBoxLayout()
        # レイヤー更新ボタン（手動更新）
        self.layer_refresh_button = QPushButton("レイヤーを更新")
        self.layer_refresh_button.setToolTip("最新のドットデータでレイヤー設定を更新します")
        self.layer_refresh_button.clicked.connect(self.update_layer_controls)
        layer_group_layout.addWidget(self.layer_refresh_button)
        # 色統合ボタン
        self.merge_color_button = QPushButton("色統合")
        self.merge_color_button.setToolTip("選択した色を統一します")
        self.merge_color_button.clicked.connect(self.merge_selected_colors)
        layer_group_layout.addWidget(self.merge_color_button)
        # 現在プレビューに使用されている色数を表示
        self.color_count_label = QLabel("使用色数: 0色")
        layer_group_layout.addWidget(self.color_count_label)
        # レイヤーの色を明度でソート
        sort_layout = QHBoxLayout()
        asc_btn = QPushButton("明度昇順")
        asc_btn.setToolTip("色の明度が低い(暗い)順から高い(明るい)順に並べ替えます")
        asc_btn.clicked.connect(lambda: self.sort_layers_by_brightness(True))
        desc_btn = QPushButton("明度降順")
        desc_btn.setToolTip("色の明度が高い(明るい)順から低い(暗い)順に並べ替えます")
        desc_btn.clicked.connect(lambda: self.sort_layers_by_brightness(False))
        sort_layout.addWidget(asc_btn)
        sort_layout.addWidget(desc_btn)
        # レイヤーの色を彩度(濃さ)でソート
        sat_asc_btn = QPushButton("濃さ昇順")
        sat_asc_btn.setToolTip("色の濃さが低い(淡い)順から高い(濃い)順に並べ替えます")
        sat_asc_btn.clicked.connect(lambda: self.sort_layers_by_saturation(True))
        sat_desc_btn = QPushButton("濃さ降順")
        sat_desc_btn.setToolTip("色の濃さが高い(濃い)順から低い(淡い)順に並べ替えます")
        sat_desc_btn.clicked.connect(lambda: self.sort_layers_by_saturation(False))
        sort_layout.addWidget(sat_asc_btn)
        sort_layout.addWidget(sat_desc_btn)
        layer_group_layout.addLayout(sort_layout)
        # パス最適順ボタン：島間接続パスの配置数を最大化するレイヤー順を提案
        self.optim_order_button = QPushButton("パス最適順")
        self.optim_order_button.setToolTip("島間接続パスの配置数を最大化するレイヤー順を提案します")
        self.optim_order_button.clicked.connect(self.optimize_layer_order)
        layer_group_layout.addWidget(self.optim_order_button)
        # 各色ごとの高さ設定用スクロール領域
        layer_group_layout.addWidget(self.layer_scroll)
        self.layer_group.setLayout(layer_group_layout)
        # 別Dock化のためここにはレイヤー設定を追加しません
        
        # カラム2：ペイント操作、プレビュー、ズームバー
        column2_panel = QWidget()
        column2_layout = QVBoxLayout(column2_panel)
        
        # ペイント操作ツールバー
        self.paint_tools_group = QGroupBox("ペイントツール")
        paint_tools_layout = QVBoxLayout()
        
        # ドット編集用ツールバー（複数行に分割して配置）
        
        # ペイントモード切り替えボタン
        paint_mode_btn = QPushButton("ペン")
        paint_mode_btn.setToolTip("ペンでドットを描く")
        paint_mode_btn.setCheckable(True)
        paint_mode_btn.setChecked(True)
        paint_mode_btn.setMinimumWidth(60)  # 最小幅を設定
        paint_mode_btn.clicked.connect(lambda checked: self.set_paint_mode(True))
        
        # バケツ（塗りつぶし）モード切り替えボタン
        bucket_mode_btn = QPushButton("塗潰")
        bucket_mode_btn.setToolTip("同じ色のドットを塗りつぶす")
        bucket_mode_btn.setCheckable(True)
        bucket_mode_btn.setMinimumWidth(60)  # 最小幅を設定
        bucket_mode_btn.clicked.connect(lambda checked: self.set_bucket_mode(checked))
        
        # 選択モード切り替えボタン
        select_mode_btn = QPushButton("選択")
        select_mode_btn.setToolTip("クリックで色を選択")
        select_mode_btn.setCheckable(True)
        select_mode_btn.setMinimumWidth(60)  # 最小幅を設定
        select_mode_btn.clicked.connect(lambda checked: self.set_paint_mode(False))
        
        # モードボタンをグループ化
        self.mode_buttons = [paint_mode_btn, select_mode_btn]
        # AIブラシモード切り替えボタン
        self.ai_brush_btn = QPushButton("AIブラシ")
        self.ai_brush_btn.setToolTip("AIブラシモード: クリックしてAIブラシ機能を使用")
        self.ai_brush_btn.setCheckable(True)
        self.ai_brush_btn.setMinimumWidth(60)
        self.ai_brush_btn.clicked.connect(self.toggle_ai_brush_mode)
        # モードボタンに追加
        self.mode_buttons.append(self.ai_brush_btn)
        
        # カラーピッカーボタン（現在のペイント色表示）
        self.color_pick_btn = QPushButton()
        self.color_pick_btn.setFixedSize(30, 30)
        self.set_button_color(self.color_pick_btn, self.current_paint_color)
        self.color_pick_btn.setToolTip("クリックして描画色を変更")
        self.color_pick_btn.clicked.connect(self.select_paint_color)
        
        # スポイトボタン
        eyedropper_btn = QPushButton("🔍")
        eyedropper_btn.setToolTip("クリックでドットの色を取得")
        eyedropper_btn.clicked.connect(self.toggle_eyedropper_mode)
        
        # 透明色ボタン（トグル式）
        self.transparent_btn = QPushButton("透明")
        self.transparent_btn.setToolTip("透明色（黒=0,0,0）で描画")
        self.transparent_btn.setCheckable(True)
        self.transparent_btn.setMinimumWidth(60)  # 最小幅を設定
        self.transparent_btn.toggled.connect(self.toggle_transparent_paint_color)
        
        # 元に戻す（Undo）ボタン
        undo_btn = QPushButton("元に戻す")
        undo_btn.setToolTip("直前の編集を元に戻す")
        undo_btn.setMinimumWidth(40)  # 最小幅を設定
        undo_btn.clicked.connect(self.undo_edit)
        
        # やり直し（Redo）ボタン
        redo_btn = QPushButton("やり直し")
        redo_btn.setToolTip("元に戻した編集をやり直す")
        redo_btn.setMinimumWidth(40)  # 最小幅を設定
        redo_btn.clicked.connect(self.redo_edit)
        
        # ツールバーにボタンを追加
        mode_toolbar = QHBoxLayout()
        mode_toolbar.addWidget(paint_mode_btn)
        mode_toolbar.addWidget(bucket_mode_btn)
        mode_toolbar.addWidget(select_mode_btn)
        mode_toolbar.addWidget(self.ai_brush_btn)
        # AIブラシ閾値設定
        threshold_label = QLabel("AIしきい値:")
        mode_toolbar.addWidget(threshold_label)
        self.ai_threshold_spin = QSpinBox()
        self.ai_threshold_spin.setRange(0, 255)
        self.ai_threshold_spin.setValue(30)
        self.ai_threshold_spin.setToolTip("AIブラシで同色判定する距離のしきい値")
        mode_toolbar.addWidget(self.ai_threshold_spin)
        # 全体AIブラシ: プレビュー内の特徴色を指定色数に丸め込む
        self.global_brush_btn = QPushButton("全体AIブラシ")
        self.global_brush_btn.setToolTip("プレビュー内の特徴色を指定色数に丸め込む")
        self.global_brush_btn.setMinimumWidth(80)
        self.global_brush_btn.clicked.connect(self.handle_global_brush)
        mode_toolbar.addWidget(self.global_brush_btn)
        # 全色ワントーン明るく/暗く
        self.lighten_btn = QPushButton("明るく")
        self.lighten_btn.setToolTip("プレビュー内の全色をワントーン明るくします")
        self.lighten_btn.setMinimumWidth(60)
        self.lighten_btn.clicked.connect(self.handle_lighten_all)
        mode_toolbar.addWidget(self.lighten_btn)
        self.darken_btn = QPushButton("暗く")
        self.darken_btn.setToolTip("プレビュー内の全色をワントーン暗くします")
        self.darken_btn.setMinimumWidth(60)
        self.darken_btn.clicked.connect(self.handle_darken_all)
        mode_toolbar.addWidget(self.darken_btn)

        color_toolbar = QHBoxLayout()
        color_toolbar.addWidget(self.color_pick_btn)
        color_toolbar.addWidget(eyedropper_btn)
        color_toolbar.addWidget(self.transparent_btn)
        # コピー・ペースト機能
        self.copy_btn = QPushButton("コピー")
        self.copy_btn.setToolTip("ドラッグで範囲を選択してコピー")
        self.copy_btn.setCheckable(True)
        self.copy_btn.clicked.connect(self.start_copy_mode)
        color_toolbar.addWidget(self.copy_btn)
        self.paste_btn = QPushButton("ペースト")
        self.paste_btn.setToolTip("クリック位置を起点にペースト")
        self.paste_btn.setCheckable(True)
        self.paste_btn.clicked.connect(self.start_paste_mode)
        color_toolbar.addWidget(self.paste_btn)
        
        history_toolbar = QHBoxLayout()
        history_toolbar.addWidget(undo_btn)
        history_toolbar.addWidget(redo_btn)
        
        # ブラシサイズコントロール
        brush_size_toolbar = QHBoxLayout()
        brush_size_label = QLabel("ブラシサイズ:")
        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setMinimum(1)
        self.brush_size_slider.setMaximum(10)
        self.brush_size_slider.setValue(self.brush_size)  # 初期値
        self.brush_size_slider.setFixedWidth(100)
        self.brush_size_slider.setToolTip("ブラシのサイズを調整します (1-10)")
        self.brush_size_slider.valueChanged.connect(self.on_brush_size_changed)
        self.brush_size_value_label = QLabel(str(self.brush_size))
        
        brush_size_toolbar.addWidget(brush_size_label)
        brush_size_toolbar.addWidget(self.brush_size_slider)
        brush_size_toolbar.addWidget(self.brush_size_value_label)
        
        # モード切替行
        paint_tools_layout.addLayout(mode_toolbar)
        # カラー選択とブラシサイズ行
        row_toolbar = QHBoxLayout()
        row_toolbar.addLayout(color_toolbar)
        row_toolbar.addSpacing(10)
        row_toolbar.addLayout(brush_size_toolbar)
        paint_tools_layout.addLayout(row_toolbar)
        # 履歴行
        paint_tools_layout.addLayout(history_toolbar)
        self.paint_tools_group.setLayout(paint_tools_layout)
        column2_layout.addWidget(self.paint_tools_group)
        
        # プレビュー表示エリア
        preview_group = QGroupBox("プレビュー")
        preview_layout = QVBoxLayout()
        
        self.preview_scroll = QScrollArea()
        # Disable automatic resizing: keep preview_label at its own size for proper scrolling
        self.preview_scroll.setWidgetResizable(False)
        self.preview_scroll.setMinimumHeight(400)
        
        # クリック可能なカスタムラベルを使用
        self.preview_label = ClickableLabel("プレビューが表示されます")
        self.preview_label.setAlignment(Qt.AlignCenter)
        # Do not expand label; keep size matching pixmap for correct scrolling
        self.preview_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        # シグナルを接続
        self.preview_label.clicked.connect(self.on_preview_clicked)
        self.preview_label.hover.connect(self.on_preview_hover)
        self.preview_label.dragPaint.connect(self.on_preview_drag_paint)
        # プレビュー上の矩形選択を有効化（コピー/ペースト用）
        self.preview_label.installEventFilter(self)
        # QScrollArea のビューポートにもイベントフィルタを設定（マウスドラッグ検出用）
        self.preview_scroll.viewport().installEventFilter(self)
        self.preview_rubber_band = QRubberBand(QRubberBand.Rectangle, self.preview_label)
        # マウスホイールはズームではなくスクロールで移動
        # self.preview_label.mouseWheel.connect(self.on_preview_mouse_wheel)
        
        self.preview_scroll.setWidget(self.preview_label)
        preview_layout.addWidget(self.preview_scroll)
        
        # ズームコントロール
        zoom_layout = QHBoxLayout()
        self.zoom_label = QLabel("ズーム:")
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(1)
        self.zoom_slider.setMaximum(40)  # より広いズーム範囲
        self.zoom_slider.setValue(10)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        
        # スクラッチでドット絵をペイントするためのクリアボタン
        self.clear_preview_btn = QPushButton("クリア")
        self.clear_preview_btn.setToolTip("減色プレビューをクリアし、新しいドット絵を描きます")
        self.clear_preview_btn.clicked.connect(self.clear_preview_for_scratch)
        
        zoom_layout.addWidget(self.zoom_label)
        zoom_layout.addWidget(self.zoom_slider)
        zoom_layout.addWidget(self.clear_preview_btn)
        preview_layout.addLayout(zoom_layout)
        
        preview_group.setLayout(preview_layout)
        column2_layout.addWidget(preview_group)
        
        # 現在モードの変数
        self.eyedropper_mode = False  # スポイトモード
        
        # カラム3：STLプレビュー
        column3_panel = QWidget()
        column3_layout = QVBoxLayout(column3_panel)
        
        # STLプレビュー領域（1:1の正方形比率で表示）
        stl_preview_group = QGroupBox("STLプレビュー")
        stl_preview_layout = QVBoxLayout()
        
        # カスタムの正方形ウィジェットを作成
        class SquareWidget(QWidget):
            def __init__(self):
                super().__init__()
                self.setMinimumSize(250, 250)
                
                # 1:1の比率を維持するためのポリシー
                self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                size_policy = self.sizePolicy()
                size_policy.setHeightForWidth(True)
                self.setSizePolicy(size_policy)
                
                # 内部レイアウト
                self.layout = QVBoxLayout(self)
                self.layout.setContentsMargins(0, 0, 0, 0)
            
            def heightForWidth(self, width):
                return width  # 幅と同じ高さを返す（正確な1:1の比率）
            
            def hasHeightForWidth(self):
                return True
            
            # サイズヒントも1:1で提供
            def minimumSizeHint(self):
                size = QSize(250, 250)
                return size
            
            def sizeHint(self):
                size = super().sizeHint()
                return QSize(size.width(), size.width())  # 幅と同じ高さ
        
        # 正方形ウィジェットを作成
        square_widget = SquareWidget()
        
        # STLプレビューラベル
        self.stl_preview_label = QLabel("STLプレビューが表示されます")
        self.stl_preview_label.setAlignment(Qt.AlignCenter)
        self.stl_preview_label.setMinimumSize(200, 200)
        
        # 正方形ウィジェットにラベルを追加
        square_widget.layout.addWidget(self.stl_preview_label)
        
        # 正方形ウィジェットをレイアウトに追加（中央揃え）
        stl_preview_layout.addWidget(square_widget, 0, Qt.AlignCenter)
        
        # STL情報表示部分（スクロール可能）
        info_frame = QFrame()
        info_frame.setFrameShape(QFrame.StyledPanel)
        info_frame.setFrameShadow(QFrame.Sunken)
        info_frame.setLineWidth(1)
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(0, 0, 0, 0)
        
        # スクロールエリアを追加
        info_scroll = QScrollArea()
        info_scroll.setWidgetResizable(True)
        info_scroll.setFrameShape(QFrame.NoFrame)
        # スクロール領域を上下左右に拡張可能に設定
        info_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        info_scroll.setMinimumHeight(180)  # 最小の高さを設定
        # 最大高さ制限を解除（全体レイアウトに合わせて拡張可）
        # info_scroll.setMaximumHeight(250)
        info_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 横スクロールバーを非表示
        
        # STL情報ラベル（クリック可能なHTML表示）
        self.stl_info_label = QLabel("STL情報が表示されます")
        self.stl_info_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.stl_info_label.setWordWrap(True)
        self.stl_info_label.setTextFormat(Qt.RichText)
        self.stl_info_label.setMargin(5)
        self.stl_info_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.stl_info_label.setOpenExternalLinks(False)
        # スクロールビュー内でテーブルが幅いっぱいに表示されるようサイズポリシーを設定
        self.stl_info_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # カラーセルのクリックイベントを接続
        self.stl_info_label.linkActivated.connect(self.on_color_cell_clicked)
        
        # 色ハイライト用のタイマー
        self.highlight_timer = QTimer(self)
        self.highlight_timer.setSingleShot(True)
        self.highlight_timer.timeout.connect(self.clear_color_highlight)
        
        # スクロールエリアにラベルを設定
        info_scroll.setWidget(self.stl_info_label)
        
        # スクロールエリアをフレームに追加
        info_layout.addWidget(info_scroll)
        
        # 情報フレームをレイアウトに追加
        stl_preview_layout.addWidget(info_frame)
        
        stl_preview_group.setLayout(stl_preview_layout)
        column3_layout.addWidget(stl_preview_group)
        
        # 中央にペイント＋プレビューをセット
        self.setCentralWidget(column2_panel)
        # ファイル操作／画像パネルをドッキング
        self.file_dock = QDockWidget("ファイル操作", self)
        self.file_dock.setWidget(column1_panel)
        self.file_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.file_dock.setObjectName("FileDock")
        self.addDockWidget(Qt.LeftDockWidgetArea, self.file_dock)
        # パラメータ設定パネルをドッキング
        self.param_dock = QDockWidget("パラメータ設定", self)
        self.param_dock.setWidget(self.param_group)
        self.param_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.param_dock.setObjectName("ParamDock")
        self.addDockWidget(Qt.LeftDockWidgetArea, self.param_dock)
        # レイヤー設定パネルをドッキング
        self.layer_dock = QDockWidget("レイヤー設定", self)
        self.layer_dock.setWidget(self.layer_group)
        # 色統合機能用フラグ
        self.layer_merge_enable = {}
        self.layer_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.layer_dock.setObjectName("LayerDock")
        self.addDockWidget(Qt.LeftDockWidgetArea, self.layer_dock)
        # ハイブリッドモード用の有効色フラグ
        self.layer_hybrid_enable = {}
        # タブ化：左側のファイル／パラメータ／レイヤー設定をタブでまとめる
        self.tabifyDockWidget(self.file_dock, self.param_dock)
        self.tabifyDockWidget(self.file_dock, self.layer_dock)
        self.file_dock.raise_()
        # STLプレビューをドッキング
        self.stl_dock = QDockWidget("STL プレビュー", self)
        self.stl_dock.setWidget(column3_panel)
        self.stl_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.stl_dock.setObjectName("StlDock")
        self.addDockWidget(Qt.RightDockWidgetArea, self.stl_dock)
        # 「表示」メニューにパネルの表示/ドッキング切替と管理を追加
        view_menu = self.menuBar().addMenu("表示")
        # ファイル操作
        file_act = QAction("ファイル操作", self, checkable=True)
        file_act.setChecked(self.file_dock.isVisible())
        file_act.toggled.connect(lambda checked, d=self.file_dock: (d.setFloating(False) if checked else None) or d.setVisible(checked))
        view_menu.addAction(file_act)
        # パラメータ設定
        param_act = QAction("パラメータ設定", self, checkable=True)
        param_act.setChecked(self.param_dock.isVisible())
        param_act.toggled.connect(lambda checked, d=self.param_dock: (d.setFloating(False) if checked else None) or d.setVisible(checked))
        view_menu.addAction(param_act)
        # レイヤー設定
        layer_act = QAction("レイヤー設定", self, checkable=True)
        layer_act.setChecked(self.layer_dock.isVisible())
        layer_act.toggled.connect(lambda checked, d=self.layer_dock: (d.setFloating(False) if checked else None) or d.setVisible(checked))
        view_menu.addAction(layer_act)
        # STL プレビュー
        stl_act = QAction("STL プレビュー", self, checkable=True)
        stl_act.setChecked(self.stl_dock.isVisible())
        stl_act.toggled.connect(lambda checked, d=self.stl_dock: (d.setFloating(False) if checked else None) or d.setVisible(checked))
        view_menu.addAction(stl_act)
        view_menu.addSeparator()
        # パネル管理ダイアログ
        panel_manager_action = QAction("パネル管理...", self)
        panel_manager_action.triggered.connect(self.show_panel_manager)
        view_menu.addAction(panel_manager_action)
        
        self.image_path = None
        self.zoom_factor = 10
        
        # ドット編集用の変数
        self.current_grid_size = 32  # デフォルト値
        self.pixels_rounded_np = None  # 減色後の画像データ
        
        # 元に戻す（undo）機能のための履歴
        self.edit_history = []  # ピクセルデータの履歴
        self.history_position = -1  # 現在の履歴位置
        self.pixels_rounded_np = None  # 初期化
        
        # 各カラムの設定完了
        
        self.image_path = None
        self.zoom_factor = 10
        
        # ドット編集用の変数
        self.current_grid_size = 32  # デフォルト値
        self.pixels_rounded_np = None  # 減色後の画像データ
        
        # 元に戻す（undo）機能のための履歴
        self.edit_history = []  # ピクセルデータの履歴
        self.history_position = -1  # 現在の履歴位置
        self.pixels_rounded_np = None  # 初期化
    
    def show_parameter_help(self, parameter_name):
        dialog = ParameterHelpDialog(parameter_name, self)
        dialog.exec_()
    def show_panel_manager(self):
        """パネル管理ダイアログを表示"""
        dlg = PanelManagerDialog(self)
        dlg.exec_()
    def on_original_zoom_changed(self, value):
        """オリジナル画像のズームスライダー変更時"""
        if hasattr(self, 'original_pixmap_source'):
            self.applyOriginalZoom()

    def applyOriginalZoom(self):
        """オリジナル画像に現在のズーム率を適用"""
        pix = self.original_pixmap_source
        zoom = self.original_zoom_slider.value()
        factor = zoom / 10.0
        w = int(pix.width() * factor)
        h = int(pix.height() * factor)
        scaled = pix.scaled(w, h, Qt.KeepAspectRatio)
        self.original_image_label.setPixmap(scaled)
        self.original_image_label.adjustSize()
        
    def set_button_color(self, button, color):
        """ボタンの背景色を設定する"""
        button.setStyleSheet(f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); border: 1px solid black;")
        
    def select_wall_color(self):
        """壁の色を選択するダイアログを表示"""
        color = QColorDialog.getColor(self.wall_color, self, "壁の色を選択")
        if color.isValid():
            self.wall_color = color
            self.set_button_color(self.wall_color_button, color)
    
    def select_transparent_color(self):
        """透過色を選択するダイアログを表示"""
        color = QColorDialog.getColor(self.transparent_color, self, "透過色を選択")
        if color.isValid():
            self.transparent_color = color
            self.set_button_color(self.transparent_color_button, color)
            # 透明ペイントボタンのツールチップを更新
            tc = self.transparent_color
            if hasattr(self, 'transparent_btn'):
                self.transparent_btn.setToolTip(f"透明モード: RGB({tc.red()},{tc.green()},{tc.blue()}) で描画")
            # プレビュー更新
            self.update_preview()
            
    def on_preview_hover(self, grid_x, grid_y):
        """ドット上をマウスがホバーした時の処理"""
        if self.pixels_rounded_np is None or not isinstance(self.pixels_rounded_np, np.ndarray):
            return
            
        # NumPy配列は[row, col]=[y, x]の順でアクセス
        # クリック座標(x,y)を入れ替えて[y,x]の順でアクセスする
        array_y = grid_y  # Y軸は反転しない
        array_x = grid_x  # X軸はそのまま
        
        try:
            # ホバー位置のドットの色を取得 - numpy配列は[y, x]の順
            current_color = self.pixels_rounded_np[array_y, array_x]
            # 透明色と比較
            tc = (self.transparent_color.red(), self.transparent_color.green(), self.transparent_color.blue())
            is_transparent = tuple(current_color) == tc
            # ステータス表示文字列
            color_str = "透明" if is_transparent else f"RGB({current_color[0]}, {current_color[1]}, {current_color[2]})"
            self.statusBar().showMessage(f"位置(x,y): [{grid_x}, {grid_y}] → 配列位置[行,列]=[{array_y}, {array_x}] 色: {color_str}")
            
            # ホバー表示でプレビューを更新
            self.update_hover_preview(grid_x, grid_y)
        except Exception as e:
            print(f"ホバー処理エラー: {str(e)}")
    
    def update_hover_preview(self, hover_x, hover_y):
        """ホバー位置のハイライトだけを更新"""
        # 表示更新の負荷を下げるため、常にフル更新せず軽量更新する
        params = {key: spin.value() for key, spin in self.controls.items()}
        
        try:
            # 最後にクリックされた位置があれば取得
            highlight_pos = None
            if hasattr(self.preview_label, 'last_clicked_pos') and self.preview_label.last_clicked_pos is not None:
                highlight_pos = self.preview_label.last_clicked_pos
                
            # ホバー位置
            hover_pos = (hover_x, hover_y)
            
            # 軽量なプレビュー更新（既存のピクセルデータを使用）
            preview_img = generate_preview_image(
                self.image_path,
                self.current_grid_size,
                int(params["Color Step"]),
                int(params["Top Colors"]),
                self.zoom_factor,
                custom_pixels=self.pixels_rounded_np,
                highlight_pos=highlight_pos,
                hover_pos=hover_pos
            )
            
            # プレビュー画像を更新（QPixmapに変換）
            preview_buffer = BytesIO()
            preview_img.save(preview_buffer, format="PNG")
            preview_qimg = QImage()
            preview_qimg.loadFromData(preview_buffer.getvalue())
            preview_pixmap = QPixmap.fromImage(preview_qimg)
            
            # ラベルに表示
            self.preview_label.setPixmap(preview_pixmap)
        except Exception as e:
            print(f"ホバープレビュー更新エラー: {str(e)}")
    
    def on_zoom_changed(self, value):
        """ズームスライダーの値が変更されたときの処理"""
        self.zoom_factor = value
        self.update_preview(custom_pixels=self.pixels_rounded_np)
        
    def on_preview_mouse_wheel(self, zoom_change):
        """マウスホイールでズームを変更する処理"""
        current_zoom = self.zoom_slider.value()
        new_zoom = max(1, min(self.zoom_slider.maximum(), current_zoom + zoom_change))
        self.zoom_slider.setValue(new_zoom)
        
    def toggle_ai_brush_mode(self, checked):
        """AIブラシモードの切り替え処理"""
        self.ai_brush_mode = checked
        if checked:
            # 他の編集モードを解除
            self.is_paint_mode = False
            self.is_bucket_mode = False
            self.eyedropper_mode = False
            # モードボタン状態の更新
            for btn in self.mode_buttons:
                btn.setChecked(btn == self.ai_brush_btn)
            self.statusBar().showMessage("AIブラシモード")
            self.preview_label.setCursor(Qt.ArrowCursor)
        else:
            # ペンモードに戻す
            self.set_paint_mode(True)
            self.statusBar().clearMessage()

    def set_paint_mode(self, is_paint):
        """ペイントモードと選択モードの切り替え"""
        self.is_paint_mode = is_paint
        
        # モードボタンの状態を更新
        for btn in self.mode_buttons:
            btn.setChecked(False)
        
        self.mode_buttons[0 if is_paint else 1].setChecked(True)
        
        # 塗りつぶしモードはペイントモードの時のみ有効
        if not is_paint:
            self.is_bucket_mode = False
        
        # カーソルの更新
        if is_paint and not self.eyedropper_mode and not self.is_bucket_mode:
            # ペイントモードならブラシサイズに合わせたカーソルに
            self.update_paint_cursor()
        elif hasattr(self, 'preview_label'):
            # 選択モードなら通常カーソルに
            self.preview_label.setCursor(Qt.ArrowCursor)
            
        # ステータスバー更新
        mode_name = "ペンモード" if is_paint else "選択モード"
        self.statusBar().showMessage(f"モード: {mode_name}")
        
    def set_bucket_mode(self, is_bucket):
        """塗りつぶしモードの切り替え"""
        self.is_bucket_mode = is_bucket
        
        # 塗りつぶしモードはペイントモードの時のみ有効
        if is_bucket:
            self.is_paint_mode = True
            self.mode_buttons[0].setChecked(True)
            # 塗りつぶしモードのカーソル
            if hasattr(self, 'preview_label'):
                self.preview_label.setCursor(Qt.PointingHandCursor)
        else:
            # ペイントモードならブラシサイズに合わせたカーソル
            if self.is_paint_mode and not self.eyedropper_mode and hasattr(self, 'preview_label'):
                self.update_paint_cursor()
            
        # ステータスバー更新
        mode_name = "塗りつぶしモード" if is_bucket else "ペンモード"
        self.statusBar().showMessage(f"モード: {mode_name}")
    
    def toggle_eyedropper_mode(self):
        """スポイトモードの切り替え"""
        self.eyedropper_mode = not self.eyedropper_mode
        
        # スポイトモード中はカーソルを変更するなどの処理を追加可能
        if self.eyedropper_mode:
            self.statusBar().showMessage("スポイトモード: クリックして色を取得")
            # カーソルを十字に変更
            self.preview_label.setCursor(Qt.CrossCursor)
        else:
            self.statusBar().showMessage("準備完了")
            # ペイントモードではブラシサイズに合わせたカーソルを表示
            if self.is_paint_mode and not self.is_bucket_mode:
                self.update_paint_cursor()
            else:
                # 通常カーソル
                self.preview_label.setCursor(Qt.ArrowCursor)
    
    def select_paint_color(self):
        """ペイントに使用する色を選択"""
        color = QColorDialog.getColor(self.current_paint_color, self, "描画色を選択")
        if color.isValid():
            self.current_paint_color = color
            self.set_button_color(self.color_pick_btn, color)
    
    def toggle_transparent_paint_color(self, checked):
        """透明色（黒=0,0,0）のトグル"""
        if checked:
            # 現在の色を保存して透明色に切り替え
            self.prev_paint_color = self.current_paint_color
            # 透明色はユーザー設定の透過色
            tc = self.transparent_color
            self.current_paint_color = QColor(tc.red(), tc.green(), tc.blue())
            self.set_button_color(self.color_pick_btn, self.current_paint_color)
            self.statusBar().showMessage(f"透明モード: RGB({tc.red()},{tc.green()},{tc.blue()}) で描画")
        else:
            # 前の色に戻す（保存されていなければデフォルト赤）
            if hasattr(self, 'prev_paint_color'):
                self.current_paint_color = self.prev_paint_color
            else:
                self.current_paint_color = QColor(255, 0, 0)
            self.set_button_color(self.color_pick_btn, self.current_paint_color)
            self.statusBar().showMessage("通常モード")
    
    def start_copy_mode(self, checked):
        """コピー操作モードの開始/終了"""
        self.is_copy_mode = checked
        if checked:
            # 他モードを解除
            self.is_paste_mode = False
            if hasattr(self, 'paste_btn'):
                self.paste_btn.setChecked(False)
            self.statusBar().showMessage("コピー範囲をドラッグで指定してください")
        else:
            # キャンセル時はラバーバンドを非表示
            self.preview_rubber_band.hide()
            self.statusBar().showMessage("コピーキャンセル")
    
    def start_paste_mode(self, checked):
        """ペースト操作モードの開始/終了"""
        self.is_paste_mode = checked
        if checked:
            # 他モードを解除
            self.is_copy_mode = False
            if hasattr(self, 'copy_btn'):
                self.copy_btn.setChecked(False)
            # コピー済みバッファ確認
            if self.copy_buffer is None:
                QMessageBox.warning(self, "ペーストエラー", "コピー範囲がありません")
                self.paste_btn.setChecked(False)
                self.is_paste_mode = False
            else:
                self.statusBar().showMessage("ペースト位置をクリックしてください")
        else:
            self.statusBar().showMessage("ペーストキャンセル")
    
    def get_pixel_color(self, grid_x, grid_y):
        """指定位置のピクセル色を取得する"""
        if self.pixels_rounded_np is None or not isinstance(self.pixels_rounded_np, np.ndarray):
            return None
            
        try:
            # NumPy配列は[row, col]=[y, x]の順でアクセス
            array_y = grid_y
            array_x = grid_x
            current_color = self.pixels_rounded_np[array_y, array_x]
            return current_color
        except IndexError:
            print(f"座標[{array_y}, {array_x}]はインデックス範囲外です")
            return None
    
    def paint_pixel(self, grid_x, grid_y, color=None):
        """ピクセルを指定色で塗る（デフォルトは現在のペイント色）"""
        if self.pixels_rounded_np is None or not isinstance(self.pixels_rounded_np, np.ndarray):
            return False
            
        if color is None:
            # QColorからRGB配列に変換
            color = [self.current_paint_color.red(), 
                     self.current_paint_color.green(), 
                     self.current_paint_color.blue()]
        
        # ブラシサイズの取得
        brush_size = self.brush_size
            
        # 編集前の状態を履歴に保存（最初の変更時のみ）
        has_painted = False
        self.save_edit_history()
        
        # ブラシサイズに基づいてピクセルを塗る
        # ブラシの形は円形に近い形にする
        for dy in range(-brush_size+1, brush_size):
            for dx in range(-brush_size+1, brush_size):
                # 円形のブラシパターンを作る（ユークリッド距離）
                if dx*dx + dy*dy < brush_size*brush_size:
                    array_x = grid_x + dx
                    array_y = grid_y + dy
                    
                # グリッド範囲内かチェック (幅と高さを使用)
                    h, w = self.pixels_rounded_np.shape[:2]
                    if 0 <= array_x < w and 0 <= array_y < h:
                        try:
                            # 現在の色と同じなら変更しない
                            current_color = self.pixels_rounded_np[array_y, array_x]
                            if tuple(current_color) != tuple(color):
                                # ピクセルの色を更新
                                self.pixels_rounded_np[array_y, array_x] = color
                                has_painted = True
                        except IndexError:
                            print(f"座標[{array_y}, {array_x}]はインデックス範囲外です")
        
        return has_painted
            
    def bucket_fill(self, grid_x, grid_y):
        """塗りつぶし処理 - 同じ色の隣接ドットを全て指定色で塗る"""
        if self.pixels_rounded_np is None or not isinstance(self.pixels_rounded_np, np.ndarray):
            return
            
        # 編集前の状態を履歴に保存
        self.save_edit_history()
        
        # 塗りつぶす元の色
        target_color = tuple(self.get_pixel_color(grid_x, grid_y))
        if target_color is None:
            return
            
        # 新しい色（現在のペイント色）
        new_color = [self.current_paint_color.red(), 
                     self.current_paint_color.green(), 
                     self.current_paint_color.blue()]
                     
        # 同じ色なら塗りつぶす必要なし
        if target_color == tuple(new_color):
            return
        
        # 幅優先探索で塗りつぶし
        grid_size = self.pixels_rounded_np.shape[0]  # グリッドサイズ
        visited = set()  # 訪問済み座標
        queue = [(grid_x, grid_y)]  # 処理待ちキュー
        
        while queue:
            x, y = queue.pop(0)
            
            # 既に訪問済みならスキップ
            if (x, y) in visited:
                continue
                
            # 範囲外ならスキップ
            if not (0 <= x < grid_size and 0 <= y < grid_size):
                continue
                
            # 色が異なればスキップ
            current = tuple(self.pixels_rounded_np[y, x])
            if current != target_color:
                continue
                
            # 色を変更
            self.pixels_rounded_np[y, x] = new_color
            visited.add((x, y))
            
            # 隣接する4方向をキューに追加
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            for nx, ny in neighbors:
                if (nx, ny) not in visited:
                    queue.append((nx, ny))
        
        # プレビューを更新
        self.update_preview(custom_pixels=self.pixels_rounded_np)
    
    def on_preview_drag_paint(self, grid_x, grid_y):
        """ドラッグ中のペイント処理"""
        if not self.is_paint_mode or self.eyedropper_mode or self.pixels_rounded_np is None:
            return
            
        # ペイントモードの場合は色を塗る
        self.paint_pixel(grid_x, grid_y)
        
        # プレビューを更新
        self.update_preview(custom_pixels=self.pixels_rounded_np)
    
    def on_preview_clicked(self, grid_x, grid_y):
        """減色後のプレビュー画像内のドットがクリックされたときの処理"""
        if self.pixels_rounded_np is None:
            return
        # AIブラシモードの場合は専用処理
        if self.ai_brush_mode:
            self.handle_ai_brush_click(grid_x, grid_y)
            return
        
        # スポイトモードの場合は色を取得
        if self.eyedropper_mode:
            color = self.get_pixel_color(grid_x, grid_y)
            if color is not None:
                self.current_paint_color = QColor(color[0], color[1], color[2])
                self.set_button_color(self.color_pick_btn, self.current_paint_color)
                self.statusBar().showMessage(f"色を取得: RGB({color[0]}, {color[1]}, {color[2]})")
                # スポイト使用後は透明モードを解除
                self.transparent_btn.setChecked(False)
                self.eyedropper_mode = False  # 取得後にモードを解除
            return
        
        # ペイントモードの場合は直接描画
        if self.is_paint_mode:
            # 塗りつぶしモードの場合
            if self.is_bucket_mode:
                self.bucket_fill(grid_x, grid_y)
            else:
                # 通常のペイントモード
                self.paint_pixel(grid_x, grid_y)
                self.update_preview(custom_pixels=self.pixels_rounded_np)
            return
        
        # 以下は選択モード
        try:
            # NumPy配列は[row, col]=[y, x]の順でアクセス
            array_y = grid_y
            array_x = grid_x
            
            # 配列アクセス
            current_color = self.pixels_rounded_np[array_y, array_x]
            
            # 選択したドットの色をQColorに変換
            rgb_color = QColor(current_color[0], current_color[1], current_color[2])
            
            # コンテキストメニューを作成
            from PyQt5.QtWidgets import QMenu, QAction
            
            menu = QMenu(self)
            
            # この色をペイント色に設定
            pick_action = QAction(f"この色を使用 RGB({current_color[0]}, {current_color[1]}, {current_color[2]})", self)
            pick_action.triggered.connect(lambda: self.pick_color_for_paint(rgb_color, None))
            
            # 色変更ダイアログを表示
            change_action = QAction("この位置の色を変更...", self)
            change_action.triggered.connect(lambda: self.show_color_dialog_simple(rgb_color, grid_x, grid_y))
            
            # 透明にする
            is_transparent = tuple(current_color) == (0, 0, 0)
            transparent_action = QAction("透明にする", self)
            transparent_action.setEnabled(not is_transparent)  # 既に透明なら無効化
            transparent_action.triggered.connect(lambda: self.set_transparent_color_simple(grid_x, grid_y))
            
            # 同じ色のドットをすべて置換する
            replace_action = QAction("同じ色のすべてのドットを置換...", self)
            replace_action.setEnabled(not is_transparent)  # 透明色なら無効化
            replace_action.triggered.connect(lambda: self.show_replace_color_dialog(current_color))
            
            # メニューにアクションを追加
            menu.addAction(pick_action)
            menu.addAction(change_action)
            menu.addAction(transparent_action)
            menu.addAction(replace_action)
            
            # カーソル位置にメニューを表示
            from PyQt5.QtGui import QCursor
            menu.exec_(QCursor.pos())
            
        except IndexError as e:
            print(f"座標変換エラー: {e}")
            return
    # AIブラシモード用クリック処理
    def handle_ai_brush_click(self, grid_x, grid_y):
        """AIブラシモードでのクリック処理"""
        if self.pixels_rounded_np is None:
            return
        try:
            selected_color = tuple(self.pixels_rounded_np[grid_y, grid_x])
        except Exception:
            return
        self.ai_brush_target_color = selected_color
        # ハイライト表示
        self.update_preview(custom_pixels=self.pixels_rounded_np, highlight_color=selected_color)
        # 実行確認
        reply = QMessageBox.question(self, "AIブラシ適用確認", "AIブラシを実行しますか？", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            updated_pixels = self.call_ai_brush_api(selected_color, grid_x, grid_y)
            if updated_pixels is not None:
                # 編集履歴に追加
                self.edit_history = self.edit_history[:self.history_position+1]
                self.edit_history.append(updated_pixels.copy())
                self.history_position += 1
                self.pixels_rounded_np = updated_pixels
                self.update_preview(custom_pixels=self.pixels_rounded_np)
        # モード解除後はペンモードに戻す
        self.ai_brush_btn.setChecked(False)
        self.ai_brush_mode = False
        self.set_paint_mode(True)

    def call_ai_brush_api(self, selected_color, grid_x, grid_y):
        """AIブラシ: 選択した色に近いピクセルを同色系で統合し置換するローカル処理"""
        # ピクセルデータを取得
        pixels = self.pixels_rounded_np
        if pixels is None:
            return None
        # 選択色をNumPy配列に
        sel = np.array(selected_color, dtype=int)
        # 各ピクセルとの距離（Euclid）を計算
        diff = np.linalg.norm(pixels.astype(int) - sel[None, None, :], axis=2)
        # 類似色検出の閾値（調整可能: AIしきい値スピンボックスの値を使用）
        threshold = int(self.ai_threshold_spin.value())
        # マスクを作成
        mask = diff <= threshold
        # 類似色がない場合は通知して終了
        if not np.any(mask):
            QMessageBox.information(self, "AIブラシ", "選択色に近い色のドットが見つかりませんでした。閾値を調整してください。")
            return None
        # 類似色のピクセル色を抽出
        similar_colors = pixels[mask].reshape(-1, 3)
        # 最頻出色を取得
        unique, counts = np.unique(similar_colors, axis=0, return_counts=True)
        target_color = unique[counts.argmax()]
        # ピクセルデータをコピーして更新
        new_pixels = pixels.copy()
        new_pixels[mask] = target_color
        # ステータスバーに結果を表示
        self.statusBar().showMessage(f"AIブラシ: {counts.max()} ドットを色 {tuple(target_color)} に統一しました", 3000)
        return new_pixels
    
    def handle_global_brush(self):
        """全体AIブラシ: プレビュー内の特徴色を指定色数に丸め込む処理"""
        # ピクセルデータがない場合は何もしない
        if self.pixels_rounded_np is None:
            return
        # 色数入力ダイアログを表示
        n_colors, ok = QInputDialog.getInt(self, "全体AIブラシ", "変換後の色数を入力してください:", 8, 1, 256)
        if not ok:
            return
        # 既存ピクセルを取得
        pixels = self.pixels_rounded_np
        h, w, _ = pixels.shape
        flat = pixels.reshape(-1, 3)
        # 透過色は変更せず残す
        tc = (self.transparent_color.red(), self.transparent_color.green(), self.transparent_color.blue())
        mask = ~np.all(flat == tc, axis=1)
        valid = flat[mask]
        if valid.size == 0:
            QMessageBox.information(self, "全体AIブラシ", "変換対象のドットがありませんでした。透過色設定を確認してください。")
            return
        # K-meansで代表色を抽出（失敗時はメディアンカット）
        try:
            palette = get_kmeans_palette(valid, n_colors)
        except Exception:
            palette = get_median_cut_palette(valid, n_colors)
        # 透過色をパレットから除外
        palette = [tuple(c) for c in palette if tuple(c) != tc]
        # 各ピクセルを最も近い代表色に丸め込む
        new_flat = flat.copy()
        for idx, pix in enumerate(flat):
            if mask[idx]:
                new_flat[idx] = map_to_closest_color(pix, palette)
        new_pixels = new_flat.reshape((h, w, 3)).astype(np.uint8)
        # 履歴に追加
        self.edit_history = self.edit_history[:self.history_position + 1]
        self.edit_history.append(new_pixels.copy())
        self.history_position += 1
        # 更新
        self.pixels_rounded_np = new_pixels
        self.update_preview(custom_pixels=self.pixels_rounded_np)
        self.statusBar().showMessage(f"全体AIブラシ: {n_colors} 色に変換しました", 3000)

    def handle_lighten_all(self):
        """プレビュー内の全色をワントーン明るくする"""
        if self.pixels_rounded_np is None:
            return
        # 履歴保存
        self.save_edit_history()
        pix = self.pixels_rounded_np
        tc = (self.transparent_color.red(), self.transparent_color.green(), self.transparent_color.blue())
        # 明るくする増分
        step = 16
        new_pix = pix.copy().astype(int)
        mask = ~(np.all(pix == tc, axis=2))
        new_pix[mask] = np.clip(new_pix[mask] + step, 0, 255)
        new_pix = new_pix.astype(np.uint8)
        self.pixels_rounded_np = new_pix
        self.update_preview(custom_pixels=new_pix)
        self.statusBar().showMessage("全色をワントーン明るくしました", 3000)

    def handle_darken_all(self):
        """プレビュー内の全色をワントーン暗くする"""
        if self.pixels_rounded_np is None:
            return
        # 履歴保存
        self.save_edit_history()
        pix = self.pixels_rounded_np
        tc = (self.transparent_color.red(), self.transparent_color.green(), self.transparent_color.blue())
        step = 16
        new_pix = pix.copy().astype(int)
        mask = ~(np.all(pix == tc, axis=2))
        new_pix[mask] = np.clip(new_pix[mask] - step, 0, 255)
        new_pix = new_pix.astype(np.uint8)
        self.pixels_rounded_np = new_pix
        self.update_preview(custom_pixels=new_pix)
        self.statusBar().showMessage("全色をワントーン暗くしました", 3000)

    def pick_color_for_paint(self, color, dialog=None):
        """選択したドットの色をペイント色として設定"""
        self.current_paint_color = color
        self.set_button_color(self.color_pick_btn, color)
        # 透明色モードが有効なら無効化
        if self.transparent_btn.isChecked():
            self.transparent_btn.setChecked(False)
        if dialog:
            dialog.accept()
        
    def show_replace_color_dialog(self, target_color):
        """同じ色のすべてのドットを置換するためのダイアログを表示"""
        if self.pixels_rounded_np is None or not isinstance(self.pixels_rounded_np, np.ndarray):
            return
            
        color_dialog = QColorDialog(self)
        color_dialog.setWindowTitle("新しい色を選択")
        color_dialog.setOption(QColorDialog.ShowAlphaChannel, True)
        
        if color_dialog.exec_():
            new_color = color_dialog.selectedColor()
            if new_color.isValid():
                # (r,g,b)形式に変換
                rgb_new_color = (new_color.red(), new_color.green(), new_color.blue())
                self.replace_all_same_color(target_color, rgb_new_color)
    
    def show_color_dialog_simple(self, current_color, grid_x, grid_y):
        """シンプル版の色選択ダイアログ（コンテキストメニュー用）"""
        if self.pixels_rounded_np is None or not isinstance(self.pixels_rounded_np, np.ndarray):
            return
            
        color_dialog = QColorDialog(self)
        color_dialog.setCurrentColor(current_color)
        color_dialog.setOption(QColorDialog.ShowAlphaChannel, True)
        
        if color_dialog.exec_():
            new_color = color_dialog.selectedColor()
            if new_color.isValid():
                try:
                    # 編集前の状態を履歴に保存
                    self.save_edit_history()
                    
                    # NumPy配列は[row, col]=[y, x]の順でアクセス
                    array_y = grid_y
                    array_x = grid_x
                    
                    # 新しい色をRGB値に変換
                    new_rgb = [new_color.red(), new_color.green(), new_color.blue()]
                    
                    # ピクセルの色を更新
                    self.pixels_rounded_np[array_y, array_x] = new_rgb
                    
                    # プレビューを更新
                    self.update_preview(custom_pixels=self.pixels_rounded_np)
                except Exception as e:
                    print(f"色設定エラー: {str(e)}")
    
    def show_color_dialog(self, current_color, grid_x, grid_y, parent_dialog, transparent_check):
        """色選択ダイアログを表示（旧処理）"""
        if self.pixels_rounded_np is None:
            print("エラー: pixels_rounded_np がNoneです")
            parent_dialog.reject()
            return
            
        # 型チェック: pixels_rounded_npが正しくnumpy配列であることを確認
        if not isinstance(self.pixels_rounded_np, np.ndarray):
            print(f"エラー: pixels_rounded_npが正しいnumpy配列ではありません: {type(self.pixels_rounded_np)}")
            parent_dialog.reject()
            return
            
        color_dialog = QColorDialog(self)
        color_dialog.setCurrentColor(current_color)
        color_dialog.setOption(QColorDialog.ShowAlphaChannel, True)
        
        if color_dialog.exec_():
            new_color = color_dialog.selectedColor()
            if new_color.isValid():
                try:
                    # 編集前の状態を履歴に保存
                    self.save_edit_history()
                    
                    # 透過色チェックがある場合は外す
                    transparent_check.setChecked(False)
                    
                    # NumPy配列は[row, col]=[y, x]の順でアクセス
                    array_y = grid_y
                    array_x = grid_x
                    
                    # 新しい色の確認
                    new_rgb = [new_color.red(), new_color.green(), new_color.blue()]
                    
                    # ピクセルの色を更新
                    self.pixels_rounded_np[array_y, array_x] = new_rgb
                    
                    # プレビューを更新（編集したピクセルデータを使用）
                    self.update_preview(custom_pixels=self.pixels_rounded_np)
                    
                    # 親ダイアログを閉じる
                    parent_dialog.accept()
                except Exception as e:
                    print(f"色設定エラー: {str(e)}")
                    parent_dialog.reject()
                
    def set_transparent_color_simple(self, grid_x, grid_y):
        """ドットを透明（黒色=0,0,0）に設定 - シンプル版"""
        if self.pixels_rounded_np is None or not isinstance(self.pixels_rounded_np, np.ndarray):
            return
        
        # 編集前の状態を履歴に保存
        self.save_edit_history()
        
        # NumPy配列は[row, col]=[y, x]の順でアクセス
        array_y = grid_y
        array_x = grid_x
        
        try:
            # 透過色をユーザー設定の透過色で扱う
            tc = self.transparent_color
            self.pixels_rounded_np[array_y, array_x] = [tc.red(), tc.green(), tc.blue()]
            
            # プレビューを更新
            self.update_preview(custom_pixels=self.pixels_rounded_np)
        except Exception as e:
            print(f"透明色設定エラー: {str(e)}")
    
    def set_transparent_color(self, grid_x, grid_y, dialog):
        """ドットを透明（黒色=0,0,0）に設定 - ダイアログ版（旧処理）"""
        if self.pixels_rounded_np is None:
            print("エラー: pixels_rounded_np がNoneです")
            dialog.reject()
            return
            
        # 型チェック: pixels_rounded_npが正しくnumpy配列であることを確認
        if not isinstance(self.pixels_rounded_np, np.ndarray):
            print(f"エラー: pixels_rounded_npが正しいnumpy配列ではありません: {type(self.pixels_rounded_np)}")
            dialog.reject()
            return
        
        # 編集前の状態を履歴に保存
        self.save_edit_history()
        
        # NumPy配列は[row, col]=[y, x]の順でアクセス
        array_y = grid_y
        array_x = grid_x
        
        print(f"透明化: クリック位置(x,y)=({grid_x}, {grid_y}) → 配列アクセス[y,x]=[{array_y}, {array_x}]")
            
        try:
            # 透過色をユーザー設定の透過色で扱う
            tc = self.transparent_color
            self.pixels_rounded_np[array_y, array_x] = [tc.red(), tc.green(), tc.blue()]
            
            # プレビューを更新
            self.update_preview(custom_pixels=self.pixels_rounded_np)
            
            # ダイアログを閉じる
            dialog.accept()
        except Exception as e:
            print(f"透明色設定エラー: {str(e)}")
            dialog.reject()
        
    def save_edit_history(self):
        """現在のピクセルデータを履歴に保存"""
        if self.pixels_rounded_np is None:
            print("警告: 履歴保存に失敗 - pixels_rounded_npがNoneです")
            return
            
        # 型チェック
        if not isinstance(self.pixels_rounded_np, np.ndarray):
            print(f"警告: 履歴保存に失敗 - pixels_rounded_npが正しいnumpy配列ではありません: {type(self.pixels_rounded_np)}")
            return
            
        try:
            # 履歴が空でない場合は、現在の位置以降の履歴を削除
            if self.history_position < len(self.edit_history) - 1:
                self.edit_history = self.edit_history[:self.history_position + 1]
                
            # 現在のピクセルデータのコピーを作成して履歴に追加
            self.edit_history.append(self.pixels_rounded_np.copy())
            self.history_position = len(self.edit_history) - 1
            print(f"履歴保存: 位置 {self.history_position}, 履歴数 {len(self.edit_history)}")
        except Exception as e:
            print(f"履歴保存エラー: {str(e)}")
        
    def undo_edit(self):
        """直前の編集を元に戻す"""
        try:
            if not hasattr(self, 'edit_history') or not self.edit_history:
                print("履歴がありません")
                return
                
            if self.history_position <= 0:
                print("これ以上戻れる履歴がありません")
                return
                
            # 一つ前の履歴に戻る
            self.history_position -= 1
            print(f"Undo: 履歴位置 {self.history_position + 1} → {self.history_position}")
            
            if self.history_position < len(self.edit_history):
                self.pixels_rounded_np = self.edit_history[self.history_position].copy()
                
                # プレビューを更新
                self.update_preview(custom_pixels=self.pixels_rounded_np)
            else:
                print(f"エラー: 無効な履歴位置 {self.history_position}, 履歴数: {len(self.edit_history)}")
        except Exception as e:
            print(f"Undoエラー: {str(e)}")
        
    def redo_edit(self):
        """元に戻した編集をやり直す"""
        try:
            if not hasattr(self, 'edit_history') or not self.edit_history:
                print("履歴がありません")
                return
                
            if self.history_position >= len(self.edit_history) - 1:
                print("これ以上進める履歴がありません")
                return
                
            # 次の履歴に進む
            self.history_position += 1
            print(f"Redo: 履歴位置 {self.history_position - 1} → {self.history_position}")
            
            if 0 <= self.history_position < len(self.edit_history):
                self.pixels_rounded_np = self.edit_history[self.history_position].copy()
                
                # プレビューを更新
                self.update_preview(custom_pixels=self.pixels_rounded_np)
            else:
                print(f"エラー: 無効な履歴位置 {self.history_position}, 履歴数: {len(self.edit_history)}")
        except Exception as e:
            print(f"Redoエラー: {str(e)}")
            
    def event(self, event):
        """カスタムイベントの処理"""
        from PyQt5.QtCore import QEvent
        
        # 画像保存完了イベント
        if event.type() == QEvent.User + 10:  # ImageSavedEvent
            # ファイル名に "top" が含まれているかどうかで上面/正面を判断
            if "top" in event.filename:
                message = f"上面からの画像を {event.filename} として保存しました"
            else:
                message = f"正面からの画像を {event.filename} として保存しました"
                
            # 既存のメッセージに追加
            current_text = self.input_label.text()
            # "保存しました" が含まれていなければ追加
            if "保存しました" not in current_text:
                self.input_label.setText(f"{current_text} {message}")
            else:
                # 既に画像保存メッセージがある場合は、そのメッセージの後に追加
                self.input_label.setText(f"{current_text}、{message}")
                
            return True
            
        # 画像保存エラーイベント
        elif event.type() == QEvent.User + 11:  # ImageSaveErrorEvent
            self.input_label.setText(f"{self.input_label.text()} 画像の保存に失敗しました: {event.error_msg}")
            return True
            
        return super().event(event)
    
    def select_image(self):
        # 画像ファイルまたは MRPAF ファイルを選択可能にする
        path, _ = QFileDialog.getOpenFileName(
            self,
            "画像を開く",
            "",
            "画像ファイル (*.png *.jpg *.jpeg *.gif *.bmp *.mrpaf);;PNG (*.png);;JPEG (*.jpg *.jpeg);;その他 (*.*)"
        )
        if path:
            self.image_path = path
            self.input_label.setText(path)
            # 元画像をロードしてズーム適用 (.mrpaf対応)
            ext = os.path.splitext(path)[1].lower()
            if ext == ".mrpaf":
                try:
                    # MRPAFファイルから画像を生成
                    pil_img = load_mrpaf(path)
                    buf = BytesIO()
                    pil_img.save(buf, format="PNG")
                    qimg = QImage()
                    qimg.loadFromData(buf.getvalue())
                    self.original_pixmap_source = QPixmap.fromImage(qimg)
                    self.applyOriginalZoom()
                except Exception as e:
                    print(f"MRPAF読み込みエラー: {e}")
            elif ext == ".gif":
                try:
                    # GIFファイルをPILで読み込み（最初のフレーム）
                    pil_img = Image.open(path)
                    pil_img = pil_img.convert("RGBA")
                    buf = BytesIO()
                    pil_img.save(buf, format="PNG")
                    qimg = QImage()
                    qimg.loadFromData(buf.getvalue())
                    self.original_pixmap_source = QPixmap.fromImage(qimg)
                    self.applyOriginalZoom()
                except Exception as e:
                    print(f"GIF読み込みエラー: {e}")
                    try:
                        self.original_pixmap_source = QPixmap(self.image_path)
                        self.applyOriginalZoom()
                    except Exception:
                        pass
            else:
                try:
                    self.original_pixmap_source = QPixmap(self.image_path)
                    self.applyOriginalZoom()
                except Exception:
                    pass
            # 画像を読み込んだらグリッド幅を自動検出（幅を優先）
            try:
                img = Image.open(path)
                w, h = img.size
                if hasattr(self, 'controls') and 'Grid Size' in self.controls:
                    self.controls['Grid Size'].setValue(int(w))
            except Exception:
                # 自動検出失敗時は何もしない
                pass
            # 新しい画像を選択したらハイライトをクリア
            if hasattr(self.preview_label, 'last_clicked_pos'):
                self.preview_label.last_clicked_pos = None
            # プレビュー更新
            self.update_preview()
    
    def convert_to_line_art(self):
        """オリジナル画像を線画に変換してプレビュー表示"""
        if not hasattr(self, 'image_path') or not self.image_path:
            QMessageBox.warning(self, "線画変換エラー", "先に画像を読み込んでください。")
            return
        try:
            # プレビュー表示されている画像（ズーム適用済み）を使用して線画生成
            pixmap = None
            if hasattr(self.preview_label, 'pixmap') and self.preview_label.pixmap() is not None:
                pixmap = self.preview_label.pixmap()
            elif hasattr(self, 'original_pixmap_source'):
                pixmap = self.original_pixmap_source
            if pixmap:
                # QPixmap -> QImage -> NumPy配列 (BGR)
                qimg = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
                w, h = qimg.width(), qimg.height()
                ptr = qimg.bits()
                ptr.setsize(h * w * 3)
                arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 3))
                img_np = arr[..., ::-1]  # RGB->BGR
            else:
                # ファイルからロード（RGB->BGR）
                img_pil = Image.open(self.image_path).convert("RGB")
                img_np = np.array(img_pil)[..., ::-1]
            # build_preview_and_svgによる線画プレビュー作成
            preview_bgr, contours, mask_regions = build_preview_and_svg(img_np)
            # BGR->RGB (reverse channels without OpenCV, ensure contiguous memory)
            preview_rgb = np.ascontiguousarray(preview_bgr[..., ::-1])
            h, w, ch = preview_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(preview_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.preview_label.setPixmap(pixmap)
        except Exception as e:
            QMessageBox.critical(self, "線画変換エラー", f"線画への変換に失敗しました: {e}")
    
    def on_lineart_toggled(self, state):
        """Rebuild preview according to line-art checkbox state."""
        # Simply refresh preview; update_preview will handle line-art mode
        self.update_preview()

    def on_color_algo_changed(self, index):
        """減色アルゴリズムが変更されたときの処理"""
        algo_map = {
            0: "simple",          # 単純量子化
            1: "median_cut",      # メディアンカット法
            2: "kmeans",          # K-means法
            3: "octree",          # オクトツリー法
            4: "toon",            # トゥーンアニメ風
            5: "fixed_palette",   # 固定パレットディザリング
            6: "none"             # 減色なし
        }
        
        self.current_color_algo = algo_map.get(index, "simple")
        
        # ステータスメッセージ更新
        status_messages = {
            "simple":       "単純量子化アルゴリズムを使用します",
            "median_cut":   "メディアンカット法（色空間分割による減色）を使用します",
            "kmeans":       "K-means法（機械学習ベースのクラスタリング）を使用します",
            "octree":       "オクトツリー法（階層的色空間分割）を使用します",
            "toon":         "トゥーンアニメ風の鮮やかな色使いで減色します",
            "fixed_palette":"固定パレットディザリングで減色します",
            "none":         "減色せず元画像の色をそのまま使用します"
        }
        
        self.statusBar().showMessage(status_messages.get(self.current_color_algo, "減色アルゴリズムを変更しました"))
        
        # 画像がロードされていればプレビューを更新
        if hasattr(self, 'image_path') and self.image_path:
            # 編集履歴をリセット
            if hasattr(self, 'pixels_rounded_np'):
                self.pixels_rounded_np = None
            self.update_preview()
    
    def clear_preview_for_scratch(self):
        """減色プレビューをクリアし、新しいドット絵を描くための空白キャンバスを作成する"""
        if not self.image_path or not hasattr(self, 'current_grid_size'):
            return
            
        try:
            # 現在のグリッドサイズに合わせて全て透明(黒)の配列を作成
            grid_size = self.current_grid_size
            blank_pixels = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
            
            # 編集履歴を保存
            self.save_edit_history()
            
            # 空のピクセルデータを設定
            self.pixels_rounded_np = blank_pixels
            
            # プレビューを更新
            self.update_preview(custom_pixels=blank_pixels)
            
            # メッセージを表示
            print("プレビューをクリアしました。スクラッチからドット絵をペイントできます。")
            self.input_label.setText("プレビューをクリアしました。スクラッチからドット絵をペイントできます。")
            
        except Exception as e:
            print(f"プレビュークリアエラー: {str(e)}")
            self.input_label.setText(f"プレビュークリアエラー: {str(e)}")
    
    def on_brush_size_changed(self, value):
        """ブラシサイズが変更されたときの処理"""
        self.brush_size = value
        self.brush_size_value_label.setText(str(value))
        self.statusBar().showMessage(f"ブラシサイズ: {value}")
        
        # ペイントモードのときはカーソルを更新
        if self.is_paint_mode and not self.eyedropper_mode and not self.is_bucket_mode:
            self.update_paint_cursor()
    
    def update_paint_cursor(self):
        """現在のブラシサイズに合わせてカーソルを更新"""
        if not hasattr(self, 'preview_label'):
            return
            
        # ブラシサイズに合わせたカーソルを作成
        if self.brush_size <= 1:
            # サイズ1ならデフォルトのカーソル
            self.preview_label.setCursor(Qt.ArrowCursor)
        else:
            # カスタムカーソルを作成
            cursor_size = min(64, max(16, self.brush_size * 6))  # ブラシサイズに比例したカーソルサイズ
            pixmap = QPixmap(cursor_size, cursor_size)
            pixmap.fill(Qt.transparent)  # 透明で初期化
            
            # 円を描画
            painter = QPainter(pixmap)
            painter.setPen(QPen(Qt.black, 1))
            painter.setBrush(Qt.transparent)  # 塗りつぶさない
            painter.drawEllipse(2, 2, cursor_size-4, cursor_size-4)  # 少し小さめに描画
            painter.end()
            
            # カーソルのホットスポットは中心
            hotspot = QPoint(cursor_size // 2, cursor_size // 2)
            cursor = QCursor(pixmap, hotspot.x(), hotspot.y())
            self.preview_label.setCursor(cursor)
    
    def update_preview(self, custom_pixels=None, highlight_color=None):
        """プレビュー画像を更新する（custom_pixelsが指定された場合はそれを使用）"""
        # If line-art mode is active, delegate to line-art conversion and skip color preview
        if hasattr(self, 'lineart_checkbox') and self.lineart_checkbox.isChecked():
            self.convert_to_line_art()
            return
        # If pixel data (edited) already exists and no explicit custom_pixels passed,
        # reuse existing pixels to avoid resetting on parameter changes
        if custom_pixels is None and hasattr(self, 'pixels_rounded_np') and self.pixels_rounded_np is not None:
            custom_pixels = self.pixels_rounded_np
        if not self.image_path:
            return
        
        try:
            self.zoom_factor = self.zoom_slider.value()
            params = {key: spin.value() for key, spin in self.controls.items()}
            
            # 現在のグリッドサイズを保存
            self.current_grid_size = int(params["Grid Size"])
            
            # オリジナル画像はズームスライダーで制御
            if hasattr(self, 'original_pixmap_source'):
                self.applyOriginalZoom()
            
            # カーソル位置を常にハイライト表示
            highlight_pos = None
            if hasattr(self.preview_label, 'last_clicked_pos') and self.preview_label.last_clicked_pos is not None:
                highlight_pos = self.preview_label.last_clicked_pos
                
            # ホバー位置の取得（スポイトモード時は明確に表示）
            hover_pos = None
            if hasattr(self.preview_label, 'hover_grid_pos') and self.preview_label.hover_grid_pos is not None:
                hover_pos = self.preview_label.hover_grid_pos
            
            # 減色後の画像を生成または更新
            try:
                if custom_pixels is not None:
                    # カスタムピクセルデータ（編集済み）を使用
                    self.pixels_rounded_np = custom_pixels
                    preview_img = generate_preview_image(
                        self.image_path,
                        self.current_grid_size,
                        int(params["Color Step"]),
                        int(params["Top Colors"]),
                        self.zoom_factor,
                        custom_pixels=self.pixels_rounded_np,
                        highlight_pos=highlight_pos,
                        hover_pos=hover_pos,
                        color_algo=self.current_color_algo,
                        highlight_color=highlight_color
                    )
                else:
                    # 新たに画像を生成
                    preview_img = generate_preview_image(
                        self.image_path,
                        self.current_grid_size,
                        int(params["Color Step"]),
                        int(params["Top Colors"]),
                        self.zoom_factor,
                        highlight_pos=highlight_pos,
                        hover_pos=hover_pos,
                        color_algo=self.current_color_algo,
                        highlight_color=highlight_color
                    )
            except Exception as e:
                # エラーが発生した場合、カスタムピクセルを無視して再試行
                print(f"プレビュー生成エラー: {str(e)}、単純アルゴリズムで再試行します")
                self.current_color_algo = "simple"  # 単純アルゴリズムにフォールバック
                self.color_algo_combo.setCurrentIndex(0)  # UIも更新
                preview_img = generate_preview_image(
                    self.image_path,
                    self.current_grid_size,
                    int(params["Color Step"]),
                    int(params["Top Colors"]),
                    self.zoom_factor,
                    color_algo="simple",
                    highlight_color=highlight_color
                )
            
            # カスタムピクセルを使用していない場合のみ、ピクセルデータを生成
            if custom_pixels is None:
                try:
                    # ピクセルデータを保存（後でドット編集時に使用）
                    # 画像を幅 self.current_grid_size、高さはアスペクト比に応じてリサイズ
                    img_full = Image.open(self.image_path).convert("RGB")
                    orig_w, orig_h = img_full.size
                    grid_w = self.current_grid_size
                    grid_h = int(round(grid_w * orig_h / orig_w)) if orig_w > 0 else grid_w
                    grid_h = max(1, grid_h)
                    img_resized = img_full.resize((grid_w, grid_h), resample=Image.NEAREST)
                    pixels = np.array(img_resized).reshape(-1, 3)
                    
                    # 選択されたアルゴリズムで減色処理
                    if self.current_color_algo == "simple":
                        # 単純な量子化アルゴリズム
                        pixels_normalized = normalize_colors(pixels, int(params["Color Step"]))
                        colors = [tuple(c) for c in pixels_normalized]
                        color_counts = Counter(colors)
                        top_colors = [c for c, _ in color_counts.most_common(int(params["Top Colors"]))]
                        pixels_rounded = [map_to_closest_color(c, top_colors) for c in colors]
                        
                    elif self.current_color_algo == "median_cut":
                        # メディアンカット法
                        palette = get_median_cut_palette(pixels, int(params["Top Colors"]))
                        pixels_rounded = [map_to_closest_color(c, palette) for c in pixels]
                        
                    elif self.current_color_algo == "kmeans":
                        # K-means法
                        try:
                            palette = get_kmeans_palette(pixels, int(params["Top Colors"]))
                            pixels_rounded = [map_to_closest_color(c, palette) for c in pixels]
                        except ImportError:
                            # scikit-learnがインストールされていない場合
                            print("K-means減色にはscikit-learnが必要です。単純アルゴリズムを使用します。")
                            self.current_color_algo = "simple"
                            self.color_algo_combo.setCurrentIndex(0)
                            # 単純アルゴリズムでフォールバック
                            pixels_normalized = normalize_colors(pixels, int(params["Color Step"]))
                            colors = [tuple(c) for c in pixels_normalized]
                            color_counts = Counter(colors)
                            top_colors = [c for c, _ in color_counts.most_common(int(params["Top Colors"]))]
                            pixels_rounded = [map_to_closest_color(c, top_colors) for c in colors]
                            
                    elif self.current_color_algo == "octree":
                        # オクトツリー法
                        palette = get_octree_palette(pixels, int(params["Top Colors"]))
                        pixels_rounded = [map_to_closest_color(c, palette) for c in pixels]
                        
                    elif self.current_color_algo == "none":
                        # 減色なし - 元の色をそのまま使用
                        pixels_rounded = pixels.tolist()  # NumPy配列をリストに変換
                        
                    elif self.current_color_algo == "toon":
                        # トゥーンアニメ風
                        try:
                            palette = get_toon_palette(pixels, int(params["Top Colors"]))
                            pixels_rounded = [map_to_closest_color(c, palette) for c in pixels]
                        except Exception as e:
                            print(f"トゥーンアニメ風減色エラー: {str(e)}。単純アルゴリズムを使用します。")
                            self.current_color_algo = "simple"
                            self.color_algo_combo.setCurrentIndex(0)
                            # 単純アルゴリズムでフォールバック
                            pixels_normalized = normalize_colors(pixels, int(params["Color Step"]))
                            colors = [tuple(c) for c in pixels_normalized]
                            color_counts = Counter(colors)
                            top_colors = [c for c, _ in color_counts.most_common(int(params["Top Colors"]))]
                            pixels_rounded = [map_to_closest_color(c, top_colors) for c in colors]
                        
                    else:
                        # デフォルトは単純アルゴリズム
                        pixels_normalized = normalize_colors(pixels, int(params["Color Step"]))
                        colors = [tuple(c) for c in pixels_normalized]
                        color_counts = Counter(colors)
                        top_colors = [c for c, _ in color_counts.most_common(int(params["Top Colors"]))]
                        pixels_rounded = [map_to_closest_color(c, top_colors) for c in colors]
                    
                    # 適切な形状のnumpy配列に変換
                    pixels_array = np.array(pixels_rounded, dtype=np.uint8)
                    self.pixels_rounded_np = pixels_array.reshape((grid_h, grid_w, 3))
                    # 初期状態を履歴に追加（元に戻す機能のため）
                    self.edit_history = [self.pixels_rounded_np.copy()]
                    self.history_position = 0
                except Exception as e:
                    print(f"ピクセルデータ生成エラー: {str(e)}")
                    return
            
            # プレビュー画像をQPixmapに変換して表示
            try:
                preview_buffer = BytesIO()
                preview_img.save(preview_buffer, format="PNG")
                preview_qimg = QImage()
                preview_qimg.loadFromData(preview_buffer.getvalue())
                preview_pixmap = QPixmap.fromImage(preview_qimg)
                
                # クリックイベント用にピクセルサイズ情報を設定
                self.preview_label.pixmap_size = (preview_pixmap.width(), preview_pixmap.height())
                self.preview_label.grid_size = self.current_grid_size
                self.preview_label.zoom_factor = self.zoom_factor
                
                self.preview_label.setPixmap(preview_pixmap)
                self.preview_label.adjustSize()
                
                # カーソルをモードに応じて変更
                if self.eyedropper_mode:
                    self.preview_label.setCursor(Qt.CrossCursor)  # スポイトモード
                elif self.is_bucket_mode:
                    self.preview_label.setCursor(Qt.PointingHandCursor)  # 塗りつぶしモード
                elif self.is_paint_mode:
                    self.update_paint_cursor()  # ブラシサイズに合わせたカーソル
                else:
                    self.preview_label.setCursor(Qt.ArrowCursor)  # 選択モード
                
            except Exception as e:
                print(f"プレビュー表示エラー: {str(e)}")
                self.input_label.setText(f"プレビュー表示エラー: {str(e)}")
                
        except Exception as e:
            print(f"update_preview全体エラー: {str(e)}")
            self.input_label.setText(f"画像表示エラー: {str(e)}")
    
    def generate_html_report(self, stl_path, mesh):
        """STL情報とアプリの情報をHTMLレポートとして保存する"""
        try:
            # HTMLファイルパスを取得（STLと同じ名前＋.html）
            html_path = f"{os.path.splitext(stl_path)[0]}.html"
            
            # パラメータ値を取得
            params = {key: spin.value() for key, spin in self.controls.items()}
            
            # オリジナル画像と減色プレビュー画像のパス
            timestamp = int(time.time())
            original_img_path = f"{os.path.splitext(stl_path)[0]}_original_{timestamp}.png"
            preview_img_path = f"{os.path.splitext(stl_path)[0]}_preview_{timestamp}.png"
            stl_preview_img_path = f"{os.path.splitext(stl_path)[0]}_stl_preview_{timestamp}.png"
            
            # オリジナル画像の保存
            original_img = Image.open(self.image_path).convert("RGB")
            original_img.save(original_img_path)
            
            # 減色プレビュー画像の保存
            if hasattr(self, 'pixels_rounded_np') and self.pixels_rounded_np is not None:
                preview_img = Image.fromarray(self.pixels_rounded_np, mode='RGB')
                preview_img.save(preview_img_path)
            else:
                # 減色プレビューがなければオリジナルをコピー
                preview_img_path = original_img_path
            
            # STLプレビュー画像の保存（すでに保存されている場合は再利用）
            if hasattr(self, 'stl_preview_img_path') and os.path.exists(self.stl_preview_img_path):
                # 既存のSTLプレビュー画像をコピー
                import shutil
                shutil.copy(self.stl_preview_img_path, stl_preview_img_path)
            else:
                # STLプレビュー画像を新規生成
                self.generate_stl_preview_image(mesh, stl_preview_img_path)
            
            # 壁の色をRGBタプルに変換
            if isinstance(self.wall_color, QColor):
                wall_color = (self.wall_color.red(), self.wall_color.green(), self.wall_color.blue())
            else:
                wall_color = self.wall_color
            
            # アルゴリズム名の取得
            algo_names = {
                "simple": "単純量子化",
                "median_cut": "メディアンカット法",
                "kmeans": "K-means法",
                "octree": "オクトツリー法",
                "toon": "トゥーンアニメ風",
                "none": "減色なし"
            }
            algo_name = algo_names.get(self.current_color_algo, "単純量子化")
            
            # STLの情報を取得
            bounds = mesh.bounds
            min_bounds = bounds[0]
            max_bounds = bounds[1]
            
            width = max_bounds[0] - min_bounds[0]  # X方向の幅
            depth = max_bounds[1] - min_bounds[1]  # Y方向の深さ
            height = max_bounds[2] - min_bounds[2]  # Z方向の高さ
            
            # ドット数と色情報
            total_dots = 0
            unique_colors = 0
            color_counts = Counter()
            color_volumes = {}
            
            if hasattr(self, 'pixels_rounded_np') and self.pixels_rounded_np is not None:
                # 透明でないピクセルをカウント (RGB(0,0,0)は透明として扱う)
                non_transparent_mask = (self.pixels_rounded_np != 0).any(axis=2)
                total_dots = np.sum(non_transparent_mask)
                
                # 色ごとのドット数をカウント
                colors = [tuple(pixel) for pixel in self.pixels_rounded_np.reshape(-1, 3) 
                         if tuple(pixel) != (0, 0, 0)]
                color_counts = Counter(colors)
                unique_colors = len(color_counts)
                
                # 体積計算（ドットのサイズと各色のドット数から概算）
                dot_size = float(params["Dot Size"])
                base_height = float(params["Base Height"])
                wall_height = float(params["Wall Height"])
                dot_height = wall_height + base_height  # mm
                dot_volume = dot_size * dot_size * dot_height  # mm³
                
                # 色ごとの体積を計算
                for color, count in color_counts.items():
                    color_volumes[color] = count * dot_volume
            
            # HTMLレポートの生成
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Dot Plate Generator - レポート</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        h1, h2 {{ color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .images {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }}
        .image-container {{ flex: 1; min-width: 300px; }}
        .image-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        .info-section {{ margin-bottom: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .color-cell {{ width: 20px; height: 20px; display: inline-block; border: 1px solid #ccc; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Dot Plate Generator - プロジェクトレポート</h1>
        
        <div class="info-section">
            <h2>ファイル情報</h2>
            <table>
                <tr><th>項目</th><th>値</th></tr>
                <tr><td>入力ファイル</td><td>{self.image_path}</td></tr>
                <tr><td>STL出力ファイル</td><td>{stl_path}</td></tr>
                <tr><td>レポート作成日時</td><td>{time.strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
            </table>
        </div>
        
        <div class="images">
            <div class="image-container">
                <h3>オリジナル画像</h3>
                <img src="{os.path.basename(original_img_path)}" alt="オリジナル画像">
            </div>
            <div class="image-container">
                <h3>減色プレビュー</h3>
                <img src="{os.path.basename(preview_img_path)}" alt="減色プレビュー">
            </div>
            <div class="image-container">
                <h3>STLプレビュー</h3>
                <img src="{os.path.basename(stl_preview_img_path)}" alt="STLプレビュー">
            </div>
        </div>
        
        <div class="info-section">
            <h2>パラメータ設定</h2>
            <table>
                <tr><th>パラメータ</th><th>値</th></tr>
                <tr><td>グリッドサイズ</td><td>{params["Grid Size"]}</td></tr>
                <tr><td>ドットサイズ</td><td>{params["Dot Size"]} mm</td></tr>
                <tr><td>壁の厚さ</td><td>{params["Wall Thickness"]} mm</td></tr>
                <tr><td>壁の高さ</td><td>{params["Wall Height"]} mm</td></tr>
                <tr><td>ベースの高さ</td><td>{params["Base Height"]} mm</td></tr>
                <tr><td>外壁の厚さ</td><td>{params["Out Thickness"]} mm</td></tr>
                <tr><td>色ステップ</td><td>{params["Color Step"]}</td></tr>
                <tr><td>上位色数</td><td>{params["Top Colors"]}</td></tr>
                <tr><td>減色アルゴリズム</td><td>{algo_name}</td></tr>
                <tr><td>壁の色</td><td style="display: flex; align-items: center;">
                    <div class="color-cell" style="background-color: rgb{wall_color};"></div>
                    &nbsp;RGB{wall_color}
                </td></tr>
                <tr><td>同色間内壁省略</td><td>{"あり" if getattr(self, 'stl_mode', 0) == 1 else "なし"}</td></tr>
            </table>
        </div>
        
        <div class="info-section">
            <h2>STL情報</h2>
            <table>
                <tr><th>項目</th><th>値</th></tr>
                <tr><td>最大幅 (X)</td><td>{width:.2f} mm</td></tr>
                <tr><td>最大奥行き (Y)</td><td>{depth:.2f} mm</td></tr>
                <tr><td>最大高さ (Z)</td><td>{height:.2f} mm</td></tr>
                <tr><td>ドット数</td><td>{total_dots}</td></tr>
                <tr><td>使用色数</td><td>{unique_colors}</td></tr>
            </table>
        </div>
        
        <div class="info-section">
            <h2>色情報</h2>
            <table>
                <tr>
                    <th style="width: 10%;">色</th>
                    <th style="width: 30%;">RGB値</th>
                    <th style="width: 30%;">ドット数</th>
                    <th style="width: 30%;">推定体積 (mm³)</th>
                </tr>
"""

            # 色ごとの詳細テーブルを追加
            if color_counts:
                # 色を使用頻度順にソート
                sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
                
                # 各色の行を追加
                for color, count in sorted_colors:
                    r, g, b = color
                    volume = color_volumes.get(color, 0)
                    hex_color = f"#{r:02x}{g:02x}{b:02x}"
                    
                    html_content += f"""
                <tr>
                    <td style="text-align: center;">
                        <div class="color-cell" style="background-color: {hex_color}; display: inline-block; width: 20px; height: 20px; border: 1px solid #ccc;"></div>
                    </td>
                    <td>RGB({r}, {g}, {b})</td>
                    <td>{count}</td>
                    <td>{volume:.2f}</td>
                </tr>"""
            
            html_content += """
            </table>
        </div>
    </div>
</body>
</html>"""
            
            # HTMLファイルに保存
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            self.statusBar().showMessage(f"HTMLレポートを保存しました: {html_path}")
            return html_path
            
        except Exception as e:
            print(f"HTMLレポート生成エラー: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def generate_stl_preview_image(self, mesh, output_path):
        """STLプレビュー画像を生成して保存"""
        try:
            # MatplotlibでのSTLプレビュー画像生成
            import matplotlib.pyplot as plt
            from matplotlib import rcParams
            from mpl_toolkits.mplot3d import Axes3D
            import numpy as np
            
            # プロット設定
            rcParams['axes.labelsize'] = 8
            rcParams['xtick.labelsize'] = 8
            rcParams['ytick.labelsize'] = 8
            
            # 描画スペース確保
            fig = plt.figure(figsize=(6, 6), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            
            # メッシュの三角形を描画
            vertices = mesh.vertices
            faces = mesh.faces
            
            # 上面視点になるようにZ方向から見下ろす角度に設定
            ax.view_init(elev=90, azim=-90)
            
            # 三角形をポリゴンとして描画
            for face in faces:
                verts = vertices[face]
                tri = Axes3D.art3d.Poly3DCollection([verts])
                tri.set_color('lightgray')
                tri.set_edgecolor('black')
                ax.add_collection3d(tri)
            
            # 軸の設定
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # 視点の調整
            bounds = mesh.bounds
            center = [(bounds[0][i] + bounds[1][i]) / 2 for i in range(3)]
            max_range = max([bounds[1][i] - bounds[0][i] for i in range(3)])
            
            # すべての次元で等しいスケール
            ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
            ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
            ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)
            
            # 余白を小さく
            plt.tight_layout()
            
            # 画像として保存
            plt.savefig(output_path)
            plt.close()
            
            return output_path
            
        except Exception as e:
            print(f"STLプレビュー画像生成エラー: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def update_layer_controls(self):
        """Refresh the layer settings controls based on current pixels_rounded_np"""
        # Clear existing layer list
        self.layer_list.clear()
        # Initialize layer_heights dict if not present
        if not hasattr(self, 'layer_heights'):
            self.layer_heights = {}
        # Collect unique colors excluding transparent (black)
        if hasattr(self, 'pixels_rounded_np') and self.pixels_rounded_np is not None:
            # 現在のプレビューで使用されている色数をカウント (透過色は除外)
            arr = self.pixels_rounded_np.reshape(-1, 3)
            tc = (self.transparent_color.red(), self.transparent_color.green(), self.transparent_color.blue())
            counts = Counter([tuple(c) for c in arr])
            present = [color for color, _ in counts.most_common() if color != tc]
            # ラベルを更新
            self.color_count_label.setText(f"使用色数: {len(present)}色")
            # Initialize or update order: keep existing order, append new
            if not hasattr(self, 'layer_color_order') or not self.layer_color_order:
                self.layer_color_order = present.copy()
            else:
                new_order = [c for c in self.layer_color_order if c in present]
                for c in present:
                    if c not in new_order:
                        new_order.append(c)
                self.layer_color_order = new_order
            # Build controls in layer order using drag-and-drop list
            for color in self.layer_color_order:
                if color not in present:
                    continue
                # Ensure a default height entry exists
                default_h = self.layer_heights.get(color, 0.2)
                self.layer_heights[color] = default_h
                # Create item widget
                item_widget = QWidget()
                row_layout = QHBoxLayout(item_widget)
                # Thumbnail
                label = QLabel()
                pixmap = QPixmap(20, 20)
                pixmap.fill(QColor(*color))
                label.setPixmap(pixmap)
                row_layout.addWidget(label)
                # Hybrid enable checkbox
                if not color in self.layer_hybrid_enable:
                    self.layer_hybrid_enable[color] = False
                cb = QCheckBox("色レイヤー化")
                cb.setChecked(self.layer_hybrid_enable.get(color, False))
                cb.stateChanged.connect(lambda st, c=color: self.layer_hybrid_enable.__setitem__(c, st == Qt.Checked))
                row_layout.addWidget(cb)
                # 色統合用チェックボックス
                if color not in self.layer_merge_enable:
                    self.layer_merge_enable[color] = False
                cbm = QCheckBox("色統合")
                cbm.setChecked(self.layer_merge_enable.get(color, False))
                cbm.stateChanged.connect(lambda st, c=color: self.layer_merge_enable.__setitem__(c, st == Qt.Checked))
                row_layout.addWidget(cbm)
                # Show palette mix ratios for this layer color
                mix = self.get_palette_mix(color)
                for mc in mix:
                    pal_label = QLabel()
                    pal_pix = QPixmap(14, 14)
                    pal_pix.fill(QColor(*mc))
                    pal_label.setPixmap(pal_pix)
                    pal_label.setToolTip(f"Mix palette: RGB{mc}")
                    row_layout.addWidget(pal_label)
                # Spin box for height
                spin = QDoubleSpinBox()
                spin.setRange(0.0, 10.0)
                spin.setSingleStep(0.1)
                spin.setValue(default_h)
                spin.valueChanged.connect(lambda val, c=color: self.layer_heights.__setitem__(c, val))
                spin.valueChanged.connect(lambda val: self.update_layer_controls())
                row_layout.addWidget(spin)
                # Up/Down buttons for ordering
                up_btn = QPushButton("▲")
                up_btn.setFixedSize(30, 20)
                up_btn.clicked.connect(lambda _, c=color: self.move_layer_up(c))
                row_layout.addWidget(up_btn)
                down_btn = QPushButton("▼")
                down_btn.setFixedSize(30, 20)
                down_btn.clicked.connect(lambda _, c=color: self.move_layer_down(c))
                row_layout.addWidget(down_btn)
                # Add to drag-and-drop list
                item = QListWidgetItem(self.layer_list)
                item.setSizeHint(item_widget.sizeHint())
                item.setData(Qt.UserRole, color)
                self.layer_list.addItem(item)
                self.layer_list.setItemWidget(item, item_widget)
    
    # Layer ordering controls
    def move_layer_up(self, color):
        """Move the specified color one layer up in the order."""
        if hasattr(self, 'layer_color_order'):
            idx = self.layer_color_order.index(color)
            if idx > 0:
                self.layer_color_order[idx], self.layer_color_order[idx-1] = (
                    self.layer_color_order[idx-1], self.layer_color_order[idx])
                self.update_layer_controls()

    def move_layer_down(self, color):
        """Move the specified color one layer down in the order."""
        if hasattr(self, 'layer_color_order'):
            idx = self.layer_color_order.index(color)
            if idx < len(self.layer_color_order) - 1:
                self.layer_color_order[idx], self.layer_color_order[idx+1] = (
                    self.layer_color_order[idx+1], self.layer_color_order[idx])
                self.update_layer_controls()
    
    def sort_layers_by_brightness(self, ascending=True):
        """Sort layer_color_order by perceived brightness (ascending or descending)."""
        if not hasattr(self, 'layer_color_order'):
            return
        # brightness: Y = 0.299R + 0.587G + 0.114B
        def brightness(c):
            return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        # Sort in place
        self.layer_color_order.sort(key=brightness, reverse=not ascending)
        self.update_layer_controls()
    
    def sort_layers_by_saturation(self, ascending=True):
        """Sort layer_color_order by color saturation (ascending or descending)."""
        if not hasattr(self, 'layer_color_order'):
            return
        def saturation(c):
            r, g, b = c
            # Normalize to [0,1]
            r, g, b = r / 255.0, g / 255.0, b / 255.0
            mx = max(r, g, b)
            mn = min(r, g, b)
            return (mx - mn) / mx if mx != 0 else 0
        self.layer_color_order.sort(key=saturation, reverse=not ascending)
        self.update_layer_controls()
    
    def optimize_layer_order(self):
        """Optimize layer order to maximize inter-island connection paths."""
        # Ensure pixel data is available
        if not hasattr(self, 'pixels_rounded_np') or self.pixels_rounded_np is None:
            QMessageBox.warning(self, "警告", "レイヤーの最適化には画像の読み込みと減色処理が必要です")
            return
        import numpy as np
        from collections import deque
        # Compute connection path counts per color
        counts = {}
        height, width = self.pixels_rounded_np.shape[:2]
        for color in self.layer_color_order:
            if color == (0, 0, 0):
                counts[color] = 0
                continue
            # Create mask for the color
            mask = np.all(self.pixels_rounded_np == color, axis=2)
            visited = set()
            islands = []
            # Detect islands (connected components)
            for y in range(height):
                for x in range(width):
                    if mask[y, x] and (x, y) not in visited:
                        queue = deque([(x, y)])
                        visited.add((x, y))
                        island = []
                        while queue:
                            xx, yy = queue.popleft()
                            island.append((xx, yy))
                            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                                nx, ny = xx + dx, yy + dy
                                if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and (nx, ny) not in visited:
                                    visited.add((nx, ny))
                                    queue.append((nx, ny))
                        islands.append(island)
            # Generate paths between islands and count unique positions
            paths = generate_connection_paths_between_islands(islands)
            counts[color] = len(paths)
        # Sort layers by descending number of paths
        self.layer_color_order.sort(key=lambda c: counts.get(c, 0), reverse=True)
        self.update_layer_controls()
        QMessageBox.information(self, "パス最適順", "レイヤー順序をパス数が多い順に並び替えました")
    
    def merge_selected_colors(self):
        """色統合: チェックされたレイヤー色を1色に統一する"""
        if not hasattr(self, 'pixels_rounded_np') or self.pixels_rounded_np is None:
            QMessageBox.warning(self, "エラー", "先にプレビューを生成してください。")
            return
        # 色選択ダイアログ
        color = QColorDialog.getColor(parent=self, title="統一後の色を選択")
        if not color.isValid():
            return
        target = (color.red(), color.green(), color.blue())
        arr = self.pixels_rounded_np.copy()
        # 統合対象の色を置換
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                pix = tuple(arr[y, x])
                if self.layer_merge_enable.get(pix, False):
                    arr[y, x] = target
        # 履歴に追加
        self.edit_history = self.edit_history[:self.history_position+1]
        self.edit_history.append(arr.copy())
        self.history_position = len(self.edit_history) - 1
        # 更新
        self.pixels_rounded_np = arr
        # Update preview and layer controls to reflect new colors
        self.update_preview(custom_pixels=arr)
        self.update_layer_controls()
        QMessageBox.information(self, "色統合", "選択された色を統一しました。プレビューと使用色数を更新しました。")

    def on_layer_reordered(self, parent, start, end, destination, row):
        """Update layer_color_order after drag-and-drop reordering."""
        new_order = []
        for i in range(self.layer_list.count()):
            item = self.layer_list.item(i)
            color = item.data(Qt.UserRole)
            new_order.append(color)
        self.layer_color_order = new_order
    
    def show_layer_settings_dialog(self):
        """ポップアップウィンドウでレイヤー設定を開く"""
        # Ensure layer order and heights are up-to-date
        self.update_layer_controls()
        dialog = QDialog(self)
        dialog.setWindowTitle("レイヤー設定 (別ウィンドウ)")
        dialog.resize(400, 500)
        self.layer_dialog = dialog
        layout = QVBoxLayout(dialog)
        # 更新ボタン
        refresh_btn = QPushButton("レイヤーを更新")
        refresh_btn.setToolTip("最新のドットデータでレイヤー設定を更新します")
        def on_refresh():
            refresh_content()
            self.update_layer_controls()
        refresh_btn.clicked.connect(on_refresh)
        layout.addWidget(refresh_btn)
        # ポップアップ：レイヤーの色を明度・彩度でソートするボタン
        sort_popup_layout = QHBoxLayout()
        # 明度ソート
        popup_asc_btn = QPushButton("明度昇順")
        popup_asc_btn.setToolTip("色の明度が低い(暗い)順から高い(明るい)順に並べ替えます")
        popup_asc_btn.clicked.connect(lambda: [self.sort_layers_by_brightness(True), refresh_content()])
        popup_desc_btn = QPushButton("明度降順")
        popup_desc_btn.setToolTip("色の明度が高い(明るい)順から低い(暗い)順に並べ替えます")
        popup_desc_btn.clicked.connect(lambda: [self.sort_layers_by_brightness(False), refresh_content()])
        # 彩度(濃さ)ソート
        popup_sat_asc_btn = QPushButton("濃さ昇順")
        popup_sat_asc_btn.setToolTip("色の濃さが低い(淡い)順から高い(濃い)順に並べ替えます")
        popup_sat_asc_btn.clicked.connect(lambda: [self.sort_layers_by_saturation(True), refresh_content()])
        popup_sat_desc_btn = QPushButton("濃さ降順")
        popup_sat_desc_btn.setToolTip("色の濃さが高い(濃い)順から低い(淡い)順に並べ替えます")
        popup_sat_desc_btn.clicked.connect(lambda: [self.sort_layers_by_saturation(False), refresh_content()])
        sort_popup_layout.addWidget(popup_asc_btn)
        sort_popup_layout.addWidget(popup_desc_btn)
        sort_popup_layout.addWidget(popup_sat_asc_btn)
        sort_popup_layout.addWidget(popup_sat_desc_btn)
        layout.addLayout(sort_popup_layout)
        # スクロールエリア
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        # コンテンツ更新関数
        def refresh_content():
            # クリア
            while content_layout.count():
                item = content_layout.takeAt(0)
                w = item.widget()
                if w:
                    w.deleteLater()
                l = item.layout()
                if l:
                    while l.count():
                        sub = l.takeAt(0)
                        w2 = sub.widget()
                        if w2:
                            w2.deleteLater()
            # 各色ごとに行を作成
            for color in self.layer_color_order:
                if color == (0, 0, 0):
                    continue
                h = self.layer_heights.get(color, 0.2)
                # 行レイアウトとアイテム作成
                row = QHBoxLayout()
                # カラーサムネイル
                label = QLabel()
                pixmap = QPixmap(20, 20)
                pixmap.fill(QColor(*color))
                label.setPixmap(pixmap)
                row.addWidget(label)
                # 登録パレットから混色比を表示
                mix = self.get_palette_mix(color)
                for mc in mix:
                    pal_label = QLabel()
                    pal_pix = QPixmap(14, 14)
                    pal_pix.fill(QColor(*mc))
                    pal_label.setPixmap(pal_pix)
                    pal_label.setToolTip(f"Mix palette: RGB{mc}")
                    row.addWidget(pal_label)
                # 高さスピンボックス
                spin = QDoubleSpinBox()
                spin.setRange(0.0, 10.0)
                spin.setSingleStep(0.1)
                spin.setValue(h)
                spin.valueChanged.connect(lambda val, c=color: self.layer_heights.__setitem__(c, val))
                spin.valueChanged.connect(lambda val: self.update_layer_controls())
                row.addWidget(spin)
                # 順序移動ボタン
                up_btn = QPushButton("▲")
                up_btn.setFixedSize(30, 20)
                up_btn.clicked.connect(lambda _, c=color: [self.move_layer_up(c), refresh_content(), self.update_layer_controls()])
                row.addWidget(up_btn)
                down_btn = QPushButton("▼")
                down_btn.setFixedSize(30, 20)
                down_btn.clicked.connect(lambda _, c=color: [self.move_layer_down(c), refresh_content(), self.update_layer_controls()])
                row.addWidget(down_btn)
                content_layout.addLayout(row)
        # 初期表示
        refresh_content()
        dialog.show()

    def export_stl(self):
        # Ensure PIL Image is available
        from PIL import Image
        if not self.image_path:
            self.input_label.setText("画像が選択されていません")
            return
            
        # パラメータ取得
        params = {key: spin.value() for key, spin in self.controls.items()}
        # 市松模様モードの処理: 透過色(黒)を除外し輪郭検知
        # STL出力モードで「市松模様 (チェックボード)」は index 2
        if getattr(self, 'stl_mode', 0) == 2:
            cb_path, _ = QFileDialog.getSaveFileName(
                self, "チェックボードSTLを保存", "checkerboard.stl", "STLファイル (*.stl)"
            )
            if cb_path:
                # パラメータ取得
                grid_size = int(params.get("Grid Size", 0))
                dot_size = float(params.get("Dot Size", 0.0))
                base_height = float(params.get("Base Height", 0.0))
                wall_thickness = float(params.get("Wall Thickness", 0.0))
                wall_height = float(params.get("Wall Height", 0.0))
                # マスク生成: カスタム編集後のピクセルデータがあれば使用し、黒色(0,0,0)を透過扱い。それ以外は元画像からフォールバック。
                if hasattr(self, 'pixels_rounded_np') and isinstance(self.pixels_rounded_np, np.ndarray):
                    # 編集・減色後のRGB配列を利用 (shape: H x W x 3)
                    arr = self.pixels_rounded_np
                    mask = (arr != 0).any(axis=2)
                else:
                    # 元画像からマスク生成 (RGBA or RGB or グレースケール)
                    from PIL import Image as _Image
                    import numpy as _np
                    img_tmp = _Image.open(self.image_path)
                    img_small = img_tmp.resize((grid_size, grid_size), resample=_Image.NEAREST)
                    arr = _np.array(img_small)
                    if arr.ndim == 3 and arr.shape[2] == 4:
                        # RGBA: alpha > 0 を有効
                        mask = arr[:, :, 3] > 0
                    elif arr.ndim == 3:
                        # RGB: 非黒ピクセルを有効
                        mask = _np.any(arr[:, :, :3] != 0, axis=2)
                    elif arr.ndim == 2:
                        # グレースケール: 非ゼロピクセルを有効
                        mask = arr != 0
                    else:
                        # フォールバック: 全ピクセル有効
                        mask = _np.ones((grid_size, grid_size), dtype=bool)
                # チェックボードSTL生成 (輪郭検知付き)
                mesh = generate_checkerboard_stl(
                    grid_size, dot_size, base_height,
                    wall_thickness, wall_height, mask
                )
                mesh.export(cb_path)
                self.input_label.setText(f"{cb_path} に市松模様STLをエクスポートしました")
            return
        # 色レイヤーモードの処理
        # STL出力モードで「色レイヤーモード」は index 3
        if getattr(self, 'stl_mode', 0) == 3:
            layer_path, _ = QFileDialog.getSaveFileName(
                self, "色レイヤーモードSTLを保存", "layered.stl", "STLファイル (*.stl)"
            )
            if layer_path:
                params = {key: spin.value() for key, spin in self.controls.items()}
                # 色レイヤーモード：編集済みピクセルデータを使用して積層STLを生成
                mesh = generate_layered_stl(
                    self.pixels_rounded_np,
                    layer_path,
                    int(params.get("Grid Size", 0)),
                    float(params.get("Dot Size", 0.0)),
                    float(params.get("Base Height", 0.0)),
                    float(params.get("Wall Thickness", 0.0)),
                    float(params.get("Wall Height", 0.0)),
                    self.layer_heights,
                    self.layer_color_order
                )
                # プレビューとレポート表示
                self.show_stl_preview(mesh)
                _ = self.generate_html_report(layer_path, mesh)
                self.input_label.setText(f"{layer_path} に色レイヤーモードSTLをエクスポートしました")
            return
        
        # レイヤースタックモードの処理
        if getattr(self, 'stl_mode', 0) == 4:
            import os
            stack_path, _ = QFileDialog.getSaveFileName(
                self, "レイヤースタックSTLを保存（ベースファイル名）", "layer_stack", "STLファイル (*.stl)"
            )
            if stack_path:
                base_path = os.path.splitext(stack_path)[0]
                params = {key: spin.value() for key, spin in self.controls.items()}
                
                # 前提条件チェック
                if not hasattr(self, 'layer_color_order') or not self.layer_color_order:
                    QMessageBox.warning(self, "レイヤー設定エラー", "レイヤー設定が見つかりません。先にレイヤー設定を行ってください。")
                    return
                
                if not hasattr(self, 'pixels_rounded_np') or self.pixels_rounded_np is None:
                    QMessageBox.warning(self, "ピクセルデータエラー", "編集可能なピクセルデータがありません。先に画像を読み込んでプレビューを生成してください。")
                    return
                
                try:
                    self.input_label.setText("レイヤースタックSTLファイルを生成中...")
                    QApplication.processEvents()
                    
                    # レイヤースタック用STL生成
                    meshes = generate_layer_stack_stl(
                        self.pixels_rounded_np,
                        base_path,
                        int(params.get("Grid Size", 0)),
                        float(params.get("Dot Size", 0.0)),
                        float(params.get("Wall Thickness", 0.0)),
                        float(params.get("Wall Height", 0.0)),
                        float(params.get("Base Height", 0.0)),
                        float(params.get("Out Thickness", 0.0)),
                        self.layer_color_order,
                        getattr(self, 'layer_heights', {})
                    )
                    
                    if meshes:
                        # 最初のレイヤーをプレビュー表示
                        self.show_stl_preview(meshes[0])
                        
                        # HTMLレポート生成
                        first_layer_path = f"{base_path}_layer_01_{self.layer_color_order[0][0]:03d}_{self.layer_color_order[0][1]:03d}_{self.layer_color_order[0][2]:03d}.stl"
                        html_path = self.generate_html_report(first_layer_path, meshes[0])
                        
                        layer_count = len(meshes)
                        message = f"{layer_count}個のレイヤースタックSTLを {base_path}_layer_XX.stl として出力しました"
                        if html_path:
                            message += f"、HTMLレポート {html_path} も生成しました"
                        self.input_label.setText(message)
                    else:
                        self.input_label.setText("レイヤースタックSTLの生成に失敗しました")
                        
                except Exception as e:
                    print(f"レイヤースタックSTL生成エラー: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    self.input_label.setText(f"レイヤースタックSTL生成エラー: {str(e)}")
            return

        # プラモデル組み立て式モードの処理
        if getattr(self, 'stl_mode', 0) == 5:
            import os
            plastic_path, _ = QFileDialog.getSaveFileName(
                self, "プラモデルSTLを保存（ベースファイル名）", "plastic_model", "STLファイル (*.stl)"
            )
            if plastic_path:
                base_path = os.path.splitext(plastic_path)[0]
                params = {key: spin.value() for key, spin in self.controls.items()}
                
                # 前提条件チェック
                if not hasattr(self, 'layer_color_order') or not self.layer_color_order:
                    QMessageBox.warning(self, "レイヤー設定エラー", "レイヤー設定が見つかりません。先にレイヤー設定を行ってください。")
                    return
                
                if not hasattr(self, 'pixels_rounded_np') or self.pixels_rounded_np is None:
                    QMessageBox.warning(self, "ピクセルデータエラー", "編集可能なピクセルデータがありません。先に画像を読み込んでプレビューを生成してください。")
                    return
                
                try:
                    self.input_label.setText("プラモデル組み立て式STLファイルを生成中...")
                    QApplication.processEvents()
                    
                    # プラモデル用STL生成
                    meshes = generate_plastic_model_stl(
                        self.pixels_rounded_np,
                        base_path,
                        int(params.get("Grid Size", 0)),
                        float(params.get("Dot Size", 0.0)),
                        float(params.get("Wall Thickness", 0.0)),
                        float(params.get("Wall Height", 0.0)),
                        float(params.get("Base Height", 0.0)),
                        float(params.get("Out Thickness", 0.0)),
                        self.layer_color_order,
                        connection_thickness=0.1,  # 連結薄皮の厚さ
                        sprue_width=0.3           # ランナーの幅
                    )
                    
                    if meshes:
                        # 最初のパーツをプレビュー表示
                        self.show_stl_preview(meshes[0])
                        
                        # HTMLレポート生成
                        first_part_path = f"{base_path}_plastic_{self.layer_color_order[0][0]:03d}_{self.layer_color_order[0][1]:03d}_{self.layer_color_order[0][2]:03d}.stl"
                        html_path = self.generate_html_report(first_part_path, meshes[0])
                        
                        parts_count = len(meshes)
                        instructions_path = f"{base_path}_assembly_instructions.html"
                        message = f"{parts_count}個のプラモデルパーツを {base_path}_plastic_XXX.stl として出力、組み立て説明書 {instructions_path} も生成しました"
                        if html_path:
                            message += f"、HTMLレポート {html_path} も生成しました"
                        self.input_label.setText(message)
                    else:
                        self.input_label.setText("プラモデル組み立て式STLの生成に失敗しました")
                        
                except Exception as e:
                    print(f"プラモデルSTL生成エラー: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    self.input_label.setText(f"プラモデルSTL生成エラー: {str(e)}")
            return
        
        # 色別レイヤー分離出力モードの処理
        if getattr(self, 'stl_mode', 0) == 6:
            import os
            separation_path, _ = QFileDialog.getSaveFileName(
                self, "色別レイヤー分離STLを保存（ベースファイル名）", "color_separated", "STLファイル (*.stl)"
            )
            if separation_path:
                base_path = os.path.splitext(separation_path)[0]
                params = {key: spin.value() for key, spin in self.controls.items()}
                
                # 前提条件チェック
                if not hasattr(self, 'layer_color_order') or not self.layer_color_order:
                    QMessageBox.warning(self, "レイヤー設定エラー", "レイヤー設定が見つかりません。先にレイヤー設定を行ってください。")
                    return
                
                if not hasattr(self, 'pixels_rounded_np') or self.pixels_rounded_np is None:
                    QMessageBox.warning(self, "ピクセルデータエラー", "編集可能なピクセルデータがありません。先に画像を読み込んでプレビューを生成してください。")
                    return
                
                try:
                    self.input_label.setText("色別レイヤー分離STLファイルを生成中...")
                    QApplication.processEvents()
                    
                    # 色別レイヤー分離STL生成
                    meshes = generate_color_separated_layers_stl(
                        self.pixels_rounded_np,
                        base_path,
                        int(params.get("Grid Size", 0)),
                        float(params.get("Dot Size", 0.0)),
                        float(params.get("Wall Thickness", 0.0)),
                        float(params.get("Wall Height", 0.0)),
                        float(params.get("Base Height", 0.0)),
                        float(params.get("Out Thickness", 0.0)),
                        self.layer_color_order
                    )
                    
                    if meshes:
                        # 最初のレイヤーをプレビュー表示
                        self.show_stl_preview(meshes[0])
                        
                        # HTMLレポート生成
                        first_layer_path = f"{base_path}_color_01_RGB{self.layer_color_order[0][0]:03d}_{self.layer_color_order[0][1]:03d}_{self.layer_color_order[0][2]:03d}.stl"
                        html_path = self.generate_html_report(first_layer_path, meshes[0])
                        
                        layers_count = len(meshes)
                        report_path = f"{base_path}_color_separation_report.html"
                        message = f"{layers_count}個の色別レイヤーを {base_path}_color_XX_RGBXXX_XXX_XXX.stl として出力、分離レポート {report_path} も生成しました"
                        if html_path:
                            message += f"、HTMLレポート {html_path} も生成しました"
                        self.input_label.setText(message)
                    else:
                        self.input_label.setText("色別レイヤー分離STLの生成に失敗しました")
                        
                except Exception as e:
                    print(f"色別レイヤー分離STL生成エラー: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    self.input_label.setText(f"色別レイヤー分離STL生成エラー: {str(e)}")
            return

        # ハイブリッドモード: ドットプレート (同色内壁省略) + 色レイヤーモード の複合
        # インデックス7
        if getattr(self, 'stl_mode', 0) == 7:
            # ファイル選択
            out_path, _ = QFileDialog.getSaveFileName(self, "ハイブリッドSTLを保存", "hybrid.stl", "STLファイル (*.stl)")
            if not out_path:
                return
            params = {key: spin.value() for key, spin in self.controls.items()}
            # カスタムピクセル配列の取得
            arr_full = self.pixels_rounded_np if hasattr(self, 'pixels_rounded_np') and self.pixels_rounded_np is not None else None
            if arr_full is None:
                QMessageBox.warning(self, "エラー", "先にプレビューを生成してください。")
                return
            # 分割色リスト
            off_colors = [c for c in self.layer_color_order if not self.layer_hybrid_enable.get(c, False)]
            on_colors = [c for c in self.layer_color_order if self.layer_hybrid_enable.get(c, False)]
            meshes = []
            import tempfile, os
            # ベース部 (OFF色) をドットプレート生成
            # マスクOFF色を残し、他は透明化
            mask_off = np.zeros_like(arr_full)
            for y in range(arr_full.shape[0]):
                for x in range(arr_full.shape[1]):
                    if tuple(arr_full[y, x]) in off_colors:
                        mask_off[y, x] = arr_full[y, x]
                    else:
                        mask_off[y, x] = (0, 0, 0)
            # 一時PNG作成
            tmp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            Image.fromarray(mask_off, mode='RGB').save(tmp_img.name)
            # 一時STL作成
            tmp_stl_off = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
            # determine wall_color tuple
            if isinstance(self.wall_color, QColor):
                wc = (self.wall_color.red(), self.wall_color.green(), self.wall_color.blue())
            else:
                wc = tuple(self.wall_color)
            mesh_off, _ = generate_dot_plate_stl(
                tmp_img.name, tmp_stl_off.name,
                int(params.get("Grid Size", 0)), float(params.get("Dot Size", 0.0)),
                float(params.get("Wall Thickness", 0.0)), float(params.get("Wall Height", 0.0)),
                float(params.get("Base Height", 0.0)),
                1, 1000,
                float(params.get("Out Thickness", 0.0)),
                wall_color=wc,
                merge_same_color=True,
                return_colors=True
            )
            meshes.append(mesh_off)
            # クリーンアップオフ
            tmp_img.close(); os.unlink(tmp_img.name)
            tmp_stl_off.close(); os.unlink(tmp_stl_off.name)
            # 色レイヤー部 (ON色) を生成 (壁厚0)
            if on_colors:
                mask_on = np.zeros_like(arr_full)
                for y in range(arr_full.shape[0]):
                    for x in range(arr_full.shape[1]):
                        if tuple(arr_full[y, x]) in on_colors:
                            mask_on[y, x] = arr_full[y, x]
                        else:
                            mask_on[y, x] = (0, 0, 0)
                # 一時STLレイヤー生成
                tmp_stl_on = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
                # 高さパラメータの抽出
                heights = {c: self.layer_heights.get(c, 0.0) for c in on_colors}
                mesh_on = generate_layered_stl(
                    mask_on,
                    tmp_stl_on.name,
                    int(params.get("Grid Size", 0)), float(params.get("Dot Size", 0.0)),
                    float(params.get("Base Height", 0.0)),
                    0.0, float(params.get("Wall Height", 0.0)),
                    heights, on_colors
                )
                meshes.append(mesh_on)
                tmp_stl_on.close(); os.unlink(tmp_stl_on.name)
            # マージして出力
            mesh = trimesh.util.concatenate(meshes)
            mesh.export(out_path)
            self.show_stl_preview(mesh)
            self.input_label.setText(f"{out_path} にハイブリッドSTLをエクスポートしました")
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "STLを保存", "dot_plate.stl", "STLファイル (*.stl)")
        if out_path:
            params = {key: spin.value() for key, spin in self.controls.items()}
            
            try:
                # STLファイル生成（時間がかかる可能性がある）
                self.input_label.setText("カラーSTLファイルを生成中...")
                QApplication.processEvents()  # UIを更新
                
                # 壁の色をRGBタプルに変換
                if isinstance(self.wall_color, QColor):
                    wall_color = (self.wall_color.red(), self.wall_color.green(), self.wall_color.blue())
                else:
                    # すでにタプルかリストの場合
                    wall_color = self.wall_color
                
                # カスタム編集されたピクセルデータがあるかチェック
                custom_pixels = self.pixels_rounded_np if hasattr(self, 'pixels_rounded_np') and self.pixels_rounded_np is not None else None
                
                # メッシュ生成（メッシュも返すように指定）
                if custom_pixels is not None:
                    # カスタムピクセルからSTLを直接生成
                    import tempfile
                    
                    # 一時ファイルに画像を保存
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        tmp_path = tmp.name
                        # カスタムピクセルデータから画像を作成
                        custom_img = Image.fromarray(custom_pixels, mode='RGB')
                        custom_img.save(tmp_path)
                    
                    # 同色内壁省略オプション (stl_mode == 1で有効)
                    merge_same_color = (getattr(self, 'stl_mode', 0) == 1)
                    
                    # 生成された一時画像を使用してSTLを生成
                    mesh = generate_dot_plate_stl(
                        tmp_path,  # 一時画像パス
                        out_path,
                        int(params["Grid Size"]),
                        float(params["Dot Size"]),
                        float(params["Wall Thickness"]),
                        float(params["Wall Height"]),
                        float(params["Base Height"]),
                        1,  # 色ステップは1（既に減色済み）
                        1000,  # 上位色制限は高く設定（全ての色を使用）
                        float(params["Out Thickness"]),
                        wall_color=wall_color,  # 選択した壁の色を使用
                        merge_same_color=merge_same_color,  # 同色間の内壁省略オプション
                        return_colors=True  # メッシュを返すように指定
                    )
                    
                    # 一時ファイルを削除
                    import os
                    os.unlink(tmp_path)
                else:
                    # 同色内壁省略オプション (stl_mode == 1で有効)
                    merge_same_color = (getattr(self, 'stl_mode', 0) == 1)
                    
                    # 選択されたアルゴリズムの情報を表示
                    algo_names = {
                        "simple": "単純量子化",
                        "median_cut": "メディアンカット法",
                        "kmeans": "K-means法",
                        "octree": "オクトツリー法"
                    }
                    algo_name = algo_names.get(self.current_color_algo, "単純量子化")
                    self.input_label.setText(f"減色アルゴリズム「{algo_name}」でSTLを生成中...")
                    QApplication.processEvents()  # UIを更新
                    
                    # 元の画像から新たにSTLを生成（減色アルゴリズムを指定）
                    if hasattr(self, "generate_dot_plate_stl_with_algorithm"):
                        # 将来的に実装する場合のコード
                        mesh = self.generate_dot_plate_stl_with_algorithm(
                            self.image_path,
                            out_path,
                            int(params["Grid Size"]),
                            float(params["Dot Size"]),
                            float(params["Wall Thickness"]),
                            float(params["Wall Height"]),
                            float(params["Base Height"]),
                            int(params["Color Step"]),
                            int(params["Top Colors"]),
                            float(params["Out Thickness"]),
                            wall_color=wall_color,
                            merge_same_color=merge_same_color,
                            return_colors=True,
                            color_algo=self.current_color_algo
                        )
                    else:
                        # 現状の実装（すでに減色済みの場合はカスタムピクセルを使用）
                        if self.pixels_rounded_np is not None:
                            # 減色済みデータから一時画像を作成してSTL生成
                            import tempfile
                            
                            # 一時ファイルに画像を保存
                            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                                tmp_path = tmp.name
                                # カスタムピクセルデータから画像を作成
                                custom_img = Image.fromarray(self.pixels_rounded_np, mode='RGB')
                                custom_img.save(tmp_path)
                                
                            mesh = generate_dot_plate_stl(
                                tmp_path,  # 一時画像パス
                                out_path,
                                int(params["Grid Size"]),
                                float(params["Dot Size"]),
                                float(params["Wall Thickness"]),
                                float(params["Wall Height"]),
                                float(params["Base Height"]),
                                1,  # 色ステップは1（既に減色済み）
                                1000,  # 上位色制限は高く設定（全ての色を使用）
                                float(params["Out Thickness"]),
                                wall_color=wall_color,
                                merge_same_color=merge_same_color,
                                return_colors=True
                            )
                            
                            # 一時ファイルを削除
                            import os
                            os.unlink(tmp_path)
                        else:
                            # 通常の方法でSTL生成
                            mesh = generate_dot_plate_stl(
                                self.image_path,
                                out_path,
                                int(params["Grid Size"]),
                                float(params["Dot Size"]),
                                float(params["Wall Thickness"]),
                                float(params["Wall Height"]),
                                float(params["Base Height"]),
                                int(params["Color Step"]),
                                int(params["Top Colors"]),
                                float(params["Out Thickness"]),
                                wall_color=wall_color,
                                merge_same_color=merge_same_color,
                                return_colors=True
                            )
                
                # メッシュオブジェクトを取得
                if isinstance(mesh, tuple) and len(mesh) > 0:
                    # return_colors=Trueの場合、最初の要素がメッシュ
                    preview_mesh = mesh[0]
                else:
                    # 単一のメッシュオブジェクトの場合
                    preview_mesh = mesh
                
                # STLプレビューを表示
                self.show_stl_preview(preview_mesh)
                
                # HTMLレポートを生成
                html_path = self.generate_html_report(out_path, preview_mesh)
                
                # 壁の色の文字列表現を作成
                if isinstance(self.wall_color, QColor):
                    color_name = f"RGB({self.wall_color.red()}, {self.wall_color.green()}, {self.wall_color.blue()})"
                else:
                    # タプルやリストの場合
                    color_name = f"RGB{self.wall_color}"
                if html_path:
                    self.input_label.setText(f"{out_path} にカラーSTL（壁の色：{color_name}）をエクスポートし、HTMLレポート {html_path} も生成しました")
                else:
                    self.input_label.setText(f"{out_path} にカラーSTL（壁の色：{color_name}）をエクスポートしました")
                
            except Exception as e:
                print(f"STL生成エラー: {str(e)}")
                import traceback
                traceback.print_exc()
                self.input_label.setText(f"STL生成エラー: {str(e)}")

    def export_preview_image(self):
        """プレビュー中のドット絵を透過背景込みでSTLと同じサイズの画像として保存"""
        # 必要条件チェック
        if not hasattr(self, 'pixels_rounded_np') or self.pixels_rounded_np is None:
            QMessageBox.warning(self, "プレビュー画像保存エラー", "先にプレビューを生成してください。")
            return
        # 保存先選択
        path, _ = QFileDialog.getSaveFileName(self, "プレビュー画像を保存", "preview.png", "PNG画像 (*.png)")
        if not path:
            return
        # DPI入力
        dpi, ok = QInputDialog.getInt(self, "画像解像度(DPI)", "DPIを入力してください:", 300, 1, 2400)
        if not ok:
            return
        # パラメータ取得
        params = {key: spin.value() for key, spin in self.controls.items()}
        grid_size = self.current_grid_size
        dot_size_mm = float(params.get("Dot Size", 1.0))
        # 1インチ=25.4mmとしてピクセル数を計算
        px_per_mm = dpi / 25.4
        cell_px = max(1, int(round(dot_size_mm * px_per_mm)))
        # プレビュー画像生成 (checkerboard背景込み)
        try:
            img = generate_preview_image(
                self.image_path,
                grid_size,
                int(params.get("Color Step", 1)),
                int(params.get("Top Colors", 256)),
                zoom_factor=cell_px,
                custom_pixels=self.pixels_rounded_np
            )
            # 保存
            # 保存時にDPIを埋め込む
            img.save(path, dpi=(dpi, dpi))
            self.input_label.setText(f"{path} にプレビュー画像を保存しました")
        except Exception as e:
            QMessageBox.critical(self, "プレビュー画像保存エラー", f"画像保存中にエラーが発生しました: {e}")
        
    def export_mrpaf(self):
        """プレビュー中のピクセルデータをMRPAF形式で保存"""
        # プレビューが生成されているかチェック
        if not hasattr(self, 'pixels_rounded_np') or self.pixels_rounded_np is None:
            QMessageBox.warning(self, "MRPAF保存エラー", "先にプレビューを生成してください。")
            return
        # 保存先ファイル選択
        path, _ = QFileDialog.getSaveFileName(self, "MRPAFを保存", "image.mrpaf", "MRPAFファイル (*.mrpaf *.json)")
        if not path:
            return
        # データ準備
        import datetime
        grid_h, grid_w, _ = self.pixels_rounded_np.shape
        dot_size = float(self.controls.get("Dot Size", QDoubleSpinBox()).value())
        # 透過色
        tc = (self.transparent_color.red(), self.transparent_color.green(), self.transparent_color.blue())
        # パレット作成
        flat = self.pixels_rounded_np.reshape(-1, 3)
        seen = []
        for pix in map(tuple, flat):
            if pix not in seen:
                seen.append(pix)
        palette = []
        index_map = {}
        for i, pix in enumerate(seen):
            # numpy.uint8 型を Python int に変換
            r_i, g_i, b_i = int(pix[0]), int(pix[1]), int(pix[2])
            a_i = 0 if pix == tc else 255
            hexcode = f"#{r_i:02X}{g_i:02X}{b_i:02X}{a_i:02X}"
            palette.append({"id": i, "hex": hexcode, "rgb": [r_i, g_i, b_i, a_i]})
            index_map[pix] = i
        # ピクセルインデックス化
        data = []
        for y in range(grid_h):
            row = []
            for x in range(grid_w):
                pix = tuple(int(c) for c in self.pixels_rounded_np[y, x])
                row.append(index_map.get(pix, 0))
            data.append(row)
        # JSON組み立て
        mrpaf = {
            "format": "MRPAF",
            "version": "1.1",
            "metadata": {
                "tool": {"name": "DotPlateGenerator", "version": "1.0"},
                "created": datetime.datetime.utcnow().isoformat() + 'Z'
            },
            "canvas": {
                "baseWidth": grid_w,
                "baseHeight": grid_h,
                "pixelUnit": dot_size,
                "backgroundColor": f"#{tc[0]:02X}{tc[1]:02X}{tc[2]:02X}00"
            },
            "palette": palette,
            "layers": [
                {
                    "id": 0,
                    "name": "PreviewLayer",
                    "type": "raster",
                    "visible": True,
                    "locked": False,
                    "opacity": 1.0,
                    "blending": {"mode": "normal", "resolution": "target", "interpolation": "nearest"},
                    "resolution": {"pixelArraySize": {"width": grid_w, "height": grid_h},
                                     "scale": 1, "effectiveSize": {"width": grid_w, "height": grid_h}},
                    "placement": {"x": 0, "y": 0, "width": grid_w, "height": grid_h,
                                  "unit": "base", "allowSubPixel": False},
                    "viewport": {"x": 0, "y": 0, "width": grid_w, "height": grid_h},
                    "pixels": {"encoding": "array", "data": data}
                }
            ],
            "animations": {},
            "resources": {}
        }
        # 保存
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(mrpaf, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "保存完了", f"{path} にMRPAFファイルを保存しました。")
        except Exception as e:
            QMessageBox.critical(self, "MRPAF保存エラー", f"保存中にエラーが発生しました: {e}")
    def export_svg(self):
        """Fusion360用SVGをエクスポート"""
        if not hasattr(self, 'pixels_rounded_np') or self.pixels_rounded_np is None:
            QMessageBox.warning(self, "SVG出力エラー", "先にプレビューを生成してください。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "SVGを保存", "output.svg", "SVGファイル (*.svg)")
        if not path:
            return
        import numpy as np
        pixels = self.pixels_rounded_np
        # ドットサイズ（mm）
        try:
            dot_size = float(self.controls["Dot Size"].value())
        except Exception:
            dot_size = 1.0
        h, w, _ = pixels.shape
        # 透過色判定用
        tc = (self.transparent_color.red(), self.transparent_color.green(), self.transparent_color.blue())
        # SVGヘッダー
        svg = []
        svg.append('<?xml version="1.0" encoding="UTF-8"?>')
        svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w * dot_size}mm" height="{h * dot_size}mm" viewBox="0 0 {w * dot_size} {h * dot_size}">')
        # 出力モード判定: 1=同色内壁省略 (領域結合)、それ以外はピクセル毎矩形
        mode = getattr(self, 'stl_mode', 0)
        if mode == 1:
            # 同色領域ごとに外周輪郭をSVGパスとして出力
            import cv2
            for color in getattr(self, 'layer_color_order', []):
                if color == tc:
                    continue
                mask = (np.all(pixels == color, axis=2).astype(np.uint8) * 255)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                hexcol = f'#{color[0]:02X}{color[1]:02X}{color[2]:02X}'
                for cnt in contours:
                    pts = cnt.reshape(-1, 2)
                    path_coords = []
                    for x, y in pts:
                        path_coords.append(f"{x * dot_size:.3f},{y * dot_size:.3f}")
                    # パスを閉じる
                    d = 'M ' + ' L '.join(path_coords) + ' Z'
                    svg.append(f'  <path d="{d}" fill="{hexcol}" stroke="none"/>')
        else:
            # 各ピクセルを矩形として出力
            for y in range(h):
                for x in range(w):
                    pix = tuple(int(c) for c in pixels[y, x])
                    if pix == tc:
                        continue
                    hexcol = f'#{pix[0]:02X}{pix[1]:02X}{pix[2]:02X}'
                    x0 = x * dot_size
                    y0 = y * dot_size
                    svg.append(f'  <rect x="{x0:.3f}" y="{y0:.3f}" width="{dot_size:.3f}" height="{dot_size:.3f}" fill="{hexcol}" stroke="none"/>')
        svg.append('</svg>')
        # ファイルへ書き出し
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write("\n".join(svg))
            QMessageBox.information(self, "SVG出力完了", f"{path} にSVGを保存しました")
        except Exception as e:
            QMessageBox.critical(self, "SVG保存エラー", f"保存中にエラーが発生しました: {e}")
    def show_stl_preview(self, mesh):
        """メインウィンドウにSTLプレビューを表示し、別スレッドで画像も保存"""
        try:
            # VEDO使用可能ならvedoで描画、なければmatplotlibにフォールバック
            if VEDO_AVAILABLE:
                # Vedoを使用したプレビュー生成
                self._show_stl_preview_vedo(mesh)
            else:
                # MatplotlibでのプレビューにフォールバックAgg
                self._show_stl_preview_matplotlib(mesh)
            
            # STL情報を表示
            self.update_stl_info(mesh)
                
            # 別スレッドで画像を保存
            self.input_label.setText(f"{self.input_label.text()} STLプレビュー画像を保存中...")
            QApplication.processEvents()  # UIを更新
                
            # 別スレッドで画像保存
            save_thread = threading.Thread(
                target=self.save_front_view_image, 
                args=(mesh,)
            )
            save_thread.daemon = True  # メインスレッド終了時にこのスレッドも終了
            save_thread.start()
            
        except Exception as e:
            print(f"STLプレビュー表示エラー: {str(e)}")
            if hasattr(self, 'stl_preview_label'):
                self.stl_preview_label.setText(f"STLプレビュー表示失敗: {str(e)}")
            else:
                print(f"stl_preview_label属性が見つかりません: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def update_stl_info(self, mesh):
        """STLの情報を計算して表示する"""
        try:
            # メッシュが存在しない場合は終了
            if mesh is None:
                self.stl_info_label.setText("STLデータがありません")
                return
                
            # STLのサイズ情報を取得
            bounds = mesh.bounds
            min_bounds = bounds[0]
            max_bounds = bounds[1]
            
            width = max_bounds[0] - min_bounds[0]  # X方向の幅
            depth = max_bounds[1] - min_bounds[1]  # Y方向の深さ
            height = max_bounds[2] - min_bounds[2]  # Z方向の高さ
            
            # ドット数の計算（グリッドサイズから）
            params = {key: spin.value() for key, spin in self.controls.items()}
            grid_size = int(params["Grid Size"])
            total_dots = 0
            color_counts = {}
            total_volume = 0
            
            # 現在のピクセルデータがあれば、それを使ってドット数と色の分布を計算
            if hasattr(self, 'pixels_rounded_np') and self.pixels_rounded_np is not None:
                # 透明でないピクセルをカウント (RGB(0,0,0)は透明として扱う)
                non_transparent_mask = (self.pixels_rounded_np != 0).any(axis=2)
                total_dots = np.sum(non_transparent_mask)
                
                # 色ごとのドット数をカウント
                colors = [tuple(pixel) for pixel in self.pixels_rounded_np.reshape(-1, 3) 
                         if tuple(pixel) != (0, 0, 0)]
                color_counts = Counter(colors)
                
                # 色の数
                unique_colors = len(color_counts)
                
                # 体積計算（ドットのサイズと各色のドット数から概算）
                dot_size = float(params["Dot Size"])
                base_height = float(params["Base Height"])
                wall_height = float(params["Wall Height"])
                dot_height = wall_height + base_height  # mm
                dot_volume = dot_size * dot_size * dot_height  # mm³
                
                # 総体積
                total_volume = total_dots * dot_volume
                
                # 色ごとの体積を計算
                color_volumes = {}
                for color, count in color_counts.items():
                    color_volumes[color] = count * dot_volume
            else:
                # ピクセルデータがない場合は推定
                total_dots = "不明"
                unique_colors = "不明"
                color_counts = {}
                color_volumes = {}
                
            # 情報テキストの組み立て
            info_html = f"""
            <html>
            <body>
            <style>
                body {{ margin: 0; padding: 0; width: 100%; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 8px; table-layout: fixed; }}
                th, td {{ padding: 5px; text-align: left; border: 1px solid #ddd; overflow: hidden; }}
                th {{ background-color: #f2f2f2; }}
                .color-cell {{ width: 20px; height: 20px; display: inline-block; border: 1px solid #ccc; border-radius: 3px; }}
                .color-column {{ width: 10%; }}
                .rgb-column {{ width: 30%; }}
                .count-column {{ width: 30%; }}
                .volume-column {{ width: 30%; }}
            </style>
            <table>
                <tr><th colspan="2">STL情報</th></tr>
                <tr><td>最大幅 (X):</td><td>{width:.2f} mm</td></tr>
                <tr><td>最大奥行き (Y):</td><td>{depth:.2f} mm</td></tr>
                <tr><td>最大高さ (Z):</td><td>{height:.2f} mm</td></tr>
                <tr><td>ドット数:</td><td>{total_dots}</td></tr>
                <tr><td>使用色数:</td><td>{unique_colors}</td></tr>
                <tr><td>総推定体積:</td><td>{total_volume:.2f} mm³</td></tr>
            </table>
            <br/>
            """
            
            # 色ごとの詳細テーブルを追加
            if color_counts:
                # 色を使用頻度順にソート
                sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
                
                # 色ごとの詳細テーブル
                info_html += """
                <table>
                    <tr>
                        <th class="color-column">色</th>
                        <th class="rgb-column">RGB</th>
                        <th class="count-column">ドット数</th>
                        <th class="volume-column">推定体積 (mm³)</th>
                    </tr>
                """
                
                # 各色の行を追加
                for color, count in sorted_colors:
                    r, g, b = color
                    volume = color_volumes[color]
                    hex_color = f"#{r:02x}{g:02x}{b:02x}"
                    
                    # 色情報とともにURLリンクを埋め込み（クリック可能にする）
                    color_id = f"color_{r}_{g}_{b}"  # リンク識別用のID
                    info_html += f"""
                    <tr>
                        <td class="color-column">
                            <a href="color://{r},{g},{b}" title="この色を持つドットをハイライト表示">
                                <div class="color-cell" style="background-color: {hex_color};"></div>
                            </a>
                        </td>
                        <td class="rgb-column">
                            <a href="color://{r},{g},{b}" title="この色を持つドットをハイライト表示">
                                ({r}, {g}, {b})
                            </a>
                        </td>
                        <td class="count-column">{count}</td>
                        <td class="volume-column">{volume:.2f}</td>
                    </tr>
                    """
                
                info_html += """
                </table>
                """
            
            info_html += """
            </body>
            </html>
            """
            
            # 情報表示を更新
            self.stl_info_label.setText(info_html)
            
        except Exception as e:
            print(f"STL情報更新エラー: {str(e)}")
            self.stl_info_label.setText(f"情報取得エラー: {str(e)}")
    
    def _show_stl_preview_vedo(self, mesh):
        """Vedoを使用したSTLプレビュー生成"""
        # 一時的なSTLファイルを作成してvedo用にメッシュを準備
        temp_stl_path = f"temp_preview_{int(time.time())}.stl"
        mesh.export(temp_stl_path)
        
        try:
            # Vedoのオフスクリーンレンダリング設定
            vedo.settings.useOffScreen = True
            
            # メッシュを読み込み
            vmesh = vedo.Mesh(temp_stl_path)
            
            # メッシュの中心と大きさを取得
            center = vmesh.center_of_mass()
            bounds = vmesh.bounds()
            max_length = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
            z_pos = bounds[5] + max_length * 2  # モデルの最大Z値より十分高い位置
            
            # プレビュー用のプロット設定
            plt = vedo.Plotter(offscreen=True, size=(600, 600))
            plt.add(vmesh)
            
            # カメラをZ軸正方向から真上に配置（Z軸真正面から見る）
            cam = plt.camera
            # 完全に真上からの視点に設定
            cam.SetPosition(center[0], center[1], z_pos)
            cam.SetFocalPoint(center[0], center[1], center[2])
            cam.SetViewUp(1, 0, 0)  # X軸正方向が上になるよう設定（XY平面で180度回転）
            
            # 背景色を白にし、軸を非表示に
            plt.background('white')
            plt.axes(False)
            
            # 画像として保存
            img_path = f"temp_preview_img_{int(time.time())}.png"
            plt.screenshot(img_path)
            plt.close()
            
            # 画像を読み込んでプレビューに表示
            pixmap = QPixmap(img_path)
            
            # 画像を正方形にトリミング（1:1の比率を確保）
            size = min(pixmap.width(), pixmap.height())
            square_pixmap = pixmap.copy(
                (pixmap.width() - size) // 2,
                (pixmap.height() - size) // 2,
                size, size
            )
            
            self.stl_preview_label.setPixmap(square_pixmap)
            self.stl_preview_label.setScaledContents(True)
            
            # 一時ファイルを削除
            os.remove(temp_stl_path)
            os.remove(img_path)
            
        except Exception as e:
            print(f"Vedoプレビューエラー: {str(e)}")
            # 一時ファイルの削除を試行
            if os.path.exists(temp_stl_path):
                os.remove(temp_stl_path)
            # エラー時はMatplotlibにフォールバック
            self._show_stl_preview_matplotlib(mesh)
    
    def _show_stl_preview_matplotlib(self, mesh):
        """MatplotlibでのSTLプレビュー生成（フォールバック用）"""
        # MatplotlibでのAggバックエンド使用（スレッドセーフ）
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
            
        # UIプレビュー用の画像生成（上面斜めからのビュー）
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # メッシュの中心と大きさを取得して最適な視点を設定
        center = mesh.center_mass
        min_bounds = mesh.bounds[0]
        max_bounds = mesh.bounds[1]
        
        # Z軸正方向から真上に見る角度に設定
        ax.view_init(elev=90, azim=270)  # 真上から見て、XY平面で180度回転した状態（azimuthを270度に）
        
        # メッシュを表示 (trimesh.Trimesh.show()はmatplotlibのax引数を受け付けない問題の修正)
        # trimeshのvisuals.plotterでマニュアルで描画
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # メッシュの頂点と面を取得
        verts = mesh.vertices
        faces = mesh.faces
        
        # 頂点をプロット
        ax.scatter3D(verts[:, 0], verts[:, 1], verts[:, 2], c='k', s=0.1)
        
        # 面をプロット
        mesh_collection = Poly3DCollection([verts[face] for face in faces], 
                                          alpha=1.0, 
                                          linewidths=0.1, 
                                          edgecolors='k')
        
        # 面の色を設定
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'face_colors'):
            face_colors = mesh.visual.face_colors
            rgba_colors = face_colors / 255.0  # 0-1の範囲に正規化
            mesh_collection.set_facecolors(rgba_colors)
        else:
            mesh_collection.set_facecolors((0.8, 0.8, 0.8))
            
        ax.add_collection3d(mesh_collection)
        
        # 軸の範囲を設定
        all_verts = verts.reshape(-1, 3)
        min_x, max_x = all_verts[:, 0].min(), all_verts[:, 0].max()
        min_y, max_y = all_verts[:, 1].min(), all_verts[:, 1].max()
        min_z, max_z = all_verts[:, 2].min(), all_verts[:, 2].max()
        
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_zlim(min_z, max_z)
        
        ax.set_axis_off()
        plt.tight_layout()
        
        # 画像として保存
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)  # 必ずfigを閉じる
        buf.seek(0)
        
        # QPixmapとして読み込み
        qimg = QImage()
        qimg.loadFromData(buf.getvalue())
        pixmap = QPixmap.fromImage(qimg)
        
        # 画像を正方形にトリミング（1:1の比率を確保）
        size = min(pixmap.width(), pixmap.height())
        square_pixmap = pixmap.copy(
            (pixmap.width() - size) // 2,
            (pixmap.height() - size) // 2,
            size, size
        )
        
        # プレビューラベルに表示
        self.stl_preview_label.setPixmap(square_pixmap)
        self.stl_preview_label.setScaledContents(True)
    
    def save_front_view_image(self, mesh):
        """別スレッドで正面からの画像と上面からの画像を保存"""
        try:
            timestamp = int(time.time())
            top_filename = f"stl_top_view_{timestamp}.png"
            top_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), top_filename)
            
            if VEDO_AVAILABLE:
                # Vedoを使って上面からの画像を保存
                self._save_top_view_vedo(mesh, top_save_path, top_filename)
            else:
                # Matplotlibで上面からの画像を保存
                self._save_top_view_matplotlib(mesh, top_save_path, top_filename)
            
        except Exception as e:
            print(f"画像保存エラー: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # エラー通知
            from PyQt5.QtCore import QEvent
            
            class ImageSaveErrorEvent(QEvent):
                def __init__(self, error_msg):
                    super().__init__(QEvent.Type(QEvent.User + 11))
                    self.error_msg = error_msg
            
            QApplication.instance().postEvent(self, ImageSaveErrorEvent(str(e)))
    
    def _save_front_view_vedo(self, mesh, save_path, filename):
        """Vedoを使った正面からの画像保存"""
        # 一時的なSTLファイルを作成
        temp_stl_path = f"temp_front_{int(time.time())}.stl"
        mesh.export(temp_stl_path)
        
        try:
            # Vedoのオフスクリーンレンダリング設定
            vedo.settings.useOffScreen = True
            
            # メッシュを読み込み
            vmesh = vedo.Mesh(temp_stl_path)
            
            # 正面からの視点に設定
            plt = vedo.Plotter(offscreen=True, size=(800, 800))
            plt.add(vmesh)
            plt.camera.elevation(0)
            plt.camera.azimuth(0)
            
            # 背景色を白にし、軸を非表示に
            plt.background('white')
            plt.axes(False)
            
            # 画像として保存（高解像度）
            plt.screenshot(save_path, scale=2)
            plt.close()
            
            # 一時ファイルを削除
            if os.path.exists(temp_stl_path):
                os.remove(temp_stl_path)
            
            # 完了通知をGUIスレッドに送信
            from PyQt5.QtCore import QEvent
            
            class ImageSavedEvent(QEvent):
                def __init__(self, filename):
                    super().__init__(QEvent.Type(QEvent.User + 10))
                    self.filename = filename
            
            QApplication.instance().postEvent(self, ImageSavedEvent(filename))
            
        except Exception as e:
            # エラー時はMatplotlibにフォールバック
            print(f"Vedo画像保存エラー: {str(e)}, Matplotlibにフォールバックします")
            # 一時ファイルの削除を試行
            if os.path.exists(temp_stl_path):
                os.remove(temp_stl_path)
            self._save_front_view_matplotlib(mesh, save_path, filename)
            
    def _save_top_view_vedo(self, mesh, save_path, filename):
        """Vedoを使った上面（Z軸上から）の画像保存"""
        # 一時的なSTLファイルを作成
        temp_stl_path = f"temp_top_{int(time.time())}.stl"
        mesh.export(temp_stl_path)
        
        try:
            # Vedoのオフスクリーンレンダリング設定
            vedo.settings.useOffScreen = True
            
            # メッシュを読み込み
            vmesh = vedo.Mesh(temp_stl_path)
            
            # メッシュの中心と大きさを取得
            center = vmesh.center_of_mass()
            bounds = vmesh.bounds()
            max_length = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
            z_pos = bounds[5] + max_length * 2  # モデルの最大Z値より十分高い位置
            
            # 上面からの視点に設定 (Z軸正方向から原点を見る)
            plt = vedo.Plotter(offscreen=True, size=(800, 800))
            plt.add(vmesh)
            
            # カメラをZ軸正方向に配置し、メッシュの中心を見るよう設定
            cam = plt.camera
            cam.SetPosition(center[0], center[1], z_pos)
            cam.SetFocalPoint(center[0], center[1], center[2])
            cam.SetViewUp(-1, 0, 0)  # X軸負方向が上になるよう設定（反時計回りに90度回転）
            
            # 背景色を白にし、軸を非表示に
            plt.background('white')
            plt.axes(False)
            
            # 画像として保存（高解像度）
            plt.screenshot(save_path, scale=2)
            plt.close()
            
            # 一時ファイルを削除
            if os.path.exists(temp_stl_path):
                os.remove(temp_stl_path)
            
            # 完了通知をGUIスレッドに送信
            from PyQt5.QtCore import QEvent
            
            class ImageSavedEvent(QEvent):
                def __init__(self, filename):
                    super().__init__(QEvent.Type(QEvent.User + 10))
                    self.filename = filename
            
            QApplication.instance().postEvent(self, ImageSavedEvent(filename))
            
        except Exception as e:
            # エラー時はMatplotlibにフォールバック
            print(f"Vedo上面画像保存エラー: {str(e)}, Matplotlibにフォールバックします")
            # 一時ファイルの削除を試行
            if os.path.exists(temp_stl_path):
                os.remove(temp_stl_path)
            self._save_top_view_matplotlib(mesh, save_path, filename)
    
    def _save_front_view_matplotlib(self, mesh, save_path, filename):
        """Matplotlibでの正面からの画像保存（フォールバック用）"""
        # MatplotlibでのAggバックエンド使用（スレッドセーフ）
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # 正面からのビュー生成
        front_fig = plt.figure(figsize=(8, 8))
        front_ax = front_fig.add_subplot(111, projection='3d')
        front_ax.view_init(elev=0, azim=0)  # 正面から
        
        # メッシュを表示 (trimesh.Trimesh.show()はmatplotlibのax引数を受け付けない問題の修正)
        # trimeshのvisuals.plotterでマニュアルで描画
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # メッシュの頂点と面を取得
        verts = mesh.vertices
        faces = mesh.faces
        
        # 頂点をプロット
        front_ax.scatter3D(verts[:, 0], verts[:, 1], verts[:, 2], c='k', s=0.1)
        
        # 面をプロット
        mesh_collection = Poly3DCollection([verts[face] for face in faces], 
                                          alpha=1.0, 
                                          linewidths=0.1, 
                                          edgecolors='k')
        
        # 面の色を設定
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'face_colors'):
            face_colors = mesh.visual.face_colors
            rgba_colors = face_colors / 255.0  # 0-1の範囲に正規化
            mesh_collection.set_facecolors(rgba_colors)
        else:
            mesh_collection.set_facecolors((0.8, 0.8, 0.8))
            
        front_ax.add_collection3d(mesh_collection)
        
        # 軸の範囲を設定
        all_verts = verts.reshape(-1, 3)
        min_x, max_x = all_verts[:, 0].min(), all_verts[:, 0].max()
        min_y, max_y = all_verts[:, 1].min(), all_verts[:, 1].max()
        min_z, max_z = all_verts[:, 2].min(), all_verts[:, 2].max()
        
        front_ax.set_xlim(min_x, max_x)
        front_ax.set_ylim(min_y, max_y)
        front_ax.set_zlim(min_z, max_z)
        
        front_ax.set_axis_off()
        plt.tight_layout()
        
        # 画像を保存
        plt.savefig(save_path, format='png', dpi=150)
        plt.close(front_fig)
        
        # 完了通知をGUIスレッドに送信
        from PyQt5.QtCore import QEvent
        
        class ImageSavedEvent(QEvent):
            def __init__(self, filename):
                super().__init__(QEvent.Type(QEvent.User + 10))
                self.filename = filename
        
        QApplication.instance().postEvent(self, ImageSavedEvent(filename))
    
    def _save_top_view_matplotlib(self, mesh, save_path, filename):
        """Matplotlibでの上面からの画像保存（Z軸上から見下ろす視点）"""
        # MatplotlibでのAggバックエンド使用（スレッドセーフ）
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # 上面からのビュー生成
        top_fig = plt.figure(figsize=(8, 8))
        top_ax = top_fig.add_subplot(111, projection='3d')
        
        # メッシュを表示 (trimesh.Trimesh.show()はmatplotlibのax引数を受け付けない問題の修正)
        # trimeshのvisuals.plotterでマニュアルで描画
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # メッシュの頂点と面を取得
        verts = mesh.vertices
        faces = mesh.faces
        
        # メッシュの中心と大きさを取得
        center = mesh.center_mass
        min_bounds = mesh.bounds[0]
        max_bounds = mesh.bounds[1]
        max_length = max(max_bounds[0] - min_bounds[0], 
                          max_bounds[1] - min_bounds[1], 
                          max_bounds[2] - min_bounds[2])
        
        # Z軸正方向からメッシュの中心を見るようにカメラを設定
        # matplotlibでは直接カメラ位置は設定できないので、視点角度と距離で調整
        top_ax.view_init(elev=90, azim=90)  # 真上から見下ろす角度、azimuth=90で反時計回りに90度回転
        
        # 頂点をプロット
        top_ax.scatter3D(verts[:, 0], verts[:, 1], verts[:, 2], c='k', s=0.1)
        
        # 面をプロット
        mesh_collection = Poly3DCollection([verts[face] for face in faces], 
                                          alpha=1.0, 
                                          linewidths=0.1, 
                                          edgecolors='k')
        
        # 面の色を設定
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'face_colors'):
            face_colors = mesh.visual.face_colors
            rgba_colors = face_colors / 255.0  # 0-1の範囲に正規化
            mesh_collection.set_facecolors(rgba_colors)
        else:
            mesh_collection.set_facecolors((0.8, 0.8, 0.8))
            
        top_ax.add_collection3d(mesh_collection)
        
        # 軸の範囲を設定
        all_verts = verts.reshape(-1, 3)
        min_x, max_x = all_verts[:, 0].min(), all_verts[:, 0].max()
        min_y, max_y = all_verts[:, 1].min(), all_verts[:, 1].max()
        min_z, max_z = all_verts[:, 2].min(), all_verts[:, 2].max()
        
        # 視点調整のため、Z軸の範囲を広げる
        extra_z = max_length * 1.5
        top_ax.set_xlim(min_x, max_x)
        top_ax.set_ylim(min_y, max_y)
        top_ax.set_zlim(min_z, max_z + extra_z)  # 上方向に余裕を持たせる
        
        # カメラ位置をZ軸正方向に設定（matplotlibでは間接的に）
        top_ax.dist = 8  # カメラと対象物の距離
        
        top_ax.set_axis_off()
        plt.tight_layout()
        
        # 画像を保存
        plt.savefig(save_path, format='png', dpi=150)
        plt.close(top_fig)
        
        # 完了通知をGUIスレッドに送信
        from PyQt5.QtCore import QEvent
        
        class ImageSavedEvent(QEvent):
            def __init__(self, filename):
                super().__init__(QEvent.Type(QEvent.User + 10))
                self.filename = filename
        
        QApplication.instance().postEvent(self, ImageSavedEvent(filename))

# -------------------------------
# 実行エントリポイント
# -------------------------------
    def export_fusion360_script(self):
        """Fusion360用スケッチ作成スクリプトを出力"""
        # プレビューが生成されているかチェック
        if not hasattr(self, 'pixels_rounded_np') or self.pixels_rounded_np is None:
            QMessageBox.warning(self, "スクリプト出力エラー", "先にプレビューを生成してください。")
            return
        # 保存先選択
        path, _ = QFileDialog.getSaveFileName(self, "Fusion360スクリプトを保存", "fusion360_script.py", "Pythonファイル (*.py)")
        if not path:
            return
        # グリッドサイズ取得
        try:
            grid_size = self.controls["Grid Size"].value()
        except Exception:
            grid_size = getattr(self, 'current_grid_size', 1)
        import cv2
        import numpy as np
        layers = []
        # 各レイヤー色に対して輪郭抽出
        for color in getattr(self, 'layer_color_order', []):
            mask = (np.all(self.pixels_rounded_np == color, axis=2).astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pts_list = []
            for cnt in contours:
                pts = cnt.reshape(-1, 2)
                # スケール適用
                scaled = [(float(x) * grid_size, float(y) * grid_size) for x, y in pts]
                if len(scaled) >= 2:
                    pts_list.append(scaled)
            if pts_list:
                layers.append((color, pts_list))
        # スクリプト生成
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write("import adsk.core, adsk.fusion, adsk.cam, traceback\n\n")
                f.write("def run(context):\n")
                f.write("    ui = None\n")
                f.write("    try:\n")
                f.write("        app = adsk.core.Application.get()\n")
                f.write("        ui = app.userInterface\n")
                f.write("        design = adsk.fusion.Design.cast(app.activeProduct)\n")
                f.write("        root = design.rootComponent\n")
                f.write("        sketches = root.sketches\n")
                f.write("        plane = root.xYConstructionPlane\n")
                for idx, (color, contours) in enumerate(layers):
                    f.write(f"        # Layer {idx}: color RGB{color}\n")
                    f.write("        sk = sketches.add(plane)\n")
                    f.write("        lines = sk.sketchCurves.sketchLines\n")
                    for pts in contours:
                        for i in range(len(pts)):
                            x1, y1 = pts[i]
                            x2, y2 = pts[(i+1) % len(pts)]
                            f.write(f"        lines.addByTwoPoints(adsk.core.Point3D.create({x1}, {y1}, 0), adsk.core.Point3D.create({x2}, {y2}, 0))\n")
                f.write("    except:\n")
                f.write("        if ui:\n")
                f.write("            ui.messageBox('Fusion360スクリプト実行中にエラーが発生しました')\n")
        except Exception as e:
            QMessageBox.critical(self, "スクリプト出力エラー", f"ファイル書き込み中にエラーが発生しました: {e}")
            return
        QMessageBox.information(self, "完了", f"Fusion360スクリプトを {path} に保存しました")
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DotPlateApp()
    window.show()
    sys.exit(app.exec_())