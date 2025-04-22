# dot_plate_generator_gui.py
# 必要ライブラリ: PyQt5, PIL, numpy, trimesh, shapely, skimage, scipy, matplotlib

import sys
import os
import json
import pickle
import base64
import numpy as np
from PIL import Image
from collections import Counter
from scipy.spatial import distance
import trimesh
from trimesh.creation import box
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QScrollArea,
    QListWidget, QListWidgetItem, QVBoxLayout, QHBoxLayout, QSlider, QSpinBox,
    QGridLayout, QDoubleSpinBox, QToolButton, QDialog, QGroupBox, QFrame,
    QSizePolicy, QToolTip, QMainWindow, QColorDialog, QCheckBox, QComboBox,
    QMenu, QAction, QMenuBar, QRubberBand, QAbstractItemView
)
# 以下のウィジェットを追加インポート（APIキーダイアログ・メッセージボックス用）
from PyQt5.QtWidgets import QMessageBox, QInputDialog, QLineEdit
from PyQt5.QtCore import Qt, QSize, QTimer, QPoint, QSettings, QEvent, QRect
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QPen, QCursor, QIcon
from shapely.geometry import Polygon
from skimage import measure
from scipy.ndimage import binary_fill_holes
from io import BytesIO
import threading
import openai  # OpenAI API for AIブラシ機能
import ast
import time
import tempfile

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
    # 型チェックと値チェック
    if not isinstance(grid_size, int) or grid_size <= 0:
        raise ValueError("grid_size must be a positive integer")
    
    if custom_pixels is not None:
        # カスタムピクセルデータが提供されている場合、それを使用
        # 型チェック: カスタムピクセルがnumpy配列で、適切な形状か確認
        if not isinstance(custom_pixels, np.ndarray) or custom_pixels.ndim != 3 or custom_pixels.shape[2] != 3:
            raise ValueError("custom_pixels must be a 3D numpy array with shape (height, width, 3)")
        pixels_array = custom_pixels
    else:
        # 画像からピクセルデータを生成
        img = Image.open(image_path).convert("RGB")
        img_resized = img.resize((grid_size, grid_size), resample=Image.NEAREST)
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
        pixels_array = np.array(pixels_rounded, dtype=np.uint8).reshape((grid_size, grid_size, 3))
    
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
    checkerboard = Image.new('RGBA', (grid_size * zoom_factor, grid_size * zoom_factor), (255, 255, 255, 255))
    pattern = Image.new('RGBA', (zoom_factor * 2, zoom_factor * 2), (255, 255, 255, 0))
    draw = ImageDraw.Draw(pattern)
    draw.rectangle((0, 0, zoom_factor, zoom_factor), fill=(200, 200, 200, 255))
    draw.rectangle((zoom_factor, zoom_factor, zoom_factor * 2, zoom_factor * 2), fill=(200, 200, 200, 255))
    
    # 市松模様パターンを繰り返し配置
    for y in range(0, grid_size * zoom_factor, zoom_factor * 2):
        for x in range(0, grid_size * zoom_factor, zoom_factor * 2):
            checkerboard.paste(pattern, (x, y), pattern)
    
    # 拡大したプレビュー画像
    img_preview = img_preview.resize((grid_size * zoom_factor, grid_size * zoom_factor), resample=Image.NEAREST)
    
    # 市松模様の背景と合成
    result = Image.alpha_composite(checkerboard, img_preview)
    
    # 共通の枠線描画関数
    def draw_grid_highlight(grid_pos, color, width_factor=10):
        grid_x, grid_y = grid_pos
        # 有効なグリッド位置かチェック
        if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
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
        for y in range(grid_size):
            for x in range(grid_size):
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

def generate_dot_plate_stl(image_path, output_path, grid_size, dot_size,
                           wall_thickness, wall_height, base_height,
                           color_step, top_color_limit, out_thickness=0.1, 
                           wall_color=(255, 255, 255), # 壁の色（デフォルトは白）
                           merge_same_color=False,     # 同じ色のドット間の内壁を省略するオプション
                           return_colors=False):
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
    # Create base blocks for non-transparent pixels (exclude transparent color)
    blocks = []
    cumulative_z = base_height
    transparent_color = (0, 0, 0)
    # Base layer: for each pixel not transparent, add a block of base_height
    for y in range(grid_size):
        for x in range(grid_size):
            if tuple(pixels_rounded_np[y, x]) != transparent_color:
                x0 = x * dot_size
                y0 = (grid_size - 1 - y) * dot_size
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
        mask_support = np.zeros((grid_size, grid_size), dtype=bool)
        for sc in support_colors:
            sc_arr = np.array(sc, dtype=np.uint8)
            mask_support |= np.all(pixels_rounded_np == sc_arr, axis=2)
        # Add support blocks
        for y, x in np.argwhere(mask_support):
            x0 = x * dot_size
            y0 = (grid_size - 1 - y) * dot_size
            block = box(extents=[dot_size, dot_size, h])
            block.apply_translation([x0 + dot_size/2, y0 + dot_size/2, z0 + h/2])
            blocks.append(block)
        # Add perimeter walls for this layer's actual color region (height = wall_height)
        color_arr = np.array(color, dtype=np.uint8)
        mask_color = np.all(pixels_rounded_np == color_arr, axis=2)
        wt = wall_thickness
        for y, x in np.argwhere(mask_color):
            x0 = x * dot_size
            y0 = (grid_size - 1 - y) * dot_size
            y_center = y0 + dot_size/2
            # Left wall
            if x == 0 or not mask_color[y, x-1]:
                w = box(extents=[wt, dot_size, wall_height])
                w.apply_translation([x0 - wt/2, y_center, z0 + wall_height/2])
                blocks.append(w)
            # Right wall
            if x == grid_size-1 or not mask_color[y, x+1]:
                w = box(extents=[wt, dot_size, wall_height])
                w.apply_translation([x0 + dot_size + wt/2, y_center, z0 + wall_height/2])
                blocks.append(w)
            # Top wall (positive Y direction)
            if y == 0 or not mask_color[y-1, x]:
                w = box(extents=[dot_size, wt, wall_height])
                w.apply_translation([x0 + dot_size/2, y0 + dot_size + wt/2, z0 + wall_height/2])
                blocks.append(w)
            # Bottom wall (negative Y direction)
            if y == grid_size-1 or not mask_color[y+1, x]:
                w = box(extents=[dot_size, wt, wall_height])
                w.apply_translation([x0 + dot_size/2, y0 - wt/2, z0 + wall_height/2])
                blocks.append(w)
        cumulative_z += h
    # Concatenate all blocks and export
    mesh = trimesh.util.concatenate(blocks)
    mesh.export(output_path)
    return mesh

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


# -------------------------------
# GUI クラス
# -------------------------------
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
    
    def get_nearest_palette_color(self, color):
        """登録パレットから最も近い色を返す (R,G,B tuple)"""
        if not hasattr(self, 'palette_colors') or not self.palette_colors:
            return None
        r, g, b = color
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
        r, g, b = color
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
        if obj is getattr(self, 'original_image_label', None) and getattr(self, 'trim_selecting', False):
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._trim_origin = event.pos()
                self.rubber_band.setGeometry(QRect(self._trim_origin, QSize()))
                self.rubber_band.show()
                return True
            elif event.type() == QEvent.MouseMove and self.rubber_band.isVisible():
                # 四角形を常に縦横比1:1に固定
                current_pos = event.pos()
                dx = current_pos.x() - self._trim_origin.x()
                dy = current_pos.y() - self._trim_origin.y()
                side = min(abs(dx), abs(dy))
                # 押下方向を保持
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
            # BMP画像をトリミングした場合は、ドット数に合わせてGrid Sizeを更新
            try:
                if suffix.lower() == '.bmp' and hasattr(self, 'controls') and 'Grid Size' in self.controls:
                    # croppedはPIL Imageのまま利用
                    w_crop, h_crop = cropped.size
                    grid_size = w_crop if w_crop == h_crop else w_crop
                    # スピンボックスに反映
                    self.controls['Grid Size'].setValue(int(grid_size))
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
            
            # 同色マージオプションを保存
            if hasattr(self, 'merge_same_color_checkbox'):
                project_data['merge_same_color'] = self.merge_same_color_checkbox.isChecked()
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
            
            # 同色マージオプションを復元
            if 'merge_same_color' in project_data and hasattr(self, 'merge_same_color_checkbox'):
                self.merge_same_color_checkbox.setChecked(project_data['merge_same_color'])
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
                # ensure layer mode checkbox is visible
                if hasattr(self, 'layer_mode_checkbox'):
                    self.layer_mode_checkbox.setChecked(True)
            
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
            
        # 現在のパラメータ取得
        params = {key: spin.value() for key, spin in self.controls.items()}
        
        # ハイライト表示用の一時的なピクセルデータを作成
        highlighted_pixels = self.pixels_rounded_np.copy()
        r, g, b = target_color
        
        # ハイライトする色を保存
        self.highlighted_color = target_color
        
        try:
            # プレビュー更新（ハイライト表示）
            preview_img = generate_preview_image(
                self.image_path,
                self.current_grid_size,
                int(params["Color Step"]),
                int(params["Top Colors"]),
                self.zoom_factor,
                custom_pixels=highlighted_pixels,
                highlight_color=target_color  # ハイライト色を指定
            )
            
            # プレビュー画像を更新（QPixmapに変換）
            preview_buffer = BytesIO()
            preview_img.save(preview_buffer, format="PNG")
            preview_qimg = QImage()
            preview_qimg.loadFromData(preview_buffer.getvalue())
            preview_pixmap = QPixmap.fromImage(preview_qimg)
            
            # ラベルに表示
            self.preview_label.setPixmap(preview_pixmap)
            
            # ハイライトはダイアログが閉じるまで維持するので、タイマーは使わない
            # self.highlight_timer.start(1000)
            
        except Exception as e:
            print(f"ハイライト表示エラー: {str(e)}")
    
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
            # 透明色に置換（黒=0,0,0として扱う）
            self.replace_all_same_color(target_color, (0, 0, 0))
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
            # 壁色とマージオプション
            if hasattr(self, 'wall_color') and isinstance(self.wall_color, QColor):
                wall_clr = (self.wall_color.red(), self.wall_color.green(), self.wall_color.blue())
            else:
                wall_clr = getattr(self, 'wall_color', (255, 255, 255))
            merge_same = self.merge_same_color_checkbox.isChecked() if hasattr(self, 'merge_same_color_checkbox') else False
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
        file_group = QGroupBox("ファイル操作")
        file_layout = QVBoxLayout()
        
        self.input_label = QLabel("画像が選択されていません")
        self.input_label.setWordWrap(True)
        
        file_btn_layout = QHBoxLayout()
        self.select_button = QPushButton("画像を選択")
        self.select_button.clicked.connect(self.select_image)
        
        self.export_button = QPushButton("STLをエクスポート")
        self.export_button.clicked.connect(self.export_stl)
        
        file_btn_layout.addWidget(self.select_button)
        file_btn_layout.addWidget(self.export_button)
        
        file_layout.addWidget(self.input_label)
        file_layout.addLayout(file_btn_layout)
        
        file_group.setLayout(file_layout)
        column1_layout.addWidget(file_group)
        
        # オリジナル画像表示エリア
        original_group = QGroupBox("オリジナル画像")
        original_layout = QVBoxLayout()
        
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
        original_group.setLayout(original_layout)
        column1_layout.addWidget(original_group)
        
        # パラメータ設定グループ（スクロール対応）
        param_group = QGroupBox("パラメータ設定")
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
            "減色なし (No Quantization)"
        ])
        self.color_algo_combo.setToolTip(
            "減色アルゴリズムの選択:\n"
            "・単純量子化: 最も高速で簡単なアルゴリズム\n"
            "・メディアンカット法: 色空間を分割し、各領域の代表色を使用\n"
            "・K-means法: 機械学習ベースの色のクラスタリング\n"
            "・オクトツリー法: 色空間の階層的分割による高品質な減色\n"
            "・トゥーンアニメ風: 鮮やかな色とはっきりした色の差を持つアニメ風の配色\n"
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
        
        # 同じ色のドット間の内壁を省略するオプション
        self.merge_same_color_checkbox = QCheckBox("同じ色のドット間の内壁を省略")
        self.merge_same_color_checkbox.setChecked(False)  # デフォルトはオフ
        self.merge_same_color_checkbox.setToolTip("このオプションを有効にすると、同じ色のドット同士の間の内壁が作られなくなります。")
        
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
                
                # グリッドサイズの範囲内かチェック
                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
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
                
        # パラメータのグリッドレイアウト
        self.param_grid = QGridLayout()
        self.controls = {}
        self.sliders = {}
        
        # パラメータ定義
        parameters = [
            ("Grid Size", 32, 8, 512),  # 512x512までの大きな画像に対応
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
        
        # レイアウトに追加
        param_layout.addLayout(color_algo_layout)
        param_layout.addLayout(wall_color_layout)
        param_layout.addWidget(self.merge_same_color_checkbox)
        param_layout.addLayout(self.param_grid)
        param_layout.addStretch()  # 下部に余白を追加
        
        # スクロールエリアの設定
        param_scroll.setWidget(param_scroll_content)
        param_scroll.setMinimumHeight(250)  # 最小の高さを設定
        
        param_group_layout = QVBoxLayout()
        param_group_layout.addWidget(param_scroll)
        param_group.setLayout(param_group_layout)
        column1_layout.addWidget(param_group)
        # レイヤー設定パネル
        self.layer_group = QGroupBox("レイヤー設定")
        # レイヤーモード有効化オプション
        self.layer_mode_checkbox = QCheckBox("色レイヤーモードを有効にする")
        self.layer_mode_checkbox.setChecked(False)
        # レイヤー設定リスト (ドラッグで順序変更可能)
        self.layer_scroll = QScrollArea()
        self.layer_scroll.setWidgetResizable(True)
        self.layer_list = QListWidget()
        # 内部移動を有効にし、アイテムをドラッグで並び替え
        self.layer_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.layer_list.setDefaultDropAction(Qt.MoveAction)
        self.layer_list.setSelectionMode(QAbstractItemView.NoSelection)
        # 順序変更時にカラー順序を更新
        self.layer_list.model().rowsMoved.connect(self.on_layer_reordered)
        self.layer_scroll.setWidget(self.layer_list)
        # レイアウト設定
        layer_group_layout = QVBoxLayout()
        # レイヤーモード有効化チェック
        layer_group_layout.addWidget(self.layer_mode_checkbox)
        # レイヤー更新ボタン（手動更新）
        self.layer_refresh_button = QPushButton("レイヤーを更新")
        self.layer_refresh_button.setToolTip("最新のドットデータでレイヤー設定を更新します")
        self.layer_refresh_button.clicked.connect(self.update_layer_controls)
        layer_group_layout.addWidget(self.layer_refresh_button)
        # レイヤー設定を別ウィンドウで開くボタン
        self.layer_popup_button = QPushButton("別ウィンドウで開く")
        self.layer_popup_button.setToolTip("レイヤー設定を別ウィンドウで開きます")
        self.layer_popup_button.clicked.connect(self.show_layer_settings_dialog)
        layer_group_layout.addWidget(self.layer_popup_button)
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
        # 各色ごとの高さ設定用スクロール領域
        layer_group_layout.addWidget(self.layer_scroll)
        self.layer_group.setLayout(layer_group_layout)
        column1_layout.addWidget(self.layer_group)
        
        # カラム2：ペイント操作、プレビュー、ズームバー
        column2_panel = QWidget()
        column2_layout = QVBoxLayout(column2_panel)
        
        # ペイント操作ツールバー
        paint_tools_group = QGroupBox("ペイントツール")
        paint_tools_layout = QVBoxLayout()
        
        # ドット編集用ツールバー
        edit_toolbar = QHBoxLayout()
        
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
        undo_btn = QPushButton("←")
        undo_btn.setToolTip("直前の編集を元に戻す")
        undo_btn.setMinimumWidth(40)  # 最小幅を設定
        undo_btn.clicked.connect(self.undo_edit)
        
        # やり直し（Redo）ボタン
        redo_btn = QPushButton("→")
        redo_btn.setToolTip("元に戻した編集をやり直す")
        redo_btn.setMinimumWidth(40)  # 最小幅を設定
        redo_btn.clicked.connect(self.redo_edit)
        
        # ツールバーにボタンを追加
        mode_toolbar = QHBoxLayout()
        mode_toolbar.addWidget(paint_mode_btn)
        mode_toolbar.addWidget(bucket_mode_btn)
        mode_toolbar.addWidget(select_mode_btn)
        mode_toolbar.addWidget(self.ai_brush_btn)
        
        color_toolbar = QHBoxLayout()
        color_toolbar.addWidget(self.color_pick_btn)
        color_toolbar.addWidget(eyedropper_btn)
        color_toolbar.addWidget(self.transparent_btn)
        
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
        
        # ツールバーをメインレイアウトに追加
        edit_toolbar.addLayout(mode_toolbar)
        edit_toolbar.addLayout(color_toolbar)
        edit_toolbar.addLayout(brush_size_toolbar)
        edit_toolbar.addLayout(history_toolbar)
        
        # 操作方法説明用のツールチップ
        info_label = QLabel("編集方法")
        info_label.setToolTip(
            "ドット編集方法:\n"
            "・ペンモード: クリック・ドラッグでドットを描画\n"
            "・塗りつぶし: 同じ色のドットをクリックで塗りつぶし\n"
            "・選択モード: ドットをクリックして色の変更や透明化\n"
            "・スクロール: ズームイン/アウト\n"
            "・透明にする: 黒色(0,0,0)として処理されます\n"
            "・元に戻す/やり直し: 編集履歴の操作が可能です"
        )
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: blue; text-decoration: underline;")
        
        paint_tools_layout.addLayout(edit_toolbar)
        paint_tools_layout.addWidget(info_label)
        paint_tools_group.setLayout(paint_tools_layout)
        column2_layout.addWidget(paint_tools_group)
        
        # プレビュー表示エリア
        preview_group = QGroupBox("プレビュー")
        preview_layout = QVBoxLayout()
        
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setMinimumHeight(400)
        
        # クリック可能なカスタムラベルを使用
        self.preview_label = ClickableLabel("プレビューが表示されます")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # シグナルを接続
        self.preview_label.clicked.connect(self.on_preview_clicked)
        self.preview_label.hover.connect(self.on_preview_hover)
        self.preview_label.dragPaint.connect(self.on_preview_drag_paint)
        self.preview_label.mouseWheel.connect(self.on_preview_mouse_wheel)
        
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
        info_scroll.setMinimumHeight(180)  # やや高めに設定
        info_scroll.setMaximumHeight(250)  # 最大高さを増やす
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
        
        # 3つのカラムをメインレイアウトに追加（1と2を優先）
        main_layout.addWidget(column1_panel, 4)  # カラム1の幅を4
        main_layout.addWidget(column2_panel, 4)  # カラム2の幅を4
        main_layout.addWidget(column3_panel, 2)  # カラム3の幅を2（やや狭め）
        
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
        
    def set_button_color(self, button, color):
        """ボタンの背景色を設定する"""
        button.setStyleSheet(f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); border: 1px solid black;")
        
    def select_wall_color(self):
        """壁の色を選択するダイアログを表示"""
        color = QColorDialog.getColor(self.wall_color, self, "壁の色を選択")
        if color.isValid():
            self.wall_color = color
            self.set_button_color(self.wall_color_button, color)
            
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
            is_transparent = tuple(current_color) == (0, 0, 0)
            
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
            # 現在の色を保存して透明に切り替え
            self.prev_paint_color = self.current_paint_color
            self.current_paint_color = QColor(0, 0, 0)
            self.set_button_color(self.color_pick_btn, self.current_paint_color)
            self.statusBar().showMessage("透明モード: 黒色(0,0,0)で描画")
        else:
            # 前の色に戻す（保存されていなければデフォルト赤）
            if hasattr(self, 'prev_paint_color'):
                self.current_paint_color = self.prev_paint_color
            else:
                self.current_paint_color = QColor(255, 0, 0)
            self.set_button_color(self.color_pick_btn, self.current_paint_color)
            self.statusBar().showMessage("通常モード")
    
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
                    
                    # グリッド範囲内かチェック
                    if 0 <= array_x < self.current_grid_size and 0 <= array_y < self.current_grid_size:
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
        """OpenAI APIを使用してピクセルデータを更新する"""
        if not getattr(self, 'openai_api_key', None):
            QMessageBox.warning(self, "APIキー未設定", "まず[設定]メニューからAPIキーを設定してください。")
            return None
        self.statusBar().showMessage("AIブラシ処理中...")
        try:
            # ピクセルデータをJSONに変換
            pixel_list = self.pixels_rounded_np.tolist()
            # プロンプト作成
            brush_color = (self.current_paint_color.red(), self.current_paint_color.green(), self.current_paint_color.blue())
            # プロンプト：選択した色に近いピクセルを指定色に変更するよう指示
            prompt = (
                f"ピクセルデータは三重配列のJSONです。選択された座標({grid_x},{grid_y})の色{selected_color}に近い色のすべてのピクセルを"
                f"指定の色{brush_color}に変更してください。変更後のピクセルデータ配列をJSON形式のみで返してください。"
            )
            messages = [
                {"role": "system", "content": "You are a helpful assistant for pixel editing."},
                {"role": "user", "content": prompt + " ピクセルデータ: " + json.dumps(pixel_list)}
            ]
            # 設定されたAPIキーを適用
            openai.api_key = self.openai_api_key
            try:
                # OpenAI Python >=1.0.0: 新クライアントAPIを使用
                client = openai.OpenAI(api_key=self.openai_api_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0
                )
            except AttributeError:
                # OpenAI Python <1.0.0: 従来のインターフェースにフォールバック
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0
                )
            # モデル応答からテキストを取得
            raw = response.choices[0].message.content.strip()
            text = raw
            # Markdownコードフェンスを除去
            if text.startswith('```'):
                parts = text.split('```')
                # フェンス内の先頭にあるJSON/リストを探す
                for part in parts:
                    p = part.strip()
                    if p.startswith('[') or p.startswith('{'):
                        text = p
                        break
            # 先頭のリスト/オブジェクトをバランスマッチで抽出
            def extract_balance(s, open_ch, close_ch):
                start = s.find(open_ch)
                if start < 0:
                    return None
                lvl = 1
                for idx in range(start+1, len(s)):
                    c = s[idx]
                    if c == open_ch:
                        lvl += 1
                    elif c == close_ch:
                        lvl -= 1
                        if lvl == 0:
                            return s[start:idx+1]
                return None
            content = None
            # 優先してリストを抽出
            content = extract_balance(text, '[', ']')
            if content is None:
                # 次にオブジェクトを抽出
                content = extract_balance(text, '{', '}')
            # 抽出できなければ生テキストを使う
            if content is None:
                content = text
            # JSONパース or Pythonリテラル評価
            try:
                new_pixels = json.loads(content)
            except Exception:
                new_pixels = ast.literal_eval(content)
            arr = np.array(new_pixels, dtype=np.uint8)
            return arr
        except Exception as e:
            QMessageBox.critical(self, "AIブラシエラー", f"AIブラシ処理中にエラーが発生しました: {e}")
            return None
        finally:
            self.statusBar().clearMessage()
    
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
            # 透過色を黒（0,0,0）として扱う
            self.pixels_rounded_np[array_y, array_x] = [0, 0, 0]
            
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
            # 透過色を黒（0,0,0）として扱う
            self.pixels_rounded_np[array_y, array_x] = [0, 0, 0]
            
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
        path, _ = QFileDialog.getOpenFileName(
            self,
            "画像を開く",
            "",
            "画像ファイル (*.png *.jpg *.jpeg *.gif *.bmp)"
        )
        if path:
            self.image_path = path
            self.input_label.setText(path)
            # BMPファイルの場合はグリッドサイズを自動検出
            try:
                if path.lower().endswith('.bmp'):
                    img = Image.open(path)
                    w, h = img.size
                    # 正方形グリッドを想定し、幅を優先
                    grid_size = w if w == h else w
                    # コントロールが存在すれば値を設定（範囲外は自動的にクランプされる）
                    if hasattr(self, 'controls') and 'Grid Size' in self.controls:
                        self.controls['Grid Size'].setValue(int(grid_size))
            except Exception:
                # 自動検出失敗時は何もしない
                pass
            # 新しい画像を選択したらハイライトをクリア
            if hasattr(self.preview_label, 'last_clicked_pos'):
                self.preview_label.last_clicked_pos = None
            # プレビュー更新
            self.update_preview()
    
    def on_color_algo_changed(self, index):
        """減色アルゴリズムが変更されたときの処理"""
        algo_map = {
            0: "simple",     # 単純量子化
            1: "median_cut", # メディアンカット法
            2: "kmeans",     # K-means法
            3: "octree",     # オクトツリー法
            4: "toon",       # トゥーンアニメ風
            5: "none"        # 減色なし
        }
        
        self.current_color_algo = algo_map.get(index, "simple")
        
        # ステータスメッセージ更新
        status_messages = {
            "simple": "単純量子化アルゴリズムを使用します",
            "median_cut": "メディアンカット法（色空間分割による減色）を使用します",
            "kmeans": "K-means法（機械学習ベースのクラスタリング）を使用します",
            "toon": "トゥーンアニメ風の鮮やかな色使いで減色します",
            "none": "減色せず元画像の色をそのまま使用します",
            "octree": "オクトツリー法（階層的色空間分割）を使用します"
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
            
            # オリジナル画像の表示
            original_img = Image.open(self.image_path)
            
            # GIF画像の場合は最初のフレームを取得
            if hasattr(original_img, 'format') and original_img.format == 'GIF' and 'duration' in original_img.info:
                # アニメーションGIFの場合
                original_img = original_img.convert('RGBA')  # 透明部分を適切に処理
            
            # 画像が大きすぎる場合はリサイズ（BMPは最近傍、その他は高品質リサンプリング）
            max_display_size = 500
            if max(original_img.width, original_img.height) > max_display_size:
                ratio = max_display_size / max(original_img.width, original_img.height)
                new_size = (int(original_img.width * ratio), int(original_img.height * ratio))
                # BMP画像はドット単位を維持するために最近傍を使用
                resample_method = Image.NEAREST if self.image_path.lower().endswith('.bmp') else Image.LANCZOS
                original_img = original_img.resize(new_size, resample_method)
            
            # オリジナル画像をQPixmapに変換して表示
            original_buffer = BytesIO()
            original_img.save(original_buffer, format="PNG")
            original_qimg = QImage()
            original_qimg.loadFromData(original_buffer.getvalue())
            original_pixmap = QPixmap.fromImage(original_qimg)
            
            self.original_image_label.setPixmap(original_pixmap)
            self.original_image_label.adjustSize()
            
            # ペイントモードではハイライト表示しない
            highlight_pos = None
            # 選択モードの場合のみ、最後にクリックされた位置をハイライト表示
            if not self.is_paint_mode:
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
                    img_resized = Image.open(self.image_path).convert("RGB").resize(
                        (self.current_grid_size, self.current_grid_size), resample=Image.NEAREST)
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
                    self.pixels_rounded_np = pixels_array.reshape((self.current_grid_size, self.current_grid_size, 3))
                    
                    # デバッグのために型と形状を確認
                    print(f"生成されたpixels_rounded_np の型: {type(self.pixels_rounded_np)}")
                    print(f"生成されたpixels_rounded_np の形状: {self.pixels_rounded_np.shape}")
                        
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
                <tr><td>同色間内壁省略</td><td>{"あり" if self.merge_same_color_checkbox.isChecked() else "なし"}</td></tr>
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
            arr = self.pixels_rounded_np.reshape(-1, 3)
            counts = Counter([tuple(c) for c in arr])
            present = [color for color, _ in counts.most_common() if color != (0, 0, 0)]
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
        # レイヤーモードチェックボックス
        cb = QCheckBox("色レイヤーモードを有効にする")
        cb.setChecked(self.layer_mode_checkbox.isChecked())
        cb.toggled.connect(self.layer_mode_checkbox.setChecked)
        layout.addWidget(cb)
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
        if not self.image_path:
            self.input_label.setText("画像が選択されていません")
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
                # レイヤーモードが有効なら色ごとに積層生成
                if hasattr(self, 'layer_mode_checkbox') and self.layer_mode_checkbox.isChecked():
                    # 色レイヤーモード：編集済みピクセルを使用して積層STLを生成
                    mesh = generate_layered_stl(
                        self.pixels_rounded_np,
                        out_path,
                        int(params["Grid Size"]),
                        float(params["Dot Size"]),
                        float(params["Base Height"]),
                        float(params["Wall Thickness"]),
                        float(params["Wall Height"]),
                        self.layer_heights,
                        self.layer_color_order
                    )
                    # プレビューとレポート生成
                    preview_mesh = mesh
                    self.show_stl_preview(preview_mesh)
                    html_path = self.generate_html_report(out_path, preview_mesh)
                    self.input_label.setText(f"{out_path} に色レイヤーモードSTLをエクスポートしました")
                    return
                
                # メッシュ生成（メッシュも返すように指定）
                if custom_pixels is not None:
                    # カスタムピクセルからSTLを直接生成
                    from PIL import Image
                    import tempfile
                    
                    # 一時ファイルに画像を保存
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        tmp_path = tmp.name
                        # カスタムピクセルデータから画像を作成
                        custom_img = Image.fromarray(custom_pixels, mode='RGB')
                        custom_img.save(tmp_path)
                    
                    # 同じ色のドット間の内壁を省略するオプションの状態を取得
                    merge_same_color = self.merge_same_color_checkbox.isChecked()
                    
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
                    # 同じ色のドット間の内壁を省略するオプションの状態を取得
                    merge_same_color = self.merge_same_color_checkbox.isChecked()
                    
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
                            from PIL import Image
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
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DotPlateApp()
    window.show()
    sys.exit(app.exec_())