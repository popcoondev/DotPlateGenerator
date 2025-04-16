# dot_plate_generator_gui.py
# 必要ライブラリ: PyQt5, PIL, numpy, trimesh, shapely, skimage, scipy, matplotlib

import sys
import os
import numpy as np
from PIL import Image
from collections import Counter
from scipy.spatial import distance
import trimesh
from trimesh.creation import box
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QScrollArea,
    QVBoxLayout, QHBoxLayout, QSlider, QSpinBox, QGridLayout, QDoubleSpinBox,
    QToolButton, QDialog, QGroupBox, QFrame, QSizePolicy, QToolTip, QMainWindow,
    QColorDialog, QCheckBox, QComboBox, QMenu, QAction
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QImage, QColor
from shapely.geometry import Polygon
from skimage import measure
from io import BytesIO
import threading
import time

# Vedoをインポート (VTKベースの3D可視化ライブラリ)
try:
    import vedo
    VEDO_AVAILABLE = True
except ImportError:
    print("vedo library not available, please install with: pip install vedo")
    import matplotlib.pyplot as plt
    VEDO_AVAILABLE = False

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
                       custom_pixels=None, highlight_pos=None, hover_pos=None, color_algo="simple"):
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
    
    return result

# -------------------------------
# モデル生成関数
# -------------------------------
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
    mask = np.array([[tuple(px) != (0, 0, 0) for px in row] for row in pixels_rounded_np]).astype(np.uint8)
    
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
                
                # 各方向の拡張量を計算
                extend_left = 0 if has_left else out_thickness
                extend_right = 0 if has_right else out_thickness
                extend_top = 0 if has_top else out_thickness
                extend_bottom = 0 if has_bottom else out_thickness
                
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
                
                # ドットの区切り壁とベースの輪郭壁を分けて処理
                # 通常の内側壁と外周壁で厚みを区別する
                
                # 壁の長さを計算（ベースの寸法に合わせる）
                left_wall_length = base_depth
                right_wall_length = base_depth
                top_wall_length = base_width
                bottom_wall_length = base_width
                
                # まずすべてのドットに対して基本的な内壁を作成
                # 左・右の内側壁（基本壁）
                lr_wall_boxes = [
                    box(extents=[wall_thickness, left_wall_length, wall_height]),
                    box(extents=[wall_thickness, right_wall_length, wall_height]),
                ]
                
                # 上・下の内側壁（基本壁）
                tb_wall_boxes = [
                    box(extents=[top_wall_length, wall_thickness, wall_height]),
                    box(extents=[bottom_wall_length, wall_thickness, wall_height]),
                ]
                
                # 外周壁（追加の厚みあり）- 外部に面しているドットのみに適用
                # 左・右の外周壁
                lr_outer_wall_boxes = [
                    box(extents=[wall_thickness + out_thickness, left_wall_length, wall_height]),
                    box(extents=[wall_thickness + out_thickness, right_wall_length, wall_height]),
                ]
                # 上・下の外周壁
                tb_outer_wall_boxes = [
                    box(extents=[top_wall_length, wall_thickness + out_thickness, wall_height]),
                    box(extents=[bottom_wall_length, wall_thickness + out_thickness, wall_height]),
                ]
                
                # 壁ボックスのリスト
                wall_boxes = []
                
                # 左側の壁 - 同じ色の場合は壁を作らない
                if is_left_edge:  # 左端または左が空白または隣接ドットが異なる色（外周）
                    wall_boxes.append(lr_outer_wall_boxes[0])  # 厚い外周壁
                elif not merge_same_color:  # 同色でも壁を作る場合
                    wall_boxes.append(lr_wall_boxes[0])  # 通常の内側壁
                # merge_same_color=True かつ同色の場合は壁を追加しない
                
                # 右側の壁 - 同じ色の場合は壁を作らない
                if is_right_edge:  # 右端または右が空白または隣接ドットが異なる色（外周）
                    wall_boxes.append(lr_outer_wall_boxes[1])  # 厚い外周壁
                elif not merge_same_color:  # 同色でも壁を作る場合
                    wall_boxes.append(lr_wall_boxes[1])  # 通常の内側壁
                # merge_same_color=True かつ同色の場合は壁を追加しない
                
                # 上側の壁 - 同じ色の場合は壁を作らない
                if is_top_edge:  # 上端または上が空白または隣接ドットが異なる色（外周）
                    wall_boxes.append(tb_outer_wall_boxes[0])  # 厚い外周壁
                elif not merge_same_color:  # 同色でも壁を作る場合
                    wall_boxes.append(tb_wall_boxes[0])  # 通常の内側壁
                # merge_same_color=True かつ同色の場合は壁を追加しない
                
                # 下側の壁 - 同じ色の場合は壁を作らない
                if is_bottom_edge:  # 下端または下が空白または隣接ドットが異なる色（外周）
                    wall_boxes.append(tb_outer_wall_boxes[1])  # 厚い外周壁
                elif not merge_same_color:  # 同色でも壁を作る場合
                    wall_boxes.append(tb_wall_boxes[1])  # 通常の内側壁
                # merge_same_color=True かつ同色の場合は壁を追加しない
                
                # 壁の位置を設定する
                positions = []
                
                # 左側の壁の位置 - wall_boxesに追加された分だけ位置も計算
                if is_left_edge:  # 左端または左が空白または隣接ドットが異なる色（外周）
                    # 左外周壁の位置（外側に厚みを追加）
                    positions.append([
                        x0 - extend_left + (wall_thickness + out_thickness) / 2, 
                        y_center,  # ベースの中心Y座標を使用
                        base_height + wall_height / 2
                    ])
                elif not merge_same_color:  # 同色でも壁を作る場合
                    # 通常の左内側壁
                    positions.append([
                        x0 + wall_thickness / 2,
                        y_center,
                        base_height + wall_height / 2
                    ])
                
                # 右側の壁の位置 - wall_boxesに追加された分だけ位置も計算
                if is_right_edge:  # 右端または右が空白または隣接ドットが異なる色（外周）
                    # 右外周壁の位置（外側に厚みを追加）
                    positions.append([
                        x0 + dot_size + extend_right - (wall_thickness + out_thickness) / 2,
                        y_center,
                        base_height + wall_height / 2
                    ])
                elif not merge_same_color:  # 同色でも壁を作る場合
                    # 通常の右内側壁
                    positions.append([
                        x0 + dot_size - wall_thickness / 2,
                        y_center,
                        base_height + wall_height / 2
                    ])
                
                # 上側の壁の位置 - wall_boxesに追加された分だけ位置も計算
                if is_top_edge:  # 上端または上が空白または隣接ドットが異なる色（外周）
                    # 上外周壁の位置（外側に厚みを追加）
                    positions.append([
                        x_center,  # ベースの中心X座標を使用
                        y0 + dot_size + extend_top - (wall_thickness + out_thickness) / 2,
                        base_height + wall_height / 2
                    ])
                elif not merge_same_color:  # 同色でも壁を作る場合
                    # 通常の上内側壁
                    positions.append([
                        x_center,
                        y0 + wall_thickness / 2,
                        base_height + wall_height / 2
                    ])
                
                # 下側の壁の位置 - wall_boxesに追加された分だけ位置も計算
                if is_bottom_edge:  # 下端または下が空白または隣接ドットが異なる色（外周）
                    # 下外周壁の位置（外側に厚みを追加）
                    positions.append([
                        x_center,
                        y0 - extend_bottom + (wall_thickness + out_thickness) / 2,
                        base_height + wall_height / 2
                    ])
                elif not merge_same_color:  # 同色でも壁を作る場合
                    # 通常の下内側壁
                    positions.append([
                        x_center,
                        y0 + dot_size - wall_thickness / 2,
                        base_height + wall_height / 2
                    ])
                
                for i, (wbox, pos) in enumerate(zip(wall_boxes, positions)):
                    wbox.apply_translation(pos)
                    # 壁には独自の色情報を付けない（後で一律に指定色にする）
                    wall_blocks.append(wbox)
    
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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dot Plate Generator")
        self.setMinimumSize(1200, 700)
        
        # ステータスバーを初期化
        self.statusBar().showMessage("準備完了")
        
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
            "オクトツリー法 (Octree)"
        ])
        self.color_algo_combo.setToolTip(
            "減色アルゴリズムの選択:\n"
            "・単純量子化: 最も高速で簡単なアルゴリズム\n"
            "・メディアンカット法: 色空間を分割し、各領域の代表色を使用\n"
            "・K-means法: 機械学習ベースの色のクラスタリング\n"
            "・オクトツリー法: 色空間の階層的分割による高品質な減色"
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
            ("Grid Size", 32, 8, 64),
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
                    self.update_preview()
                return spin_changed
            
            def make_slider_changed(label, is_int, slider_factor):
                def slider_changed(value):
                    self.controls[label].setValue(value / slider_factor)
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
        
        color_toolbar = QHBoxLayout()
        color_toolbar.addWidget(self.color_pick_btn)
        color_toolbar.addWidget(eyedropper_btn)
        color_toolbar.addWidget(self.transparent_btn)
        
        history_toolbar = QHBoxLayout()
        history_toolbar.addWidget(undo_btn)
        history_toolbar.addWidget(redo_btn)
        
        # ツールバーをメインレイアウトに追加
        edit_toolbar.addLayout(mode_toolbar)
        edit_toolbar.addLayout(color_toolbar)
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
        
        zoom_layout.addWidget(self.zoom_label)
        zoom_layout.addWidget(self.zoom_slider)
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
        
        # 正方形のフレームを作成するためのウィジェット
        square_frame = QWidget()
        square_frame.setMinimumSize(300, 300)
        square_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # アスペクト比を1:1に保つためのポリシーを設定
        square_frame_policy = square_frame.sizePolicy()
        square_frame_policy.setHeightForWidth(True)
        square_frame.setSizePolicy(square_frame_policy)
        
        # sizeHintを上書きするためのサブクラス化
        class SquareWidget(QWidget):
            def __init__(self):
                super().__init__()
            
            def heightForWidth(self, width):
                return width  # 幅と同じ高さを返す（1:1の比率）
            
            def hasHeightForWidth(self):
                return True
        
        # 正方形ウィジェットを作成
        square_widget = SquareWidget()
        square_layout = QVBoxLayout(square_widget)
        square_layout.setContentsMargins(0, 0, 0, 0)
        
        # STLプレビューラベル
        self.stl_preview_label = QLabel("STLプレビューが表示されます")
        self.stl_preview_label.setAlignment(Qt.AlignCenter)
        self.stl_preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        square_layout.addWidget(self.stl_preview_label)
        stl_preview_layout.addWidget(square_widget)
        stl_preview_group.setLayout(stl_preview_layout)
        column3_layout.addWidget(stl_preview_group)
        
        # 3つのカラムをメインレイアウトに追加
        main_layout.addWidget(column1_panel, 1)  # カラム1の幅を1
        main_layout.addWidget(column2_panel, 1)  # カラム2の幅を1
        main_layout.addWidget(column3_panel, 1)  # カラム3の幅を1:1
        
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
            # カーソルを元に戻す
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
            
        try:
            # NumPy配列は[row, col]=[y, x]の順でアクセス
            array_y = grid_y
            array_x = grid_x
            
            # 現在の色と同じなら変更しない
            current_color = self.pixels_rounded_np[array_y, array_x]
            if tuple(current_color) == tuple(color):
                return False
                
            # 編集前の状態を履歴に保存（最初の変更時のみ）
            self.save_edit_history()
                
            # ピクセルの色を更新
            self.pixels_rounded_np[array_y, array_x] = color
            return True
            
        except IndexError:
            print(f"座標[{array_y}, {array_x}]はインデックス範囲外です")
            return False
            
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
            
            # メニューにアクションを追加
            menu.addAction(pick_action)
            menu.addAction(change_action)
            menu.addAction(transparent_action)
            
            # カーソル位置にメニューを表示
            from PyQt5.QtGui import QCursor
            menu.exec_(QCursor.pos())
            
        except IndexError as e:
            print(f"座標変換エラー: {e}")
            return
    
    def pick_color_for_paint(self, color, dialog=None):
        """選択したドットの色をペイント色として設定"""
        self.current_paint_color = color
        self.set_button_color(self.color_pick_btn, color)
        # 透明色モードが有効なら無効化
        if self.transparent_btn.isChecked():
            self.transparent_btn.setChecked(False)
        if dialog:
            dialog.accept()
        
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
        path, _ = QFileDialog.getOpenFileName(self, "画像を開く", "", "画像ファイル (*.png *.jpg *.jpeg *.gif)")
        if path:
            self.image_path = path
            self.input_label.setText(path)
            # 新しい画像を選択したらハイライトをクリア
            if hasattr(self.preview_label, 'last_clicked_pos'):
                self.preview_label.last_clicked_pos = None
            self.update_preview()
    
    def on_color_algo_changed(self, index):
        """減色アルゴリズムが変更されたときの処理"""
        algo_map = {
            0: "simple",     # 単純量子化
            1: "median_cut", # メディアンカット法
            2: "kmeans",     # K-means法
            3: "octree"      # オクトツリー法
        }
        
        self.current_color_algo = algo_map.get(index, "simple")
        
        # ステータスメッセージ更新
        status_messages = {
            "simple": "単純量子化アルゴリズムを使用します",
            "median_cut": "メディアンカット法（色空間分割による減色）を使用します",
            "kmeans": "K-means法（機械学習ベースのクラスタリング）を使用します",
            "octree": "オクトツリー法（階層的色空間分割）を使用します"
        }
        
        self.statusBar().showMessage(status_messages.get(self.current_color_algo, "減色アルゴリズムを変更しました"))
        
        # 画像がロードされていればプレビューを更新
        if hasattr(self, 'image_path') and self.image_path:
            # 編集履歴をリセット
            if hasattr(self, 'pixels_rounded_np'):
                self.pixels_rounded_np = None
            self.update_preview()
    
    def update_preview(self, custom_pixels=None):
        """プレビュー画像を更新する（custom_pixelsが指定された場合はそれを使用）"""
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
            
            # 画像が大きすぎる場合はリサイズ
            max_display_size = 500
            if max(original_img.width, original_img.height) > max_display_size:
                # アスペクト比を維持しながらリサイズ
                ratio = max_display_size / max(original_img.width, original_img.height)
                new_size = (int(original_img.width * ratio), int(original_img.height * ratio))
                original_img = original_img.resize(new_size, Image.LANCZOS)
            
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
                        color_algo=self.current_color_algo
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
                        color_algo=self.current_color_algo
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
                    color_algo="simple"
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
                    self.preview_label.setCursor(Qt.ArrowCursor)  # ペイントモード
                else:
                    self.preview_label.setCursor(Qt.ArrowCursor)  # 選択モード
                
            except Exception as e:
                print(f"プレビュー表示エラー: {str(e)}")
                self.input_label.setText(f"プレビュー表示エラー: {str(e)}")
                
        except Exception as e:
            print(f"update_preview全体エラー: {str(e)}")
            self.input_label.setText(f"画像表示エラー: {str(e)}")
    
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
                wall_color = (self.wall_color.red(), self.wall_color.green(), self.wall_color.blue())
                
                # カスタム編集されたピクセルデータがあるかチェック
                custom_pixels = self.pixels_rounded_np if hasattr(self, 'pixels_rounded_np') and self.pixels_rounded_np is not None else None
                
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
                
                color_name = f"RGB({self.wall_color.red()}, {self.wall_color.green()}, {self.wall_color.blue()})"
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
            self.stl_preview_label.setPixmap(pixmap)
            self.stl_preview_label.setScaledContents(True)
            
            # プレビューを正方形にするために、高さ=幅を設定
            self.stl_preview_label.setFixedHeight(self.stl_preview_label.width())
            
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
        
        # プレビューラベルに表示
        self.stl_preview_label.setPixmap(pixmap)
        self.stl_preview_label.setScaledContents(True)
        
        # プレビューを正方形にするために、高さ=幅を設定
        self.stl_preview_label.setFixedHeight(self.stl_preview_label.width())
    
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