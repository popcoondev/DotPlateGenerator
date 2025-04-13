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
    QColorDialog
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
    return (pixels // step) * step

def map_to_closest_color(pixel, palette):
    return min(palette, key=lambda c: distance.euclidean(pixel, c))

def generate_preview_image(image_path, grid_size, color_step, top_color_limit, zoom_factor=10, custom_pixels=None, highlight_pos=None, hover_pos=None):
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
        pixels_normalized = normalize_colors(pixels, color_step)
        colors = [tuple(c) for c in pixels_normalized]
        color_counts = Counter(colors)
        top_colors = [c for c, _ in color_counts.most_common(top_color_limit)]
        pixels_rounded = [map_to_closest_color(c, top_colors) for c in colors]
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
                
                # まず左右方向の壁を設定（基本は内壁）
                # 左側の壁
                if is_left_edge:  # 左端または左が空白（外周）
                    wall_boxes.append(lr_outer_wall_boxes[0])  # 厚い外周壁
                else:
                    wall_boxes.append(lr_wall_boxes[0])  # 通常の内側壁
                
                # 右側の壁
                if is_right_edge:  # 右端または右が空白（外周）
                    wall_boxes.append(lr_outer_wall_boxes[1])  # 厚い外周壁
                else:
                    wall_boxes.append(lr_wall_boxes[1])  # 通常の内側壁
                
                # 次に上下方向の壁を設定
                # 上側の壁
                if is_top_edge:  # 上端または上が空白（外周）
                    wall_boxes.append(tb_outer_wall_boxes[0])  # 厚い外周壁
                else:
                    wall_boxes.append(tb_wall_boxes[0])  # 通常の内側壁
                
                # 下側の壁
                if is_bottom_edge:  # 下端または下が空白（外周）
                    wall_boxes.append(tb_outer_wall_boxes[1])  # 厚い外周壁
                else:
                    wall_boxes.append(tb_wall_boxes[1])  # 通常の内側壁
                
                # 壁の位置を設定する
                positions = []
                
                # 左側の壁の位置
                if is_left_edge:  # 左端または左が空白（外周）
                    # 左外周壁の位置（外側に厚みを追加）
                    positions.append([
                        x0 - extend_left + (wall_thickness + out_thickness) / 2, 
                        y_center,  # ベースの中心Y座標を使用
                        base_height + wall_height / 2
                    ])
                else:
                    # 通常の左内側壁
                    positions.append([
                        x0 + wall_thickness / 2,
                        y_center,
                        base_height + wall_height / 2
                    ])
                
                # 右側の壁の位置
                if is_right_edge:  # 右端または右が空白（外周）
                    # 右外周壁の位置（外側に厚みを追加）
                    positions.append([
                        x0 + dot_size + extend_right - (wall_thickness + out_thickness) / 2,
                        y_center,
                        base_height + wall_height / 2
                    ])
                else:
                    # 通常の右内側壁
                    positions.append([
                        x0 + dot_size - wall_thickness / 2,
                        y_center,
                        base_height + wall_height / 2
                    ])
                
                # 上側の壁の位置
                if is_top_edge:  # 上端または上が空白（外周）
                    # 上外周壁の位置（外側に厚みを追加）
                    positions.append([
                        x_center,  # ベースの中心X座標を使用
                        y0 + dot_size + extend_top - (wall_thickness + out_thickness) / 2,
                        base_height + wall_height / 2
                    ])
                else:
                    # 通常の上内側壁
                    positions.append([
                        x_center,
                        y0 + wall_thickness / 2,
                        base_height + wall_height / 2
                    ])
                
                # 下側の壁の位置
                if is_bottom_edge:  # 下端または下が空白（外周）
                    # 下外周壁の位置（外側に厚みを追加）
                    positions.append([
                        x_center,
                        y0 - extend_bottom + (wall_thickness + out_thickness) / 2,
                        base_height + wall_height / 2
                    ])
                else:
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
        self.setMinimumSize(900, 700)
        
        # ステータスバーを初期化
        self.statusBar().showMessage("準備完了")
        
        # メインウィジェットとレイアウト
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # 左パネル（プレビュー）
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 画像プレビュー領域
        preview_group = QGroupBox("画像プレビュー")
        preview_layout = QVBoxLayout()
        
        # プレビュー表示用の水平レイアウト（オリジナルと減色後の画像を並べる）
        preview_images_layout = QHBoxLayout()
        
        # オリジナル画像表示エリア
        original_area = QVBoxLayout()
        original_label = QLabel("オリジナル画像")
        original_label.setAlignment(Qt.AlignCenter)
        
        self.original_scroll = QScrollArea()
        self.original_scroll.setWidgetResizable(True)
        self.original_scroll.setMinimumHeight(350)
        self.original_scroll.setMinimumWidth(250)
        
        self.original_image_label = QLabel("オリジナル画像が表示されます")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.original_scroll.setWidget(self.original_image_label)
        
        original_area.addWidget(original_label)
        original_area.addWidget(self.original_scroll)
        
        # クリック可能なカスタムラベルの定義
        from PyQt5.QtCore import pyqtSignal
        
        class ClickableLabel(QLabel):
            clicked = pyqtSignal(int, int)  # x, y座標を返すシグナル
            hover = pyqtSignal(int, int)    # ホバー時のx, y座標を返すシグナル
            
            def __init__(self, text):
                super().__init__(text)
                self.pixmap_size = None
                self.grid_size = None
                self.zoom_factor = None
                self.last_clicked_pos = None  # 最後にクリックされたグリッド位置を保存
                self.hover_grid_pos = None    # ホバー中のグリッド位置
                self.setMouseTracking(True)   # マウスの移動を追跡
            
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
                
                # デバッグ出力
                # print(f"Mouse Position: {pos.x()}, {pos.y()}")
                # print(f"Scale: {scale_x}, {scale_y}")
                # print(f"Pixel Position: {pixel_x}, {pixel_y}")
                
                # グリッド座標に変換（ズームを考慮）
                grid_x = pixel_x // self.zoom_factor
                grid_y = pixel_y // self.zoom_factor
                
                # 座標の反転は行わない - on_preview_clicked などのハンドラ側で行う
                
                # グリッドサイズの範囲内かチェック
                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    # print(f"Grid Position: {grid_x}, {grid_y}")
                    return (grid_x, grid_y)
                return None
            
            def mouseMoveEvent(self, event):
                """マウス移動時のイベントハンドラ - ホバー効果を提供"""
                grid_pos = self.get_grid_position(event.pos())
                if grid_pos:
                    # 前回のホバー位置と異なる場合のみシグナル発信
                    if grid_pos != self.hover_grid_pos:
                        self.hover_grid_pos = grid_pos
                        self.hover.emit(grid_pos[0], grid_pos[1])
                        # ツールチップでグリッド位置を表示
                        QToolTip.showText(event.globalPos(), f"位置: [{grid_pos[0]}, {grid_pos[1]}]", self)
                super().mouseMoveEvent(event)
            
            def mousePressEvent(self, event):
                """マウスクリック時のイベントハンドラ"""
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
        
        # 減色後画像表示エリア
        reduced_area = QVBoxLayout()
        
        # ドット編集用ツールバー
        edit_toolbar = QHBoxLayout()
        
        # 元に戻す（Undo）ボタン
        undo_btn = QPushButton("↩ 元に戻す")
        undo_btn.setToolTip("直前の編集を元に戻す")
        undo_btn.clicked.connect(self.undo_edit)
        
        # やり直し（Redo）ボタン
        redo_btn = QPushButton("↪ やり直し")
        redo_btn.setToolTip("元に戻した編集をやり直す")
        redo_btn.clicked.connect(self.redo_edit)
        
        # ツールバーにボタンを追加
        edit_toolbar.addWidget(undo_btn)
        edit_toolbar.addWidget(redo_btn)
        
        # ドット画像ラベル
        reduced_label = QLabel("減色後のドット画像（クリックで色を変更）")
        reduced_label.setAlignment(Qt.AlignCenter)
        
        # 操作方法説明用のツールチップ
        info_label = QLabel("ℹ️ 編集方法")
        info_label.setToolTip(
            "ドット編集方法:\n"
            "・ドットをクリック: 色の変更や透明化ができます\n"
            "・透明にする: 黒色(0,0,0)として処理されます\n"
            "・元に戻す/やり直し: 編集履歴の操作が可能です\n"
            "・選択中のドット: 赤色の枠線でハイライト表示されます"
        )
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: blue; text-decoration: underline;")
        
        # ツールバーとラベルをレイアウトに追加
        reduced_area.addLayout(edit_toolbar)
        reduced_area.addWidget(reduced_label)
        reduced_area.addWidget(info_label)
        
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setMinimumHeight(350)
        self.preview_scroll.setMinimumWidth(250)
        
        # クリック可能なカスタムラベルを使用
        self.preview_label = ClickableLabel("プレビューが表示されます")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # クリックシグナルとホバーシグナルを接続
        self.preview_label.clicked.connect(self.on_preview_clicked)
        self.preview_label.hover.connect(self.on_preview_hover)
        
        self.preview_scroll.setWidget(self.preview_label)
        reduced_area.addWidget(self.preview_scroll)
        
        # 両方の画像エリアを水平に並べる
        preview_images_layout.addLayout(original_area)
        preview_images_layout.addLayout(reduced_area)
        
        # STLプレビュー用のラベル
        self.stl_preview_label = QLabel("STLプレビューが表示されます")
        self.stl_preview_label.setAlignment(Qt.AlignCenter)
        self.stl_preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stl_preview_label.setMinimumSize(300, 300)
        
        # ズームコントロール
        zoom_layout = QHBoxLayout()
        self.zoom_label = QLabel("ズーム:")
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(1)
        self.zoom_slider.setMaximum(20)
        self.zoom_slider.setValue(10)
        self.zoom_slider.valueChanged.connect(self.update_preview)
        
        zoom_layout.addWidget(self.zoom_label)
        zoom_layout.addWidget(self.zoom_slider)
        
        preview_layout.addLayout(preview_images_layout)
        preview_layout.addLayout(zoom_layout)
        
        preview_group.setLayout(preview_layout)
        left_layout.addWidget(preview_group)
        
        # STLプレビュー領域
        stl_preview_group = QGroupBox("STLプレビュー")
        stl_preview_layout = QVBoxLayout()
        stl_preview_layout.addWidget(self.stl_preview_label)
        stl_preview_group.setLayout(stl_preview_layout)
        left_layout.addWidget(stl_preview_group)
        
        # 右パネル（コントロール）
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
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
        
        # パラメータ設定グループ
        param_group = QGroupBox("パラメータ設定")
        param_layout = QVBoxLayout()
        
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
        
        param_layout.addLayout(wall_color_layout)
        
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
        
        param_layout.addLayout(self.param_grid)
        param_group.setLayout(param_layout)
        
        # レイアウトを組み立てる
        right_layout.addWidget(file_group)
        right_layout.addWidget(param_group)
        right_layout.addStretch()
        
        # スプリッター比率を設定
        main_layout.addWidget(left_panel, 3)  # 左パネルの幅を60%
        main_layout.addWidget(right_panel, 2)  # 右パネルの幅を40%
        
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
    
    def on_preview_clicked(self, grid_x, grid_y):
        """減色後のプレビュー画像内のドットがクリックされたときの処理"""
        if self.pixels_rounded_np is None:
            return
        
        # デバッグ情報
        print(f"ドットクリック: grid_x={grid_x}, grid_y={grid_y}")
        
        # 型チェック: pixels_rounded_npが正しくnumpy配列であることを確認
        if not isinstance(self.pixels_rounded_np, np.ndarray):
            print(f"エラー: pixels_rounded_npが正しいnumpy配列ではありません: {type(self.pixels_rounded_np)}")
            return
            
        print(f"pixels_rounded_np.shape = {self.pixels_rounded_np.shape}")
        
        # ピクセル配列の中身をテスト表示
        grid_height = self.pixels_rounded_np.shape[0]
        print(f"グリッド高さ: {grid_height}")
        
        # 座標変換前の位置の色を確認
        try:
            # 反転前
            orig_color = self.pixels_rounded_np[grid_y, grid_x]
            print(f"変換前座標[{grid_y}, {grid_x}]の色: RGB({orig_color[0]}, {orig_color[1]}, {orig_color[2]})")
        except IndexError:
            print(f"変換前座標[{grid_y}, {grid_x}]はインデックス範囲外です")
            
        # グリッド座標の調整（表示上の座標からピクセル配列の座標に変換）
        # 重要: Y座標を反転するのは、配列と表示が上下逆の場合のみ
        array_y = grid_y  # まずは反転せず試す
        array_x = grid_x  # X軸は調整不要の可能性が高い
        
        # 反転後の座標も表示（デバッグ用）
        print(f"変換後配列座標: array_x={array_x}, array_y={array_y}")
        
        try:
            # 本質的な問題: PILやUIの座標系とNumPy配列のインデックスには2つの違いがある
            # 1. UI/画像では (x, y) の順だが、NumPy配列では [row, col] = [y, x] の順
            # 2. PILやQtのY軸は上から下、配列でも同様に上から下の行インデックスが増える
            
            # 座標変換方法:
            # 1. x,y を入れ替えずに配列にアクセス
            # 2. Y軸反転は必要ない (UIと配列の座標系が同じ向き)
            
            # 正しい配列アクセス - [grid_y, grid_x] でアクセス
            # つまり、クリック位置(16, 18)なら配列[18, 16]にアクセスする
            array_y = grid_y  # Y軸は反転しない
            array_x = grid_x  # X軸はそのまま
            
            # 座標変換と色の確認 (デバッグ用)
            print(f"クリック位置(x,y): ({grid_x}, {grid_y})")
            print(f"配列アクセス[row,col]=[y,x]: [{array_y}, {array_x}]")
            
            # 配列は[row, column]=[y, x]の順でアクセス
            try:
                # まず正しいと思われる順序でアクセス
                current_color = self.pixels_rounded_np[array_y, array_x]
                print(f"配列[{array_y}, {array_x}]の色: RGB({current_color[0]}, {current_color[1]}, {current_color[2]})")
            except IndexError:
                print(f"配列[{array_y}, {array_x}]はインデックス範囲外です")
                
            # X,Yを入れ替えてアクセスしてみる (デバッグ用)
            try:
                swapped_color = self.pixels_rounded_np[array_x, array_y]
                print(f"配列[{array_x}, {array_y}](x,y順)の色: RGB({swapped_color[0]}, {swapped_color[1]}, {swapped_color[2]})")
            except IndexError:
                print(f"配列[{array_x}, {array_y}]はインデックス範囲外です")
                
            # 選択したドットの色をQColorに変換
            rgb_color = QColor(current_color[0], current_color[1], current_color[2])
        
        except IndexError as e:
            print(f"座標変換エラー: {e}")
            return
        
        # 色選択オプションを作成
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("ドットの色を選択")
        layout = QVBoxLayout(dialog)
        
        # 透過色（透明）オプション
        transparent_check = QCheckBox("透過色（透明）に設定")
        is_transparent = tuple(current_color) == (0, 0, 0)  # 黒色（0,0,0）を透過色として扱う
        transparent_check.setChecked(is_transparent)
        layout.addWidget(transparent_check)
        
        # 色選択ダイアログボタン
        color_btn = QPushButton("色を選択")
        color_btn.clicked.connect(lambda: self.show_color_dialog(rgb_color, grid_x, grid_y, dialog, transparent_check))
        
        # 透明にするボタン
        transparent_btn = QPushButton("透明にする")
        transparent_btn.clicked.connect(lambda: self.set_transparent_color(grid_x, grid_y, dialog))
        
        # キャンセルボタン
        cancel_btn = QPushButton("キャンセル")
        cancel_btn.clicked.connect(dialog.reject)
        
        # ボタンレイアウト
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(color_btn)
        btn_layout.addWidget(transparent_btn)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
        
        # ダイアログを表示
        dialog.exec_()
        
    def show_color_dialog(self, current_color, grid_x, grid_y, parent_dialog, transparent_check):
        """色選択ダイアログを表示"""
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
                    # クリック座標(x,y)を入れ替えて[y,x]の順でアクセスする
                    array_y = grid_y  # Y軸は反転しない
                    array_x = grid_x  # X軸はそのまま
                    
                    print(f"色変更: クリック位置(x,y)=({grid_x}, {grid_y}) → 配列アクセス[y,x]=[{array_y}, {array_x}]")
                    
                    # 新しい色の確認
                    new_rgb = [new_color.red(), new_color.green(), new_color.blue()]
                    print(f"新しい色: RGB({new_rgb[0]}, {new_rgb[1]}, {new_rgb[2]})")
                    print(f"更新座標: np配列[{array_y}, {array_x}] (行,列=[y,x]順)")
                    
                    # ピクセルの色を更新 - numpy配列は[row, column] = [y, x]の順でアクセス
                    # つまり、クリック位置(16, 18)なら配列[18, 16]にアクセスする
                    self.pixels_rounded_np[array_y, array_x] = new_rgb
                    
                    # プレビューを更新（編集したピクセルデータを使用）
                    self.update_preview(custom_pixels=self.pixels_rounded_np)
                    
                    # 親ダイアログを閉じる
                    parent_dialog.accept()
                except Exception as e:
                    print(f"色設定エラー: {str(e)}")
                    parent_dialog.reject()
                
    def set_transparent_color(self, grid_x, grid_y, dialog):
        """ドットを透明（黒色=0,0,0）に設定"""
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
        # クリック座標(x,y)を入れ替えて[y,x]の順でアクセスする
        array_y = grid_y  # Y軸は反転しない
        array_x = grid_x  # X軸はそのまま
        
        print(f"透明化: クリック位置(x,y)=({grid_x}, {grid_y}) → 配列アクセス[y,x]=[{array_y}, {array_x}]")
            
        try:
            # 透過色を黒（0,0,0）として扱う
            print(f"透明化座標: np配列[{array_y}, {array_x}] (行,列=[y,x]順)")
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
            
            # 最後にクリックされた位置があれば取得
            highlight_pos = None
            if hasattr(self.preview_label, 'last_clicked_pos') and self.preview_label.last_clicked_pos is not None:
                highlight_pos = self.preview_label.last_clicked_pos
                
            # ホバー位置の取得
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
                        highlight_pos=highlight_pos,  # ハイライト位置を渡す
                        hover_pos=hover_pos  # ホバー位置を渡す
                    )
                else:
                    # 新たに画像を生成
                    preview_img = generate_preview_image(
                        self.image_path,
                        self.current_grid_size,
                        int(params["Color Step"]),
                        int(params["Top Colors"]),
                        self.zoom_factor,
                        highlight_pos=highlight_pos,  # ハイライト位置を渡す
                        hover_pos=hover_pos  # ホバー位置を渡す
                    )
            except Exception as e:
                # エラーが発生した場合、カスタムピクセルを無視して再試行
                print(f"プレビュー生成エラー: {str(e)}、カスタムピクセルなしで再試行します")
                preview_img = generate_preview_image(
                    self.image_path,
                    self.current_grid_size,
                    int(params["Color Step"]),
                    int(params["Top Colors"]),
                    self.zoom_factor
                )
            
            # カスタムピクセルを使用していない場合のみ、ピクセルデータを生成
            if custom_pixels is None:
                try:
                    # ピクセルデータを保存（後でドット編集時に使用）
                    img_resized = Image.open(self.image_path).convert("RGB").resize(
                        (self.current_grid_size, self.current_grid_size), resample=Image.NEAREST)
                    pixels = np.array(img_resized).reshape(-1, 3)
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
                        return_colors=True  # メッシュを返すように指定
                    )
                    
                    # 一時ファイルを削除
                    import os
                    os.unlink(tmp_path)
                else:
                    # 元の画像から新たにSTLを生成
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
                        wall_color=wall_color,  # 選択した壁の色を使用
                        return_colors=True  # メッシュを返すように指定
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
            cam.SetViewUp(-1, 0, 0)  # X軸負方向が上になるよう設定（反時計回りに90度回転）
            
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
        
        # Z軸正方向から真上に見る角度に設定（完全に90度）
        ax.view_init(elev=90, azim=90)  # 真上から見て反時計回りに90度回転（azimuthを90度に）
        
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
    
    def save_front_view_image(self, mesh):
        """別スレッドで正面からの画像と上面からの画像を保存"""
        try:
            timestamp = int(time.time())
            front_filename = f"stl_front_view_{timestamp}.png"
            top_filename = f"stl_top_view_{timestamp}.png"
            
            front_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), front_filename)
            top_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), top_filename)
            
            if VEDO_AVAILABLE:
                # Vedoを使って正面からの画像を保存
                self._save_front_view_vedo(mesh, front_save_path, front_filename)
                # Vedoを使って上面からの画像を保存
                self._save_top_view_vedo(mesh, top_save_path, top_filename)
            else:
                # Matplotlibで保存
                self._save_front_view_matplotlib(mesh, front_save_path, front_filename)
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