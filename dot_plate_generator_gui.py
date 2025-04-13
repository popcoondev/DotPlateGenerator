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

def generate_preview_image(image_path, grid_size, color_step, top_color_limit, zoom_factor=10):
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((grid_size, grid_size), resample=Image.NEAREST)
    pixels = np.array(img_resized).reshape(-1, 3)
    pixels_normalized = normalize_colors(pixels, color_step)
    colors = [tuple(c) for c in pixels_normalized]
    color_counts = Counter(colors)
    top_colors = [c for c, _ in color_counts.most_common(top_color_limit)]
    pixels_rounded = [map_to_closest_color(c, top_colors) for c in colors]
    img_preview = Image.fromarray(np.array(pixels_rounded, dtype=np.uint8).reshape((grid_size, grid_size, 3)), mode="RGB")
    img_preview = img_preview.resize((grid_size * zoom_factor, grid_size * zoom_factor), resample=Image.NEAREST)
    return img_preview

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
                
                x0 = x * dot_size - out_thickness
                y0 = (grid_size - 1 - y) * dot_size - out_thickness
                
                # ベースブロックを外壁厚み分大きくする
                block = box(extents=[dot_size + out_thickness * 2, dot_size + out_thickness * 2, base_height])
                block.apply_translation([x0 + (dot_size + out_thickness * 2) / 2, y0 + (dot_size + out_thickness * 2) / 2, base_height / 2])
                
                # 色情報を追加
                color_mapping[len(base_blocks)] = {
                    'type': 'base', 
                    'color': pixel_color, 
                    'position': [x, y]
                }
                
                base_blocks.append(block)
                
                # ドットの区切り壁とベースの輪郭壁を分けて処理
                # 左・右の壁
                lr_wall_boxes = [
                    box(extents=[wall_thickness, dot_size, wall_height]),
                    box(extents=[wall_thickness, dot_size, wall_height]),
                ]
                # 上・下の壁
                tb_wall_boxes = [
                    box(extents=[dot_size, wall_thickness, wall_height]),
                    box(extents=[dot_size, wall_thickness, wall_height]),
                ]
                # 外壁壁ボックス（ベースプレートの輪郭にあたる壁のみ）
                wall_boxes = []
                # X方向（左右）端のチェック
                if x == 0 or not mask[y, x-1]:  # 左端または左が空白
                    wall_boxes.append(box(extents=[wall_thickness, dot_size + out_thickness * 2, wall_height]))
                else:
                    wall_boxes.append(lr_wall_boxes[0])
                    
                if x == grid_size - 1 or not mask[y, x+1]:  # 右端または右が空白
                    wall_boxes.append(box(extents=[wall_thickness, dot_size + out_thickness * 2, wall_height]))
                else:
                    wall_boxes.append(lr_wall_boxes[1])
                
                # Y方向（上下）端のチェック
                if y == 0 or not mask[y-1, x]:  # 上端または上が空白
                    wall_boxes.append(box(extents=[dot_size + out_thickness * 2, wall_thickness, wall_height]))
                else:
                    wall_boxes.append(tb_wall_boxes[0])
                    
                if y == grid_size - 1 or not mask[y+1, x]:  # 下端または下が空白
                    wall_boxes.append(box(extents=[dot_size + out_thickness * 2, wall_thickness, wall_height]))
                else:
                    wall_boxes.append(tb_wall_boxes[1])
                
                # 壁の位置を設定する
                positions = []
                # X方向（左右）壁の位置
                if x == 0 or not mask[y, x-1]:  # 左端または左が空白
                    positions.append([x0 + wall_thickness / 2, y0 + (dot_size + out_thickness * 2) / 2, base_height + wall_height / 2])
                else:
                    # 通常の左壁
                    positions.append([x0 + out_thickness + wall_thickness / 2, y0 + out_thickness + dot_size / 2, base_height + wall_height / 2])
                
                if x == grid_size - 1 or not mask[y, x+1]:  # 右端または右が空白
                    positions.append([x0 + (dot_size + out_thickness * 2) - wall_thickness / 2, y0 + (dot_size + out_thickness * 2) / 2, base_height + wall_height / 2])
                else:
                    # 通常の右壁
                    positions.append([x0 + out_thickness + dot_size - wall_thickness / 2, y0 + out_thickness + dot_size / 2, base_height + wall_height / 2])
                
                # Y方向（上下）壁の位置
                if y == 0 or not mask[y-1, x]:  # 上端または上が空白
                    positions.append([x0 + (dot_size + out_thickness * 2) / 2, y0 + wall_thickness / 2, base_height + wall_height / 2])
                else:
                    # 通常の上壁
                    positions.append([x0 + out_thickness + dot_size / 2, y0 + out_thickness + wall_thickness / 2, base_height + wall_height / 2])
                
                if y == grid_size - 1 or not mask[y+1, x]:  # 下端または下が空白
                    positions.append([x0 + (dot_size + out_thickness * 2) / 2, y0 + (dot_size + out_thickness * 2) - wall_thickness / 2, base_height + wall_height / 2])
                else:
                    # 通常の下壁
                    positions.append([x0 + out_thickness + dot_size / 2, y0 + out_thickness + dot_size - wall_thickness / 2, base_height + wall_height / 2])
                
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
        
        # メインウィジェットとレイアウト
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # 左パネル（プレビュー）
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 画像プレビュー領域
        preview_group = QGroupBox("ドットプレビュー")
        preview_layout = QVBoxLayout()
        
        # プレビュー表示のためのスクロールエリア
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setMinimumHeight(400)
        self.preview_scroll.setMinimumWidth(400)
        
        self.preview_label = QLabel("プレビューが表示されます")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.preview_scroll.setWidget(self.preview_label)
        
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
        
        preview_layout.addWidget(self.preview_scroll)
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
            ("Dot Size", 1.0, 0.2, 5.0),
            ("Wall Thickness", 0.2, 0.0, 5.0),
            ("Wall Height", 1.0, 0.0, 5.0),
            ("Base Height", 2.0, 0.0, 5.0),
            ("Out Thickness", 0.1, 0.0, 5.0),
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
        path, _ = QFileDialog.getOpenFileName(self, "画像を開く", "", "画像ファイル (*.png *.jpg *.jpeg)")
        if path:
            self.image_path = path
            self.input_label.setText(path)
            self.update_preview()
    
    def update_preview(self):
        if not self.image_path:
            return
        
        self.zoom_factor = self.zoom_slider.value()
        params = {key: spin.value() for key, spin in self.controls.items()}
        
        img = generate_preview_image(
            self.image_path,
            int(params["Grid Size"]),
            int(params["Color Step"]),
            int(params["Top Colors"]),
            self.zoom_factor
        )
        
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        qimg = QImage()
        qimg.loadFromData(buffer.getvalue())
        pixmap = QPixmap.fromImage(qimg)
        
        self.preview_label.setPixmap(pixmap)
        self.preview_label.adjustSize()
    
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
                
                # メッシュ生成（メッシュも返すように指定）
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
            
            # カメラをZ軸正方向から少し斜めに配置
            cam = plt.camera
            # 真上からではなく、少し斜めから見る位置に設定
            cam.SetPosition(center[0] + max_length*0.5, center[1] + max_length*0.5, z_pos)
            cam.SetFocalPoint(center[0], center[1], center[2])
            cam.SetViewUp(0, 1, 0)  # Y軸が上になるよう設定
            
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
        
        # Z軸正方向から少し斜めに見る角度に設定（真上90度ではなく80度程度）
        ax.view_init(elev=80, azim=30)  # 真上から少し斜めに
        
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
            cam.SetViewUp(0, 1, 0)  # Y軸が上になるよう設定
            
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
        top_ax.view_init(elev=90, azim=0)  # 真上から見下ろす角度
        
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