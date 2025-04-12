# dot_plate_generator_gui.py
# 必要ライブラリ: PyQt5, PIL, numpy, trimesh, shapely, skimage, scipy, matplotlib

import sys
import numpy as np
from PIL import Image
from collections import Counter
from scipy.spatial import distance
import trimesh
from trimesh.creation import box
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QScrollArea,
    QVBoxLayout, QHBoxLayout, QSlider, QSpinBox, QGridLayout, QDoubleSpinBox,
    QToolButton, QDialog, QGroupBox, QFrame, QSizePolicy, QToolTip, QMainWindow
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QImage
from shapely.geometry import Polygon
from skimage import measure
import matplotlib.pyplot as plt
from io import BytesIO
import threading

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
                           color_step, top_color_limit, out_thickness=0.1, return_colors=False):
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
                    # 壁の色情報も追加（ベースと同じ色）
                    color_mapping[len(base_blocks) + len(wall_blocks)] = {
                        'type': 'wall', 
                        'color': pixel_color,
                        'position': [x, y], 
                        'wall_index': i
                    }
                    wall_blocks.append(wbox)
    
    # メッシュを作成
    mesh = trimesh.util.concatenate(base_blocks + wall_blocks)
    
    # 色情報を設定
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'face_colors'):
        mesh.visual.face_colors = np.ones((len(mesh.faces), 4), dtype=np.uint8) * 200  # デフォルト色（薄いグレー）
        
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
        
        # 壁ブロックはやや暗くする
        for i, block in enumerate(wall_blocks):
            idx = i + len(base_blocks)
            if idx in color_mapping:
                color_info = color_mapping[idx]
                r, g, b = color_info['color']
                
                # 壁は少し暗くする
                r = int(r * 0.7)
                g = int(g * 0.7)
                b = int(b * 0.7)
                
                color = np.array([r, g, b, 255], dtype=np.uint8)
                
                # このブロックの面数
                num_faces = len(block.faces)
                
                # 該当する面すべてに色を設定
                mesh.visual.face_colors[face_index:face_index + num_faces] = color
                
                # 次のブロックの最初の面インデックス
                face_index += num_faces
    
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
# STLプレビューダイアログ
# -------------------------------
class STLPreviewDialog(QDialog):
    def __init__(self, stl_path, parent=None, mesh=None):
        super().__init__(parent)
        self.setWindowTitle("STLプレビュー")
        self.setMinimumSize(600, 600)
        # ダイアログのフラグ設定（×ボタンで閉じられるようにする）
        from PyQt5.QtCore import Qt
        self.setWindowFlags(self.windowFlags() | Qt.WindowSystemMenuHint | Qt.WindowCloseButtonHint)
        
        layout = QVBoxLayout()
        
        self.preview_label = QLabel("STLモデルの読み込み中...")
        self.preview_label.setAlignment(Qt.AlignCenter)
        
        # 視点コントロール
        view_group = QGroupBox("視点コントロール")
        view_layout = QHBoxLayout()
        
        self.azim_slider = QSlider(Qt.Horizontal)
        self.azim_slider.setMinimum(0)
        self.azim_slider.setMaximum(360)
        self.azim_slider.setValue(45)
        self.azim_slider.setTickPosition(QSlider.TicksBelow)
        self.azim_slider.setTickInterval(30)
        self.azim_slider.valueChanged.connect(self.update_view)
        
        self.elev_slider = QSlider(Qt.Horizontal)
        self.elev_slider.setMinimum(-90)
        self.elev_slider.setMaximum(90)
        self.elev_slider.setValue(20)
        self.elev_slider.setTickPosition(QSlider.TicksBelow)
        self.elev_slider.setTickInterval(15)
        self.elev_slider.valueChanged.connect(self.update_view)
        
        view_layout.addWidget(QLabel("水平角度:"))
        view_layout.addWidget(self.azim_slider)
        view_layout.addWidget(QLabel("垂直角度:"))
        view_layout.addWidget(self.elev_slider)
        
        view_group.setLayout(view_layout)
        
        close_button = QPushButton("閉じる")
        close_button.clicked.connect(self.accept)
        
        layout.addWidget(self.preview_label)
        layout.addWidget(view_group)
        layout.addWidget(close_button)
        
        self.setLayout(layout)
        
        # メッシュがすでに提供されている場合は直接使用
        self.mesh = mesh
        self.stl_path = stl_path
        
        # 別スレッドでSTLを読み込み
        threading.Thread(target=self.load_stl_preview).start()
    
    def update_view(self):
        # メッシュがロードされていない場合は何もしない
        if not hasattr(self, 'mesh') or self.mesh is None:
            return
            
        # スライダーから値を取得
        azim = self.azim_slider.value()
        elev = self.elev_slider.value()
        
        # メインスレッドでのGUI更新を要求
        from PyQt5.QtCore import QEvent
        
        class UpdateViewEvent(QEvent):
            def __init__(self, mesh, azim, elev):
                super().__init__(QEvent.Type(QEvent.User + 3))
                self.mesh = mesh
                self.azim = azim
                self.elev = elev
        
        # メインスレッドでのGUI更新を要求
        QApplication.instance().postEvent(self, UpdateViewEvent(self.mesh, azim, elev))
    
    def load_stl_preview(self):
        if self.mesh is None:
            try:
                # メッシュファイルデータを読み込む（プロットしない）
                self.mesh = trimesh.load(self.stl_path)
                
                # ここではメッシュデータをメインスレッドに渡すだけ
                from PyQt5.QtCore import QEvent
                
                class MeshLoadedEvent(QEvent):
                    def __init__(self, mesh):
                        super().__init__(QEvent.Type(QEvent.User + 1))
                        self.mesh = mesh
                
                # メインスレッドでのGUI更新を要求
                QApplication.instance().postEvent(self, MeshLoadedEvent(self.mesh))
                
            except Exception as e:
                # エラー表示もメインスレッドで
                from PyQt5.QtCore import QEvent
                
                class ErrorEvent(QEvent):
                    def __init__(self, error_msg):
                        super().__init__(QEvent.Type(QEvent.User + 2))
                        self.error_msg = error_msg
                
                QApplication.instance().postEvent(self, ErrorEvent(str(e)))
        else:
            # すでにメッシュがある場合は直接ビュー更新イベントを発行
            self.update_view()
    
    def closeEvent(self, event):
        # ダイアログが閉じられるときの処理
        # スレッドの終了などクリーンアップが必要な場合はここで行う
        event.accept()  # イベントを受け入れて、ウィンドウを閉じる
    
    def event(self, event):
        from PyQt5.QtCore import QEvent
        
        # カスタムイベントの処理
        if event.type() == QEvent.User + 1:  # メッシュロードイベント
            # メインスレッドでMatplotlibを使ってメッシュを描画
            try:
                print("STLメッシュ描画開始")
                self.mesh = event.mesh
                
                # 初期ビューの描画
                azim = self.azim_slider.value()
                elev = self.elev_slider.value()
                self.render_mesh(self.mesh, elev, azim)
                
                print("STLメッシュ描画完了")
            except Exception as e:
                print(f"STLプレビュー描画エラー: {str(e)}")
                self.preview_label.setText(f"STLプレビュー描画失敗: {str(e)}")
            return True
        elif event.type() == QEvent.User + 2:  # エラーイベント
            self.preview_label.setText(f"STLプレビュー失敗: {event.error_msg}")
            return True
        elif event.type() == QEvent.User + 3:  # ビュー更新イベント
            # ビューアングル変更時の描画更新
            try:
                mesh = event.mesh
                azim = event.azim
                elev = event.elev
                self.render_mesh(mesh, elev, azim)
            except Exception as e:
                print(f"ビュー更新エラー: {str(e)}")
            return True
            
        return super().event(event)
        
    def render_mesh(self, mesh, elev, azim):
        """メッシュを指定のビューアングルで描画"""
        try:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=elev, azim=azim)
            
            # メッシュを表示（色情報がある場合は色を付けて表示）
            mesh.show(ax=ax)
            
            ax.set_axis_off()
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close(fig)
            buf.seek(0)
            
            qimg = QImage()
            qimg.loadFromData(buf.getvalue())
            pixmap = QPixmap.fromImage(qimg)
            self.preview_label.setPixmap(pixmap)
        except Exception as e:
            print(f"メッシュ描画エラー: {str(e)}")
            self.preview_label.setText(f"メッシュ描画失敗: {str(e)}")

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
        self.preview_dialog = None  # STLプレビューダイアログの参照
    
    def show_parameter_help(self, parameter_name):
        dialog = ParameterHelpDialog(parameter_name, self)
        dialog.exec_()
    
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
                
                # メッシュと色情報を取得
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
                    return_colors=True  # メッシュを返すように指定
                )
                
                self.input_label.setText(f"{out_path} にカラーSTLをエクスポートしました")
                
                # 別ウィンドウでプレビューを表示（モードレス）
                # メッシュオブジェクトを直接渡す
                if isinstance(mesh, tuple) and len(mesh) > 0:
                    # return_colors=Trueの場合、最初の要素がメッシュ
                    preview_mesh = mesh[0]
                else:
                    # 単一のメッシュオブジェクトの場合
                    preview_mesh = mesh
                    
                preview_dialog = STLPreviewDialog(out_path, self, mesh=preview_mesh)
                preview_dialog.show()  # モードレスダイアログとして表示
                # ダイアログがガベージコレクションされないように参照を保持
                self.preview_dialog = preview_dialog
                
            except Exception as e:
                print(f"STL生成エラー: {str(e)}")
                import traceback
                traceback.print_exc()
                self.input_label.setText(f"STL生成エラー: {str(e)}")

# -------------------------------
# 実行エントリポイント
# -------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DotPlateApp()
    window.show()
    sys.exit(app.exec_())