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
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QSlider, QSpinBox, QGridLayout, QDoubleSpinBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from shapely.geometry import Polygon
from skimage import measure
import matplotlib.pyplot as plt
from io import BytesIO

# -------------------------------
# 補助関数
# -------------------------------
def normalize_colors(pixels, step):
    return (pixels // step) * step

def map_to_closest_color(pixel, palette):
    return min(palette, key=lambda c: distance.euclidean(pixel, c))

def generate_preview_image(image_path, grid_size, color_step, top_color_limit):
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((grid_size, grid_size), resample=Image.NEAREST)
    pixels = np.array(img_resized).reshape(-1, 3)
    pixels_normalized = normalize_colors(pixels, color_step)
    colors = [tuple(c) for c in pixels_normalized]
    color_counts = Counter(colors)
    top_colors = [c for c, _ in color_counts.most_common(top_color_limit)]
    pixels_rounded = [map_to_closest_color(c, top_colors) for c in colors]
    img_preview = Image.fromarray(np.array(pixels_rounded, dtype=np.uint8).reshape((grid_size, grid_size, 3)), mode="RGB")
    img_preview = img_preview.resize((grid_size * 10, grid_size * 10), resample=Image.NEAREST)
    return img_preview

# -------------------------------
# モデル生成関数
# -------------------------------
def generate_dot_plate_stl(image_path, output_path, grid_size, dot_size,
                           wall_thickness, wall_height, base_height,
                           color_step, top_color_limit):
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
    for y in range(grid_size):
        for x in range(grid_size):
            if mask[y, x]:
                x0 = x * dot_size
                y0 = (grid_size - 1 - y) * dot_size
                block = box(extents=[dot_size, dot_size, base_height])
                block.apply_translation([x0 + dot_size / 2, y0 + dot_size / 2, base_height / 2])
                base_blocks.append(block)
                wall_boxes = [
                    box(extents=[wall_thickness, dot_size, wall_height]),
                    box(extents=[wall_thickness, dot_size, wall_height]),
                    box(extents=[dot_size, wall_thickness, wall_height]),
                    box(extents=[dot_size, wall_thickness, wall_height]),
                ]
                positions = [
                    [x0 + wall_thickness / 2, y0 + dot_size / 2, base_height + wall_height / 2],
                    [x0 + dot_size - wall_thickness / 2, y0 + dot_size / 2, base_height + wall_height / 2],
                    [x0 + dot_size / 2, y0 + wall_thickness / 2, base_height + wall_height / 2],
                    [x0 + dot_size / 2, y0 + dot_size - wall_thickness / 2, base_height + wall_height / 2],
                ]
                for wbox, pos in zip(wall_boxes, positions):
                    wbox.apply_translation(pos)
                    wall_blocks.append(wbox)
    mesh = trimesh.util.concatenate(base_blocks + wall_blocks)
    mesh.export(output_path)

# -------------------------------
# GUI クラス
# -------------------------------
class DotPlateApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dot Plate Generator")
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        self.layout = QVBoxLayout()
        control_layout = QVBoxLayout()
        self.mesh_view = QLabel("3D STL preview will appear here")
        self.mesh_view.setAlignment(Qt.AlignCenter)
        self.mesh_view.setFixedHeight(240)
        self.setLayout(self.layout)

        self.input_label = QLabel("No image selected")
        self.preview_label = QLabel("Preview will appear here")
        self.preview_label.setFixedHeight(320)
        self.preview_label.setAlignment(Qt.AlignCenter)

        self.select_button = QPushButton("Select Image")
        self.select_button.clicked.connect(self.select_image)

        self.export_button = QPushButton("Export STL")
        self.export_button.clicked.connect(self.export_stl)

        self.param_grid = QGridLayout()
        self.controls = {}
        for i, (label, default, minv, maxv) in enumerate([
            ("Grid Size", 32, 8, 64),
            ("Dot Size", 1.0, 0.2, 5.0),
            ("Wall Thickness", 0.2, 0.0, 5.0),
            ("Wall Height", 1.0, 0.0, 5.0),
            ("Base Height", 2.0, 0.0, 5.0),
            ("Color Step", 8, 1, 64),
            ("Top Colors", 36, 1, 64)
        ]):
            label_widget = QLabel(label)
            spin = QSpinBox() if isinstance(default, int) else QDoubleSpinBox()
            spin.setMinimum(minv)
            spin.setMaximum(maxv)
            spin.setValue(default)
            spin.valueChanged.connect(self.update_preview)
            self.param_grid.addWidget(label_widget, i, 0)
            self.param_grid.addWidget(spin, i, 1)
            self.controls[label] = spin

        control_layout.addWidget(self.input_label)
        control_layout.addWidget(self.select_button)
        self.layout.addWidget(self.preview_label)
        
        control_layout.addLayout(self.param_grid)
        self.layout.addWidget(self.mesh_view)
        control_layout.addWidget(self.export_button)

        main_layout.addLayout(self.layout)
        main_layout.addLayout(control_layout)

        self.image_path = None

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.image_path = path
            self.input_label.setText(path)
            self.update_preview()

    def update_preview(self):
        if not self.image_path:
            return
        params = {key: spin.value() for key, spin in self.controls.items()}
        img = generate_preview_image(
            self.image_path,
            int(params["Grid Size"]),
            int(params["Color Step"]),
            int(params["Top Colors"])
        )
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        qimg = QImage()
        qimg.loadFromData(buffer.getvalue())
        pixmap = QPixmap.fromImage(qimg)
        self.preview_label.setPixmap(pixmap)

    def export_stl(self):
        if not self.image_path:
            self.input_label.setText("No image selected")
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "Save STL", "dot_plate.stl", "STL Files (*.stl)")
        if out_path:
            params = {key: spin.value() for key, spin in self.controls.items()}
            generate_dot_plate_stl(
                self.image_path,
                out_path,
                int(params["Grid Size"]),
                float(params["Dot Size"]),
                float(params["Wall Thickness"]),
                float(params["Wall Height"]),
                float(params["Base Height"]),
                int(params["Color Step"]),
                int(params["Top Colors"]),
            )
            self.input_label.setText(f"Exported to {out_path}")
            try:
                mesh = trimesh.load(out_path)
                fig = plt.figure(figsize=(4, 4))
                ax = fig.add_subplot(111, projection='3d')
                ax.view_init(elev=20, azim=45)
                mesh.show(ax=ax)
                ax.set_axis_off()
                buf = BytesIO()
                plt.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                qimg = QImage()
                qimg.loadFromData(buf.getvalue())
                self.mesh_view.setPixmap(QPixmap.fromImage(qimg))
            except Exception as e:
                self.mesh_view.setText(f"STL preview failed: {e}")

# -------------------------------
# 実行エントリポイント
# -------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DotPlateApp()
    window.show()
    sys.exit(app.exec_())
