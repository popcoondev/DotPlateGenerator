# dot_plate_generator.py
# PIL, numpy, shapely, trimesh, matplotlib などが必要

import numpy as np
from PIL import Image
from collections import Counter
from scipy.spatial import distance
import trimesh
from trimesh.creation import box
from shapely.geometry import Polygon
from shapely.ops import unary_union
from skimage import measure

# -------------------------------
# ユーザー設定パラメータ
# -------------------------------
DOT_SIZE = 2.0            # 1ドットの一辺のサイズ (mm)
WALL_THICKNESS = 0.2      # 凹み壁の厚み (mm)
WALL_HEIGHT = 0.4         # 凹み壁の高さ (mm)
BASE_HEIGHT = 1.0         # ベースプレートの厚み (mm)
GRID_SIZE = 32            # ドット数（縦横）
COLOR_STEP = 8            # 色正規化のステップ
TOP_COLOR_LIMIT = 36      # 使用する上位色数（近似）

# -------------------------------
# 補助関数群
# -------------------------------
def normalize_colors(pixels, step):
    return (pixels // step) * step

def map_to_closest_color(pixel, palette):
    return min(palette, key=lambda c: distance.euclidean(pixel, c))

# -------------------------------
# メイン処理関数
# -------------------------------
def generate_dot_plate_stl(image_path, output_path):
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((GRID_SIZE, GRID_SIZE), resample=Image.NEAREST)
    pixels = np.array(img_resized).reshape(-1, 3)

    # 色正規化
    pixels_normalized = normalize_colors(pixels, COLOR_STEP)
    colors = [tuple(c) for c in pixels_normalized]
    color_counts = Counter(colors)

    # 頻出色への丸め
    top_colors = [c for c, _ in color_counts.most_common(TOP_COLOR_LIMIT)]
    pixels_rounded = [map_to_closest_color(c, top_colors) for c in colors]
    pixels_rounded_np = np.array(pixels_rounded, dtype=np.uint8).reshape((GRID_SIZE, GRID_SIZE, 3))

    # マスク生成（黒以外）
    mask = np.array([[tuple(px) != (0, 0, 0) for px in row] for row in pixels_rounded_np]).astype(np.uint8)

    # ベース生成（簡易カットアウト）
    base_blocks = []
    wall_blocks = []
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if mask[y, x]:
                x0 = x * DOT_SIZE
                y0 = (GRID_SIZE - 1 - y) * DOT_SIZE

                # ベースプレート部分
                block = box(extents=[DOT_SIZE, DOT_SIZE, BASE_HEIGHT])
                block.apply_translation([x0 + DOT_SIZE / 2, y0 + DOT_SIZE / 2, BASE_HEIGHT / 2])
                base_blocks.append(block)

                # 壁生成
                wall_boxes = [
                    box(extents=[WALL_THICKNESS, DOT_SIZE, WALL_HEIGHT]),
                    box(extents=[WALL_THICKNESS, DOT_SIZE, WALL_HEIGHT]),
                    box(extents=[DOT_SIZE, WALL_THICKNESS, WALL_HEIGHT]),
                    box(extents=[DOT_SIZE, WALL_THICKNESS, WALL_HEIGHT]),
                ]
                positions = [
                    [x0 + WALL_THICKNESS / 2, y0 + DOT_SIZE / 2, BASE_HEIGHT + WALL_HEIGHT / 2],
                    [x0 + DOT_SIZE - WALL_THICKNESS / 2, y0 + DOT_SIZE / 2, BASE_HEIGHT + WALL_HEIGHT / 2],
                    [x0 + DOT_SIZE / 2, y0 + WALL_THICKNESS / 2, BASE_HEIGHT + WALL_HEIGHT / 2],
                    [x0 + DOT_SIZE / 2, y0 + DOT_SIZE - WALL_THICKNESS / 2, BASE_HEIGHT + WALL_HEIGHT / 2],
                ]
                for wbox, pos in zip(wall_boxes, positions):
                    wbox.apply_translation(pos)
                    wall_blocks.append(wbox)

    # 全体結合
    mesh = trimesh.util.concatenate(base_blocks + wall_blocks)
    mesh.export(output_path)

# -------------------------------
# 使用例（必要に応じて修正）
# -------------------------------
# generate_dot_plate_stl("input_image.png", "output_model.stl")

# ここにmain関数を実装
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a dot plate STL file from an image.")
    parser.add_argument("input_image", type=str, help="Path to the input image file.")
    parser.add_argument("output_stl", type=str, help="Path to the output STL file.")
    args = parser.parse_args()

    generate_dot_plate_stl(args.input_image, args.output_stl)
