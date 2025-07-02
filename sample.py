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
import json
import os

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
    # build palette id->RGBA
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
        # Treat id==0 as transparent regardless of hex alpha
        if pid == 0:
            a = 0
        palette[pid] = (r, g, b, a)
    base_img = Image.new("RGBA", (base_w, base_h), (0, 0, 0, 0))
    for layer in sorted(data.get("layers", []), key=lambda l: l.get("id", 0)):
        res = layer.get("resolution", {})
        w_hi = int(res.get("pixelArraySize", {}).get("width", 0))
        h_hi = int(res.get("pixelArraySize", {}).get("height", 0))
        scale = float(res.get("scale", 1.0))
        pix = layer.get("pixels")
        # skip layers without pixel data
        if not pix or (pix.get("encoding") is None and pix.get("format") is None):
            continue
        # support both 'encoding' and legacy 'format' fields
        fmt_field = pix.get("encoding") or pix.get("format")
        enc = fmt_field.lower() if isinstance(fmt_field, str) else fmt_field
        # decode pixel indices
        if enc == "array":
            data_list = pix.get("data", []) or []
            # support nested row arrays or flat list
            arr = []
            if data_list and isinstance(data_list[0], list):
                # nested rows: None => transparent
                for row_data in data_list[:h_hi]:
                    row = []
                    for v in row_data:
                        row.append(v if isinstance(v, int) else None)
                    if len(row) < w_hi:
                        row += [None] * (w_hi - len(row))
                    arr.append(row[:w_hi])
                # pad missing rows
                if len(arr) < h_hi:
                    arr += [[None] * w_hi for _ in range(h_hi - len(arr))]
            else:
                # flat list: None => transparent
                flat = data_list
                for i in range(h_hi):
                    row = []
                    for j in range(w_hi):
                        idx = i * w_hi + j
                        v = flat[idx] if idx < len(flat) else None
                        row.append(v if isinstance(v, int) else None)
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
        # create high-res layer image with transparent background
        img_hi = Image.new("RGBA", (w_hi, h_hi), (0, 0, 0, 0))
        px_hi = img_hi.load()
        for y in range(h_hi):
            for x in range(w_hi):
                cid = arr[y][x] if y < len(arr) and x < len(arr[y]) else None
                if cid is None:
                    continue
                px_hi[x, y] = palette.get(cid, (0, 0, 0, 0))
        # downscale to base resolution
        if scale != 1.0 and scale > 0:
            w_lo = int(round(w_hi / scale))
            h_lo = int(round(h_hi / scale))
            img_lo = img_hi.resize((w_lo, h_lo), resample=Image.NEAREST)
        else:
            img_lo = img_hi
        # placement
        place = layer.get("placement", {})
        x0 = place.get("x", 0)
        y0 = place.get("y", 0)
        # composite layer image onto base using alpha channel
        base_img.paste(img_lo, (int(x0), int(y0)), img_lo)
    return base_img
    
def load_mrpaf_layers(path):
    """
    Load each layer from MRPAF as a separate PIL.Image (RGBA) at base resolution.
    Returns list of (layer_id, PIL.Image, scale) tuples.
    """
    data = json.load(open(path, "r", encoding="utf-8"))
    canvas_info = data.get("canvas", {})
    base_w = int(canvas_info.get("baseWidth", 0))
    base_h = int(canvas_info.get("baseHeight", 0))
    # build palette id->RGBA
    palette = {}
    for entry in data.get("palette", []):
        pid = entry.get("id")
        hexstr = entry.get("hex", "#00000000").lstrip("#")
        if len(hexstr) == 8:
            r = int(hexstr[0:2], 16); g = int(hexstr[2:4], 16)
            b = int(hexstr[4:6], 16); a = int(hexstr[6:8], 16)
        elif len(hexstr) == 6:
            r = int(hexstr[0:2], 16); g = int(hexstr[2:4], 16)
            b = int(hexstr[4:6], 16); a = 255
        else:
            r = g = b = a = 0
        # Palette index 0 reserved for transparency
        if pid == 0:
            a = 0
        palette[pid] = (r, g, b, a)
    layers_out = []
    for layer in data.get("layers", []):
        layer_id = layer.get("id")
        res = layer.get("resolution", {})
        w_hi = int(res.get("pixelArraySize", {}).get("width", 0))
        h_hi = int(res.get("pixelArraySize", {}).get("height", 0))
        scale = float(res.get("scale", 1.0))
        pix = layer.get("pixels")
        if not pix or (pix.get("encoding") is None and pix.get("format") is None):
            continue
        fmt_field = pix.get("encoding") or pix.get("format")
        enc = fmt_field.lower() if isinstance(fmt_field, str) else fmt_field
        # decode pixel indices
        if enc == "array":
            data_list = pix.get("data", []) or []
            arr = []
            if data_list and isinstance(data_list[0], list):
                # nested rows: None => transparent
                for row_data in data_list[:h_hi]:
                    row = [v if isinstance(v, int) else None for v in row_data]
                    if len(row) < w_hi:
                        row += [None] * (w_hi - len(row))
                    arr.append(row[:w_hi])
                if len(arr) < h_hi:
                    arr += [[None] * w_hi for _ in range(h_hi - len(arr))]
            else:
                # flat list: None => transparent
                flat = data_list
                for i in range(h_hi):
                    row = []
                    for j in range(w_hi):
                        idx = i * w_hi + j
                        v = flat[idx] if idx < len(flat) else None
                        row.append(v if isinstance(v, int) else None)
                    arr.append(row)
        elif enc == "rle":
            arr = decode_rle(pix.get("data", ""), w_hi, h_hi)
        elif enc == "sparse":
            dims = pix.get("dimensions", {"width": w_hi, "height": h_hi})
            w_hi = int(dims.get("width", w_hi)); h_hi = int(dims.get("height", h_hi))
            default = pix.get("defaultValue", 0)
            arr = [[default] * w_hi for _ in range(h_hi)]
            for item in pix.get("data", []):
                x = item.get("x", 0); y = item.get("y", 0); col = item.get("color", default)
                if 0 <= x < w_hi and 0 <= y < h_hi:
                    arr[y][x] = col
        else:
            raise NotImplementedError(f"Unsupported encoding: {enc}")
        # create high-res layer image with transparent background
        img_hi = Image.new("RGBA", (w_hi, h_hi), (0, 0, 0, 0))
        px_hi = img_hi.load()
        for yy in range(h_hi):
            for xx in range(w_hi):
                cid = arr[yy][xx] if yy < len(arr) and xx < len(arr[yy]) else None
                if cid is None:
                    continue
                px_hi[xx, yy] = palette.get(cid, (0, 0, 0, 0))
        # downscale to base resolution
        if scale != 1.0 and scale > 0:
            w_lo = int(round(w_hi / scale)); h_lo = int(round(h_hi / scale))
            img_lo = img_hi.resize((w_lo, h_lo), resample=Image.NEAREST)
        else:
            img_lo = img_hi
        # For standalone layer export, use the downscaled layer image directly
        layers_out.append((layer_id, img_lo, scale))
    return layers_out
    
# -------------------------------
# メイン処理関数
# -------------------------------
def generate_dot_plate_stl(img_or_path, output_path):
    # MRPAF レイヤー画像 (RGBA) の場合: color boundary walls
    if isinstance(img_or_path, Image.Image) and 'A' in img_or_path.getbands():
        img_layer = img_or_path
        arr = np.array(img_layer)
        # 透過判定と色配列
        alpha = arr[:, :, 3]
        mask = alpha > 0
        rgb = arr[:, :, :3]
        grid_h, grid_w = mask.shape
        base_blocks = []
        wall_blocks = []
        dirs = [(-1,0),(1,0),(0,1),(0,-1)]
        wall_extents = [[WALL_THICKNESS, DOT_SIZE, WALL_HEIGHT],
                        [WALL_THICKNESS, DOT_SIZE, WALL_HEIGHT],
                        [DOT_SIZE, WALL_THICKNESS, WALL_HEIGHT],
                        [DOT_SIZE, WALL_THICKNESS, WALL_HEIGHT]]
        for y in range(grid_h):
            for x in range(grid_w):
                if not mask[y, x]:
                    continue
                # ベースブロック
                x0 = x * DOT_SIZE; y0 = (grid_h-1 - y) * DOT_SIZE
                blk = box(extents=[DOT_SIZE, DOT_SIZE, BASE_HEIGHT])
                blk.apply_translation([x0+DOT_SIZE/2, y0+DOT_SIZE/2, BASE_HEIGHT/2])
                base_blocks.append(blk)
                # 色境界／透明境界で壁生成
                curr_color = tuple(rgb[y, x])
                for idx, (dx, dy) in enumerate(dirs):
                    nx, ny = x+dx, y+dy
                    make_wall = False
                    if not (0 <= nx < grid_w and 0 <= ny < grid_h):
                        # 画像外は常に壁
                        make_wall = True
                    else:
                        if not mask[ny, nx]:
                            # 透過境界
                            make_wall = True
                        else:
                            neigh_color = tuple(rgb[ny, nx])
                            if neigh_color != curr_color:
                                # 色が異なる隣接境界
                                make_wall = True
                    if not make_wall:
                        continue
                    ext = wall_extents[idx]
                    w = box(extents=ext)
                    # 壁の位置
                    if idx == 0:
                        pos = [x0+ext[0]/2, y0+DOT_SIZE/2]
                    elif idx == 1:
                        pos = [x0+DOT_SIZE-ext[0]/2, y0+DOT_SIZE/2]
                    elif idx == 2:
                        pos = [x0+DOT_SIZE/2, y0+ext[1]/2]
                    else:
                        pos = [x0+DOT_SIZE/2, y0+DOT_SIZE-ext[1]/2]
                    w.apply_translation([pos[0], pos[1], BASE_HEIGHT+WALL_HEIGHT/2])
                    wall_blocks.append(w)
        mesh = trimesh.util.concatenate(base_blocks + wall_blocks) if base_blocks or wall_blocks else None
        if mesh:
            mesh.export(output_path)
            return mesh
        return None
    # 以降: 汎用画像処理 (RGB) の場合
    if isinstance(img_or_path, str):
        ext = os.path.splitext(img_or_path)[1].lower()
        if ext == ".mrpaf":
            img = load_mrpaf(img_or_path).convert("RGB")
        else:
            img = Image.open(img_or_path).convert("RGB")
    else:
        img = img_or_path.convert("RGB")
    img_resized = img.resize((GRID_SIZE, GRID_SIZE), resample=Image.NEAREST)
    pixels = np.array(img_resized).reshape(-1, 3)
    # 色正規化～マスク生成 (従来の黒マスク)
    pixels_normalized = normalize_colors(pixels, COLOR_STEP)
    colors = [tuple(c) for c in pixels_normalized]
    top_colors = [c for c,_ in Counter(colors).most_common(TOP_COLOR_LIMIT)]
    pixels_rounded = [map_to_closest_color(c, top_colors) for c in colors]
    pixels_rounded_np = np.array(pixels_rounded, dtype=np.uint8).reshape((GRID_SIZE, GRID_SIZE, 3))
    mask = np.array([[tuple(px)!=(0,0,0) for px in row] for row in pixels_rounded_np]).astype(np.uint8)

    # ベース生成および壁生成（同色隣接内壁を省略）
    base_blocks = []
    wall_blocks = []
    # 方向と対応する壁の形状・位置インデックス
    dirs = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    # extents: left, right walls share same extents; bottom, top share
    wall_extents = [
        [WALL_THICKNESS, DOT_SIZE, WALL_HEIGHT],
        [WALL_THICKNESS, DOT_SIZE, WALL_HEIGHT],
        [DOT_SIZE, WALL_THICKNESS, WALL_HEIGHT],
        [DOT_SIZE, WALL_THICKNESS, WALL_HEIGHT],
    ]
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if not mask[y, x]:
                continue
            # 基本ピクセル位置
            x0 = x * DOT_SIZE
            y0 = (GRID_SIZE - 1 - y) * DOT_SIZE
            # ベースブロック
            block = box(extents=[DOT_SIZE, DOT_SIZE, BASE_HEIGHT])
            block.apply_translation([x0 + DOT_SIZE / 2, y0 + DOT_SIZE / 2, BASE_HEIGHT / 2])
            base_blocks.append(block)
            # 現在ピクセルの色（Numpy配列からタプルへ）
            curr_color = tuple(pixels_rounded_np[y, x])
            # 各方向に壁を生成（隣が同色なら省略）
            for idx, (dx, dy) in enumerate(dirs):
                nx, ny = x + dx, y + dy
                same = False
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    neigh = tuple(pixels_rounded_np[ny, nx])
                    if neigh == curr_color:
                        same = True
                if same:
                    continue
                # 壁の作成と配置
                wbox = box(extents=wall_extents[idx])
                # 位置計算
                if idx == 0:  # 左
                    pos = [x0 + WALL_THICKNESS / 2, y0 + DOT_SIZE / 2, BASE_HEIGHT + WALL_HEIGHT / 2]
                elif idx == 1:  # 右
                    pos = [x0 + DOT_SIZE - WALL_THICKNESS / 2, y0 + DOT_SIZE / 2, BASE_HEIGHT + WALL_HEIGHT / 2]
                elif idx == 2:  # 下
                    pos = [x0 + DOT_SIZE / 2, y0 + WALL_THICKNESS / 2, BASE_HEIGHT + WALL_HEIGHT / 2]
                else:  # 上
                    pos = [x0 + DOT_SIZE / 2, y0 + DOT_SIZE - WALL_THICKNESS / 2, BASE_HEIGHT + WALL_HEIGHT / 2]
                wbox.apply_translation(pos)
                wall_blocks.append(wbox)

    # 全体結合
    mesh = trimesh.util.concatenate(base_blocks + wall_blocks)
    mesh.export(output_path)
    return mesh

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
    parser.add_argument("-l2", "--layer2", action="store_true",
                        help="Process only layer 2 (for MRPAF files)")
    args = parser.parse_args()

    # MRPAF input: export per layer
    def parse_layer_index(lid):
        try:
            # support numeric or strings ending with number
            if isinstance(lid, int): return lid
            import re
            m = re.search(r"(\d+)$", str(lid))
            if m: return int(m.group(1))
        except:
            pass
        return None

    ext = os.path.splitext(args.input_image)[1].lower()
    if ext == ".mrpaf":
        layers = load_mrpaf_layers(args.input_image)
        if args.layer2:
            # filter for layer index 2
            layers = [(lid, img, scale) for lid, img, scale in layers if parse_layer_index(lid) == 2]
        stem = os.path.splitext(args.input_image)[0]
        meshes = []
        layer_scales = []
        layer_ids = []
        for lid, img, scale in layers:
            outp = f"{stem}_layer_{lid}.stl"
            mesh = generate_dot_plate_stl(img, outp)
            if mesh is not None:
                meshes.append(mesh)
                layer_scales.append(scale)
                layer_ids.append(lid)
            print(f"Wrote {outp}")
        # Combine layers by stacking them vertically (sumSTL)
        if meshes and not args.layer2:
            stacked_meshes = []
            layer_height = BASE_HEIGHT + WALL_HEIGHT
            # Get layer placement info from MRPAF
            data = json.load(open(args.input_image, "r", encoding="utf-8"))
            layer_placements = {}
            for layer in data.get("layers", []):
                layer_id = layer.get("id")
                placement = layer.get("placement", {})
                layer_placements[layer_id] = {
                    "x": placement.get("x", 0),
                    "y": placement.get("y", 0)
                }
            
            for i, (mesh, scale, lid) in enumerate(zip(meshes, layer_scales, layer_ids)):
                mesh_copy = mesh.copy()
                # Apply scale to the mesh (scale around center)
                if scale != 1.0:
                    mesh_copy.apply_scale(scale)
                    # Center the scaled mesh
                    bounds = mesh_copy.bounds
                    center_offset = -(bounds[1] + bounds[0]) / 2
                    mesh_copy.apply_translation([center_offset[0], center_offset[1], 0])
                
                # Apply MRPAF placement offset
                placement = layer_placements.get(lid, {"x": 0, "y": 0})
                placement_x = placement["x"] * DOT_SIZE
                placement_y = placement["y"] * DOT_SIZE
                mesh_copy.apply_translation([placement_x, placement_y, 0])
                
                # Stack each layer on top of the previous one
                if i > 0:
                    z_offset = i * layer_height
                    mesh_copy.apply_translation([0, 0, z_offset])
                stacked_meshes.append(mesh_copy)
            sum_mesh = trimesh.util.concatenate(stacked_meshes)
            sum_outp = f"{stem}_sum.stl"
            sum_mesh.export(sum_outp)
            print(f"Wrote {sum_outp}")
    else:
        generate_dot_plate_stl(args.input_image, args.output_stl)
