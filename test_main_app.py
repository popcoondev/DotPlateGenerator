#!/usr/bin/env python3
"""
test_main_app.py - メインアプリのレイヤースタックモードをテスト
"""

import numpy as np
import sys
import os

# パスを追加してメインモジュールをインポート
sys.path.append('/Users/mn/development/DotPlateGenerator')
from dot_plate_generator_gui import generate_layer_stack_stl

def test_layer_stack_mode():
    """メインアプリのレイヤースタックモードをテスト"""
    
    # テストデータ（3x3のピクセル配列）
    # building.pyと同じデータを使用
    pixels_data = [
        [(255, 0, 0), (0, 255, 0), (255, 0, 0)],    # 赤, 緑, 赤
        [(0, 255, 0), (0, 0, 255), (0, 255, 0)],    # 緑, 青, 緑  
        [(0, 0, 255), (0, 0, 255), (255, 0, 0)]     # 青, 青, 赤
    ]
    
    # NumPy配列に変換
    pixels_rounded_np = np.array(pixels_data, dtype=np.uint8)
    
    print("テストデータ:")
    for y, row in enumerate(pixels_data):
        print(f"行{y}: {row}")
    
    # レイヤー色順序（値の小さい順: 赤=1, 緑=2, 青=3）
    layer_color_order = [
        (255, 0, 0),  # 赤 (レイヤー1)
        (0, 255, 0),  # 緑 (レイヤー2)
        (0, 0, 255),  # 青 (レイヤー3)
    ]
    
    # パラメータ（building.pyと同じ）
    grid_size = 3
    dot_size = 1
    wall_thickness = 0.5
    out_thickness = 0.5
    base_height = 1
    wall_height = 2
    layer_heights = None  # 未使用
    
    output_base_path = "/Users/mn/development/DotPlateGenerator/test_main_layer"
    
    print(f"\nパラメータ:")
    print(f"grid_size: {grid_size}")
    print(f"dot_size: {dot_size}")
    print(f"wall_thickness: {wall_thickness}")
    print(f"out_thickness: {out_thickness}")
    print(f"base_height: {base_height}")
    print(f"wall_height: {wall_height}")
    
    print(f"\nレイヤー色順序:")
    for i, color in enumerate(layer_color_order):
        print(f"レイヤー{i+1}: RGB{color}")
    
    print(f"\n=== メインアプリのレイヤースタックモード実行 ===")
    
    try:
        # レイヤースタックSTL生成
        generated_meshes = generate_layer_stack_stl(
            pixels_rounded_np=pixels_rounded_np,
            output_base_path=output_base_path,
            grid_size=grid_size,
            dot_size=dot_size,
            wall_thickness=wall_thickness,
            wall_height=wall_height,
            base_height=base_height,
            out_thickness=out_thickness,
            layer_color_order=layer_color_order,
            layer_heights=layer_heights
        )
        
        print(f"\n=== 実行完了 ===")
        print(f"生成されたメッシュ数: {len(generated_meshes)}")
        
        # 生成されたファイルの確認
        import glob
        generated_files = glob.glob(f"{output_base_path}*.stl")
        print(f"\n生成されたSTLファイル:")
        for filepath in sorted(generated_files):
            filename = os.path.basename(filepath)
            file_size = os.path.getsize(filepath)
            print(f"  {filename} ({file_size:,} bytes)")
            
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_layer_stack_mode()