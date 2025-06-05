#!/usr/bin/env python3

"""
更新されたレジンフレーム機能のテスト
- 天井だけが空いた箱型フレーム
- レイヤー数に基づく動的高さ計算
- 1mm壁厚
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dot_plate_generator_gui import generate_color_separated_layers_stl
import numpy as np

def test_updated_resin_frame():
    """更新されたレジンフレーム機能をテスト"""
    
    print("=== 更新されたレジンフレーム機能テスト ===")
    
    # テストケース1: 3レイヤーの場合
    print("\n--- テストケース1: 3レイヤー ---")
    pixels_3_layer = np.array([
        [(255, 0, 0), (0, 255, 0)],    # 赤, 緑
        [(0, 0, 255), (255, 255, 0)]   # 青, 黄
    ])
    
    generate_color_separated_layers_stl(
        pixels_rounded_np=pixels_3_layer,
        output_base_path="test_updated_3layer",
        grid_size=2,
        dot_size=2.0,
        wall_thickness=1.0,
        wall_height=1.0,
        base_height=0.1,
        out_thickness=1.0,
        layer_color_order=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    )
    
    # テストケース2: 1レイヤーの場合  
    print("\n--- テストケース2: 1レイヤー ---")
    pixels_1_layer = np.array([[(255, 0, 0)]])
    
    generate_color_separated_layers_stl(
        pixels_rounded_np=pixels_1_layer,
        output_base_path="test_updated_1layer",
        grid_size=1,
        dot_size=2.0,
        wall_thickness=1.0,
        wall_height=1.0,
        base_height=0.1,
        out_thickness=1.0,
        layer_color_order=[(255, 0, 0)]
    )
    
    # テストケース3: 5レイヤーの場合
    print("\n--- テストケース3: 5レイヤー ---")
    pixels_5_layer = np.array([
        [(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        [(255, 255, 0), (255, 0, 255), (128, 128, 128)],
        [(64, 64, 64), (192, 192, 192), (255, 128, 0)]
    ])
    
    generate_color_separated_layers_stl(
        pixels_rounded_np=pixels_5_layer,
        output_base_path="test_updated_5layer",
        grid_size=3,
        dot_size=2.0,
        wall_thickness=1.0,
        wall_height=1.0,
        base_height=0.1,
        out_thickness=1.0,
        layer_color_order=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    )
    
    print("\n=== 更新フレーム機能テスト完了 ===")

def analyze_updated_frames():
    """生成されたフレームを解析"""
    
    print("\n=== フレーム解析 ===")
    
    import trimesh
    
    test_cases = [
        ("test_updated_1layer_resin_frame.stl", 1),
        ("test_updated_3layer_resin_frame.stl", 4), 
        ("test_updated_5layer_resin_frame.stl", 5)
    ]
    
    for filename, expected_layers in test_cases:
        if os.path.exists(filename):
            try:
                frame = trimesh.load(filename)
                print(f"\n{filename}:")
                print(f"  レイヤー数: {expected_layers}")
                print(f"  頂点数: {len(frame.vertices)}")
                print(f"  面数: {len(frame.faces)}")
                print(f"  境界ボックス: {frame.bounds}")
                print(f"  体積: {frame.volume:.2f} mm³")
                print(f"  高さ: {frame.bounds[1][2] - frame.bounds[0][2]:.2f} mm")
                
                # 期待される高さ計算
                layer_height = 0.1 + 1.0 + 2.0  # base + wall + margin
                expected_height = expected_layers * layer_height + 5.0  # frame margin
                print(f"  期待高さ: {expected_height:.2f} mm")
                
                actual_height = frame.bounds[1][2] - frame.bounds[0][2]
                height_diff = abs(actual_height - expected_height)
                print(f"  高さ差: {height_diff:.2f} mm")
                
                if height_diff < 0.1:
                    print("  ✅ 高さ計算正常")
                else:
                    print("  ❌ 高さ計算異常")
                    
            except Exception as e:
                print(f"  エラー: {str(e)}")
        else:
            print(f"{filename}: ファイルが見つかりません")

if __name__ == "__main__":
    test_updated_resin_frame()
    analyze_updated_frames()