#!/usr/bin/env python3

"""
溝（くぼみ）ガイドライン付きフレーム機能のテスト
- カップラーメン風の溝状ガイドライン
- ブール演算による溝作成
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dot_plate_generator_gui import generate_color_separated_layers_stl
import numpy as np

def test_groove_frame():
    """溝ガイドライン付きフレーム機能をテスト"""
    
    print("=== 溝ガイドライン付きフレーム機能テスト ===")
    
    # テストケース1: 2レイヤーの溝ガイドライン
    print("\n--- テストケース1: 2レイヤー溝ガイドライン ---")
    pixels_2_layer = np.array([
        [(255, 0, 0), (0, 255, 0)]    # 赤, 緑
    ])
    
    generate_color_separated_layers_stl(
        pixels_rounded_np=pixels_2_layer,
        output_base_path="test_groove_2layer",
        grid_size=1,
        dot_size=2.0,
        wall_thickness=1.0,
        wall_height=1.0,
        base_height=0.1,
        out_thickness=1.0,
        layer_color_order=[(255, 0, 0), (0, 255, 0)]
    )
    
    # テストケース2: 4レイヤーの溝ガイドライン  
    print("\n--- テストケース2: 4レイヤー溝ガイドライン ---")
    pixels_4_layer = np.array([
        [(255, 0, 0), (0, 255, 0)],
        [(0, 0, 255), (255, 255, 0)]
    ])
    
    generate_color_separated_layers_stl(
        pixels_rounded_np=pixels_4_layer,
        output_base_path="test_groove_4layer",
        grid_size=2,
        dot_size=2.0,
        wall_thickness=1.0,
        wall_height=1.0,
        base_height=0.1,
        out_thickness=1.0,
        layer_color_order=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    )
    
    print("\n=== 溝ガイドライン機能テスト完了 ===")

def analyze_groove_frames():
    """溝ガイドライン付きフレームを解析"""
    
    print("\n=== 溝ガイドラインフレーム解析 ===")
    
    import trimesh
    
    test_cases = [
        ("test_groove_2layer_resin_frame.stl", 2),
        ("test_groove_4layer_resin_frame.stl", 4)
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
                
                # 溝の存在確認（体積減少の確認）
                # 基本フレーム体積を計算
                frame_dims = frame.bounds[1] - frame.bounds[0]
                outer_volume = frame_dims[0] * frame_dims[1] * frame_dims[2]
                inner_dims = frame_dims - np.array([2, 2, 0])  # 壁厚2mm分
                inner_volume = max(0, inner_dims[0] * inner_dims[1] * frame_dims[2])
                expected_volume = outer_volume - inner_volume
                
                print(f"  理論体積: {expected_volume:.2f} mm³")
                print(f"  実際体積: {frame.volume:.2f} mm³")
                volume_diff = expected_volume - frame.volume
                print(f"  体積減少: {volume_diff:.2f} mm³（溝による減少）")
                
                if volume_diff > 0:
                    print("  ✅ 溝が作成されている")
                else:
                    print("  ❌ 溝が見つからない")
                    
            except Exception as e:
                print(f"  エラー: {str(e)}")
        else:
            print(f"{filename}: ファイルが見つかりません")

def visual_groove_info():
    """溝の視覚確認用情報"""
    
    print("\n=== 溝ガイドライン確認情報 ===")
    
    layer_height = 0.1 + 1.0 + 2.0  # base + wall + margin = 3.1mm
    
    for layers in [2, 4]:
        print(f"\n{layers}レイヤーケース:")
        for i in range(1, layers + 1):
            groove_height = i * layer_height
            print(f"  レイヤー{i}: {groove_height:.1f}mm高に水平溝")
        
        total_height = layers * layer_height + 5.0
        print(f"  総フレーム高: {total_height:.1f}mm")
        print(f"  溝仕様: 高さ0.3mm, 深さ0.8mm")

if __name__ == "__main__":
    test_groove_frame()
    analyze_groove_frames()
    visual_groove_info()