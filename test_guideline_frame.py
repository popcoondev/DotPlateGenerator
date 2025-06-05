#!/usr/bin/env python3

"""
ガイドライン付きフレーム機能のテスト
- カップラーメン風の水平ガイドライン
- 各レイヤーの配置位置に明確な線
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dot_plate_generator_gui import generate_color_separated_layers_stl
import numpy as np

def test_guideline_frame():
    """ガイドライン付きフレーム機能をテスト"""
    
    print("=== ガイドライン付きフレーム機能テスト ===")
    
    # テストケース1: 3レイヤーのガイドライン
    print("\n--- テストケース1: 3レイヤーガイドライン ---")
    pixels_3_layer = np.array([
        [(255, 0, 0), (0, 255, 0)],    # 赤, 緑
        [(0, 0, 255), (0, 0, 0)]       # 青, 黒（透明）
    ])
    
    generate_color_separated_layers_stl(
        pixels_rounded_np=pixels_3_layer,
        output_base_path="test_guideline_3layer",
        grid_size=2,
        dot_size=2.0,
        wall_thickness=1.0,
        wall_height=1.0,
        base_height=0.1,
        out_thickness=1.0,
        layer_color_order=[(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    )
    
    # テストケース2: 5レイヤーのガイドライン  
    print("\n--- テストケース2: 5レイヤーガイドライン ---")
    pixels_5_layer = np.array([
        [(255, 0, 0), (0, 255, 0)],
        [(0, 0, 255), (255, 255, 0)]
    ])
    
    generate_color_separated_layers_stl(
        pixels_rounded_np=pixels_5_layer,
        output_base_path="test_guideline_5layer",
        grid_size=2,
        dot_size=2.0,
        wall_thickness=1.0,
        wall_height=1.0,
        base_height=0.1,
        out_thickness=1.0,
        layer_color_order=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (128, 128, 128)]
    )
    
    print("\n=== ガイドライン機能テスト完了 ===")

def analyze_guideline_frames():
    """ガイドライン付きフレームを解析"""
    
    print("\n=== ガイドラインフレーム解析 ===")
    
    import trimesh
    
    test_cases = [
        ("test_guideline_3layer_resin_frame.stl", 3),
        ("test_guideline_5layer_resin_frame.stl", 5)
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
                
                # ガイドライン数の推定（面数から計算）
                # 基本構造: 底面+4壁=5要素
                # ガイドライン: 各レイヤーに4面×4壁=16面
                basic_faces = 5 * 12  # 底面+4壁の基本面数
                guideline_faces = (len(frame.faces) - basic_faces) / 12  # ガイドライン面数
                estimated_guidelines = int(guideline_faces / 4)  # 4壁で割る
                
                print(f"  推定ガイドライン数: {estimated_guidelines}個")
                print(f"  期待ガイドライン数: {expected_layers}個")
                
                if estimated_guidelines == expected_layers:
                    print("  ✅ ガイドライン数正常")
                else:
                    print("  ❌ ガイドライン数異常")
                    
            except Exception as e:
                print(f"  エラー: {str(e)}")
        else:
            print(f"{filename}: ファイルが見つかりません")

def visual_check_guidelines():
    """ガイドラインの視覚的確認用情報"""
    
    print("\n=== ガイドライン視覚確認情報 ===")
    
    layer_height = 0.1 + 1.0 + 2.0  # base + wall + margin = 3.1mm
    
    for layers in [3, 5]:
        print(f"\n{layers}レイヤーケース:")
        for i in range(1, layers + 1):
            guideline_height = i * layer_height
            print(f"  レイヤー{i}: {guideline_height:.1f}mm高にガイドライン")
        
        total_height = layers * layer_height + 5.0
        print(f"  総フレーム高: {total_height:.1f}mm")

if __name__ == "__main__":
    test_guideline_frame()
    analyze_guideline_frames()
    visual_check_guidelines()