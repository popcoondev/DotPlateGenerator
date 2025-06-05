#!/usr/bin/env python3

"""
分離フレーム機能のテスト
- 底面と壁面の分離
- 拡大サイズ（ドット2つ分+1mm余裕）
- ガイドライン付き
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dot_plate_generator_gui import generate_color_separated_layers_stl
import numpy as np

def test_separated_frame():
    """分離フレーム機能をテスト"""
    
    print("=== 分離フレーム機能テスト ===")
    
    # テストケース: 2レイヤーの分離フレーム
    print("\n--- 2レイヤー分離フレーム ---")
    pixels = np.array([
        [(255, 0, 0), (0, 255, 0)]    # 赤, 緑
    ])
    
    generate_color_separated_layers_stl(
        pixels_rounded_np=pixels,
        output_base_path="test_separated_frame",
        grid_size=1,
        dot_size=2.0,
        wall_thickness=1.0,
        wall_height=1.0,
        base_height=0.1,
        out_thickness=1.0,
        layer_color_order=[(255, 0, 0), (0, 255, 0)]
    )
    
    print("\n=== 分離フレーム機能テスト完了 ===")

def analyze_separated_frame():
    """分離フレーム出力を解析"""
    
    print("\n=== 分離フレーム解析 ===")
    
    import trimesh
    
    # 出力ファイルを確認
    files_to_check = [
        "test_separated_frame_resin_frame_bottom.stl",
        "test_separated_frame_resin_frame_walls.stl", 
        "test_separated_frame_resin_frame_combined.stl"
    ]
    
    for filename in files_to_check:
        if os.path.exists(filename):
            try:
                mesh = trimesh.load(filename)
                print(f"\n{filename}:")
                print(f"  頂点数: {len(mesh.vertices)}")
                print(f"  面数: {len(mesh.faces)}")
                print(f"  境界ボックス: {mesh.bounds}")
                print(f"  体積: {mesh.volume:.2f} mm³")
                
                bounds = mesh.bounds
                size = bounds[1] - bounds[0]
                print(f"  サイズ: {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm")
                
                if "bottom" in filename:
                    print(f"  底面高さ: {size[2]:.1f}mm")
                elif "walls" in filename:
                    print(f"  壁面高さ: {size[2]:.1f}mm")
                    # ガイドライン確認
                    expected_guidelines = 2  # 2レイヤー
                    print(f"  期待ガイドライン数: {expected_guidelines}個")
                
                print("  ✅ ファイル生成成功")
                
            except Exception as e:
                print(f"  エラー: {str(e)}")
        else:
            print(f"{filename}: ファイルが見つかりません")

def show_frame_usage():
    """分離フレーム使用方法"""
    
    print("\n=== 分離フレーム使用方法 ===")
    print("1. bottom.stl: 最初に3D印刷して底面を作成")
    print("2. レイヤー1を配置してレジンで固化")
    print("3. 固化後、bottom.stlを取り外し")
    print("4. walls.stl: 壁面フレームを設置")  
    print("5. 残りのレイヤーを順次配置・固化")
    print("6. 最終的にwalls.stlも取り外し")
    print("")
    print("フレームサイズ:")
    print("- 元サイズ + ドット2つ分 + 1mm余裕")
    print("- ベースプレートも余裕で収納可能")

def check_frame_dimensions():
    """フレーム寸法の確認"""
    
    print("\n=== フレーム寸法確認 ===")
    
    # 元のレイヤーサイズ
    grid_size = 1
    dot_size = 2.0
    out_thickness = 1.0
    
    original_size = grid_size * dot_size + 2 * out_thickness
    extra_space = 2 * dot_size + 1.0  # ドット2つ分 + 1mm
    frame_inner_size = original_size + extra_space
    frame_wall_thickness = 1.0
    frame_outer_size = frame_inner_size + 2 * frame_wall_thickness
    
    print(f"元レイヤーサイズ: {original_size:.1f} mm")
    print(f"追加スペース: {extra_space:.1f} mm")
    print(f"フレーム内側: {frame_inner_size:.1f} mm")
    print(f"フレーム外側: {frame_outer_size:.1f} mm")
    print(f"余裕スペース: {(frame_inner_size - original_size)/2:.1f} mm (片側)")

if __name__ == "__main__":
    test_separated_frame()
    analyze_separated_frame()
    show_frame_usage()
    check_frame_dimensions()