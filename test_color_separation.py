#!/usr/bin/env python3
"""
色別レイヤー分離機能のテストスクリプト
"""

import numpy as np
import sys
import os

# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_single_color():
    """単色のテスト"""
    print("=== 単色テスト ===")
    
    # 2x2の赤い画像を作成
    pixels = np.array([
        [[255, 0, 0], [255, 0, 0]],
        [[255, 0, 0], [255, 0, 0]]
    ], dtype=np.uint8)
    
    layer_color_order = [(255, 0, 0)]
    
    # パラメータ設定
    grid_size = 2
    dot_size = 2.0
    wall_thickness = 0.2
    wall_height = 1.0
    base_height = 0.5
    out_thickness = 0.5
    
    try:
        from dot_plate_generator_gui import generate_color_separated_layers_stl
        
        meshes = generate_color_separated_layers_stl(
            pixels, 
            "test_single_color",
            grid_size,
            dot_size,
            wall_thickness,
            wall_height,
            base_height,
            out_thickness,
            layer_color_order
        )
        
        print(f"生成されたメッシュ数: {len(meshes)}")
        print("単色テスト: 成功")
        return True
        
    except Exception as e:
        print(f"単色テスト: 失敗 - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_colors():
    """複数色のテスト"""
    print("\n=== 複数色テスト ===")
    
    # 2x2の複数色画像を作成
    pixels = np.array([
        [[255, 0, 0], [0, 255, 0]],
        [[0, 0, 255], [255, 255, 0]]
    ], dtype=np.uint8)
    
    layer_color_order = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    # パラメータ設定
    grid_size = 2
    dot_size = 2.0
    wall_thickness = 0.2
    wall_height = 1.0
    base_height = 0.5
    out_thickness = 0.5
    
    try:
        from dot_plate_generator_gui import generate_color_separated_layers_stl
        
        meshes = generate_color_separated_layers_stl(
            pixels, 
            "test_multiple_colors",
            grid_size,
            dot_size,
            wall_thickness,
            wall_height,
            base_height,
            out_thickness,
            layer_color_order
        )
        
        print(f"生成されたメッシュ数: {len(meshes)}")
        print("複数色テスト: 成功")
        return True
        
    except Exception as e:
        print(f"複数色テスト: 失敗 - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_island_pattern():
    """島パターンのテスト"""
    print("\n=== 島パターンテスト ===")
    
    # 4x4の島パターン画像を作成（同色の離れた島）
    pixels = np.array([
        [[255, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0]],
        [[0, 0, 0], [0, 255, 0], [0, 255, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 255, 0], [0, 255, 0], [0, 0, 0]],
        [[255, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0]]
    ], dtype=np.uint8)
    
    layer_color_order = [(255, 0, 0), (0, 255, 0)]
    
    # パラメータ設定
    grid_size = 4
    dot_size = 2.0
    wall_thickness = 0.2
    wall_height = 1.0
    base_height = 0.5
    out_thickness = 0.5
    
    try:
        from dot_plate_generator_gui import generate_color_separated_layers_stl
        
        meshes = generate_color_separated_layers_stl(
            pixels, 
            "test_island_pattern",
            grid_size,
            dot_size,
            wall_thickness,
            wall_height,
            base_height,
            out_thickness,
            layer_color_order
        )
        
        print(f"生成されたメッシュ数: {len(meshes)}")
        print("島パターンテスト: 成功")
        return True
        
    except Exception as e:
        print(f"島パターンテスト: 失敗 - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_out_thickness_variation():
    """out_thickness変化のテスト"""
    print("\n=== out_thickness変化テスト ===")
    
    # 2x2の複数色画像を作成
    pixels = np.array([
        [[255, 0, 0], [0, 255, 0]],
        [[0, 0, 255], [255, 255, 0]]
    ], dtype=np.uint8)
    
    layer_color_order = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    # パラメータ設定（out_thicknessを大きくする）
    grid_size = 2
    dot_size = 2.0
    wall_thickness = 0.2
    wall_height = 1.0
    base_height = 0.5
    out_thickness = 1.5  # 大きくしてベース拡張をテスト
    
    try:
        from dot_plate_generator_gui import generate_color_separated_layers_stl
        
        meshes = generate_color_separated_layers_stl(
            pixels, 
            "test_out_thickness",
            grid_size,
            dot_size,
            wall_thickness,
            wall_height,
            base_height,
            out_thickness,
            layer_color_order
        )
        
        print(f"生成されたメッシュ数: {len(meshes)}")
        print("out_thickness変化テスト: 成功")
        return True
        
    except Exception as e:
        print(f"out_thickness変化テスト: 失敗 - {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト関数"""
    print("色別レイヤー分離機能のテストを開始します...\n")
    
    results = []
    results.append(test_single_color())
    results.append(test_multiple_colors())
    results.append(test_island_pattern())
    results.append(test_out_thickness_variation())
    
    print("\n=== テスト結果 ===")
    success_count = sum(results)
    total_count = len(results)
    
    print(f"成功: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("すべてのテストが成功しました！")
        return 0
    else:
        print("一部のテストが失敗しました。")
        return 1

if __name__ == "__main__":
    exit(main())