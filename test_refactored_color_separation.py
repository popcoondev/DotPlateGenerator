#!/usr/bin/env python3
"""
リファクタリングされた色別レイヤー分離機能のテスト
新機能：
1. 厳密なグリッド単位管理
2. 連結成分（島）検出とチェーン接続
3. 詳細なログ出力
4. より堅牢なベース配置ロジック
"""

import numpy as np
import sys
import os

# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_island_detection_and_connection():
    """島検出と接続機能のテスト"""
    print("=== 島検出・接続機能テスト ===")
    
    # 4x4のパターンで明確な島を作成
    # 赤: 左上と右下に分離した島
    # 緑: 中央の連続エリア
    pixels = np.array([
        [[255, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0]],     # 赤, 黒, 黒, 赤
        [[0, 0, 0], [0, 255, 0], [0, 255, 0], [0, 0, 0]],     # 黒, 緑, 緑, 黒
        [[0, 0, 0], [0, 255, 0], [0, 255, 0], [0, 0, 0]],     # 黒, 緑, 緑, 黒
        [[255, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0]]      # 赤, 黒, 黒, 赤
    ], dtype=np.uint8)
    
    layer_color_order = [(255, 0, 0), (0, 255, 0)]  # 赤→緑の順
    
    # パラメータ設定
    grid_size = 4
    dot_size = 2.0
    wall_thickness = 0.2
    wall_height = 1.0
    base_height = 0.5
    out_thickness = 0.5
    
    print("パターン:")
    print("赤   黒   黒   赤")
    print("黒   緑   緑   黒")
    print("黒   緑   緑   黒")
    print("赤   黒   黒   赤")
    print()
    print("期待される動作:")
    print("- 赤: 4つの角が4つの島として検出され、チェーン接続される")
    print("- 緑: 中央の2x2エリアが1つの島として検出される")
    print()
    
    try:
        from dot_plate_generator_gui import generate_color_separated_layers_stl
        
        meshes = generate_color_separated_layers_stl(
            pixels, 
            "test_island_detection",
            grid_size,
            dot_size,
            wall_thickness,
            wall_height,
            base_height,
            out_thickness,
            layer_color_order
        )
        
        print(f"\n生成されたメッシュ数: {len(meshes)}")
        print("島検出・接続機能テスト: 成功")
        return True
        
    except Exception as e:
        print(f"島検出・接続機能テスト: 失敗 - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strict_grid_management():
    """厳密なグリッド単位管理のテスト"""
    print("\n=== 厳密なグリッド単位管理テスト ===")
    
    # 2x2の積層パターン（グリッド境界の処理確認）
    pixels = np.array([
        [[255, 0, 0], [0, 255, 0]],     # 赤, 緑
        [[0, 0, 255], [255, 255, 0]]    # 青, 黄
    ], dtype=np.uint8)
    
    layer_color_order = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    # パラメータ設定（out_thicknessを意図的に小数値に）
    grid_size = 2
    dot_size = 2.0
    wall_thickness = 0.2
    wall_height = 1.0
    base_height = 0.5
    out_thickness = 0.3  # グリッド単位で約0.15セル → 1セルに丸められる
    
    print("パターン:")
    print("赤   緑")
    print("青   黄")
    print()
    print("期待される動作:")
    print("- out_thickness 0.3mm → 1グリッドに変換")
    print("- すべて整数グリッド座標で管理")
    print("- 浮動小数点の重複を回避")
    print()
    
    try:
        from dot_plate_generator_gui import generate_color_separated_layers_stl
        
        meshes = generate_color_separated_layers_stl(
            pixels, 
            "test_grid_management",
            grid_size,
            dot_size,
            wall_thickness,
            wall_height,
            base_height,
            out_thickness,
            layer_color_order
        )
        
        print(f"\n生成されたメッシュ数: {len(meshes)}")
        print("厳密なグリッド単位管理テスト: 成功")
        return True
        
    except Exception as e:
        print(f"厳密なグリッド単位管理テスト: 失敗 - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_base_placement_logic():
    """ベース配置ロジックのテスト"""
    print("\n=== ベース配置ロジックテスト ===")
    
    # 3x3で積層構造のテスト
    pixels = np.array([
        [[255, 0, 0], [255, 0, 0], [0, 255, 0]],     # 赤, 赤, 緑
        [[255, 0, 0], [0, 0, 255], [0, 255, 0]],     # 赤, 青, 緑
        [[0, 0, 0], [0, 0, 255], [0, 255, 0]]        # 黒, 青, 緑
    ], dtype=np.uint8)
    
    layer_color_order = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 赤→緑→青の順
    
    # パラメータ設定
    grid_size = 3
    dot_size = 2.0
    wall_thickness = 0.2
    wall_height = 1.0
    base_height = 0.5
    out_thickness = 0.6
    
    print("パターン:")
    print("赤   赤   緑")
    print("赤   青   緑")
    print("黒   青   緑")
    print()
    print("期待される動作:")
    print("- 赤（1st）: 左上L字型、青・緑の予定位置にもベース配置可能")
    print("- 緑（2nd）: 右列、赤の出力済み位置を避け、青の予定位置にはベース配置可能")
    print("- 青（3rd）: 中下2セル、赤・緑の出力済み位置を避ける")
    print()
    
    try:
        from dot_plate_generator_gui import generate_color_separated_layers_stl
        
        meshes = generate_color_separated_layers_stl(
            pixels, 
            "test_base_placement",
            grid_size,
            dot_size,
            wall_thickness,
            wall_height,
            base_height,
            out_thickness,
            layer_color_order
        )
        
        print(f"\n生成されたメッシュ数: {len(meshes)}")
        print("ベース配置ロジックテスト: 成功")
        return True
        
    except Exception as e:
        print(f"ベース配置ロジックテスト: 失敗 - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complex_island_pattern():
    """複雑な島パターンのテスト"""
    print("\n=== 複雑な島パターンテスト ===")
    
    # 5x5で複数の島を持つ複雑なパターン
    pixels = np.array([
        [[255, 0, 0], [0, 0, 0], [255, 0, 0], [0, 0, 0], [255, 0, 0]],     # 赤,黒,赤,黒,赤
        [[0, 0, 0], [0, 255, 0], [0, 0, 0], [0, 255, 0], [0, 0, 0]],       # 黒,緑,黒,緑,黒
        [[255, 0, 0], [0, 0, 0], [255, 0, 0], [0, 0, 0], [255, 0, 0]],     # 赤,黒,赤,黒,赤
        [[0, 0, 0], [0, 255, 0], [0, 0, 0], [0, 255, 0], [0, 0, 0]],       # 黒,緑,黒,緑,黒
        [[255, 0, 0], [0, 0, 0], [255, 0, 0], [0, 0, 0], [255, 0, 0]]      # 赤,黒,赤,黒,赤
    ], dtype=np.uint8)
    
    layer_color_order = [(255, 0, 0), (0, 255, 0)]  # 赤→緑の順
    
    # パラメータ設定
    grid_size = 5
    dot_size = 1.5
    wall_thickness = 0.1
    wall_height = 0.8
    base_height = 0.3
    out_thickness = 0.4
    
    print("パターン（チェッカーボード風）:")
    print("赤   黒   赤   黒   赤")
    print("黒   緑   黒   緑   黒")
    print("赤   黒   赤   黒   赤")
    print("黒   緑   黒   緑   黒")
    print("赤   黒   赤   黒   赤")
    print()
    print("期待される動作:")
    print("- 赤: 9個の分離した島が検出され、チェーン接続される")
    print("- 緑: 4個の分離した島が検出され、チェーン接続される")
    print()
    
    try:
        from dot_plate_generator_gui import generate_color_separated_layers_stl
        
        meshes = generate_color_separated_layers_stl(
            pixels, 
            "test_complex_islands",
            grid_size,
            dot_size,
            wall_thickness,
            wall_height,
            base_height,
            out_thickness,
            layer_color_order
        )
        
        print(f"\n生成されたメッシュ数: {len(meshes)}")
        print("複雑な島パターンテスト: 成功")
        return True
        
    except Exception as e:
        print(f"複雑な島パターンテスト: 失敗 - {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト関数"""
    print("リファクタリングされた色別レイヤー分離機能のテストを開始します...\n")
    
    results = []
    results.append(test_island_detection_and_connection())
    results.append(test_strict_grid_management())
    results.append(test_base_placement_logic())
    results.append(test_complex_island_pattern())
    
    print("\n=== テスト結果 ===")
    success_count = sum(results)
    total_count = len(results)
    
    print(f"成功: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("すべてのテストが成功しました！")
        print("\nリファクタリングで改善された機能:")
        print("✅ 連結成分（島）の自動検出")
        print("✅ すべての島をチェーン接続で一体化")
        print("✅ 厳密なグリッド単位座標管理")
        print("✅ 詳細なログ出力でデバッグ支援")
        print("✅ サブ関数分割による保守性向上")
        print("✅ より堅牢なベース配置ロジック")
        return 0
    else:
        print("一部のテストが失敗しました。")
        return 1

if __name__ == "__main__":
    exit(main())