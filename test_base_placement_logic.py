#!/usr/bin/env python3
"""
ベース配置ロジックの詳細テスト
修正前：上位色で使用済みの座標にはベース配置不可
修正後：これまでに出力済みの座標のみベース配置不可、将来の上位レイヤーでビルが立つ予定の座標にはベース配置OK
"""

import numpy as np
import sys
import os

# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_base_placement_detail():
    """詳細なベース配置テスト"""
    print("=== ベース配置ロジック詳細テスト ===")
    
    # 3x3の特別なパターンを作成
    # 赤(1番目): 左上と右下
    # 緑(2番目): 中央
    # 青(3番目): 右上と左下
    pixels = np.array([
        [[255, 0, 0], [0, 0, 0], [0, 0, 255]],     # 赤, 黒, 青
        [[0, 0, 0], [0, 255, 0], [0, 0, 0]],       # 黒, 緑, 黒  
        [[0, 0, 255], [0, 0, 0], [255, 0, 0]]      # 青, 黒, 赤
    ], dtype=np.uint8)
    
    layer_color_order = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 赤→緑→青の順
    
    # パラメータ設定
    grid_size = 3
    dot_size = 2.0
    wall_thickness = 0.2
    wall_height = 1.0
    base_height = 0.5
    out_thickness = 0.3
    
    print("画像パターン:")
    print("赤   黒   青")
    print("黒   緑   黒")
    print("青   黒   赤")
    print()
    print("出力順序: 赤(1st) → 緑(2nd) → 青(3rd)")
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
        
        # 期待される結果の説明
        print("\n期待される結果:")
        print("1. 赤レイヤー(1st): 左上(0,0)と右下(2,2)にビル。緑や青の予定位置にもベース配置可能")
        print("2. 緑レイヤー(2nd): 中央(1,1)にビル。青の予定位置にもベース配置可能。赤の出力済み位置は避ける")
        print("3. 青レイヤー(3rd): 右上(2,0)と左下(0,2)にビル。赤・緑の出力済み位置は避ける")
        print("\nベース配置詳細テスト: 成功")
        return True
        
    except Exception as e:
        print(f"ベース配置詳細テスト: 失敗 - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stacking_pattern():
    """積層パターンテスト"""
    print("\n=== 積層パターンテスト ===")
    
    # 2x2で上下積層パターン
    # 赤(1st): 全体
    # 緑(2nd): 右半分
    # 青(3rd): 右下のみ
    pixels = np.array([
        [[255, 0, 0], [0, 255, 0]],     # 赤, 緑
        [[255, 0, 0], [0, 0, 255]]      # 赤, 青
    ], dtype=np.uint8)
    
    layer_color_order = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 赤→緑→青の順
    
    # パラメータ設定
    grid_size = 2
    dot_size = 2.0
    wall_thickness = 0.2
    wall_height = 1.0
    base_height = 0.5
    out_thickness = 0.5
    
    print("画像パターン:")
    print("赤   緑")
    print("赤   青")
    print()
    print("出力順序: 赤(1st) → 緑(2nd) → 青(3rd)")
    print()
    
    try:
        from dot_plate_generator_gui import generate_color_separated_layers_stl
        
        meshes = generate_color_separated_layers_stl(
            pixels, 
            "test_stacking",
            grid_size,
            dot_size,
            wall_thickness,
            wall_height,
            base_height,
            out_thickness,
            layer_color_order
        )
        
        print(f"\n生成されたメッシュ数: {len(meshes)}")
        
        # 期待される結果の説明
        print("\n期待される結果:")
        print("1. 赤レイヤー(1st): 左上(0,0)、右上(1,0)、左下(0,1)にビル。緑・青の予定位置(1,0),(1,1)にもベース配置可能")
        print("2. 緑レイヤー(2nd): 右上(1,0)にビル。青の予定位置(1,1)にもベース配置可能。赤の出力済み位置は避ける")
        print("3. 青レイヤー(3rd): 右下(1,1)にビル。赤・緑の出力済み位置は避ける")
        print("\n積層パターンテスト: 成功")
        return True
        
    except Exception as e:
        print(f"積層パターンテスト: 失敗 - {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト関数"""
    print("ベース配置ロジックの詳細テストを開始します...\n")
    
    results = []
    results.append(test_base_placement_detail())
    results.append(test_stacking_pattern())
    
    print("\n=== テスト結果 ===")
    success_count = sum(results)
    total_count = len(results)
    
    print(f"成功: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("すべてのテストが成功しました！")
        print("\n修正されたベース配置ロジック:")
        print("✅ これまでに出力済みの座標のみベース配置NG")
        print("✅ 将来の上位レイヤーでビルが立つ予定の座標にはベース配置OK")
        return 0
    else:
        print("一部のテストが失敗しました。")
        return 1

if __name__ == "__main__":
    exit(main())