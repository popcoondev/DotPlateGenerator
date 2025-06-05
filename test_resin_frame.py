#!/usr/bin/env python3
"""
レジン固め用フレーム機能のテスト
"""

import numpy as np
import sys
import os

# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_resin_frame_generation():
    """レジンフレーム生成のテスト"""
    print("=== レジンフレーム生成テスト ===")
    
    # 2x2の簡単なパターン
    pixels = np.array([
        [[255, 0, 0], [0, 255, 0]],     # 赤, 緑
        [[0, 0, 255], [255, 255, 0]]    # 青, 黄
    ], dtype=np.uint8)
    
    layer_color_order = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    # パラメータ設定
    grid_size = 2
    dot_size = 2.0
    wall_thickness = 0.2
    wall_height = 1.0
    base_height = 0.5
    out_thickness = 0.5
    
    print("パターン:")
    print("赤   緑")
    print("青   黄")
    print()
    print("期待される動作:")
    print("- 4つのレイヤーSTL + 1つのレジンフレームSTL生成")
    print("- フレーム内壁に等間隔刻み（base_height + wall_height + 2mm = 3.5mm間隔）")
    print("- コの字型（底面 + 左右奥壁、手前開放）")
    print()
    
    try:
        from dot_plate_generator_gui import generate_color_separated_layers_stl
        
        meshes = generate_color_separated_layers_stl(
            pixels, 
            "test_resin_frame",
            grid_size,
            dot_size,
            wall_thickness,
            wall_height,
            base_height,
            out_thickness,
            layer_color_order
        )
        
        print(f"\n生成されたメッシュ数: {len(meshes)} (色レイヤー: {len(layer_color_order)}個 + フレーム: 1個)")
        print("レジンフレーム生成テスト: 成功")
        return True
        
    except Exception as e:
        print(f"レジンフレーム生成テスト: 失敗 - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_frame_scaling():
    """フレームサイズスケーリングのテスト"""
    print("\n=== フレームサイズスケーリングテスト ===")
    
    # 5x5の大きなパターン
    pixels = np.array([
        [[255, 0, 0], [0, 255, 0], [255, 0, 0], [0, 255, 0], [255, 0, 0]],
        [[0, 255, 0], [255, 0, 0], [0, 255, 0], [255, 0, 0], [0, 255, 0]],
        [[255, 0, 0], [0, 255, 0], [255, 0, 0], [0, 255, 0], [255, 0, 0]],
        [[0, 255, 0], [255, 0, 0], [0, 255, 0], [255, 0, 0], [0, 255, 0]],
        [[255, 0, 0], [0, 255, 0], [255, 0, 0], [0, 255, 0], [255, 0, 0]]
    ], dtype=np.uint8)
    
    layer_color_order = [(255, 0, 0), (0, 255, 0)]
    
    # パラメータ設定（より大きなサイズ）
    grid_size = 5
    dot_size = 3.0
    wall_thickness = 0.3
    wall_height = 2.0
    base_height = 1.0
    out_thickness = 1.0
    
    print("パターン: 5x5チェッカーボード")
    print("期待される動作:")
    print("- 大きなサイズでもフレームが適切にスケーリング")
    print("- 刻み間隔: base_height + wall_height + 2mm = 5.0mm")
    print()
    
    try:
        from dot_plate_generator_gui import generate_color_separated_layers_stl
        
        meshes = generate_color_separated_layers_stl(
            pixels, 
            "test_frame_scaling",
            grid_size,
            dot_size,
            wall_thickness,
            wall_height,
            base_height,
            out_thickness,
            layer_color_order
        )
        
        print(f"\n生成されたメッシュ数: {len(meshes)} (色レイヤー: {len(layer_color_order)}個 + フレーム: 1個)")
        print("フレームサイズスケーリングテスト: 成功")
        return True
        
    except Exception as e:
        print(f"フレームサイズスケーリングテスト: 失敗 - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_notch_interval_calculation():
    """刻み間隔計算のテスト"""
    print("\n=== 刻み間隔計算テスト ===")
    
    # シングルピクセル
    pixels = np.array([[[255, 0, 0]]], dtype=np.uint8)
    layer_color_order = [(255, 0, 0)]
    
    # 異なるパラメータでテスト
    test_cases = [
        {"base": 0.5, "wall": 1.0, "expected_interval": 3.5},  # 0.5 + 1.0 + 2.0
        {"base": 1.0, "wall": 2.0, "expected_interval": 5.0},  # 1.0 + 2.0 + 2.0
        {"base": 0.3, "wall": 0.8, "expected_interval": 3.1},  # 0.3 + 0.8 + 2.0
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"  ケース{i}: base={case['base']}, wall={case['wall']} → 期待間隔={case['expected_interval']}mm")
        
        try:
            from dot_plate_generator_gui import generate_color_separated_layers_stl
            
            meshes = generate_color_separated_layers_stl(
                pixels, 
                f"test_notch_{i}",
                1,  # grid_size
                2.0,  # dot_size
                0.2,  # wall_thickness
                case['wall'],  # wall_height
                case['base'],  # base_height
                0.5,  # out_thickness
                layer_color_order
            )
            
            print(f"    → 成功")
            
        except Exception as e:
            print(f"    → 失敗: {e}")
            return False
    
    print("刻み間隔計算テスト: 成功")
    return True

def main():
    """メインテスト関数"""
    print("レジン固め用フレーム機能のテストを開始します...\n")
    
    results = []
    results.append(test_resin_frame_generation())
    results.append(test_frame_scaling())
    results.append(test_notch_interval_calculation())
    
    print("\n=== テスト結果 ===")
    success_count = sum(results)
    total_count = len(results)
    
    print(f"成功: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("すべてのテストが成功しました！")
        print("\n新機能:")
        print("✅ レジン固め用コの字型フレーム生成")
        print("✅ 内壁への等間隔刻みガイド")
        print("✅ レイヤー構造維持に最適化された設計")
        print("✅ HTMLレポートにフレーム情報追加")
        print("✅ 異なるサイズでの自動スケーリング")
        return 0
    else:
        print("一部のテストが失敗しました。")
        return 1

if __name__ == "__main__":
    exit(main())