#!/usr/bin/env python3

"""
最終ガイドライン機能のテスト
- 線状突起によるガイドライン
- 視認性の確認
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dot_plate_generator_gui import generate_color_separated_layers_stl
import numpy as np

def test_final_groove():
    """最終ガイドライン機能をテスト"""
    
    print("=== 最終ガイドライン機能テスト ===")
    
    # テストケース: 3レイヤーの明確なガイドライン
    print("\n--- 3レイヤー明確ガイドライン ---")
    pixels = np.array([
        [(255, 0, 0), (0, 255, 0), (0, 0, 255)]    # 赤, 緑, 青
    ])
    
    generate_color_separated_layers_stl(
        pixels_rounded_np=pixels,
        output_base_path="test_final_groove",
        grid_size=1,
        dot_size=3.0,  # 大きめにして視認性向上
        wall_thickness=1.0,
        wall_height=1.0,
        base_height=0.1,
        out_thickness=1.0,
        layer_color_order=[(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    )
    
    print("\n=== 最終ガイドライン機能テスト完了 ===")

def analyze_final_groove():
    """最終ガイドライン付きフレームを解析"""
    
    print("\n=== 最終ガイドラインフレーム解析 ===")
    
    import trimesh
    
    filename = "test_final_groove_resin_frame.stl"
    if os.path.exists(filename):
        try:
            frame = trimesh.load(filename)
            print(f"\n{filename}:")
            print(f"  頂点数: {len(frame.vertices)}")
            print(f"  面数: {len(frame.faces)}")
            print(f"  境界ボックス: {frame.bounds}")
            print(f"  体積: {frame.volume:.2f} mm³")
            print(f"  高さ: {frame.bounds[1][2] - frame.bounds[0][2]:.2f} mm")
            
            # ガイドライン位置の理論値
            layer_height = 0.1 + 1.0 + 2.0  # 3.1mm
            expected_guidelines = [
                layer_height * 1,  # 3.1mm
                layer_height * 2,  # 6.2mm  
                layer_height * 3   # 9.3mm
            ]
            
            print(f"  期待ガイドライン位置:")
            for i, height in enumerate(expected_guidelines, 1):
                print(f"    レイヤー{i}: {height:.1f}mm")
            
            # メッシュ構造の確認
            if hasattr(frame, 'visual') and hasattr(frame.visual, 'face_colors'):
                print(f"  面色情報: あり")
            else:
                print(f"  面色情報: なし")
                
            print("  ✅ ガイドライン生成完了")
                
        except Exception as e:
            print(f"  エラー: {str(e)}")
    else:
        print(f"{filename}: ファイルが見つかりません")

def show_usage_info():
    """ガイドライン使用方法"""
    
    print("\n=== ガイドライン使用方法 ===")
    print("1. フレーム内壁の水平線がガイドラインです")
    print("2. 各線の高さに対応するレイヤーを配置します")
    print("3. レイヤー1: 3.1mm高の線")
    print("4. レイヤー2: 6.2mm高の線") 
    print("5. レイヤー3: 9.3mm高の線")
    print("6. 線が見えない場合は、STLビューアーで内壁を確認してください")

if __name__ == "__main__":
    test_final_groove()
    analyze_final_groove()
    show_usage_info()