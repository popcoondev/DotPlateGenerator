#!/usr/bin/env python3
"""
building.py - レイヤースタックモードのテスト用ファイル

3x3配列をレイヤー1として処理し、ビル構造を生成するテスト
配列: 
1,2,1
2,3,2
3,3,1

パラメータ:
- dot_size = 1
- wall_thickness = 1  
- out_thickness = 1
- base_height = 1
- wall_height = 2
"""

import trimesh
from trimesh.creation import box
import numpy as np

def create_layer_building(layer_num, target_value):
    """
    指定されたレイヤー用のビル構造を生成
    
    Args:
        layer_num: レイヤー番号 (1, 2, 3)
        target_value: ビル対象となる値 (1, 2, 3)
    """
    # テストデータ
    data = np.array([
        [1, 2, 1],
        [2, 3, 2], 
        [3, 3, 1]
    ])
    
    # パラメータ
    dot_size = 1
    wall_thickness = 0.5
    out_thickness = 0.5
    base_height = 0.2
    wall_height = 5
    grid_size = 3
    
    # レイヤーごとのビル高さ計算
    # レイヤー1: wall_height, レイヤー2: wall_height - base_height, レイヤー3: wall_height - base_height * 2
    # ただし、最小でもwall_height/3は確保する
    layer_building_height = max(wall_height - base_height * (layer_num - 1), wall_height / 3)
    
    print(f"\nレイヤー{layer_num}処理開始 - ターゲット値: {target_value}")
    print(f"ベース高さ: {base_height}")
    print(f"ビル高さ: {layer_building_height}")
    print(f"配列:\n{data}")
    
    # 全体のサイズ計算
    total_size = grid_size * dot_size + 2 * out_thickness
    print(f"全体サイズ: {total_size} x {total_size}")
    
    print(f"ベースプレート全体サイズ: ({total_size}, {total_size}, {base_height})")
    
    # 上位レイヤーの穴位置を計算（このレイヤーより大きい値）
    upper_layer_holes = set()
    current_layer_positions = set()
    non_building_positions = set()
    
    for y in range(grid_size):
        for x in range(grid_size):
            value = data[y, x]
            dot_center_x = x * dot_size + dot_size / 2
            dot_center_y = (grid_size - 1 - y) * dot_size + dot_size / 2
            
            if value > target_value:
                # 上位レイヤー用の穴
                upper_layer_holes.add((dot_center_x, dot_center_y))
            elif value == target_value:
                # このレイヤーのビル位置
                current_layer_positions.add((dot_center_x, dot_center_y))
            else:
                # ビル建設しない位置（空洞化対象）
                non_building_positions.add((dot_center_x, dot_center_y))
    
    print(f"  上位レイヤー穴: {len(upper_layer_holes)}個")
    print(f"  ビル建設位置: {len(current_layer_positions)}個")  
    print(f"  空洞化位置: {len(non_building_positions)}個")
    
    # ベースプレートをグリッド単位で構築（穴と空洞を除く）
    layer_blocks = []
    
    # 外周部分のベースプレート
    # 左側
    left_block = box(extents=[out_thickness, total_size, base_height])
    left_block.apply_translation([-out_thickness/2, (total_size)/2 - out_thickness, base_height/2])
    layer_blocks.append(left_block)
    
    # 右側  
    right_block = box(extents=[out_thickness, total_size, base_height])
    right_block.apply_translation([grid_size * dot_size + out_thickness/2, (total_size)/2 - out_thickness, base_height/2])
    layer_blocks.append(right_block)
    
    # 上側
    top_block = box(extents=[grid_size * dot_size, out_thickness, base_height])
    top_block.apply_translation([(grid_size * dot_size)/2, -out_thickness/2, base_height/2])
    layer_blocks.append(top_block)
    
    # 下側
    bottom_block = box(extents=[grid_size * dot_size, out_thickness, base_height])
    bottom_block.apply_translation([(grid_size * dot_size)/2, grid_size * dot_size + out_thickness/2, base_height/2])
    layer_blocks.append(bottom_block)
    
    # グリッド内のベースプレート（必要な部分のみ）
    for y in range(grid_size):
        for x in range(grid_size):
            dot_center_x = x * dot_size + dot_size / 2
            dot_center_y = (grid_size - 1 - y) * dot_size + dot_size / 2
            
            # 上位レイヤーの穴でも空洞化位置でもない場合のみベースプレートを作成
            if (dot_center_x, dot_center_y) not in upper_layer_holes and (dot_center_x, dot_center_y) not in non_building_positions:
                base_cell = box(extents=[dot_size, dot_size, base_height])
                base_cell.apply_translation([dot_center_x, dot_center_y, base_height/2])
                layer_blocks.append(base_cell)
                print(f"  ベースセル追加: ({x}, {y})")
            elif (dot_center_x, dot_center_y) in upper_layer_holes:
                print(f"  上位レイヤー穴: ({x}, {y}) - ベースなし")
            elif (dot_center_x, dot_center_y) in non_building_positions:
                print(f"  空洞化: ({x}, {y}) - ベースなし")
    
    # ビル建設（target_valueの位置のみ）
    for y in range(grid_size):
        for x in range(grid_size):
            value = data[y, x]
            
            if value == target_value:
                # ドットの中心座標計算
                dot_center_x = x * dot_size + dot_size / 2
                dot_center_y = (grid_size - 1 - y) * dot_size + dot_size / 2
                
                print(f"  ビル建設: 位置({x}, {y}), 中心({dot_center_x}, {dot_center_y})")
                
                # (c) ビル部分を (dot_size - wall_thickness) × (dot_size - wall_thickness) × layer_building_height で作成
                building_size = dot_size - wall_thickness
                building_block = box(extents=[building_size, building_size, layer_building_height])
                building_z = base_height + layer_building_height / 2
                building_block.apply_translation([dot_center_x, dot_center_y, building_z])
                layer_blocks.append(building_block)
                
                print(f"    ビルサイズ: ({building_size}, {building_size}, {layer_building_height}) (wall_thickness分細く)")
                print(f"    ビル中心: ({dot_center_x}, {dot_center_y}, {building_z})")
    
    return layer_blocks

def create_layer_1_building():
    """レイヤー1用のビル構造を生成（後方互換性のため残す）"""
    return create_layer_building(1, 1)

def create_all_layers():
    """全レイヤーのSTLファイルを生成"""
    layers = [
        (1, 1),  # レイヤー1: 値1
        (2, 2),  # レイヤー2: 値2
        (3, 3),  # レイヤー3: 値3
    ]
    
    for layer_num, target_value in layers:
        print(f"\n{'='*50}")
        print(f"レイヤー{layer_num} 生成開始")
        
        # ビル構造生成
        blocks = create_layer_building(layer_num, target_value)
        
        # メッシュ統合
        if blocks:
            try:
                combined_mesh = trimesh.util.concatenate(blocks)
                
                # STLファイル出力
                output_path = f"/Users/mn/development/DotPlateGenerator/test_layer{layer_num}.stl"
                combined_mesh.export(output_path)
                print(f"\nSTLファイル出力完了: {output_path}")
                
                # メッシュ情報表示
                print(f"頂点数: {len(combined_mesh.vertices)}")
                print(f"面数: {len(combined_mesh.faces)}")
                print(f"バウンディングボックス: {combined_mesh.bounds}")
                
            except Exception as e:
                print(f"メッシュ処理エラー: {e}")
        else:
            print("ブロックが生成されませんでした")

def main():
    """メイン処理"""
    print("=== 全レイヤービル生成テスト ===")
    print("配列:")
    print("1,2,1")
    print("2,3,2") 
    print("3,3,1")
    print("\nレイヤー1: 値1の位置 → (0,0), (2,0), (2,2)")
    print("レイヤー2: 値2の位置 → (1,0), (0,1), (2,1)")
    print("レイヤー3: 値3の位置 → (1,1), (0,2), (1,2)")
    
    create_all_layers()

if __name__ == "__main__":
    main()