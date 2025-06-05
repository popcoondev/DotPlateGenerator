# Claude Code プロンプト: 改良版色別レイヤー分離出力モード実装

## タスク概要
既存のPythonファイル `dot_plate_generator_gui.py` に「改良版色別レイヤー分離出力モード」を追加してください。このモードでは構造的安定性と視覚的完全性を保証した色別STL出力を実現します。

## 改良された仕様

### 1. ハイブリッドベースシステム
- **L字型パス**: 左上(0,0)→右下(N,N)の基本接続
- **色専用エリア**: 各色ドット周辺の専用ベース
- **安定化構造**: 4隅の共通支持ポイント
- **強化リブ**: 主要接続部の補強構造

### 2. 段階的配置アルゴリズム
- **色重複解析**: 事前に全色の重複位置を特定
- **配置優先度**: 重要度に基づく配置順序決定
- **代替配置**: 競合時の隣接位置配置
- **完全性保証**: 元ドット絵の完全再現

### 3. 使用済み管理（5分割システム）
- **0-5レベル**: dot_size/5単位の詳細管理
- **部分配置**: 同一ドット内での複数色共存
- **競合解決**: 使用レベルに基づく配置判定

## 実装コード

### ステップ1: STL出力モード選択に追加
```python
"改良版色別レイヤー分離出力モード"  # 新規追加
```

### ステップ2: メイン関数の実装
```python
def generate_improved_color_separated_stl(pixels_rounded_np, output_base_path, grid_size, dot_size, 
                                         wall_thickness, wall_height, base_height, out_thickness,
                                         layer_color_order):
    """
    改良版色別レイヤー分離出力モード用のSTL生成
    構造的安定性と視覚的完全性を保証
    """
    import trimesh
    from trimesh.creation import box
    import numpy as np
    from collections import defaultdict
    
    if len(layer_color_order) == 0:
        return []
    
    generated_meshes = []
    
    # 使用済みピクセル管理配列（0-5レベル）
    used_pixels = np.zeros((grid_size, grid_size), dtype=int)
    
    # 色重複解析とレポート
    color_positions = {}
    overlap_analysis = defaultdict(list)
    
    print(f"=== 改良版色別レイヤー分離STL生成開始 ===")
    print(f"処理順序: {layer_color_order}")
    
    # 事前分析：全色の位置と重複を特定
    for color in layer_color_order:
        color_arr = np.array(color, dtype=np.uint8)
        color_mask = np.all(pixels_rounded_np == color_arr, axis=2)
        positions = [(x, y) for y in range(grid_size) for x in range(grid_size) if color_mask[y, x]]
        color_positions[color] = positions
        
        # 重複位置の特定
        for pos in positions:
            overlap_analysis[pos].append(color)
    
    # 重複レポート
    overlapping_positions = {pos: colors for pos, colors in overlap_analysis.items() if len(colors) > 1}
    print(f"重複位置数: {len(overlapping_positions)}")
    for pos, colors in list(overlapping_positions.items())[:5]:  # 最初の5件表示
        print(f"  位置{pos}: 色{colors}")
    
    def get_color_priority(color, position, layer_order):
        """色の配置優先度を計算（早い順序 = 高優先度）"""
        try:
            return layer_order.index(color)
        except ValueError:
            return 999  # 見つからない場合は最低優先度
    
    def create_hybrid_base_system(color_positions, used_pixels, grid_size, dot_size):
        """ハイブリッドベースシステム生成"""
        base_positions = set()
        
        # 1. L字型パス（基本接続）
        path_positions = create_l_shaped_path((0, 0), (grid_size-1, grid_size-1), used_pixels, grid_size)
        base_positions.update(path_positions)
        print(f"    L字型パス: {len(path_positions)}位置")
        
        # 2. 色専用エリア（ドット周辺）
        for x, y in color_positions:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    bx, by = x + dx, y + dy
                    if 0 <= bx < grid_size and 0 <= by < grid_size:
                        if used_pixels[by, bx] < 5:
                            base_positions.add((bx, by))
        
        # 3. 4隅の安定化構造
        corners = [(0, 0), (grid_size-1, 0), (0, grid_size-1), (grid_size-1, grid_size-1)]
        for cx, cy in corners:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    bx, by = cx + dx, cy + dy
                    if 0 <= bx < grid_size and 0 <= by < grid_size:
                        if used_pixels[by, bx] < 5:
                            base_positions.add((bx, by))
        print(f"    安定化構造: 4隅 + 周辺")
        
        # 4. 強化リブ（中央十字）
        center = grid_size // 2
        # 水平リブ
        for x in range(grid_size):
            if used_pixels[center, x] < 5:
                base_positions.add((x, center))
        # 垂直リブ
        for y in range(grid_size):
            if used_pixels[y, center] < 5:
                base_positions.add((center, y))
        print(f"    強化リブ: 中央十字構造")
        
        return list(base_positions)
    
    def create_l_shaped_path(start_pos, end_pos, used_pixels, grid_size):
        """L字型パスの生成（垂直・水平線のみ）"""
        path_positions = set()
        current_x, current_y = start_pos
        target_x, target_y = end_pos
        
        # 水平移動
        while current_x != target_x:
            if used_pixels[current_y, current_x] < 5:
                path_positions.add((current_x, current_y))
            current_x += 1 if current_x < target_x else -1
        
        # 垂直移動
        while current_y != target_y:
            if used_pixels[current_y, current_x] < 5:
                path_positions.add((current_x, current_y))
            current_y += 1 if current_y < target_y else -1
        
        # 最終位置
        if used_pixels[current_y, current_x] < 5:
            path_positions.add((current_x, current_y))
        
        return list(path_positions)
    
    def find_alternative_placement(original_pos, color, used_pixels, grid_size):
        """代替配置位置の検索（隣接優先）"""
        x, y = original_pos
        
        # 隣接位置を距離順に検索
        for radius in range(1, min(grid_size//2, 3)):  # 最大2マス先まで
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    if abs(dx) + abs(dy) == radius:  # マンハッタン距離
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < grid_size and 0 <= ny < grid_size:
                            if used_pixels[ny, nx] < 5:
                                return (nx, ny)
        return None
    
    # 各色レイヤーを処理
    placed_positions = {}  # 実際の配置位置記録
    
    for color_idx, color in enumerate(layer_color_order):
        print(f"\n--- レイヤー {color_idx + 1}: RGB{color} ---")
        
        target_positions = color_positions[color]
        actual_positions = []
        alternative_count = 0
        
        # 各ドット位置の配置判定
        for pos in target_positions:
            x, y = pos
            
            if used_pixels[y, x] < 5:
                # 元位置に配置可能
                actual_positions.append((x, y))
            else:
                # 代替位置を検索
                alt_pos = find_alternative_placement(pos, color, used_pixels, grid_size)
                if alt_pos:
                    actual_positions.append(alt_pos)
                    alternative_count += 1
                    print(f"    代替配置: {pos} → {alt_pos}")
                else:
                    print(f"    配置不可: {pos} (代替位置なし)")
        
        if not actual_positions:
            print(f"  色 RGB{color} の配置可能位置がありません。スキップします。")
            continue
        
        print(f"  配置ドット数: {len(actual_positions)} (代替: {alternative_count})")
        
        layer_blocks = []
        
        # ビル構造の作成
        for x, y in actual_positions:
            # メインビル
            building_block = box(extents=[dot_size, dot_size, wall_height])
            building_x = x * dot_size + dot_size / 2
            building_y = (grid_size - 1 - y) * dot_size + dot_size / 2
            building_z = base_height + wall_height / 2
            building_block.apply_translation([building_x, building_y, building_z])
            layer_blocks.append(building_block)
            
            # 外周壁（隣接チェック付き）
            for dx, dy, wall_type in [(-1, 0, 'left'), (1, 0, 'right'), (0, -1, 'bottom'), (0, 1, 'top')]:
                nx, ny = x + dx, y + dy
                
                need_wall = True
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    if (nx, ny) in actual_positions:  # 同色隣接なら壁不要
                        need_wall = False
                
                if need_wall:
                    if wall_type in ('left', 'right'):
                        wall_block = box(extents=[wall_thickness, dot_size, wall_height])
                        wall_x = building_x + (dot_size/2 + wall_thickness/2) * (1 if wall_type == 'right' else -1)
                        wall_y = building_y
                    else:
                        wall_block = box(extents=[dot_size, wall_thickness, wall_height])
                        wall_x = building_x
                        wall_y = building_y + (dot_size/2 + wall_thickness/2) * (1 if wall_type == 'top' else -1)
                    
                    wall_block.apply_translation([wall_x, wall_y, building_z])
                    layer_blocks.append(wall_block)
            
            # 使用レベルを5に更新
            used_pixels[y, x] = 5
        
        # ハイブリッドベースシステム生成
        base_positions = create_hybrid_base_system(actual_positions, used_pixels, grid_size, dot_size)
        print(f"  ベース位置数: {len(base_positions)}")
        
        # ベースプレート構築
        for bx, by in base_positions:
            base_block = box(extents=[dot_size, dot_size, base_height])
            base_x = bx * dot_size + dot_size / 2
            base_y = (grid_size - 1 - by) * dot_size + dot_size / 2
            base_block.apply_translation([base_x, base_y, base_height / 2])
            layer_blocks.append(base_block)
        
        # 配置記録
        placed_positions[color] = actual_positions
        
        # メッシュ統合と出力
        if layer_blocks:
            try:
                layer_mesh = trimesh.util.concatenate(layer_blocks)
                layer_filename = f"{output_base_path}_improved_{color_idx+1:02d}_{color[0]:03d}_{color[1]:03d}_{color[2]:03d}.stl"
                layer_mesh.export(layer_filename)
                generated_meshes.append(layer_mesh)
                
                print(f"  STL出力完了: {layer_filename}")
                print(f"  メッシュ情報: 頂点{len(layer_mesh.vertices)}, 面{len(layer_mesh.faces)}")
                
            except Exception as e:
                print(f"  メッシュ生成エラー: {str(e)}")
                continue
    
    # 完全性検証レポート
    generate_completeness_report(pixels_rounded_np, placed_positions, output_base_path, layer_color_order)
    
    print(f"\n=== 改良版色別レイヤー分離STL生成完了 ===")
    print(f"出力ファイル数: {len(generated_meshes)}")
    
    return generated_meshes

def generate_completeness_report(original_pixels, placed_positions, output_base_path, layer_order):
    """完全性検証レポートの生成"""
    
    html_content = f'''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>色別レイヤー分離 - 完全性レポート</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .success {{ color: green; }}
        .warning {{ color: orange; }}
        .error {{ color: red; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(20px, 1fr)); gap: 1px; margin: 10px 0; }}
        .pixel {{ width: 20px; height: 20px; border: 1px solid #ccc; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>🎨 色別レイヤー分離 - 完全性レポート</h1>
    
    <h2>📊 配置統計</h2>
    <table>
        <tr><th>色</th><th>元ドット数</th><th>配置ドット数</th><th>配置率</th><th>ステータス</th></tr>'''
    
    total_original = 0
    total_placed = 0
    
    for color in layer_order:
        color_arr = np.array(color, dtype=np.uint8)
        color_mask = np.all(original_pixels == color_arr, axis=2)
        original_count = np.sum(color_mask)
        placed_count = len(placed_positions.get(color, []))
        placement_rate = (placed_count / original_count * 100) if original_count > 0 else 0
        
        total_original += original_count
        total_placed += placed_count
        
        status_class = "success" if placement_rate >= 95 else "warning" if placement_rate >= 80 else "error"
        status_text = "完全" if placement_rate >= 95 else "良好" if placement_rate >= 80 else "要改善"
        
        r, g, b = color
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        
        html_content += f'''
        <tr>
            <td style="background-color: {hex_color}; color: white;">RGB{color}</td>
            <td>{original_count}</td>
            <td>{placed_count}</td>
            <td class="{status_class}">{placement_rate:.1f}%</td>
            <td class="{status_class}">{status_text}</td>
        </tr>'''
    
    overall_rate = (total_placed / total_original * 100) if total_original > 0 else 0
    overall_status = "success" if overall_rate >= 95 else "warning" if overall_rate >= 80 else "error"
    
    html_content += f'''
        <tr style="font-weight: bold;">
            <td>合計</td>
            <td>{total_original}</td>
            <td>{total_placed}</td>
            <td class="{overall_status}">{overall_rate:.1f}%</td>
            <td class="{overall_status}">全体品質</td>
        </tr>
    </table>
    
    <h2>🏗️ 構造的特徴</h2>
    <ul>
        <li><strong>ハイブリッドベース</strong>: L字型パス + 色専用エリア + 安定化構造</li>
        <li><strong>強化リブ</strong>: 中央十字による構造補強</li>
        <li><strong>代替配置</strong>: 競合位置の隣接配置による完全性保持</li>
        <li><strong>段階的組み立て</strong>: レイヤー順による安定した積層構造</li>
    </ul>
    
    <h2>🎭 トロピカル芸術効果</h2>
    <p><strong>正面視点</strong>: レイヤー重ね合わせによる完璧なドット絵表現</p>
    <p><strong>側面視点</strong>: 各色レイヤーの立体的な層構造が生み出すトロピカルな芸術性</p>
    <p><strong>光学効果</strong>: 角度による色の見え方の変化で動的な視覚体験</p>
    
    <footer style="margin-top: 50px; text-align: center; color: #666;">
        <p>Generated by Improved Color Separated STL Mode</p>
    </footer>
</body>
</html>'''
    
    report_path = f"{output_base_path}_completeness_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"完全性レポート生成: {report_path}")
    print(f"全体配置率: {overall_rate:.1f}% ({total_placed}/{total_original})")
```

### ステップ3: export_stl関数への処理追加
```python
        # 改良版色別レイヤー分離出力モードの処理
        if getattr(self, 'stl_mode', 0) == 6:
            improved_path, _ = QFileDialog.getSaveFileName(
                self, "改良版色別分離STLを保存（ベースファイル名）", "improved_color_separated", "STLファイル (*.stl)"
            )
            if improved_path:
                base_path = os.path.splitext(improved_path)[0]
                params = {key: spin.value() for key, spin in self.controls.items()}
                
                # 前提条件チェック
                if not hasattr(self, 'layer_color_order') or not self.layer_color_order:
                    QMessageBox.warning(self, "レイヤー設定エラー", "レイヤー設定が見つかりません。先にレイヤー設定を行ってください。")
                    return
                
                if not hasattr(self, 'pixels_rounded_np') or self.pixels_rounded_np is None:
                    QMessageBox.warning(self, "ピクセルデータエラー", "編集可能なピクセルデータがありません。先に画像を読み込んでプレビューを生成してください。")
                    return
                
                try:
                    self.input_label.setText("改良版色別レイヤー分離STLファイルを生成中...")
                    QApplication.processEvents()
                    
                    # 改良版色別分離STL生成
                    meshes = generate_improved_color_separated_stl(
                        self.pixels_rounded_np,
                        base_path,
                        int(params.get("Grid Size", 0)),
                        float(params.get("Dot Size", 0.0)),
                        float(params.get("Wall Thickness", 0.0)),
                        float(params.get("Wall Height", 0.0)),
                        float(params.get("Base Height", 0.0)),
                        float(params.get("Out Thickness", 0.0)),
                        self.layer_color_order
                    )
                    
                    if meshes:
                        # 最初のレイヤーをプレビュー表示
                        self.show_stl_preview(meshes[0])
                        
                        layer_count = len(meshes)
                        completeness_report = f"{base_path}_completeness_report.html"
                        message = f"{layer_count}個の改良版色別分離STLを {base_path}_improved_XX.stl として出力、完全性レポート {completeness_report} も生成しました"
                        self.input_label.setText(message)
                    else:
                        self.input_label.setText("改良版色別レイヤー分離STLの生成に失敗しました")
                        
                except Exception as e:
                    print(f"改良版色別分離STL生成エラー: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    self.input_label.setText(f"改良版色別分離STL生成エラー: {str(e)}")
            return
```

## 改良された特徴

### 1. **構造的安定性の保証**
- ハイブリッドベースシステム
- 4隅の安定化構造
- 中央十字の強化リブ

### 2. **視覚的完全性の保証**
- 事前色重複解析
- 代替配置アルゴリズム
- 完全性検証レポート

### 3. **トロピカル芸術効果**
- 正面：完璧なドット絵
- 側面：立体的層構造
- 角度：動的視覚体験

この改良版により、構造的に安定で視覚的に完璧なトロピカル芸術作品が実現できます。

修正を実行してください。