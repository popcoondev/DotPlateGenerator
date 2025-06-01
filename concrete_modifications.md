# Claude Code プロンプト: レイヤースタックモード実装

## タスク概要
既存のPythonファイル `dot_plate_generator_gui.py` に「レイヤースタックモード」機能を追加してください。

## 修正内容

### 1. STL出力モード選択の拡張
`DotPlateApp.__init__` メソッド内（行番号約1200付近）で、`self.stl_mode_combo.addItems()` の部分に以下を追加：

```python
# 既存の4つのアイテムに加えて
"レイヤースタックモード"  # 新規追加（index=4）
```

### 2. 新規関数の追加
`generate_layered_stl` 関数の直後（行番号約300付近）に以下の関数を追加：

```python
def generate_layer_stack_stl(pixels_rounded_np, output_base_path, grid_size, dot_size, 
                            wall_thickness, wall_height, base_height, out_thickness,
                            layer_color_order, layer_heights):
    """
    レイヤースタックモード用のSTL生成
    各レイヤーを個別のSTLファイルとして出力
    
    レイヤー構造：
    - 全レイヤーのベース高さ = base_height（統一）
    - レイヤー1: ビル高さ = wall_height
    - レイヤー2: ビル高さ = wall_height - base_height × 1  
    - レイヤーn: ビル高さ = wall_height - base_height × (n-1)
    - 重ね合わせ後の最終ビル高さ = base_height + wall_height（統一）
    - 下位レイヤーのビル部分は上位レイヤーで貫通穴として処理
    """
    if len(layer_color_order) == 0:
        return []
    
    generated_meshes = []
    
    # 各レイヤーを下位から順に処理
    for layer_idx, color in enumerate(layer_color_order):
        layer_num = layer_idx + 1  # レイヤー番号は1から開始
        
        # このレイヤーの色マスクを作成
        color_arr = np.array(color, dtype=np.uint8)
        current_layer_mask = np.all(pixels_rounded_np == color_arr, axis=2)
        
        # 下位レイヤーからの貫通穴を計算
        hole_mask = np.zeros_like(current_layer_mask, dtype=bool)
        for lower_layer_idx in range(layer_idx + 1, len(layer_color_order)):
            lower_color = layer_color_order[lower_layer_idx]
            lower_color_arr = np.array(lower_color, dtype=np.uint8)
            lower_mask = np.all(pixels_rounded_np == lower_color_arr, axis=2)
            hole_mask |= lower_mask
        
        # レイヤーのベース高さとビル高さを計算
        layer_base_height = base_height  # 全レイヤー共通
        layer_building_height = wall_height - base_height * (layer_num - 1)  # レイヤー番号分を引く
        building_top_height = layer_base_height + layer_building_height
        
        # ベースプレート作成（外周拡張込み）
        base_width = grid_size * dot_size + 2 * out_thickness
        base_depth = grid_size * dot_size + 2 * out_thickness
        base_block = box(extents=[base_width, base_depth, layer_base_height])
        base_x = (grid_size * dot_size) / 2 - out_thickness
        base_y = (grid_size * dot_size) / 2 - out_thickness
        base_block.apply_translation([base_x, base_y, layer_base_height / 2])
        
        # 下位レイヤーの穴を開ける（CSG差分演算）
        for y in range(grid_size):
            for x in range(grid_size):
                if hole_mask[y, x]:
                    hole_block = box(extents=[dot_size, dot_size, layer_base_height + 0.1])
                    hole_x = x * dot_size + dot_size / 2
                    hole_y = (grid_size - 1 - y) * dot_size + dot_size / 2
                    hole_block.apply_translation([hole_x, hole_y, layer_base_height / 2])
                    
                    try:
                        base_block = base_block.difference(hole_block)
                    except Exception as e:
                        print(f"穴開け処理エラー (Layer {layer_num}, 位置 {x},{y}): {str(e)}")
                        continue
        
        layer_blocks = [base_block]
        
        # このレイヤーの色のビル部分とその外周壁を追加
        for y in range(grid_size):
            for x in range(grid_size):
                if current_layer_mask[y, x]:
                    # ビルブロック
                    building_block = box(extents=[dot_size, dot_size, layer_building_height])
                    building_x = x * dot_size + dot_size / 2
                    building_y = (grid_size - 1 - y) * dot_size + dot_size / 2
                    building_z = layer_base_height + layer_building_height / 2
                    building_block.apply_translation([building_x, building_y, building_z])
                    layer_blocks.append(building_block)
                    
                    # ビルの外周壁を追加
                    for dx, dy, wall_type in [(-1, 0, 'left'), (1, 0, 'right'), (0, -1, 'bottom'), (0, 1, 'top')]:
                        nx, ny = x + dx, y + dy
                        
                        # 隣接セルの確認（同色なら内壁不要）
                        need_wall = True
                        if 0 <= nx < grid_size and 0 <= ny < grid_size:
                            if current_layer_mask[ny, nx]:
                                need_wall = False
                        
                        if need_wall:
                            if wall_type in ('left', 'right'):
                                wall_block = box(extents=[wall_thickness, dot_size, layer_building_height])
                                wall_x = building_x + (dot_size/2 + wall_thickness/2) * (1 if wall_type == 'right' else -1)
                                wall_y = building_y
                            else:  # top, bottom
                                wall_block = box(extents=[dot_size, wall_thickness, layer_building_height])
                                wall_x = building_x
                                wall_y = building_y + (dot_size/2 + wall_thickness/2) * (1 if wall_type == 'top' else -1)
                            
                            wall_block.apply_translation([wall_x, wall_y, layer_base_height + layer_building_height / 2])
                            layer_blocks.append(wall_block)
        
        # プレート外周壁を追加
        walls_data = [
            ([-out_thickness/2, base_y, building_top_height/2], [out_thickness, base_depth, building_top_height]),
            ([grid_size * dot_size + out_thickness/2, base_y, building_top_height/2], [out_thickness, base_depth, building_top_height]),
            ([base_x, -out_thickness/2, building_top_height/2], [base_width, out_thickness, building_top_height]),
            ([base_x, grid_size * dot_size + out_thickness/2, building_top_height/2], [base_width, out_thickness, building_top_height])
        ]
        
        for pos, size in walls_data:
            wall = box(extents=size)
            wall.apply_translation(pos)
            layer_blocks.append(wall)
        
        # レイヤーメッシュを統合してファイル出力
        if layer_blocks:
            try:
                layer_mesh = trimesh.util.concatenate(layer_blocks)
                layer_filename = f"{output_base_path}_layer_{layer_num:02d}_{color[0]:03d}_{color[1]:03d}_{color[2]:03d}.stl"
                layer_mesh.export(layer_filename)
                generated_meshes.append(layer_mesh)
                print(f"レイヤー {layer_num} (色: RGB{color}) を {layer_filename} に出力しました")
            except Exception as e:
                print(f"レイヤー {layer_num} のメッシュ生成エラー: {str(e)}")
                continue
    
    return generated_meshes
```

### 3. export_stl関数への処理追加
`export_stl` メソッド内の色レイヤーモード処理（`if getattr(self, 'stl_mode', 0) == 3:`の部分、行番号約2000付近）の直後に以下を追加：

```python
        # レイヤースタックモードの処理
        if getattr(self, 'stl_mode', 0) == 4:
            stack_path, _ = QFileDialog.getSaveFileName(
                self, "レイヤースタックSTLを保存（ベースファイル名）", "layer_stack", "STLファイル (*.stl)"
            )
            if stack_path:
                base_path = os.path.splitext(stack_path)[0]
                params = {key: spin.value() for key, spin in self.controls.items()}
                
                # 前提条件チェック
                if not hasattr(self, 'layer_color_order') or not self.layer_color_order:
                    QMessageBox.warning(self, "レイヤー設定エラー", "レイヤー設定が見つかりません。先にレイヤー設定を行ってください。")
                    return
                
                if not hasattr(self, 'pixels_rounded_np') or self.pixels_rounded_np is None:
                    QMessageBox.warning(self, "ピクセルデータエラー", "編集可能なピクセルデータがありません。先に画像を読み込んでプレビューを生成してください。")
                    return
                
                try:
                    self.input_label.setText("レイヤースタックSTLファイルを生成中...")
                    QApplication.processEvents()
                    
                    # レイヤースタック用STL生成
                    meshes = generate_layer_stack_stl(
                        self.pixels_rounded_np,
                        base_path,
                        int(params.get("Grid Size", 0)),
                        float(params.get("Dot Size", 0.0)),
                        float(params.get("Wall Thickness", 0.0)),
                        float(params.get("Wall Height", 0.0)),
                        float(params.get("Base Height", 0.0)),
                        float(params.get("Out Thickness", 0.0)),
                        self.layer_color_order,
                        getattr(self, 'layer_heights', {})
                    )
                    
                    if meshes:
                        # 最初のレイヤーをプレビュー表示
                        self.show_stl_preview(meshes[0])
                        
                        # HTMLレポート生成
                        first_layer_path = f"{base_path}_layer_01_{self.layer_color_order[0][0]:03d}_{self.layer_color_order[0][1]:03d}_{self.layer_color_order[0][2]:03d}.stl"
                        html_path = self.generate_html_report(first_layer_path, meshes[0])
                        
                        layer_count = len(meshes)
                        message = f"{layer_count}個のレイヤースタックSTLを {base_path}_layer_XX.stl として出力しました"
                        if html_path:
                            message += f"、HTMLレポート {html_path} も生成しました"
                        self.input_label.setText(message)
                    else:
                        self.input_label.setText("レイヤースタックSTLの生成に失敗しました")
                        
                except Exception as e:
                    print(f"レイヤースタックSTL生成エラー: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    self.input_label.setText(f"レイヤースタックSTL生成エラー: {str(e)}")
            return
```

## 実装仕様

### レイヤー構造の原理
- 各レイヤーは独立したSTLファイルとして出力
- 全レイヤー: ベース高さ = base_height（統一）
- レイヤー1: ビル高さ = wall_height
- レイヤー2: ビル高さ = wall_height - base_height × 1
- レイヤーn: ビル高さ = wall_height - base_height × (n-1)
- 重ね合わせ後の最終ビル高さ = base_height + wall_height（全レイヤー統一）
- 下位レイヤーのビル位置は上位レイヤーで貫通穴として処理

### ファイル出力形式
```
{ベース名}_layer_{番号:02d}_{R:03d}_{G:03d}_{B:03d}.stl
例: layer_stack_layer_01_255_000_000.stl
```

### エラーハンドリング
- レイヤー設定未完了時の警告
- ピクセルデータ不正時の警告  
- CSG演算失敗時の例外処理

## テスト方法
1. 画像を読み込み、減色プレビューを生成
2. レイヤー設定で色の順序を調整
3. STL出力モードで「レイヤースタックモード」を選択
4. STLエクスポートを実行
5. 複数のSTLファイルが生成されることを確認

修正を実行してください。