# Claude Code プロンプト: プラモデル組み立て式モード実装

## タスク概要
既存のPythonファイル `dot_plate_generator_gui.py` に「プラモデル組み立て式モード」を追加してください。このモードでは、各色レイヤーを薄皮連結で繋がった組み立て式パーツとして出力し、プラモデル風の塗装・組み立てワークフローを実現します。

## 新機能の特徴

### 1. プラモデル風の設計思想
- **色別パーツ分割**: 各色ごとに独立したSTLファイル
- **薄皮連結**: 同色ドット間を薄いブリッジで連結（一体成型）
- **ランナーシステム**: プラモデル風の取り外し可能な支持構造
- **組み立てピン**: レイヤー間の位置決め用ピン・穴システム
- **塗装効率**: 色別に一括塗装可能

### 2. 実装手順

#### ステップ1: STL出力モード選択の拡張
`DotPlateApp.__init__` メソッド内のSTL出力モード選択部分に追加：

```python
# 既存の5つのモードに加えて
"プラモデル組み立て式モード"  # 新規追加（index=5）
```

#### ステップ2: 新規関数の追加
`generate_layer_stack_stl` 関数の直後に以下の関数を追加：

```python
def generate_plastic_model_stl(pixels_rounded_np, output_base_path, grid_size, dot_size, 
                               wall_thickness, wall_height, base_height, out_thickness,
                               layer_color_order, connection_thickness=0.1, sprue_width=0.3):
    """
    プラモデル組み立て式モード用のSTL生成
    各色レイヤーを薄皮で連結した組み立て式パーツとして出力
    """
    import trimesh
    from trimesh.creation import box, cylinder
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    from scipy.sparse.csgraph import minimum_spanning_tree
    
    if len(layer_color_order) == 0:
        return []
    
    generated_meshes = []
    
    def create_connection_bridges(positions, grid_size, dot_size):
        """最小全域木アルゴリズムで同色ドット間を効率的に連結"""
        if len(positions) <= 1:
            return []
        
        bridges = []
        
        # 座標を実際の物理位置に変換
        physical_positions = []
        for x, y in positions:
            phys_x = x * dot_size + dot_size / 2
            phys_y = (grid_size - 1 - y) * dot_size + dot_size / 2
            physical_positions.append([phys_x, phys_y])
        
        physical_positions = np.array(physical_positions)
        
        # 最小全域木で連結パスを計算
        distances = pdist(physical_positions)
        dist_matrix = squareform(distances)
        mst = minimum_spanning_tree(dist_matrix)
        mst_array = mst.toarray()
        
        # MST のエッジからブリッジを生成
        for i in range(len(physical_positions)):
            for j in range(i + 1, len(physical_positions)):
                if mst_array[i, j] > 0:  # エッジが存在
                    pos1 = physical_positions[i]
                    pos2 = physical_positions[j]
                    
                    # 2点間のブリッジを作成
                    bridge = create_bridge_between_points(pos1, pos2, connection_thickness, sprue_width)
                    if bridge:
                        bridges.append(bridge)
        
        return bridges
    
    def create_bridge_between_points(pos1, pos2, thickness, width):
        """2点間の薄皮ブリッジを作成"""
        vec = pos2 - pos1
        length = np.linalg.norm(vec)
        
        if length < 0.01:
            return None
        
        # ブリッジの中心位置と向き
        center = (pos1 + pos2) / 2
        center_3d = [center[0], center[1], base_height + thickness / 2]
        
        # 回転角度計算
        angle = np.arctan2(vec[1], vec[0])
        
        # ブリッジボックス作成
        bridge = box(extents=[length, width, thickness])
        bridge.apply_translation(center_3d)
        
        # Z軸周りの回転
        rotation_matrix = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        bridge.apply_transform(rotation_matrix)
        
        return bridge
    
    def create_sprue_system(positions, grid_size, dot_size):
        """ランナーシステム（取り外し可能な支持構造）を作成"""
        if len(positions) == 0:
            return []
        
        sprue_blocks = []
        
        # メインランナー（外周に配置）
        main_runner_y = grid_size * dot_size + out_thickness * 2
        main_runner = box(extents=[grid_size * dot_size, sprue_width, connection_thickness])
        main_runner.apply_translation([
            grid_size * dot_size / 2,
            main_runner_y,
            base_height + connection_thickness / 2
        ])
        sprue_blocks.append(main_runner)
        
        # 各ドットからメインランナーへの接続
        for x, y in positions:
            dot_x = x * dot_size + dot_size / 2
            dot_y = (grid_size - 1 - y) * dot_size + dot_size / 2
            
            # ドットからメインランナーまでの垂直接続
            connection_length = main_runner_y - dot_y - dot_size / 2
            
            if connection_length > 0:
                connector = box(extents=[sprue_width, connection_length, connection_thickness])
                connector_y = dot_y + dot_size / 2 + connection_length / 2
                connector.apply_translation([
                    dot_x,
                    connector_y,
                    base_height + connection_thickness / 2
                ])
                sprue_blocks.append(connector)
        
        return sprue_blocks
    
    def add_assembly_pins(blocks, color, grid_size):
        """組み立て用のピン・穴システムを追加"""
        pin_blocks = []
        
        # 4隅にピン配置
        pin_radius = 0.5
        pin_height = wall_height / 2
        corner_positions = [
            (-out_thickness / 2, -out_thickness / 2),
            (grid_size * dot_size + out_thickness / 2, -out_thickness / 2),
            (-out_thickness / 2, grid_size * dot_size + out_thickness / 2),
            (grid_size * dot_size + out_thickness / 2, grid_size * dot_size + out_thickness / 2)
        ]
        
        color_index = layer_color_order.index(color) if color in layer_color_order else 0
        
        for i, (px, py) in enumerate(corner_positions):
            if color_index == 0:  # 最下層にはピン
                pin = cylinder(radius=pin_radius, height=pin_height)
                pin.apply_translation([px, py, base_height + pin_height / 2])
                pin_blocks.append(pin)
        
        return pin_blocks
    
    # 各色レイヤーを処理
    for color in layer_color_order:
        # この色のドット位置を収集
        color_arr = np.array(color, dtype=np.uint8)
        color_mask = np.all(pixels_rounded_np == color_arr, axis=2)
        
        positions = []
        for y in range(grid_size):
            for x in range(grid_size):
                if color_mask[y, x]:
                    positions.append((x, y))
        
        if not positions:
            continue
        
        layer_blocks = []
        
        # 各ドットのビル構造を作成
        for x, y in positions:
            # メインビル
            building_block = box(extents=[dot_size, dot_size, wall_height])
            building_x = x * dot_size + dot_size / 2
            building_y = (grid_size - 1 - y) * dot_size + dot_size / 2
            building_z = base_height + wall_height / 2
            building_block.apply_translation([building_x, building_y, building_z])
            layer_blocks.append(building_block)
            
            # ビルの外周壁
            for dx, dy, wall_type in [(-1, 0, 'left'), (1, 0, 'right'), (0, -1, 'bottom'), (0, 1, 'top')]:
                nx, ny = x + dx, y + dy
                
                need_wall = True
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    if color_mask[ny, nx]:
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
        
        # 同色ドット間の連結ブリッジを作成
        if len(positions) > 1:
            bridges = create_connection_bridges(positions, grid_size, dot_size)
            layer_blocks.extend(bridges)
        
        # スプルーシステム（ランナー）を追加
        sprue_blocks = create_sprue_system(positions, grid_size, dot_size)
        layer_blocks.extend(sprue_blocks)
        
        # 組み立てピンシステムを追加
        pin_blocks = add_assembly_pins(layer_blocks, color, grid_size)
        layer_blocks.extend(pin_blocks)
        
        # ベースプレート（この色の領域のみ）
        if positions:
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            base_x1 = min_x * dot_size - wall_thickness
            base_x2 = (max_x + 1) * dot_size + wall_thickness
            base_y1 = (grid_size - 1 - max_y) * dot_size - wall_thickness
            base_y2 = (grid_size - 1 - min_y + 1) * dot_size + wall_thickness
            
            base_width = base_x2 - base_x1
            base_depth = base_y2 - base_y1
            
            base_block = box(extents=[base_width, base_depth, base_height])
            base_center_x = (base_x1 + base_x2) / 2
            base_center_y = (base_y1 + base_y2) / 2
            base_block.apply_translation([base_center_x, base_center_y, base_height / 2])
            layer_blocks.append(base_block)
        
        # レイヤーメッシュを統合
        if layer_blocks:
            try:
                layer_mesh = trimesh.util.concatenate(layer_blocks)
                layer_filename = f"{output_base_path}_plastic_{color[0]:03d}_{color[1]:03d}_{color[2]:03d}.stl"
                layer_mesh.export(layer_filename)
                generated_meshes.append(layer_mesh)
            except Exception as e:
                print(f"レイヤー RGB{color} のメッシュ生成エラー: {str(e)}")
                continue
    
    # 組み立て説明書用HTMLファイル生成
    generate_assembly_instructions(output_base_path, layer_color_order, grid_size, dot_size)
    
    return generated_meshes

def generate_assembly_instructions(output_base_path, layer_color_order, grid_size, dot_size):
    """組み立て説明書用のHTMLファイルを生成"""
    
    html_content = f'''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>プラモデル組み立て説明書</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .step {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .color-swatch {{ width: 20px; height: 20px; display: inline-block; border: 1px solid #ccc; margin-right: 10px; }}
        .warning {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>🔧 プラモデル組み立て説明書</h1>
    
    <div class="warning">
        <strong>⚠️ 注意事項</strong>
        <ul>
            <li>各パーツは薄いランナー（スプルー）で連結されています</li>
            <li>組み立て前にニッパーでランナーを切り離してください</li>
            <li>塗装は組み立て前に各色ごとに行うことを推奨します</li>
        </ul>
    </div>
    
    <h2>📦 パーツリスト</h2>
    <table border="1" style="border-collapse: collapse; width: 100%;">
        <tr><th>色</th><th>ファイル名</th><th>塗装色</th></tr>'''
    
    for i, color in enumerate(layer_color_order):
        r, g, b = color
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        filename = f"{output_base_path}_plastic_{r:03d}_{g:03d}_{b:03d}.stl"
        
        html_content += f'''
        <tr>
            <td><div class="color-swatch" style="background-color: {hex_color};"></div></td>
            <td>{filename}</td>
            <td>RGB({r}, {g}, {b})</td>
        </tr>'''
    
    html_content += f'''
    </table>
    
    <h2>🔨 組み立て手順</h2>
    <div class="step">
        <h3>ステップ1: パーツの準備</h3>
        <ol>
            <li>各STLファイルを3Dプリントします</li>
            <li>ニッパーでランナーを切り離します</li>
            <li>切り口をやすりで滑らかに仕上げます</li>
        </ol>
    </div>
    
    <div class="step">
        <h3>ステップ2: 塗装</h3>
        <ol>
            <li>各色グループごとに塗装を行います</li>
            <li>同色のパーツをまとめて塗装できるため効率的です</li>
        </ol>
    </div>
    
    <div class="step">
        <h3>ステップ3: 組み立て</h3>
        <ol>
            <li>ベースから順番に重ねていきます</li>
            <li>各コーナーのピン穴に合わせて位置を調整します</li>
            <li>必要に応じて接着剤で固定します</li>
        </ol>
    </div>
    
    <h2>📐 仕様情報</h2>
    <ul>
        <li>グリッドサイズ: {grid_size}×{grid_size}</li>
        <li>ドットサイズ: {dot_size}mm</li>
        <li>完成サイズ: 約{grid_size * dot_size}×{grid_size * dot_size}mm</li>
    </ul>
</body>
</html>'''
    
    instructions_path = f"{output_base_path}_assembly_instructions.html"
    with open(instructions_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
```

#### ステップ3: export_stl関数への処理追加
`export_stl` メソッド内のレイヤースタックモード処理（`if getattr(self, 'stl_mode', 0) == 4:`の部分）の直後に以下を追加：

```python
        # プラモデル組み立て式モードの処理
        if getattr(self, 'stl_mode', 0) == 5:
            plastic_path, _ = QFileDialog.getSaveFileName(
                self, "プラモデルSTLを保存（ベースファイル名）", "plastic_model", "STLファイル (*.stl)"
            )
            if plastic_path:
                base_path = os.path.splitext(plastic_path)[0]
                params = {key: spin.value() for key, spin in self.controls.items()}
                
                # 前提条件チェック
                if not hasattr(self, 'layer_color_order') or not self.layer_color_order:
                    QMessageBox.warning(self, "レイヤー設定エラー", "レイヤー設定が見つかりません。先にレイヤー設定を行ってください。")
                    return
                
                if not hasattr(self, 'pixels_rounded_np') or self.pixels_rounded_np is None:
                    QMessageBox.warning(self, "ピクセルデータエラー", "編集可能なピクセルデータがありません。先に画像を読み込んでプレビューを生成してください。")
                    return
                
                try:
                    self.input_label.setText("プラモデル組み立て式STLファイルを生成中...")
                    QApplication.processEvents()
                    
                    # プラモデル用STL生成
                    meshes = generate_plastic_model_stl(
                        self.pixels_rounded_np,
                        base_path,
                        int(params.get("Grid Size", 0)),
                        float(params.get("Dot Size", 0.0)),
                        float(params.get("Wall Thickness", 0.0)),
                        float(params.get("Wall Height", 0.0)),
                        float(params.get("Base Height", 0.0)),
                        float(params.get("Out Thickness", 0.0)),
                        self.layer_color_order,
                        connection_thickness=0.1,  # 連結薄皮の厚さ
                        sprue_width=0.3           # ランナーの幅
                    )
                    
                    if meshes:
                        # 最初のパーツをプレビュー表示
                        self.show_stl_preview(meshes[0])
                        
                        # HTMLレポート生成
                        first_part_path = f"{base_path}_plastic_{self.layer_color_order[0][0]:03d}_{self.layer_color_order[0][1]:03d}_{self.layer_color_order[0][2]:03d}.stl"
                        html_path = self.generate_html_report(first_part_path, meshes[0])
                        
                        parts_count = len(meshes)
                        instructions_path = f"{base_path}_assembly_instructions.html"
                        message = f"{parts_count}個のプラモデルパーツを {base_path}_plastic_XXX.stl として出力、組み立て説明書 {instructions_path} も生成しました"
                        if html_path:
                            message += f"、HTMLレポート {html_path} も生成しました"
                        self.input_label.setText(message)
                    else:
                        self.input_label.setText("プラモデル組み立て式STLの生成に失敗しました")
                        
                except Exception as e:
                    print(f"プラモデルSTL生成エラー: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    self.input_label.setText(f"プラモデルSTL生成エラー: {str(e)}")
            return
```

## 機能の特徴

### 1. **薄皮連結システム**
- 同色ドット間を薄いブリッジ（0.1mm厚）で連結
- 最小全域木アルゴリズムで効率的な連結パスを計算
- 一体成型により色別一括塗装が可能

### 2. **ランナーシステム**
- プラモデル風の取り外し可能な支持構造
- ニッパーで切り離し可能
- 組み立て時の取り扱いを簡素化

### 3. **組み立てピンシステム**
- 4隅にピン・穴システム配置
- レイヤー間の位置決めを正確化
- 組み立て時の安定性向上

### 4. **自動説明書生成**
- HTML形式の組み立て説明書を自動生成
- パーツリスト、手順、仕様を含む
- 塗装ガイドも含有

## 出力ファイル例

```
plastic_model_plastic_255_000_000.stl  # 赤色パーツ
plastic_model_plastic_000_255_000.stl  # 緑色パーツ
plastic_model_plastic_000_000_255.stl  # 青色パーツ
plastic_model_assembly_instructions.html  # 組み立て説明書
plastic_model.html  # レポートファイル
```

## 使用ワークフロー

1. **設計**: 通常通り画像読み込み・減色・レイヤー設定
2. **出力**: STL出力モードで「プラモデル組み立て式モード」選択
3. **印刷**: 各色パーツを個別に3Dプリント
4. **準備**: ランナー切り離し・表面仕上げ
5. **塗装**: 色別に一括塗装（効率的）
6. **組み立て**: 説明書に従って組み立て

この実装により、効率的な塗装ワークフローと簡単な組み立てが可能なプラモデル風ドットプレートが実現できます。

修正を実行してください。