# Multi-Resolution Pixel Art Format (MRPAF) Specification v1.1

## 概要

Multi-Resolution Pixel Art Format (MRPAF - マーパフ) は、異なる解像度のレイヤーを組み合わせることで、高品質なドット絵を効率的に管理・保存するためのJSONベースのフォーマットです。

### 主な特徴
- レイヤーごとの独立した解像度設定
- サブピクセル精度のアニメーション対応
- 包括的なメタデータ
- 拡張可能な構造

### バージョン履歴
- v1.0: 初版リリース
- v1.1: マルチ解像度とアニメーションの整合性改善

## ファイル構造

### 基本構造

```json
{
  "format": "MRPAF",
  "version": "1.1",
  "metadata": {},
  "canvas": {},
  "coordinateSystem": {},
  "palette": [],
  "layers": [],
  "animations": {},
  "resources": {}
}
```

## 詳細仕様

### 1. ルートオブジェクト

| フィールド | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| format | string | ✓ | フォーマット識別子。常に "MRPAF" |
| version | string | ✓ | フォーマットバージョン（"1.1"） |
| metadata | object | ✓ | 作品情報 |
| canvas | object | ✓ | キャンバス設定 |
| coordinateSystem | object | ✗ | 座標系の詳細設定 |
| palette | array | ✓ | カラーパレット |
| layers | array | ✓ | レイヤー配列 |
| animations | object | ✗ | アニメーション定義 |
| resources | object | ✗ | 外部リソース参照 |

### 2. メタデータ (metadata)

```json
{
  "metadata": {
    "title": "作品タイトル",
    "author": "作者名",
    "created": "2024-01-15T10:30:00Z",
    "modified": "2024-01-20T15:45:00Z",
    "description": "作品の説明",
    "tags": ["タグ1", "タグ2"],
    "license": "CC BY-SA 4.0",
    "work": {
      "series": "シリーズ名",
      "character": "キャラクター名",
      "scene": "シーン説明",
      "variation": "バリエーション"
    },
    "tool": {
      "name": "作成ツール名",
      "version": "1.0.0"
    }
  }
}
```

### 3. キャンバス設定 (canvas)

```json
{
  "canvas": {
    "baseWidth": 24,
    "baseHeight": 24,
    "pixelUnit": 1,
    "backgroundColor": "#00000000"
  }
}
```

| フィールド | 型 | 説明 |
|-----------|-----|------|
| baseWidth | integer | 基準キャンバス幅（ピクセル） |
| baseHeight | integer | 基準キャンバス高（ピクセル） |
| pixelUnit | number | 基準ピクセルサイズ（デフォルト: 1） |
| backgroundColor | string | 背景色（16進数RGBA） |

### 4. 座標系 (coordinateSystem)

```json
{
  "coordinateSystem": {
    "origin": "top-left",
    "baseUnit": 1,
    "subPixelPrecision": 4,
    "allowFloatingPoint": true
  }
}
```

| フィールド | 型 | 説明 |
|-----------|-----|------|
| origin | string | 原点位置（"top-left", "bottom-left" など） |
| baseUnit | number | 基準単位 |
| subPixelPrecision | integer | サブピクセル精度（分割数） |
| allowFloatingPoint | boolean | 浮動小数点座標の許可 |

### 5. パレット (palette)

```json
{
  "palette": [
    {
      "id": 0,
      "name": "透明",
      "hex": "#00000000",
      "rgb": [0, 0, 0, 0],
      "usage": "background"
    }
  ]
}
```

### 6. レイヤー (layers)

```json
{
  "layers": [
    {
      "id": 0,
      "name": "レイヤー名",
      "type": "raster",
      "visible": true,
      "locked": false,
      "opacity": 1.0,
      "blending": {
        "mode": "normal",
        "resolution": "target",
        "interpolation": "nearest"
      },
      "resolution": {
        "pixelArraySize": {
          "width": 48,
          "height": 48
        },
        "scale": 2,
        "effectiveSize": {
          "width": 24,
          "height": 24
        }
      },
      "placement": {
        "x": 12,
        "y": 12,
        "width": 12,
        "height": 12,
        "unit": "base",
        "allowSubPixel": true
      },
      "viewport": {
        "x": 0,
        "y": 0,
        "width": 48,
        "height": 48
      },
      "pixels": {
        "encoding": "array",
        "data": [[0, 1, 2], [1, 2, 3]]
      }
    }
  ]
}
```

#### レイヤープロパティ詳細

##### 基本プロパティ

| フィールド | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| id | integer | ✓ | レイヤーID（0から開始） |
| name | string | ✓ | レイヤー名 |
| type | string | ✗ | レイヤータイプ（"raster", "group"） |
| visible | boolean | ✗ | 表示/非表示（デフォルト: true） |
| locked | boolean | ✗ | ロック状態（デフォルト: false） |
| opacity | number | ✗ | 不透明度（0.0-1.0、デフォルト: 1.0） |

##### ブレンディング設定 (blending)

| フィールド | 型 | 説明 |
|-----------|-----|------|
| mode | string | ブレンドモード（"normal", "multiply" など） |
| resolution | string | 解像度処理方法（"source", "target"） |
| interpolation | string | 補間方法（"nearest", "bilinear"） |

##### 解像度設定 (resolution)

| フィールド | 型 | 説明 |
|-----------|-----|------|
| pixelArraySize | object | ピクセル配列の実サイズ |
| scale | number | ベース解像度に対する倍率 |
| effectiveSize | object | ベース座標系での実効サイズ |

##### 配置設定 (placement)

| フィールド | 型 | 説明 |
|-----------|-----|------|
| x | number | X座標（浮動小数点可） |
| y | number | Y座標（浮動小数点可） |
| width | number | 表示幅 |
| height | number | 表示高さ |
| unit | string | 座標単位（"base" または "actual"） |
| allowSubPixel | boolean | サブピクセル配置の許可 |

##### ビューポート (viewport)

実際のピクセル配列の中で使用する領域を指定：

| フィールド | 型 | 説明 |
|-----------|-----|------|
| x | integer | 開始X座標（ピクセル配列内） |
| y | integer | 開始Y座標（ピクセル配列内） |
| width | integer | 使用幅 |
| height | integer | 使用高さ |

### 7. ピクセルデータエンコーディング

#### 配列形式（非圧縮）
```json
{
  "encoding": "array",
  "data": [[0, 1, 2], [3, 4, 5]]
}
```

#### Base64圧縮形式
```json
{
  "encoding": "base64",
  "compression": "zlib",
  "data": "eJzt1k1rhEAQBOD..."
}
```

#### ランレングス圧縮形式
```json
{
  "encoding": "rle",
  "data": "0:3,2:3,0:2|0:2,2:1,1:3,2:1,0:1"
}
```

#### インデックス化RLE形式
```json
{
  "encoding": "indexed-rle",
  "palette": [0, 1, 2, 3],
  "data": "0:10,1:5,2:3,0:8"
}
```

#### スパース配列形式（高解像度レイヤー向け）
```json
{
  "encoding": "sparse",
  "dimensions": {"width": 48, "height": 48},
  "defaultValue": 0,
  "values": [
    {"x": 10, "y": 15, "color": 2},
    {"x": 11, "y": 15, "color": 2}
  ]
}
```

#### デルタ圧縮形式（アニメーション向け）
```json
{
  "encoding": "delta",
  "baseFrame": 0,
  "changes": [
    {"x": 10, "y": 5, "oldColor": 1, "newColor": 2}
  ]
}
```

#### 領域ベース圧縮形式
```json
{
  "encoding": "regions",
  "dimensions": {"width": 32, "height": 32},
  "regions": [
    {
      "x": 0, "y": 0,
      "width": 16, "height": 16,
      "encoding": "rle",
      "data": "0:256"
    }
  ]
}
```

#### 圧縮方式の選択ガイドライン

| 条件 | 推奨エンコーディング |
|------|---------------------|
| 解像度スケール ≥ 4 かつ 非ゼロピクセル < 30% | sparse |
| 色数 ≤ 4 | indexed-rle |
| 連続した同色ピクセルが多い | rle |
| アニメーションフレーム | delta |
| 部分的に異なるパターン | regions |
| その他 | array または base64 |

### 8. アニメーション (animations)

```json
{
  "animations": {
    "walk": {
      "fps": 8,
      "loops": true,
      "interpolation": {
        "spatial": "nearest",
        "temporal": "step"
      },
      "frames": [
        {
          "duration": 100,
          "layers": [0, 1],
          "overrides": {
            "1": {
              "placement": {
                "x": 12.25,
                "y": 11.5
              },
              "opacity": 0.8
            }
          }
        }
      ]
    },
    "eyeBlink": {
      "targetLayer": 1,
      "targetRegion": {
        "x": 0,
        "y": 0,
        "width": 8,
        "height": 8
      },
      "frames": [
        {
          "duration": 2000,
          "visible": true
        },
        {
          "duration": 100,
          "visible": false
        }
      ]
    }
  }
}
```

#### アニメーションプロパティ

| フィールド | 型 | 説明 |
|-----------|-----|------|
| fps | number | フレームレート |
| loops | boolean | ループ再生 |
| interpolation | object | 補間設定 |
| targetLayer | integer | 対象レイヤーID（部分アニメーション用） |
| targetRegion | object | 対象領域（部分アニメーション用） |

## 座標系と解像度の扱い

### 座標変換ルール

1. **ベース座標からレイヤー座標への変換**
   ```
   レイヤー座標 = ベース座標 × レイヤースケール
   ```

2. **レイヤー座標からベース座標への変換**
   ```
   ベース座標 = レイヤー座標 ÷ レイヤースケール
   ```

3. **サブピクセル配置**
   - `allowSubPixel: true` の場合、浮動小数点座標を使用可能
   - アニメーション時の滑らかな移動に対応

### 解像度混在時の描画順序

1. キャンバスを背景色で初期化
2. レイヤーをID順にソート
3. 各レイヤーについて：
   - viewportで指定された領域を切り出し
   - placementの座標に配置
   - 解像度に応じてスケーリング
   - blending設定に従って合成

## 実装例

### 基本的な16x16キャラクター

```json
{
  "format": "MRPAF",
  "version": "1.1",
  "metadata": {
    "title": "シンプルキャラクター"
  },
  "canvas": {
    "baseWidth": 16,
    "baseHeight": 16
  },
  "palette": [
    {"id": 0, "name": "透明", "hex": "#00000000"}
  ],
  "layers": [
    {
      "id": 0,
      "name": "ベース",
      "resolution": {
        "pixelArraySize": {"width": 16, "height": 16},
        "scale": 1
      },
      "placement": {
        "x": 0,
        "y": 0,
        "width": 16,
        "height": 16
      },
      "pixels": {
        "encoding": "array",
        "data": []
      }
    }
  ]
}
```

### マルチ解像度の例

```json
{
  "format": "MRPAF",
  "version": "1.1",
  "canvas": {
    "baseWidth": 24,
    "baseHeight": 24
  },
  "coordinateSystem": {
    "allowFloatingPoint": true,
    "subPixelPrecision": 4
  },
  "layers": [
    {
      "id": 0,
      "name": "ベースキャラクター",
      "resolution": {
        "pixelArraySize": {"width": 24, "height": 24},
        "scale": 1
      },
      "placement": {
        "x": 0,
        "y": 0,
        "width": 24,
        "height": 24
      }
    },
    {
      "id": 1,
      "name": "目の詳細",
      "resolution": {
        "pixelArraySize": {"width": 16, "height": 8},
        "scale": 4,
        "effectiveSize": {"width": 4, "height": 2}
      },
      "placement": {
        "x": 4,
        "y": 5,
        "width": 4,
        "height": 2,
        "allowSubPixel": true
      }
    }
  ]
}
```

## バリデーションルール

### 必須チェック
- `format` は "MRPAF" であること
- `version` は "1.1" または互換性のあるバージョンであること
- 各レイヤーの `id` は一意であること
- パレットの `id` は一意であること

### 整合性チェック
- `placement` の座標がキャンバス範囲内であること
- `viewport` がピクセル配列サイズを超えないこと
- アニメーションで参照するレイヤーIDが存在すること

## 拡張性

### カスタムプロパティ

"x-" プレフィックスを使用してカスタムプロパティを追加可能：

```json
{
  "x-customTool": {
    "brushSize": 2,
    "gridSnap": true
  }
}
```

### プラグイン拡張

```json
{
  "extensions": {
    "particleEffects": {
      "version": "1.0",
      "data": {}
    }
  }
}
```

## 圧縮戦略

### 自動圧縮選択

MRPAFは以下の条件に基づいて最適な圧縮方式を自動選択することを推奨：

```javascript
function recommendCompression(layer) {
  const scale = layer.resolution.scale;
  const pixels = layer.pixels.data;
  const nonZeroRatio = countNonZero(pixels) / (pixels.length * pixels[0].length);
  const uniqueColors = countUniqueColors(pixels);
  
  // 高解像度で疎なデータ
  if (scale >= 4 && nonZeroRatio < 0.3) {
    return "sparse";
  }
  
  // 色数が少ない
  if (uniqueColors <= 4) {
    return "indexed-rle";
  }
  
  // RLE効率が高い
  if (calculateRLEEfficiency(pixels) > 0.5) {
    return "rle";
  }
  
  // 大きなデータ
  if (pixels.length * pixels[0].length > 1024) {
    return "base64";
  }
  
  return "array";
}
```

### ファイルサイズ最適化の目安

| レイヤータイプ | 典型的な圧縮率 | 推奨方式 |
|---------------|---------------|---------|
| ベース（1x） | 30-50% | rle, indexed-rle |
| 詳細（2-4x） | 10-30% | sparse, regions |
| 高詳細（4x以上） | 5-20% | sparse |
| アニメーション | 20-40% | delta |

## ライセンス

このフォーマット仕様はCC0 1.0 Universal (Public Domain)として公開されています。