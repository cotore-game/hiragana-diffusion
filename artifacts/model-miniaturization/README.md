# Model Miniaturization Results

条件付きひらがな拡散モデルを4,754,817パラメーターから200,000未満へ縮小した実験の比較Gridです。すべてseed `20260730`、50-step DDIMで生成しています。

| ファイル | モデル | 教師画像 | 重み | 所見 |
|---|---:|---|---|---|
| `samples/v1-raw-grid.png` | 73,721 | 幾何変形あり41,400枚 | raw | 強い背景ノイズが残る |
| `samples/v1-ema-grid.png` | 73,721 | 幾何変形あり41,400枚 | EMA | rawと同様に判別困難 |
| `samples/v2-augmented-raw-grid.png` | 166,657 | 幾何変形あり41,400枚 | raw | 一部に筆画が現れるが破綻が多い |
| `samples/v2-augmented-ema-grid.png` | 166,657 | 幾何変形あり41,400枚 | EMA | ノイズが減り、線とフォント特徴が部分的に現れる |
| `samples/v2-canonical-ema-grid.png` | 166,657 | 変形なし414枚 | EMA | 筆画状になるが、特定文字が全フォントで共通して破綻 |

## 結論

約96.5%のパラメーター削減には成功しましたが、9フォント基準モデルと同等のひらがな判別品質は維持できませんでした。第2モデルではEMAによるノイズ低減と一部のフォント特徴を確認できるため、次は変形なし教師画像でbatch sizeを下げ、文字条件の学習を改善できるか検証します。

完全な条件、構成、loss、次の比較計画は[`experiments/model-miniaturization/README.md`](../../experiments/model-miniaturization/README.md)を参照してください。
