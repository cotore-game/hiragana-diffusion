# Model Miniaturization Experiment

このブランチだけで進める、Scratch上でのひらがな拡散モデル推論実験です。リポジトリ本流の多フォント・組合せ汎化実験とは分離します。

## 成功条件

最初の目標は高品質なフォント再現ではありません。

- 通常のScratchプロジェクト内だけで推論する
- 重みを1つのリストへ格納できる
- 総パラメーター数を200,000未満にする
- 64×64グレースケール画像を生成する
- 破綻が残っても、ひらがなとして認識できる
- 拡散モデルとして複数ステップでノイズから画像を生成する

## 最初のstudentモデル

| 項目 | 値 |
|---|---|
| 画像サイズ | 64 × 64 |
| 文字条件 | 46 |
| フォント条件 | 9 |
| timesteps | 1,000 |
| base channels | 8 |
| channel multipliers | 1, 2, 3 |
| condition dimension | 32 |
| 総パラメーター数 | 73,721 |
| 上限に対する使用率 | 約36.9% |

画像サイズはパラメーター数に直接影響しませんが、畳み込みの演算回数と中間特徴量には大きく影響します。この実験では品質を優先して64×64を維持し、チャンネル数と演算を削減します。

## Scratch向けに制限した演算

studentモデルは次の演算だけで構成します。

- 通常の3×3畳み込み
- 1×1 skip畳み込み
- ReLU
- 平均プーリング
- 最近傍拡大
- 特徴マップの加算と結合
- Embeddingと全結合

標準モデルで使用していた次の演算は含みません。

- GroupNorm
- SiLU
- ConvTranspose2d
- sinusoidal timestep embedding

時刻条件は、Scratchでsin/cosを毎回計算せずに済むよう、学習可能な1,000×32のEmbeddingとして保持します。

## 学習

既存の9フォント・41,400枚の64×64データセットを再利用します。

```bash
python scripts/train.py \
  --config configs/train.miniature-64.json
```

学習はまだ実行していません。開始前にGPU使用、予想時間、出力先を確認します。

## 生成

学習後は既存のDDIM生成CLIを利用できます。

```bash
python scripts/sample.py \
  --checkpoint outputs/miniature-64/latest.pt \
  --output outputs/miniature-64/samples \
  --steps 10 \
  --seed 20260730 \
  --ema-model
```

最初はPC上で50 stepsの生成能力を確認し、その後10、8、4 stepsへ減らします。少ステップで崩れる場合はteacherモデルからの蒸留を行います。

## 段階

1. 73,721パラメーターのstudentを通常のDDPM lossで学習
2. PC上で文字として認識できるか確認
3. 通常モデルとEMAモデルを比較
4. 50 stepsから10 steps以下へ削減
5. 必要なら4.75M teacherから蒸留
6. 重みを量子化
7. Scratch用の1次元リストへ書き出し
8. ScratchでforwardとDDIM推論を実装

## 未解決の制約

200,000パラメーター未満でも、Scratchで十分速いとは限りません。特に64×64の畳み込みと複数回のU-Net実行が主なボトルネックです。

最初の実測後、必要に応じて次を検討します。

- depthwise separable convolution
- チャンネル数の追加削減
- 中間解像度の削減
- 4-stepまたは1-stepへの蒸留
- 整数量子化
- TurboWarpでの比較

32×32化は最初から採用せず、64×64で認識可能性と実行時間を確認した後の退避案とします。
