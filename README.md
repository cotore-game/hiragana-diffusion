# Hiragana Diffusion

ひらがなの文字内容と書体を条件として生成する、画像拡散モデルの実験リポジトリです。

## 目標

最初にPC上でConditional DDPMを学習し、ひらがなの生成と、学習時に観測していない「文字 × 書体」の組み合わせへの汎化を検証します。

本命の実験では、明朝体・ゴシック体では全ひらがなを学習し、教科書体では「た・ち・つ・て・と」を除外します。他書体から「た行」の文字内容を、教科書体のほかの文字から書体特徴を学習し、未観測の教科書体た行を生成できるか調べます。

## 実験方針

1. 全書体・全文字を使い、Conditional DDPMが正しく学習・生成できることを確認する
2. 教科書体のた行を除外し、未観測組み合わせへの汎化を検証する
3. フォント数、条件表現、データ拡張、モデル容量を比較する

まず十分な表現力を持つ基準モデルを構築し、生成能力と条件設計を検証します。

## 想定する初期データ

- 64 × 64ピクセル
- グレースケール
- 白背景、黒文字
- グリフのバウンディングボックスを基準に中央揃え
- 平行移動: 縦横それぞれ最大 ±4 px
- スケール: 0.90〜1.10倍
- 回転: 最大 ±3度
- 学習時に連続値から動的にデータ拡張

フォントファイルはライセンスを個別に確認し、原則としてGitでは管理しません。

## ディレクトリ構成

```text
.
├── configs/       # 実験設定
├── data/          # データセットの説明とローカルデータ
├── outputs/       # 生成画像、評価結果、チェックポイント
├── scripts/       # データ生成、学習、生成、評価用の実行スクリプト
├── src/           # モデルと学習処理
└── tests/         # テスト
```

## 環境

- WSL2
- Conda環境: `hiragana-diffusion`
- Python 3.12
- NVIDIA GeForce RTX 5070 Ti

```bash
conda env create -f environment.yml
conda activate hiragana-diffusion
```

既に環境を作成済みの場合は、定義に合わせて更新します。

```bash
conda env update --file environment.yml
```

## データセット生成

`configs/dataset.example.json`をコピーして、データセット名、画像サイズ、フォント、除外文字、データ拡張範囲を設定します。

```bash
python scripts/generate_dataset.py --config configs/dataset.example.json
```

生成先は既定で`data/datasets/<name>/`です。各画像は8-bitグレースケールPNGで、使用条件と実際に適用した変形値は`config.json`と`manifest.csv`へ保存されます。各文字のサンプル0は変形なし、それ以降にはseed付きのランダム変形を適用します。

## テスト

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 学習

`configs/train.example.json`でデータセット、モデル、学習条件を設定します。

```bash
python scripts/train.py --config configs/train.example.json
```

画像は`[-1, 1]`へ正規化し、cosine scheduleを使ったDDPMのノイズ予測損失で学習します。文字ID、書体ID、時刻は独立に埋め込んでU-Netへ与えます。チェックポイントには通常モデル、EMAモデル、optimizer、mixed precision scaler、書体ID対応、設定を保存します。

番号付きチェックポイントを残す場合は、学習設定の`keep_numbered_checkpoints`を`true`にします。既定では`latest.pt`だけを更新します。

## 生成

学習済みチェックポイントから、決定論的DDIMで全条件を生成します。

```bash
python scripts/sample.py \
  --checkpoint outputs/takao-baseline-64/latest.pt \
  --steps 50
```

同じ文字の各書体には同一の初期ノイズを使用します。書体別の個別PNGと、全条件を比較する`grid.png`を`outputs/takao-baseline-64/samples/`へ保存します。

既定では直接学習したモデルを使用します。EMAモデルを比較する場合は`--ema-model`を指定します。
