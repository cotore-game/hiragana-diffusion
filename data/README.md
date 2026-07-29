# Data

このディレクトリには、データセットの定義とローカル生成物を配置します。

- `fonts/`: ローカルで使用するフォント。ライセンス確認なしにGitへ追加しません。
- `raw/`: 元データ。Git管理外です。
- `generated/`: 生成した学習画像。Git管理外です。
- `datasets/`: 設定から生成したデータセット。`.gitkeep`以外はGit管理外です。

使用フォント名、入手元、ライセンス、ファイルハッシュ、文字集合、除外条件、生成設定、乱数seedは、実験を再現できる形で記録します。

固定データセットは次の構造で生成します。

```text
datasets/<dataset-name>/
├── config.json
├── manifest.csv
└── <font-id>/<character-index>_<codepoint>/<sample-index>.png
```
