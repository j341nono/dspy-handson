# DSPy + GEPA ハンズオン

DSPyで英文難易度分類プログラムを作成し、GEPAによってプロンプトを最適化するための小規模なハンズオンです。

初期状態のプログラムには、出力ラベル `L0`、`L1`、`L2` の意味を説明していません。GEPA用の評価関数が、正誤スコアに加えて自然言語の診断フィードバックを返すことで、Reflection LMがラベルの対応関係と難易度判定基準を推測し、Signatureの指示文を改善します。

## このハンズオンで学ぶこと

- `dspy.Signature`による入出力定義
- `dspy.Module`と`dspy.Predict`によるプログラム構築
- `dspy.Example`によるtrain／validation／testデータの作成
- 最適化前のベースライン評価
- GEPA向けの`score`と`feedback`を返す評価関数
- `dspy.GEPA`によるコンパイル
- 最適化前後の指示文とtest精度の比較
- 最適化済みプログラムの保存

## ファイル構成

```text
.
├── .env.example
├── .gitignore
├── .python-version
├── README.md
├── dspy_gepa_handson.py
└── pyproject.toml
```

`uv sync`の実行後には`uv.lock`と`.venv`が生成されます。プログラムの実行後には、次の出力ディレクトリが作成されます。

```text
artifacts/
├── gepa_logs/
└── sentence_difficulty_gepa.json
```

## 1. uvのインストール

すでに`uv`を利用できる場合、この手順は不要です。

macOSまたはLinuxでは、公式インストーラーを利用できます。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Homebrewを利用する場合は、次でもインストールできます。

```bash
brew install uv
```

インストールを確認します。

```bash
uv --version
```

## 2. プロジェクトの準備

ZIPを展開し、プロジェクトディレクトリへ移動します。

```bash
unzip dspy_gepa_handson.zip
cd dspy_gepa_handson
```

`.python-version`に従ってPython 3.11を用意し、依存関係を同期します。

```bash
uv python install
uv sync
```

`uv sync`を実行すると、プロジェクト直下に`.venv`が作成されます。通常は仮想環境を手動でactivateする必要はなく、以降のコマンドを`uv run`経由で実行できます。

## 3. APIキーとモデルの設定

環境変数のひな型をコピーします。

```bash
cp .env.example .env
```

`.env`を編集し、利用するモデル名とAPIキーを設定します。

```env
TASK_MODEL=openai/使用するタスクモデル
REFLECTION_MODEL=openai/使用するリフレクションモデル

OPENAI_API_KEY=使用するAPIキー

GEPA_BUDGET=light
GEPA_THREADS=2
```

### モデルの役割

- `TASK_MODEL`: 文難易度を実際に分類するモデルです。比較的小さく安価なモデルから始められます。
- `REFLECTION_MODEL`: 評価フィードバックを分析し、より良い指示文を提案するモデルです。可能であれば、`TASK_MODEL`より強いモデルを指定します。

AnthropicやGeminiなどを利用する場合は、モデル識別子とAPIキーの環境変数を対応するものへ変更してください。

```env
# 例
# TASK_MODEL=anthropic/使用するモデル
# REFLECTION_MODEL=anthropic/使用するモデル
# ANTHROPIC_API_KEY=使用するAPIキー
```

## 4. 実行

まずは、探索コストの小さい`light`で実行します。

```bash
uv run python dspy_gepa_handson.py \
  --budget light \
  --threads 2
```

APIのレート制限が発生する場合は、並列数を1に下げます。

```bash
uv run python dspy_gepa_handson.py \
  --budget light \
  --threads 1
```

最後にtask LMへ送信された内容も確認する場合は、`--show-history`を付けます。

```bash
uv run python dspy_gepa_handson.py \
  --budget light \
  --threads 1 \
  --show-history
```

## 5. 実行時に確認する内容

### 最適化前の指示文

初期Signatureは、次のようにラベルの意味を説明していません。

```text
Assign exactly one label, L0, L1, or L2, to the English sentence.
```

実際のコードでは、`L0`、`L1`、`L2`のいずれかを出力するよう指定していますが、それぞれがどの難易度に対応するかは明示していません。

### GEPAへ渡すフィードバック

誤分類した場合、評価関数はおおむね次の情報を返します。

```text
L2と予測したが、正解はL1である。
L1は、従属節や関係節を1つ程度含む文、中程度に抽象的な文、
または複数の考えの関係がやや複雑な文を表す。
```

GEPAは、このスコアと自然言語フィードバックを使って指示文を改善します。

### 最適化後の指示文

最適化が成功すると、指示文には次のような判定規則が反映されることが期待されます。

```text
L0は、一般的な語彙と単純な構文からなる平易な文に使用する。
L1は、従属節や関係節、中程度の抽象性を含む文に使用する。
L2は、深い埋め込み構造、専門語彙、名詞化、抽象的推論を含む難解な文に使用する。
```

生成される指示文は、使用モデルやGEPAの探索結果によって変わります。

### 最適化前後のtest精度

最後に、GEPAへ一切渡していないtest setで、最適化前後のaccuracyを比較します。

```text
Baseline test accuracy : ...
Optimized test accuracy: ...
```

この比較により、validation setに対する改善だけでなく、未知データへの汎化も確認できます。

## 6. 主なコマンド

依存関係を同期する場合:

```bash
uv sync
```

プログラムを実行する場合:

```bash
uv run python dspy_gepa_handson.py
```

Pythonの構文を検査する場合:

```bash
uv run python -m py_compile dspy_gepa_handson.py
```

依存関係を追加する場合:

```bash
uv add パッケージ名
```

ロックファイルを更新する場合:

```bash
uv lock
```

## 7. 次の練習案

このコードを一度動かした後は、次の順で拡張すると理解しやすくなります。

1. `L0/L1/L2`をCEFRの`A1/A2/B1/B2/C1/C2`へ変更する
2. データをJSONLまたはCSVから読み込む
3. Accuracyに加えてMacro-F1を計算する
4. 語彙・構文・文長ごとに誤り分析を行う
5. `Predict`と`ChainOfThought`を比較する
6. MIPROv2とGEPAを同一データ分割で比較する
7. LLM-as-a-Judgeではなく、ルールベースのfeedbackと組み合わせる

## 注意点

- GEPAは複数回LMを呼び出すため、通常の推論よりAPI利用量が増えます。
- 最初は`--budget light --threads 1`で動作を確認するのが安全です。
- `.env`にはAPIキーが含まれるため、Gitへコミットしないでください。
- test setは最終評価専用です。GEPAの`trainset`または`valset`へ混ぜないでください。

