# Piper TTS CLI

こちらは、以下のリポジトリのforkです。
https://github.com/ayutaz/piper-plus

## 事前準備
こちらからONNXを入手します
https://github.com/ayutaz/piper-plus/tree/dev/huggingface-space/models


## 機能
- ONNX Runtimeを使用したテキスト音声合成


## インストール

このプロジェクトは依存関係管理にUVを使用しています。依存関係をインストール：

```bash
uv sync
```

## 使用方法

基本的な使用方法：
```bash
uv run ./piper_tts_cli.py "Hello, world!" -m model.onnx -o output.wav
```

日本語テキストの音声合成：
```bash
uv run ./piper_tts_cli.py "はじめまして" -m ./ja_JP-test-medium.onnx -o out.wav
```

パフォーマンス ベンチマーク付き：
```bash
uv run ./piper_tts_cli.py "はじめまして" -m ./ja_JP-test-medium.onnx \
  --runs 10 --warmup 2 -o out.wav
```

ファイルからテキストを読み込み：
```bash
uv run ./piper_tts_cli.py -m model.onnx --text-file input.txt -o output.wav
```

## オプション

- `-m, --model`: ONNXモデルファイルのパス（必須）
- `-o, --output`: 出力音声ファイルのパス（デフォルト: output.wav）
- `-c, --config`: モデル設定JSONファイル（指定されない場合は自動検出）
- `--text-file`: ファイルから入力テキストを読み込み
- `--runs`: ベンチマーク用の推論実行回数（デフォルト: 1）
- `--warmup`: ベンチマーク前のウォームアップ実行回数（デフォルト: 0）

## 動作要件

- Python 3.11+
- pyproject.tomlに記載されている依存関係：
  - onnxruntime
  - numpy
  - soundfile
  - espeak-phonemizer（日本語以外の言語用）
  - pyopenjtalk（日本語用）

## 実行例

```
$ uv run ./piper_tts_cli.py "はじめまして" -m ./ja_JP-test-medium.onnx \
  --runs 10 --warmup 2 -o out.wav
INFO: Loaded model: ja_JP-test-medium.onnx
INFO: Config: ja_JP-test-medium.onnx.json
INFO: Language: ja
INFO: Sample rate: 22050
INFO: Benchmark (ORT run(None, inputs)) — runs=10 warmup=2 | mean=187.366 ms median=169.398 ms p95=241.808 ms min=156.537 ms max=244.318 ms
INFO: Audio saved to: out.wav
INFO: Duration: 2.50 seconds

```

