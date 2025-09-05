# MeloTTS ONNX

MeloTTSのONNXモデルを使用したテキスト音声合成のためのシンプルなコマンドラインインターフェース。

## 特徴

- 多言語サポート (中国語、日本語、英語、韓国語、スペイン語、フランス語)
- 高速推論のためのONNXランタイム
- カスタマイズ可能なサンプルレートと音声速度
- シンプルなCLIインターフェース

## インストール

1. このリポジトリをクローンします
2. 依存関係をインストールします:
   ```bash
   uv sync
   ```

## 使用方法

### 基本的な使用方法

```bash
uv run melotts_onnx.py --sentence "Hello, world!" --encoder ./models/encoder-en.onnx --decoder ./models/decoder-en.onnx --language EN --wav output.wav
```

### 日本語の例

```bash
uv run melotts_onnx.py --sentence "はじめまして" --encoder ./models/encoder-jp.onnx --decoder ./models/decoder-jp.onnx --language JP --wav output.wav --sample_rate 44100 --speed 1.0
```

### コマンドラインオプション

- `--sentence, -s`: 合成する入力テキスト
- `--wav, -w`: 出力WAVファイルのパス
- `--encoder, -e`: エンコーダーONNXモデルのパス
- `--decoder, -d`: デコーダーONNXモデルのパス
- `--language, -l`: 言語コード (ZH, ZH_MIX_EN, JP, EN, KR, ES, SP, FR)
- `--sample_rate, -sr`: オーディオサンプルレート (デフォルト: 44100)
- `--speed`: 音声速度倍率 (デフォルト: 1.0)
- `--dec_len`: デコーダー長パラメータ

## サポートされている言語

- **ZH**: 中国語
- **ZH_MIX_EN**: 中国語と英語のミックス
- **JP**: 日本語
- **EN**: 英語
- **KR**: 韓国語
- **ES**: スペイン語
- **SP**: スペイン語 (代替)
- **FR**: フランス語

## モデルファイル

ONNXモデルファイルを`models/`ディレクトリに配置してください:
- エンコーダーモデル: `encoder-{language}.onnx`
- デコーダーモデル: `decoder-{language}.onnx`
- 追加ファイル: `g-{language}.bin`, `lexicon.txt`, `tokens.txt`

## 必要要件

- Python 3.10+
- ONNX Runtime
- NumPy
- SoundFile
- その他の言語固有の依存関係 (pyproject.tomlを参照)
