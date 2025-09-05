# TTS_benchmark_test


# TTSベンチマーク結果

## 推論時間の比較（日本語）

### MeloTTS JP @ONNX-CPU
- **エンコーダー**: 51.76 ms
- **デコード**: 1924.81 ms
- **合計**: 1976.57 ms
- **サンプル音声**: [hajimemashite_melotts.wav](https://github.com/nnn112358/TTS_benchmark_test/blob/main/melotts_onnx/hajimemashite_melotts.wav)

### PiperPlus JP @ONNX-CPU
- **合計**: 187.366 ms
- **サンプル音声**: [output.wav](https://github.com/nnn112358/TTS_benchmark_test/blob/main/piperplus_onnx/output.wav)

## 性能比較

| TTSエンジン | 処理時間 | 性能比 |
|------------|---------|--------|
| PiperPlus JP | 187.366 ms | **約10.5倍高速** |
| MeloTTS JP | 1976.57 ms | ベースライン |

*ONNX-CPU実装でのベンチマーク結果*
