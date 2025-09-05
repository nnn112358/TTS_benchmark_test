## TTSベンチマーク結果（ONNX-CPU 実装）

### 推論時間

#### MeloTTS JP @ONNX-CPU
* **エンコーダー**: 51.76 ms
* **デコーダー**: 1924.81 ms
* **合計**: 1976.57 ms
* **サンプル音声**: [hajimemashite\_melotts.wav](https://github.com/nnn112358/TTS_benchmark_test/blob/main/melotts_onnx/hajimemashite_melotts.wav)

#### PiperPlus JP @ONNX-CPU
* **合計**: 187.37 ms
* **サンプル音声**: [output.wav](https://github.com/nnn112358/TTS_benchmark_test/blob/main/piperplus_onnx/output.wav)
> 備考: PiperPlus JP での音声合成の結果が正しい日本語になっていないため、要調査。(25/09/05)

---

## 性能比較

| TTSエンジン      | 処理時間 (ms) |
| ------------ | --------- |
| PiperPlus JP@ONNX-CPU | 187.37    |
| MeloTTS JP @ONNX-CPU  | 1976.57   |

*注: ONNX-CPU 実装でのベンチマーク結果*
