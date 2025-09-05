#!/usr/bin/env python3
"""
Piper TTS Command Line Interface
Simple CLI tool for text-to-speech synthesis using ONNX models

Updates in this version:
- Forces UTF-8 on stdin/stdout/stderr to avoid decode errors on Windows/CP932.
- Makes --config optional; auto-detects JSON next to the .onnx model.
- Guards against mistakenly passing an .onnx file to --config and provides clear guidance.
- Adds --text-file to read UTF-8 text from a file.
- Better error messages around config reading/JSON decoding.
- **NEW**: ONNX Runtime inference-time benchmarking (--runs / --warmup); reports mean/median/p95/min/max.
"""

import argparse
import json
import logging
import os
import sys
import time
import statistics as stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import onnxruntime
import soundfile as sf

# --- Force UTF-8 I/O early -------------------------------------------------
try:
    # Python 3.7+
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
os.environ.setdefault("PYTHONUTF8", "1")

# Optional dependencies - check availability
try:
    import pyopenjtalk
    PYOPENJTALK_AVAILABLE = True
except ImportError:
    PYOPENJTALK_AVAILABLE = False

try:
    from espeak_phonemizer import Phonemizer
    ESPEAK_AVAILABLE = True
except ImportError:
    ESPEAK_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Basic English word to IPA mapping (fallback)
ENGLISH_IPA_MAP = {
    "hello": "hɛloʊ", "world": "wɜrld", "this": "ðɪs", "is": "ɪz", "a": "ə",
    "test": "tɛst", "text": "tɛkst", "to": "tu", "speech": "spitʃ", "demo": "dɛmoʊ",
    "welcome": "wɛlkəm", "piper": "paɪpər", "tts": "titiɛs", "the": "ðə",
    "and": "ænd", "for": "fɔr", "with": "wɪð", "you": "ju", "can": "kæn",
    "it": "ɪt", "that": "ðæt", "have": "hæv", "from": "frʌm", "time": "taɪm"
}

# Japanese multi-character phoneme to Unicode PUA mapping
PHONEME_TO_PUA = {
    "a:": "", "i:": "", "u:": "", "e:": "", "o:": "",
    "cl": "", "ky": "", "kw": "", "gy": "", "gw": "",
    "ty": "", "dy": "", "py": "", "by": "", "ch": "",
    "ts": "", "sh": "", "zy": "", "hy": "", "ny": "",
    "my": "", "ry": ""
}


class PiperTTS:
    def __init__(self, model_path: str, config_path: Optional[str]):
        """Initialize Piper TTS with model and config paths.
        If config_path is None, auto-detect JSON next to the model.
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Resolve config path (auto-detect if needed or if mistakenly given .onnx)
        self.config_path = self._resolve_config_path(config_path)

        # Load configuration (with clear errors)
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(
                e.encoding,
                e.object,
                e.start,
                e.end,
                f"Failed to decode config as UTF-8: {self.config_path}. "
                f"Make sure you pass a JSON file to --config (not an .onnx)."
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {self.config_path}: {e}") from e

        # Determine language from config or filename
        self.language = self._detect_language()

        # Initialize ONNX session
        sess_options = onnxruntime.SessionOptions()
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 1

        self.session = onnxruntime.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        logger.info(f"Loaded model: {self.model_path.name}")
        logger.info(f"Config: {self.config_path.name}")
        logger.info(f"Language: {self.language}")
        logger.info(f"Sample rate: {self.config.get('audio', {}).get('sample_rate', 22050)}")

    def _resolve_config_path(self, config_path: Optional[str]) -> Path:
        """Find a plausible config JSON near the model.
        Tries common filenames if not provided or if a .onnx was mistakenly given.
        """
        # If user passed a path and it looks like a valid JSON file, use it
        if config_path:
            p = Path(config_path)
            # If they accidentally passed an .onnx here, try to auto-fix
            if p.suffix.lower() == ".onnx":
                logger.warning(
                    "--config points to an .onnx file. Trying to auto-detect the JSON next to the model."
                )
                config_path = None  # fall through to auto-detect
            elif p.exists():
                return p
            else:
                logger.warning(f"Config file not found: {p}. Will try to auto-detect next to the model.")
                config_path = None

        # Auto-detect candidates next to the model
        candidates = [
            self.model_path.with_suffix('.json'),
            self.model_path.with_suffix('.onnx.json'),
            self.model_path.parent / (self.model_path.stem + '.json'),
            self.model_path.parent / 'config.json',
        ]
        for c in candidates:
            if c.exists():
                return c

        raise FileNotFoundError(
            "Could not find a config JSON next to the model. "
            "Pass it explicitly with --config or place a '<model_stem>.json' beside the .onnx."
        )

    def _detect_language(self) -> str:
        """Detect language from config or filename."""
        if isinstance(self.config, dict) and 'language' in self.config:
            lang = str(self.config['language']).lower()
            if lang in {"ja", "jp", "japanese"}:
                return "ja"
            if lang.startswith("en"):
                return "en"
        # Fallback to filename heuristic
        filename = self.model_path.name.lower()
        if 'ja' in filename or 'jp' in filename:
            return 'ja'
        if 'en' in filename:
            return 'en'
        return 'en'  # default

    def _map_phonemes(self, phonemes: List[str]) -> List[str]:
        """Map multi-character phonemes to Unicode PUA characters."""
        return [PHONEME_TO_PUA.get(p, p) for p in phonemes]

    def _text_to_phonemes(self, text: str) -> List[str]:
        """Convert text to phoneme strings based on language."""
        if self.language == "ja":
            if PYOPENJTALK_AVAILABLE:
                labels = pyopenjtalk.extract_fullcontext(text)
                phonemes: List[str] = []
                for label in labels:
                    if "-" in label and "+" in label:
                        phoneme = label.split("-")[1].split("+")[0]
                        if phoneme not in ["sil", "pau"]:
                            phonemes.append(phoneme)
                phonemes = ["^"] + phonemes + ["$"]
                return self._map_phonemes(phonemes)
            logger.warning("pyopenjtalk not available, using very rough fallback for Japanese")
            return ["^"] + list("aiueo") * (max(1, len(text) // 2)) + ["$"]

        # English and others
        if ESPEAK_AVAILABLE:
            phonemizer = Phonemizer("en-us")
            phoneme_str = phonemizer.phonemize(text)
            return ["^"] + list(phoneme_str.replace(" ", "")) + ["$"]

        logger.warning("espeak_phonemizer not available, using IPA fallback for English")
        words = text.lower().split()
        phonemes = ["^"]
        for i, word in enumerate(words):
            if i > 0:
                phonemes.append(" ")
            clean_word = "".join(c for c in word if c.isalpha())
            ipa = ENGLISH_IPA_MAP.get(clean_word)
            phonemes.extend(list(ipa if ipa else clean_word))
        phonemes.append("$")
        return phonemes

    def _phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        """Convert phonemes to model input IDs using config phoneme_id_map."""
        phoneme_id_map: Dict[str, List[int]] = self.config.get("phoneme_id_map", {})
        ids: List[int] = []
        for phoneme in phonemes:
            mapped = phoneme_id_map.get(phoneme)
            if mapped is None:
                ids.append(0)  # pad token fallback
            elif isinstance(mapped, list):
                ids.extend(int(x) for x in mapped)
            else:
                ids.append(int(mapped))
        return ids

    # -------- NEW: input preparation & benchmarking helpers -----------------
    def prepare_inputs(
        self,
        text: str,
        speaker_id: int = 0,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ) -> Tuple[Dict[str, np.ndarray], int]:
        """Prepare ORT feed dict and return (inputs, sample_rate)."""
        phonemes = self._text_to_phonemes(text)
        phoneme_ids = self._phonemes_to_ids(phonemes)
        if not phoneme_ids:
            raise ValueError("Failed to convert text to phonemes")

        text_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        text_lengths = np.array([text_array.shape[1]], dtype=np.int64)
        scales = np.array([noise_scale, length_scale, noise_w], dtype=np.float32)

        inputs = {
            "input": text_array,
            "input_lengths": text_lengths,
            "scales": scales,
        }
        if self.config.get("num_speakers", 1) > 1:
            inputs["sid"] = np.array([speaker_id], dtype=np.int64)

        sample_rate = self.config.get("audio", {}).get("sample_rate", 22050)
        return inputs, sample_rate

    def benchmark(self, inputs: Dict[str, np.ndarray], warmup: int = 2, runs: int = 10) -> Dict[str, float]:
        """Benchmark ORT inference time with given feed dict. Returns stats in ms."""
        if runs <= 0:
            return {}
        output_names = [o.name for o in self.session.get_outputs()]

        # Warmup
        for _ in range(max(0, warmup)):
            self.session.run(output_names, inputs)

        times_ms: List[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            self.session.run(output_names, inputs)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        times_ms.sort()
        return {
            "runs": runs,
            "warmup": max(0, warmup),
            "mean_ms": float(sum(times_ms) / len(times_ms)),
            "median_ms": float(stats.median(times_ms)),
            "p95_ms": float(times_ms[int(len(times_ms) * 0.95) - 1] if runs > 1 else times_ms[0]),
            "min_ms": float(times_ms[0]),
            "max_ms": float(times_ms[-1]),
        }

    # -----------------------------------------------------------------------
    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ) -> Tuple[int, np.ndarray]:
        """Generate speech from text (single forward)."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        inputs, sample_rate = self.prepare_inputs(
            text,
            speaker_id=speaker_id,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w=noise_w,
        )

        try:
            audio = self.session.run(None, inputs)[0]
            audio = audio.squeeze()  # remove batch/channel dims
            audio = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
            return sample_rate, audio
        except Exception as e:
            raise RuntimeError(f"Failed to generate speech: {e}") from e


def main():
    parser = argparse.ArgumentParser(description="Piper TTS Command Line Interface")
    parser.add_argument("text", nargs="?", default=None, help="Text to synthesize. If omitted, use --text-file.")
    parser.add_argument("-f", "--text-file", help="Path to a UTF-8 text file to read as input text")
    parser.add_argument("-m", "--model", default="./ja_JP-test-medium.onnx", help="Path to ONNX model file")
    parser.add_argument("-c", "--config", default=None, help="Path to model config JSON file")
    parser.add_argument("-o", "--output", default="output.wav", help="Output audio file (default: output.wav)")
    parser.add_argument("-s", "--speaker", type=int, default=0, help="Speaker ID for multi-speaker models (default: 0)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (lower = faster) (default: 1.0)")
    parser.add_argument("--noise", type=float, default=0.667, help="Expressiveness (0.0-1.0) (default: 0.667)")
    parser.add_argument("--noise-w", type=float, default=0.8, help="Phoneme duration variance (0.0-1.0) (default: 0.8)")
    # ベンチマーク関連
    parser.add_argument("--runs", type=int, default=0, help="Number of timed ORT runs (ignored unless --benchmark-only)")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs (ignored unless --benchmark-only)")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Run benchmarking only (no audio output). This may run multiple inferences.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 入力テキスト取得
    input_text = args.text
    if input_text is None and args.text_file:
        try:
            from pathlib import Path
            input_text = Path(args.text_file).read_text(encoding="utf-8")
        except Exception as e:
            parser.error(f"Failed to read --text-file '{args.text_file}': {e}")
    if not input_text:
        parser.error("No input text provided. Provide TEXT or --text-file.")

    try:
        tts = PiperTTS(args.model, args.config)

        # ここで一度だけ入力を準備（synthesizeでも内部で準備するが、ベンチで使う可能性に備えて取得）
        inputs, sample_rate = tts.prepare_inputs(
            input_text,
            speaker_id=args.speaker,
            length_scale=args.speed,
            noise_scale=args.noise,
            noise_w=args.noise_w,
        )

        # --- ベンチマーク専用モード ---
        if args.benchmark_only:
            if args.runs <= 0:
                logger.info("Benchmark-only mode requested but --runs=0. Nothing to do.")
                return
            bench = tts.benchmark(inputs, warmup=max(0, args.warmup), runs=args.runs)
            if bench:
                logger.info(
                    "Benchmark (ORT run(None, inputs)) — runs=%(runs)d warmup=%(warmup)d | "
                    "mean=%(mean_ms).3f ms median=%(median_ms).3f ms p95=%(p95_ms).3f ms "
                    "min=%(min_ms).3f ms max=%(max_ms).3f ms",
                    bench,
                )
            return  # 音声出力はしない

        # --- 通常モード：推論は1回だけ ---
        if args.runs > 0:
            logger.warning("--runs is set but ignored because --benchmark-only was not provided. "
                           "To benchmark, add --benchmark-only (no audio will be generated).")

        # ここが唯一の推論（1回）
        sr, audio = tts.synthesize(
            input_text,
            speaker_id=args.speaker,
            length_scale=args.speed,
            noise_scale=args.noise,
            noise_w=args.noise_w,
        )

        # 保存
        sf.write(args.output, audio, sr)
        logger.info(f"Audio saved to: {args.output}")
        logger.info(f"Duration: {len(audio) / sr:.2f} seconds")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()