"""Transcribe manifest audio locally and save a transcript comparison JSON."""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "tts_manifest.json"
DEFAULT_OUTPUT = ROOT / "tts_manifest_comparison.json"


def normalize_text(text: str) -> str:
    """Normalize case and punctuation for a rough transcript comparison."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def save_results(output_path: Path, results: list[dict]) -> None:
    """Write all completed comparisons, including partial runs, to JSON."""
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, ensure_ascii=False, indent=2)


class ManifestTranscriber:
    """Compare manifest transcripts with locally generated Whisper transcripts."""

    def __init__(
        self,
        manifest_path: Path = DEFAULT_MANIFEST,
        output_path: Path = DEFAULT_OUTPUT,
        model_size: str = "small",
        limit: Optional[int] = None,
        resume: bool = False,
        data_root: Path = ROOT,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        """Load the manifest and initialize the local speech-recognition model."""
        self.manifest_path = Path(manifest_path)
        self.output_path = Path(output_path)
        self.data_root = Path(data_root).resolve()

        with self.manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        if not isinstance(manifest, list):
            raise ValueError(f"Expected a JSON list in {self.manifest_path}")
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be at least 1")
            manifest = manifest[:limit]
        self.manifest = manifest

        self.results = []
        self.completed_audio = set()
        if resume and self.output_path.exists():
            with self.output_path.open("r", encoding="utf-8") as output_file:
                existing_results = json.load(output_file)
            if not isinstance(existing_results, list):
                raise ValueError(f"Expected a JSON list in {self.output_path}")
            self.results = existing_results
            self.completed_audio = {item["audio"] for item in self.results}

        print(f"Loading local Whisper model: {model_size}")
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )

    def compare(self) -> None:
        """Transcribe each audio file and save transcript comparisons to JSON."""
        total = len(self.manifest)
        for index, item in enumerate(self.manifest, start=1):
            audio = item["audio"]
            if audio in self.completed_audio:
                continue

            audio_path = (self.data_root / audio).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"Audio file listed in manifest was not found: {audio_path}"
                )

            segments, _ = self.model.transcribe(str(audio_path), language="en")
            asr_text = " ".join(
                segment.text.strip() for segment in segments
            ).strip()
            result = {
                **item,
                "asr_text": asr_text,
                "normalized_match": normalize_text(item["text"])
                == normalize_text(asr_text),
            }
            self.results.append(result)
            self.completed_audio.add(audio)

            if index % 25 == 0 or index == total:
                save_results(self.output_path, self.results)
                print(f"Transcribed {index}/{total}")

        save_results(self.output_path, self.results)
        print(f"Comparison JSON saved to: {self.output_path}")
        print(f"Completed comparisons: {len(self.results)}")
        print(
            "Review entries where normalized_match is false; ASR output can also be wrong."
        )


def transcribe_manifest(
    manifest_path: Path,
    output_path: Path,
    data_root: Path,
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    limit: Optional[int] = None,
    resume: bool = True,
) -> None:
    """Transcribe manifest audio and save original/Whisper text side by side.

    For Google Colab with a GPU, use the defaults. For CPU, pass
    device="cpu" and compute_type="int8".
    """
    transcriber = ManifestTranscriber(
        manifest_path=manifest_path,
        output_path=output_path,
        model_size=model_size,
        limit=limit,
        resume=resume,
        data_root=data_root,
        device=device,
        compute_type=compute_type,
    )
    transcriber.compare()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe manifest audio with local Faster Whisper."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size; larger models are slower but may be more accurate.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--compute-type",
        choices=["int8", "float16"],
        help="Model precision; defaults to int8 on CPU and float16 on GPU.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Transcribe only the first N manifest entries (useful for a test run).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from comparisons already present in the output JSON.",
    )
    args = parser.parse_args()

    compute_type = args.compute_type or ("float16" if args.device == "cuda" else "int8")
    transcriber = ManifestTranscriber(
        manifest_path=args.manifest,
        output_path=args.output,
        model_size=args.model,
        limit=args.limit,
        resume=args.resume,
        device=args.device,
        compute_type=compute_type,
    )
    transcriber.compare()


if __name__ == "__main__":
    main()
