"""Export transcript disagreements to a CSV for manual audio review."""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "tts_manifest_comparison.json"
DEFAULT_OUTPUT = ROOT / "transcript_review.csv"
REVIEW_COLUMNS = [
    "audio",
    "call_id",
    "caller_id",
    "speaker_id",
    "text",
    "asr_text",
    "corrected_text",
]


def normalize_text(text: str) -> str:
    """Normalize capitalization and punctuation for a rough comparison."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def needs_review(item: Dict[str, Any]) -> bool:
    """Return whether the source and ASR transcripts differ."""
    if "normalized_match" in item:
        return not item["normalized_match"]

    original = item.get("text", "")
    transcription = item.get("asr_text", "")
    return normalize_text(original) != normalize_text(transcription)


def create_review_csv(input_path: Path, output_path: Path) -> int:
    """Write disagreement rows to a CSV, leaving corrected_text blank."""
    with input_path.open("r", encoding="utf-8") as input_file:
        comparisons = json.load(input_file)

    if not isinstance(comparisons, list):
        raise ValueError(f"Expected a JSON list in {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    review_count = 0
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()

        for item in comparisons:
            if not isinstance(item, dict):
                raise ValueError("Every comparison entry must be a JSON object.")
            if needs_review(item):
                writer.writerow(
                    {
                        "audio": item.get("audio", ""),
                        "call_id": item.get("call_id", ""),
                        "caller_id": item.get("caller_id", ""),
                        "speaker_id": item.get("speaker_id", ""),
                        "text": item.get("text", ""),
                        "asr_text": item.get("asr_text", ""),
                        "corrected_text": "",
                    }
                )
                review_count += 1

    return review_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a CSV of transcript disagreements for manual review."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Comparison JSON not found: {args.input}")

    review_count = create_review_csv(args.input, args.output)
    print(f"Transcript disagreements to review: {review_count}")
    print(f"Review sheet saved to: {args.output}")
    print("Listen to each audio clip and enter verified words in corrected_text.")


if __name__ == "__main__":
    main()
