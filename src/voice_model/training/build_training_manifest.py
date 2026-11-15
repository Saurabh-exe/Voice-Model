"""Build a training manifest from transcript comparisons and manual corrections."""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COMPARISON = ROOT / "tts_manifest_comparison.json"
DEFAULT_REVIEW_CSV = ROOT / "transcript_review.csv"
DEFAULT_OUTPUT = ROOT / "tts_training_manifest.json"


def normalize_text(text: str) -> str:
    """Normalize capitalization and punctuation for rough transcript matching."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def load_review_corrections(review_path: Path) -> Dict[str, str]:
    """Load non-empty corrected transcripts indexed by audio path."""
    corrections: Dict[str, str] = {}
    with review_path.open("r", encoding="utf-8-sig", newline="") as review_file:
        reader = csv.DictReader(review_file)
        if not reader.fieldnames or "audio" not in reader.fieldnames:
            raise ValueError(f"Review CSV must have an 'audio' column: {review_path}")
        if "corrected_text" not in reader.fieldnames:
            raise ValueError(
                f"Review CSV must have a 'corrected_text' column: {review_path}"
            )

        for row in reader:
            audio = (row.get("audio") or "").strip()
            corrected_text = (row.get("corrected_text") or "").strip()
            if audio and corrected_text:
                corrections[audio] = corrected_text
    return corrections


def build_training_manifest(
    comparison_path: Path, review_path: Path, output_path: Path
) -> Dict[str, int]:
    """Merge reviewed corrections and omit unresolved transcript disagreements."""
    with comparison_path.open("r", encoding="utf-8") as comparison_file:
        comparisons = json.load(comparison_file)
    if not isinstance(comparisons, list):
        raise ValueError(f"Expected a JSON list in {comparison_path}")

    corrections = load_review_corrections(review_path)
    training_rows = []
    matching_rows = 0
    corrected_rows = 0
    excluded_rows = 0
    seen_audio = set()

    for item in comparisons:
        if not isinstance(item, dict):
            raise ValueError("Every comparison entry must be a JSON object.")

        audio = item.get("audio")
        original_text = item.get("text")
        asr_text = item.get("asr_text")
        if not isinstance(audio, str) or not isinstance(original_text, str):
            raise ValueError("Each comparison must contain string 'audio' and 'text'.")
        if audio in seen_audio:
            raise ValueError(f"Duplicate audio entry in comparison JSON: {audio}")
        seen_audio.add(audio)

        matches = item.get("normalized_match")
        if not isinstance(matches, bool):
            matches = normalize_text(original_text) == normalize_text(
                asr_text if isinstance(asr_text, str) else ""
            )

        output_item = {
            key: value
            for key, value in item.items()
            if key not in {"asr_text", "normalized_match"}
        }

        if matches:
            matching_rows += 1
        else:
            corrected_text = corrections.get(audio, "")
            if not corrected_text:
                excluded_rows += 1
                continue
            output_item["text"] = corrected_text
            corrected_rows += 1

        training_rows.append(output_item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(training_rows, output_file, ensure_ascii=False, indent=2)

    return {
        "included_matching": matching_rows,
        "included_corrected": corrected_rows,
        "excluded_unresolved": excluded_rows,
        "total_included": len(training_rows),
        "corrections_not_used": len(set(corrections) - seen_audio),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a cleaned TTS training manifest."
    )
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    for input_path in (args.comparison, args.review):
        if not input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")

    counts = build_training_manifest(args.comparison, args.review, args.output)
    print(f"Training manifest saved to: {args.output}")
    print(f"Included unchanged (normalized transcript match): {counts['included_matching']}")
    print(f"Included with manual correction: {counts['included_corrected']}")
    print(f"Excluded unresolved disagreements: {counts['excluded_unresolved']}")
    print(f"Total included: {counts['total_included']}")
    if counts["corrections_not_used"]:
        print(
            "Warning: review corrections without matching comparison entries: "
            f"{counts['corrections_not_used']}"
        )


if __name__ == "__main__":
    main()
