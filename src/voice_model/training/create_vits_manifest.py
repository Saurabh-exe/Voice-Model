#!/usr/bin/env python3
"""
Create VITS training metadata.csv from tts_manifest.json.

Reads:  tts_manifest.json (in the Voice Model root)
Writes: tts_training_data/metadata.csv  (format: text|speaker|audio_path|text_cleaned)

Run from command line providing the Voice Model root directory:
  python3 src/voice_model/create_vits_manifest.py /Users/saurabh/Documents/Voice_Model

Or just run: cd /Users/saurabh/Documents/Voice_Model && python3 src/voice_model/create_vits_manifest.py
"""

import json
import os
import csv
import sys
from pathlib import Path

# -------------------------------------------------------------------------
# Allow passing the root directory as first argument, otherwise default
# -------------------------------------------------------------------------
if len(sys.argv) > 1:
    ROOT = Path(sys.argv[1]).resolve()
else:
    # Default: assume we're inside /Users/saurabh/Documents/Voice_Model/
    ROOT = Path("/Users/saurabh/Documents/Voice_Model").resolve()

MANIFEST_PATH = ROOT / "tts_manifest.json"
OUTPUT_DIR = ROOT / "tts_training_data"
OUTPUT_CSV = OUTPUT_DIR / "metadata.csv"


def main() -> None:
    # --- Read manifest ---
    if not MANIFEST_PATH.is_file():
        print(f"❌ Manifest not found: {MANIFEST_PATH}")
        print(f"   Expected at: {MANIFEST_PATH}")
        print("   Run step 1 first, or check the path.")
        sys.exit(1)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"📖 Read {len(manifest)} entries from {MANIFEST_PATH}")

    # --- Create output directory ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}")

    # --- Write metadata.csv ---
    # Format per VITS/TTS expectation: text|speaker|audio_path|text_cleaned
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "speaker", "audio_path", "text_cleaned"])

        skipped = 0
        for i, entry in enumerate(manifest):
            # Keys observed in your manifest: audio, text, call_id, caller_id, speaker_id
            if "audio" not in entry or "text" not in entry:
                skipped += 1
                print(
                    "⚠️  Row {i}: missing 'audio' or 'text', skipping."
                )
                continue

            text = entry["text"]
            audio = entry["audio"]  # e.g. "LEGOv2/audio/20061123/000/000.wav"
            speaker = entry.get("speaker_id", "speaker_0")
            text_clean = text.replace("\ufeff", "").strip()

            writer.writerow([text, speaker, audio, text_clean])

            if (i + 1) % 1000 == 0:
                print(f"   Processed {i + 1}/{len(manifest)}")

    print(
        "\n✅ Done! {} rows created with {} rows "
        f"({skipped} skipped due to missing data)".format(
            OUTPUT_CSV, len(manifest) - skipped, skipped
        )
    )

    # Actually fix the print format - let me just do it simply
    print("\n✅ Done! Created {} with {} rows ({} skipped)".format(
        OUTPUT_CSV, len(manifest) - skipped, skipped))

    # Show first 3 rows
    print("\n👀 First 3 rows of metadata.csv:")
    with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for i, row in enumerate(reader):
            if i >= 3:
                break
            print("  {} | {} | {} | {}".format(
                row[0][:60] + "..." if len(row[0]) > 60 else row[0],
                row[1], row[2], row[3][:60] + "..." if len(row[3]) > 60 else row[3]
            ))


if __name__ == "__main__":
    main()
