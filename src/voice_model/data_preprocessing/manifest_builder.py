"""Build a JSON audio/text manifest from the LEGOv2 interactions."""

import csv
import json
import re
from pathlib import Path
import yaml

# Find the folder containing this script, which is the project root.
ROOT = Path(__file__).resolve().parents[3]

WAV_PATH = re.compile(r"(?i)\.wav$")
EMPTY_TEXT = {"", r"\n", "nan", "none", "null"}


class ManifestBuilder:
    """Build a JSON audio/text manifest from the LEGOv2 interactions."""

    def __init__(self, config_path):
        """Initialize the ManifestBuilder with paths from the config file."""
        config_path = Path(config_path).resolve()
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError(f"Expected YAML key/value settings in {config_path}")

        project_root = config_path.parents[2]
        self.interactions_csv = project_root / config["CSV_PATH"]
        self.audio_dir = project_root / config["AUDIO_BASE"]
        self.output_manifest = project_root / config["MANIFEST_PATH"]

    def build_manifest(self) -> None:
        """Build the manifest CSV file."""
        included = 0
        skipped_no_text = 0
        skipped_no_audio = 0

        manifest = []
        with self.interactions_csv.open(
            "r", encoding="cp1252", errors="replace", newline=""
        ) as source:
            reader = csv.reader(source, delimiter=";", quotechar='"')

            for row in reader:
                if len(row) < 3 or row[2].strip().lower() in EMPTY_TEXT:
                    skipped_no_text += 1
                    continue

                audio_value = next(
                    (value.strip() for value in row if WAV_PATH.search(value.strip())),
                    None,
                )
                if not audio_value:
                    skipped_no_audio += 1
                    continue

                relative_audio = Path(audio_value.replace("\\", "/"))
                if relative_audio.is_absolute() or ".." in relative_audio.parts:
                    skipped_no_audio += 1
                    continue

                audio_path = (self.audio_dir / relative_audio).resolve()
                try:
                    audio_path.relative_to(self.audio_dir.resolve())
                except ValueError:
                    skipped_no_audio += 1
                    continue

                if not audio_path.is_file():
                    skipped_no_audio += 1
                    continue

                caller_id = row[0].strip()
                manifest.append(
                    {
                        "audio": audio_path.relative_to(ROOT).as_posix(),
                        "text": " ".join(row[2].split()),
                        "call_id": caller_id,
                        "caller_id": caller_id,
                        "speaker_id": f"speaker_{caller_id}",
                    }
                )
                included += 1

        with self.output_manifest.open("w", encoding="utf-8") as destination:
            json.dump(manifest, destination, ensure_ascii=False, indent=2)
