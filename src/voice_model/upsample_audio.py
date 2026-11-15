"""Resample LEGOv2 audio to 22.05 kHz while preserving its directory tree."""

from pathlib import Path

import torchaudio


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "LEGOv2" / "audio"
OUTPUT_DIR = ROOT / "LEGOv2" / "audio_22khz_by_source"
TARGET_SAMPLE_RATE = 22050


def main() -> None:
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Source audio directory not found: {SOURCE_DIR}")

    source_files = sorted(SOURCE_DIR.rglob("*.wav"))
    if not source_files:
        raise FileNotFoundError(f"No WAV files found under: {SOURCE_DIR}")

    processed = 0
    skipped = 0
    empty_files = []
    for index, source_path in enumerate(source_files, start=1):
        relative_path = source_path.relative_to(SOURCE_DIR)
        output_path = OUTPUT_DIR / relative_path

        if output_path.is_file():
            try:
                existing_info = torchaudio.info(str(output_path))
            except RuntimeError:
                existing_info = None
            if (
                existing_info is not None
                and existing_info.sample_rate == TARGET_SAMPLE_RATE
                and existing_info.num_frames > 0
            ):
                skipped += 1
                continue

        waveform, sample_rate = torchaudio.load(str(source_path))
        if waveform.numel() == 0:
            empty_files.append(source_path)
            if output_path.is_file():
                output_path.unlink()
            continue

        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=TARGET_SAMPLE_RATE,
            )
            waveform = resampler(waveform)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(
            f".{output_path.stem}.tmp{output_path.suffix}"
        )
        torchaudio.save(str(temporary_path), waveform, TARGET_SAMPLE_RATE)
        temporary_path.replace(output_path)
        processed += 1

        if index % 100 == 0 or index == len(source_files):
            print(
                f"Processed {index}/{len(source_files)} files "
                f"({processed} written, {skipped} already present)."
            )

    if empty_files:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        skipped_path = OUTPUT_DIR / "empty_audio_skipped.txt"
        skipped_path.write_text(
            "".join(f"{path}\n" for path in empty_files),
            encoding="utf-8",
        )
        print(
            f"Skipped {len(empty_files)} empty source WAV files. "
            f"Paths recorded in: {skipped_path}"
        )

    print(f"Upsampled audio is available at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
