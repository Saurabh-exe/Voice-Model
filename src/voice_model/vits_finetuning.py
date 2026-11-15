"""Fine-tune a pretrained multi-speaker Coqui VITS model on LEGOv2."""

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import soundfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "tts_training_manifest_speaker_reviewed.json"
DEFAULT_SOURCE_AUDIO = ROOT / "LEGOv2" / "audio"
DEFAULT_UPSAMPLED_AUDIO = ROOT / "LEGOv2" / "audio_22khz_by_source"
DEFAULT_OUTPUT = ROOT / "trained_model" / "vits_lego2"
PRETRAINED_MODEL = "tts_models/en/vctk/vits"
TARGET_SAMPLE_RATE = 22050


def _install_torchaudio_compatibility() -> None:
    """Provide the metadata and WAV loading APIs expected by Coqui."""
    import torchaudio
    import torch

    if not hasattr(torchaudio, "info"):
        def audio_info(path: str) -> SimpleNamespace:
            metadata = soundfile.info(path)
            return SimpleNamespace(num_frames=metadata.frames)

        setattr(torchaudio, "info", audio_info)

    def audio_load(path: str, *args, **kwargs) -> Tuple[torch.Tensor, int]:
        waveform, sample_rate = soundfile.read(
            path,
            dtype="float32",
            always_2d=True,
        )
        return torch.from_numpy(waveform.T.copy()), sample_rate

    setattr(torchaudio, "load", audio_load)


def build_metadata(
    manifest_path: Path,
    source_audio_dir: Path,
    upsampled_audio_dir: Path,
    metadata_path: Path,
) -> Tuple[List[Dict[str, str]], int]:
    """Validate all manifest items and write the dataset metadata file."""
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"Expected a non-empty JSON list in {manifest_path}")

    source_audio_dir = source_audio_dir.resolve()
    upsampled_audio_dir = upsampled_audio_dir.resolve()
    entries: List[Dict[str, str]] = []
    seen_audio = set()

    for row_number, item in enumerate(manifest, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest row {row_number} is not a JSON object.")

        audio_value = item.get("audio")
        text = item.get("text")
        speaker_id = item.get("speaker_id")
        if not isinstance(audio_value, str) or not audio_value.strip():
            raise ValueError(f"Manifest row {row_number} has no audio path.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Manifest row {row_number} has blank text.")
        if not isinstance(speaker_id, str) or not speaker_id.strip():
            raise ValueError(f"Manifest row {row_number} has no speaker_id.")

        original_path = (ROOT / audio_value).resolve()
        try:
            relative_audio = original_path.relative_to(source_audio_dir)
        except ValueError as exc:
            raise ValueError(
                f"Audio path {audio_value!r} is not under {source_audio_dir}."
            ) from exc

        upsampled_path = upsampled_audio_dir / relative_audio
        if not upsampled_path.is_file():
            raise FileNotFoundError(
                f"Upsampled audio missing for manifest row {row_number}: "
                f"{upsampled_path}. Run src/voice_model/upsample_audio.py first."
            )
        try:
            audio_info = soundfile.info(str(upsampled_path))
        except (OSError, RuntimeError, soundfile.LibsndfileError) as exc:
            raise ValueError(f"Invalid WAV file: {upsampled_path}") from exc
        if audio_info.frames == 0:
            raise ValueError(f"Audio file is empty: {upsampled_path}")
        if audio_info.samplerate != TARGET_SAMPLE_RATE:
            raise ValueError(
                f"{upsampled_path} has sample rate {audio_info.samplerate}; "
                f"expected {TARGET_SAMPLE_RATE} Hz."
            )

        normalized_path = str(upsampled_path)
        if normalized_path in seen_audio:
            raise ValueError(f"Duplicate audio path in manifest: {audio_value}")
        seen_audio.add(normalized_path)
        entries.append(
            {
                "audio_file": normalized_path,
                "text": " ".join(text.split()),
                "speaker_name": speaker_id.strip(),
            }
        )

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8", newline="") as metadata_file:
        writer = csv.DictWriter(
            metadata_file,
            fieldnames=["audio_file", "text", "speaker_name"],
            delimiter="|",
        )
        writer.writeheader()
        writer.writerows(entries)

    return entries, len({entry["speaker_name"] for entry in entries})


def split_by_speaker(
    samples: List[Dict], seed: int, eval_fraction: float
) -> Tuple[List[Dict], List[Dict]]:
    """Split clips per speaker while keeping at least one train clip per speaker."""
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be between 0 and 1.")

    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["speaker_name"]].append(sample)

    rng = random.Random(seed)
    train_samples = []
    eval_samples = []
    for speaker_name in sorted(grouped):
        speaker_samples = list(grouped[speaker_name])
        rng.shuffle(speaker_samples)
        if len(speaker_samples) > 1:
            eval_count = min(
                len(speaker_samples) - 1,
                max(1, round(len(speaker_samples) * eval_fraction)),
            )
            eval_samples.extend(speaker_samples[:eval_count])
            train_samples.extend(speaker_samples[eval_count:])
        else:
            train_samples.extend(speaker_samples)

    if not eval_samples:
        raise ValueError(
            "Cannot create a validation split: every speaker has only one clip."
        )
    rng.shuffle(train_samples)
    rng.shuffle(eval_samples)
    return train_samples, eval_samples


def fine_tune(
    manifest_path: Path,
    source_audio_dir: Path,
    upsampled_audio_dir: Path,
    output_dir: Path,
    batch_size: int,
    epochs: int,
    seed: int,
) -> None:
    """Initialize from a pretrained VCTK VITS checkpoint and fine-tune it."""
    try:
        import torch

        _install_torchaudio_compatibility()

        from trainer import Trainer, TrainerArgs
        from TTS.config import load_config
        from TTS.tts.configs.shared_configs import BaseDatasetConfig
        from TTS.tts.datasets import load_tts_samples
        from TTS.tts.models.vits import Vits
        from TTS.tts.utils.speakers import SpeakerManager
        from TTS.tts.utils.text.tokenizer import TTSTokenizer
        from TTS.utils.audio import AudioProcessor
        from TTS.utils.manage import ModelManager
    except ImportError as exc:
        raise RuntimeError(
            "Could not import the Coqui TTS training stack. Check that the "
            "project dependencies are synchronized; Coqui TTS 0.25.3 requires "
            "transformers>=4.43.0,<=4.46.2. "
            f"Original import error: {exc}"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"
    build_metadata(
        manifest_path,
        source_audio_dir,
        upsampled_audio_dir,
        metadata_path,
    )

    model_path, config_path, _ = ModelManager().download_model(PRETRAINED_MODEL)
    config = load_config(config_path)
    if config.audio.sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(
            f"The selected pretrained model expects {config.audio.sample_rate} Hz, "
            f"but the dataset is {TARGET_SAMPLE_RATE} Hz."
        )
    if not config.model_args.use_speaker_embedding:
        raise ValueError("The selected pretrained model is not multi-speaker VITS.")
    if config.use_phonemes and not (
        shutil.which("espeak-ng") or shutil.which("espeak")
    ):
        raise RuntimeError(
            "The selected VITS checkpoint requires eSpeak phonemization, but "
            "neither espeak-ng nor espeak is installed or available on PATH. "
            "On macOS, install it with `brew install espeak-ng` "
            "(install Homebrew first if the `brew` command is unavailable). "
            "On Ubuntu/Debian, run `sudo apt-get install espeak-ng`."
        )

    dataset_config = BaseDatasetConfig(
        formatter="coqui",
        meta_file_train=metadata_path.name,
        path=str(output_dir),
        language="en",
    )
    all_samples, _ = load_tts_samples(dataset_config, eval_split=False)
    train_samples, eval_samples = split_by_speaker(
        all_samples,
        seed=seed,
        eval_fraction=0.1,
    )

    speaker_manager = SpeakerManager()
    speaker_manager.set_ids_from_data(all_samples, parse_key="speaker_name")
    target_speaker_count = speaker_manager.num_speakers
    pretrained_speaker_count = int(config.model_args.num_speakers)
    if target_speaker_count > pretrained_speaker_count:
        raise ValueError(
            f"Dataset has {target_speaker_count} speakers, but the pretrained "
            f"model supports {pretrained_speaker_count}. Choose a compatible "
            "multi-speaker VITS checkpoint."
        )

    # Keep the pretrained embedding matrix size unchanged; unused rows are
    # reserved so the checkpoint can be loaded strictly.
    for speaker_index in range(target_speaker_count, pretrained_speaker_count):
        speaker_manager.name_to_id[
            f"__unused_pretrained_speaker_{speaker_index}"
        ] = speaker_index

    config.model_args.num_speakers = pretrained_speaker_count
    config.batch_size = batch_size
    config.eval_batch_size = min(batch_size, 4)
    config.num_loader_workers = 2
    config.num_eval_loader_workers = 2
    config.epochs = epochs
    config.output_path = str(output_dir)
    config.run_name = "legov2_vits_finetune"
    config.phoneme_cache_path = str(output_dir / "phoneme_cache")
    config.mixed_precision = torch.cuda.is_available()
    config.datasets = [dataset_config]
    config.test_sentences = [
        [sample["text"], sample["speaker_name"]]
        for sample in eval_samples[:5]
    ]

    audio_processor = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    config.model_args.init_discriminator = True
    model = Vits(config, audio_processor, tokenizer, speaker_manager)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    checkpoint_weights = {
        key: value
        for key, value in checkpoint["model"].items()
        if "speaker_encoder" not in key
    }
    incompatible_keys = model.load_state_dict(checkpoint_weights, strict=False)
    unexpected_keys = incompatible_keys.unexpected_keys
    missing_non_discriminator_keys = [
        key for key in incompatible_keys.missing_keys if not key.startswith("disc.")
    ]
    if unexpected_keys or missing_non_discriminator_keys:
        raise RuntimeError(
            "Pretrained checkpoint does not match the VITS model architecture. "
            f"Missing non-discriminator weights: {missing_non_discriminator_keys}; "
            f"unexpected weights: {unexpected_keys}"
        )
    print(
        "Loaded pretrained VITS weights; initialized the discriminator from "
        "scratch because the inference checkpoint does not contain discriminator weights."
    )

    trainer = Trainer(
        TrainerArgs(),
        config,
        str(output_dir),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    print(
        f"Fine-tuning {PRETRAINED_MODEL} with "
        f"{len(train_samples)} train clips, {len(eval_samples)} validation clips, "
        f"and {target_speaker_count} speakers."
    )
    trainer.fit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune multi-speaker VITS on the reviewed LEGOv2 manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-audio-dir", type=Path, default=DEFAULT_SOURCE_AUDIO)
    parser.add_argument(
        "--upsampled-audio-dir",
        type=Path,
        default=DEFAULT_UPSAMPLED_AUDIO,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fine_tune(
        args.manifest,
        args.source_audio_dir,
        args.upsampled_audio_dir,
        args.output_dir,
        args.batch_size,
        args.epochs,
        args.seed,
    )


if __name__ == "__main__":
    main()
