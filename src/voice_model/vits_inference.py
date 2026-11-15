"""Generate speech with a fine-tuned Coqui VITS checkpoint."""

import argparse
from pathlib import Path
from typing import Optional

import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "trained_model" / "vits_lego_v2"
DEFAULT_OUTPUT_DIR = ROOT / "generated_audio"


def _speaker_name_to_id(speakers_path: Path, speaker_name: str) -> int:
    """Resolve a speaker's training label to the integer expected by VITS."""
    import torch

    speakers = torch.load(speakers_path, map_location="cpu", weights_only=False)
    # Coqui's .pth speaker map stores name -> integer directly.
    if not isinstance(speakers, dict) or not all(
        isinstance(name, str) and isinstance(index, int)
        for name, index in speakers.items()
    ):
        raise ValueError(
            f"Expected a speaker-name-to-ID mapping in {speakers_path}."
        )
    mapping = speakers
    if speaker_name not in mapping:
        available = ", ".join(sorted(mapping))
        raise ValueError(
            f"Unknown speaker {speaker_name!r}. Available speaker names: {available}"
        )
    return int(mapping[speaker_name])


def synthesize(
    text: str,
    model_dir: Path,
    speaker_name: Optional[str],
    output_path: Path,
    use_cuda: bool,
) -> None:
    """Load the fine-tuned checkpoint and save one utterance as WAV."""
    from TTS.config import load_config
    from TTS.tts.models.vits import Vits
    from TTS.tts.utils.speakers import SpeakerManager
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.utils.audio import AudioProcessor

    model_dir = model_dir.resolve()
    checkpoint_path = model_dir / "best_model.pth"
    config_path = model_dir / "config.json"
    speakers_path = model_dir / "speakers.pth"
    for required_path in (checkpoint_path, config_path, speakers_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required model file not found: {required_path}")

    # Coqui stores the GPU machine's speaker map path in config.json. Point it
    # to the map that was copied alongside this checkpoint.
    config = load_config(str(config_path))
    config.model_args.speakers_file = str(speakers_path)
    config.speakers_file = str(speakers_path)
    speaker_manager = SpeakerManager()
    speaker_manager.load_ids_from_file(str(speakers_path))

    audio_processor = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    model = Vits(config, audio_processor, tokenizer, speaker_manager)
    model.load_checkpoint(
        config,
        str(checkpoint_path),
        eval=True,
    )
    if use_cuda:
        model.cuda()

    speaker_id = None
    if config.model_args.use_speaker_embedding:
        if not speaker_name:
            raise ValueError(
                "This is a multi-speaker model. Pass --speaker with a name "
                "from speakers.pth, for example proposed_speaker_0009."
            )
        speaker_id = _speaker_name_to_id(speakers_path, speaker_name)

    import torch

    text_ids = model.tokenizer.text_to_ids(text)
    text_input = torch.tensor(text_ids, dtype=torch.long, device=next(model.parameters()).device)
    text_input = text_input.unsqueeze(0)
    aux_input = {
        "x_lengths": torch.tensor([text_input.shape[1]], device=text_input.device),
        "d_vectors": None,
        "speaker_ids": (
            torch.tensor([speaker_id], dtype=torch.long, device=text_input.device)
            if speaker_id is not None
            else None
        ),
        "language_ids": None,
        "durations": None,
    }
    result = model.inference(text_input, aux_input=aux_input)
    waveform = result["model_outputs"].squeeze().detach().cpu().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, waveform, config.audio.sample_rate)
    print(f"Saved {output_path} ({config.audio.sample_rate} Hz)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a WAV file from the fine-tuned VITS checkpoint."
    )
    parser.add_argument("--text", required=True, help="Sentence to synthesize")
    parser.add_argument("--speaker", help="Speaker name from speakers.pth")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "sample.wav")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (auto uses CUDA when available).",
    )
    args = parser.parse_args()

    if not args.text.strip():
        parser.error("--text cannot be empty")

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available in this environment")
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    synthesize(args.text.strip(), args.model_dir, args.speaker, args.output, use_cuda)


if __name__ == "__main__":
    main()
