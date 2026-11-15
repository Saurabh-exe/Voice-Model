# Voice Model

This project prepares the LEGOv2 spoken-dialogue corpus for text-to-speech
experiments. It creates audio/transcript manifests, supports transcript review,
and includes an optional workflow for grouping calls by likely speaker voice.

## Dataset layout

The scripts expect the LEGOv2 corpus and manifests at the project root:

```text
Voice Model/
├── LEGOv2/
│   ├── audio/
│   └── corpus/csv/interactions.csv
├── tts_training_manifest.json
└── src/voice_model/
```

## Download the dataset

Download the LEGOv2 corpus from the University of Bamberg's
[LEGO Spoken Dialogue Corpus page](https://www.uni-bamberg.de/ds/ressourcen/lego/).
Use the LEGOv2 download for this project; the page also describes the corpus
and provides its license. Extract it at the repository root so the files are
available under `LEGOv2/audio/` and `LEGOv2/corpus/csv/`. The dataset is not
included in Git because of its size.

## Speaker analysis

Speaker-analysis code is in
[`src/voice_model/speaker_analysis/`](src/voice_model/speaker_analysis/).
It uses a pretrained SpeechBrain ECAPA-TDNN model to create clip-level voice
embeddings, averages them by call, and proposes call groups based on cosine
similarity. These are only suggestions: short, noisy telephone clips can lead
to incorrect matches. Listen to candidates and manually review every assigned
speaker label.

Install the optional dependency in the project environment:

```bash
uv add speechbrain
```

Run the analysis from the project root. On macOS, use CPU:

```bash
uv run python -m voice_model.speaker_analysis.analyze_speakers \
  --manifest tts_training_manifest.json \
  --data-root . \
  --output-dir speaker_analysis/output \
  --device cpu \
  --threshold 0.65
```

The threshold is an initial exploratory value, not a validated setting for
this dataset. The run writes resumable clip embeddings, call embeddings,
pairwise similarities, and `speaker_analysis/output/speaker_review.csv`.
Listen to calls and review the proposed groups. The generated
`reviewed_speaker_id` column starts with the same values as
`proposed_speaker_id`; edit it if you want to correct a group.

Write the proposed labels into a separate manifest:

```bash
uv run python -m voice_model.speaker_analysis.analyze_speakers \
  --manifest tts_training_manifest.json \
  --output-dir speaker_analysis/output \
  --apply-reviewed \
  --output-manifest tts_training_manifest_speaker_reviewed.json
```

This uses `reviewed_speaker_id` from the review CSV as each entry's
`speaker_id` and does not overwrite the source manifest. See the
[speaker-analysis guide](src/voice_model/speaker_analysis/README.md) for
outputs, resume behavior, GPU notes, and review details.

## Fine-tune multi-speaker VITS

The existing flat `LEGOv2/audio_22khz/` folder cannot reliably be joined to
the manifest because multiple source clips share the same filename. Create a
new upsampled folder that preserves the original directory layout:

```bash
uv run python src/voice_model/upsample_audio.py
```

This writes to `LEGOv2/audio_22khz_by_source/` and leaves the existing
upsampled folder untouched. The current project environment is Python 3.9.
For GPU training, use a Linux/Colab environment with CUDA-compatible PyTorch
and Coqui TTS:

```bash
pip install coqui-tts
```

The VCTK VITS checkpoint also requires the system `espeak-ng` command for
English phonemization. On macOS, install it with `brew install espeak-ng`
(install Homebrew first if `brew` is not available); on Ubuntu/Debian, use
`sudo apt-get install espeak-ng`.

Then run from the project root in that environment:

```bash
python src/voice_model/vits_finetuning.py \
  --manifest tts_training_manifest_speaker_reviewed.json \
  --upsampled-audio-dir LEGOv2/audio_22khz_by_source \
  --output-dir trained_model/vits_lego2
```

The script fine-tunes the pretrained multi-speaker `tts_models/en/vctk/vits`
checkpoint, keeps each speaker represented in training, and holds out some
clips per speaker for validation. Evaluation audio uses up to five held-out
utterances paired with their speaker IDs. The inference checkpoint does not
include the VITS discriminator, so the script initializes that training
component from scratch and verifies the other checkpoint weights match. Check
the pretrained model's license for your intended use. The project manifest and
existing audio are not modified.

## Generate speech with a trained checkpoint

The GPU-trained run is stored locally in `trained_model/vits_lego_v2/`. Keep the
checkpoint, its `config.json`, and `speakers.pth` together; inference uses all
three. The checkpoint is large and is ignored by Git. Ensure `espeak-ng` is
installed for English phonemization.

From the project root, generate a WAV sample:

```bash
uv run python src/voice_model/vits_inference.py \
  --text "Thank you for calling. How can I help you today?" \
  --speaker proposed_speaker_0009 \
  --output generated_audio/sample.wav
```

Replace the speaker value with a name in `trained_model/vits_lego_v2/speakers.pth`.
The script defaults to that run folder and chooses CUDA when it is available;
use `--device cpu` or `--device cuda` to choose explicitly.

## Review training metrics

The run includes a TensorBoard event file with 300 epochs of training and
validation scalars, plus evaluation audio and spectrograms. Start TensorBoard
from the repository root:

```bash
uv run tensorboard --logdir trained_model/vits_lego_v2 --port 6006
```

Open <http://localhost:6006> and review the **Scalars**, **Images**, and
**Audio** tabs. Losses are diagnostics; listen to generated and held-out
samples when judging speech quality.

## Git and model artifacts

The LEGOv2 corpus, local trained-model runs, model checkpoints, generated samples,
and TensorBoard event files are intentionally excluded from Git because they
are large or generated. Push the source code, documentation, and the compact
Excel report; keep the corpus and model files in local storage or use a
dataset/model artifact store to share them. See `.gitignore` for the ignored
artifacts.
