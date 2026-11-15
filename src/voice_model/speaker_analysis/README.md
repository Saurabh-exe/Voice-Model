# Speaker grouping review

This workflow creates pseudonymous speaker-group suggestions from voice
similarity. It does not identify a person's real-world identity. The starting
`speaker_id` values in the training manifest are call-based placeholders; this
script compares calls to find possible repeated voices.

## Install

Use an environment with PyTorch and torchaudio installed, then install
SpeechBrain and NumPy:

```bash
uv add speechbrain
```

In Colab or Kaggle, install SpeechBrain in a notebook cell instead:

```python
%pip install speechbrain
```

The first run downloads the pretrained ECAPA-TDNN model, so network access is
needed. The script resamples audio to 16 kHz for embedding extraction.

## Generate embeddings and proposed groups

From the project root:

```bash
uv run python -m voice_model.speaker_analysis.analyze_speakers \
  --manifest tts_training_manifest.json \
  --data-root . \
  --output-dir speaker_analysis/output \
  --device cpu \
  --threshold 0.65
```

Use `--device cuda` only with a CUDA-enabled NVIDIA GPU and a CUDA-enabled
PyTorch installation. macOS does not support CUDA; use `--device cpu` there.
The threshold is only an initial exploratory setting; it is not calibrated for
this telephone corpus.

The output folder contains:

- `clip_embeddings.npz`: per-clip embeddings; saved every 50 clips so runs can
  resume. Delete this file to recompute the clip embeddings.
- `call_embeddings.npz`: averaged, normalized embeddings for each call.
- `call_similarity.csv`: pairwise cosine similarities, highest first.
- `speaker_review.csv`: one row per call, with proposed groups copied into
  `reviewed_speaker_id` as the starting labels, plus a notes column.
- `pretrained_model/`: downloaded SpeechBrain model files.

Review candidate calls by listening to clips from each call. The generated
`reviewed_speaker_id` values initially match `proposed_speaker_id`. If you
want to correct a group, edit `reviewed_speaker_id` in the review CSV; give
calls with the same voice the same label and different voices distinct labels.
Short/noisy 8 kHz phone clips can produce incorrect similarities.

## Write proposed labels into a new manifest

If you have listened to the calls and approved the proposed groups, run:

```bash
uv run python -m voice_model.speaker_analysis.analyze_speakers \
  --manifest tts_training_manifest.json \
  --output-dir speaker_analysis/output \
  --apply-reviewed \
  --output-manifest tts_training_manifest_speaker_reviewed.json
```

The command reads each call's `reviewed_speaker_id`, checks that every call in
the manifest has exactly one non-empty reviewed label, and writes those labels
as `speaker_id` to a new manifest. The source manifest is not modified.

The generated reviewed labels already match the proposals, so you can run the
command as-is or edit `reviewed_speaker_id` first to correct any labels.
