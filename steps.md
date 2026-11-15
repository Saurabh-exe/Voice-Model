# TTS Dataset Preparation: Progress and Next Steps

This document records dataset preparation and the current status of the LEGOv2
text-to-speech (TTS) experiment, including GPU fine-tuning and local inference.

## 📁 Project Structure Update

The scripts in `src/voice_model/` have been reorganized into two subdirectories:

- **`src/voice_model/data_preprocessing/`** ← Data preparation scripts
  - `manifest_builder.py` - Builds `tts_manifest.json` from `interactions.csv`
  - `transcribe_manifest.py` - Whisper transcription + comparison JSON
  - `create_transcript_review.py` - Exports disagreements to CSV for manual review

- **`src/voice_model/training/`** ← Training manifest scripts
  - `build_training_manifest.py` - Merges comparisons + corrections → `tts_training_manifest.json`
  - `create_vits_manifest.py` - Converts manifest to VITS `metadata.csv` format
- **`trained_model/`** ← Trained model checkpoints and run artifacts; ignored by Git

- **`src/voice_model/speaker_analysis/`** ← Already existed
  - `analyze_speakers.py` - ECAPA-TDNN speaker similarity analysis
- **`src/voice_model/vits_finetuning.py`** - Fine-tunes multi-speaker VITS
- **`src/voice_model/vits_inference.py`** - Generates WAV files from a trained run

---

## ✅ Completed Steps

### 1. **Inspected the LEGOv2 corpus**
- Audio clips are under `LEGOv2/audio/` (532 `.raw`/`.wav` files, ~114 MB, ~2 hours)
- Interaction records and ASR transcripts are in `LEGOv2/corpus/csv/interactions.csv`
- The interactions CSV is semicolon-separated with 60 columns per row
- Corpus documentation describes utterance audio and full-call recordings

### 2. **Created the initial audio/text manifest**
- `src/voice_model/data_preprocessing/manifest_builder.py` reads the CSV using `cp1252` encoding
- Skips rows without transcript text or valid existing audio
- Writes `tts_manifest.json` with 4,751 entries
- Each entry contains: `audio`, `text`, `call_id`, `caller_id`, and `speaker_id`
- The `speaker_id` is derived from `call_id`; it is not a verified speaker identity

### 3. **Generated ASR comparisons**
- `src/voice_model/data_preprocessing/transcribe_manifest.py` uses Faster Whisper to transcribe the manifest
- Saves `tts_manifest_comparison.json` with entries containing: `original text`, `asr_text`, and `normalized_match` flag
- Whisper Large v3 was run in a hosted notebook workflow
- Comparison output contains the original `text`, Whisper's `asr_text`, and a rough `normalized_match` flag
- A matching flag is not proof that either transcript is correct

### 4. **Created a transcript review sheet**
- `src/voice_model/data_preprocessing/create_transcript_review.py` exports disagreements to `transcript_review.csv`
- The sheet was manually edited with corrected transcripts
- **928 of 928 rows had corrections** (previously 897/928, now fully corrected)
- Blank `corrected_text` column was filled for all rows

### 5. **Built a cleaned transcript manifest**
- `src/voice_model/training/build_training_manifest.py` combines the comparison JSON and transcript review CSV into `tts_training_manifest.json`
- Uses manual corrections for disagreements with a filled-in `corrected_text`
- Retains original text when normalized transcripts matched
- Excludes unresolved disagreements
- Generated manifest: **1,519 examples**, no empty text, no missing audio files
- 103 call-based speaker labels, about 0.7 hours of audio
- Clips are mono 8 kHz recordings
- Matching ASR transcripts were not individually verified and should still be sampled by listening

### 6. **Added speaker-similarity analysis**
- `src/voice_model/speaker_analysis/` contains the workflow
- `analyze_speakers.py` computes clip-level SpeechBrain ECAPA-TDNN embeddings
- Averages them by call, proposes call clusters, and writes similarity/review files under `speaker_analysis/output/`
- `src/voice_model/speaker_analysis/README.md` documents how to run it and review the results
- Proposed clusters are suggestions, especially given short, noisy, low-sample-rate telephone audio. They must be checked by listening.
- The script can apply manually reviewed call labels into a new manifest named `tts_training_manifest_speaker_reviewed.json`

### 7. **Created VITS training-ready metadata**
- `src/voice_model/training/create_vits_manifest.py` converts `tts_manifest.json` into `tts_training_data/metadata.csv`
- Format per VITS/TTS expectation: `text|speaker|audio_path|text_cleaned`
- **4,751 rows** with no skipped entries
- Text artifacts (`\ufeff`) cleaned automatically
- Audio paths point to `LEGOv2/audio/... .wav` files

### 8. **Reorganized scripts into data_preprocessing and training folders**
- Moved `manifest_builder.py`, `transcribe_manifest.py`, `create_transcript_review.py` to `src/voice_model/data_preprocessing/`
- Moved `build_training_manifest.py`, `create_vits_manifest.py` to `src/voice_model/training/`
- Updated all `ROOT` paths from `parents[2]` to `parents[3]` to account for the new subdirectory structure
- All scripts verified working with updated paths

### 9. **Fine-tuned multi-speaker VITS on GPU**
- The completed run is in `trained_model/vits_lego_v2/`
- Run assets include `best_model.pth`, `config.json`, `speakers.pth`, the trainer log, and a TensorBoard event file
- The configuration records 109 speaker embeddings and 300 training epochs
- The event file contains training and validation scalar metrics, plus evaluation audio and spectrograms
- `src/voice_model/vits_inference.py` loads the checkpoint/config/speaker map together and writes synthesized WAV files
- Initial local inference works, but the generated voice quality is poor and needs further evaluation/improvement
- The Excel loss report is `outputs/vits-loss-report-2026-11-15/vits_training_report.xlsx`

---

## 📊 Current Data Summary

| File/Artifact | Description | Count/Size |
|--------------|-------------|------------|
| `tts_manifest.json` | Original text-audio pairs from interactions.csv | 4,751 entries |
| `transcript_review.csv` | Manual corrections for all 928 reviewed rows | 928 rows, 100% corrected |
| `tts_manifest_comparison.json` | Whisper ASR comparison output | All entries with `asr_text` + `normalized_match` |
| `tts_training_manifest.json` | Cleaned manifest with corrections applied | 1,519 examples (0.7 hrs audio) |
| `tts_training_manifest_speaker_reviewed.json` | Separate manifest with reviewed speaker labels | Depends on the review CSV and applied labels |
| `tts_training_data/metadata.csv` | **VITS-ready metadata** for fine-tuning | **4,751 rows** |
| `speaker_analysis/output/speaker_review.csv` | Speaker similarity review with 103 call entries | 103 calls, all have `reviewed_speaker_id` fields |

**Audio characteristics:** 8 kHz, 16-bit, mono WAV files

---

## Current Status and Next Steps

- GPU fine-tuning completed 300 epochs. Initial synthesized samples sound poor, so the checkpoint is experimental rather than ready for deployment.
- The corpus is small and consists of short 8 kHz telephone clips. Upsampling to 22.05 kHz enables the selected model but does not restore detail that was not recorded.
- Speaker IDs were assigned to calls and clustered by similarity; they are not confirmed real-person identities. Review speaker assignments and clips for consistency.
- Transcript corrections are complete for the reviewed disagreements, but transcripts and matching ASR rows still need listening-based spot checks.
- The Excel report summarizes epoch losses. Use TensorBoard Audio and Images tabs to inspect held-out utterances and spectrograms too.

Recommended next steps:

1. Compare several held-out TensorBoard samples with their reference audio and transcripts. Also generate the same short test sentences for a few known speaker IDs.
2. Check low-quality examples for transcript errors, clipping, silence, overlap, and incorrect speaker labels; clean the data and keep changes reproducible.
3. Review training and validation curves together. A lower loss does not guarantee natural speech; use listening tests alongside the metrics.
4. Run a controlled follow-up fine-tune after the data review, changing one training choice at a time and comparing checkpoints with the same sentences and speaker IDs.
5. Package or deploy the checkpoint only after intelligibility, pronunciation, and speaker consistency are acceptable.

### Generate a sample

```bash
uv run python src/voice_model/vits_inference.py \
  --text "Thank you for calling. How can I help you today?" \
  --speaker proposed_speaker_0009 \
  --output generated_audio/sample.wav
```

Use a speaker name present in `trained_model/vits_lego_v2/speakers.pth`.

### View TensorBoard metrics and evaluation samples

```bash
uv run tensorboard --logdir trained_model/vits_lego_v2 --port 6006
```

Open <http://localhost:6006>. The event file records 300 epochs of train and
validation metrics as well as generated evaluation audio and spectrograms.
The workbook `outputs/vits-loss-report-2026-11-15/vits_training_report.xlsx`
provides loss charts and an epoch-by-epoch scalar table.

---

For the complete project overview and speaker-analysis commands, see
`README.md` and `src/voice_model/speaker_analysis/README.md`.
