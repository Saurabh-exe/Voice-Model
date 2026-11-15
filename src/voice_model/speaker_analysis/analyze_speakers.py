"""Generate call-level speaker suggestions and apply manually reviewed labels."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "tts_training_manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "speaker_analysis" / "output"
MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
SAMPLE_RATE = 16000
CHECKPOINT_EVERY = 50


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return a unit-length vector, leaving a zero vector unchanged."""
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return vector / norm


def resolve_device(device: str, cuda_available: bool) -> str:
    """Validate the requested device and use SpeechBrain's indexed CUDA form."""
    if device == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "CUDA was requested, but this PyTorch installation has no working "
                "CUDA device. Rerun with --device cpu."
            )
        return "cuda:0"
    return device


def complete_link_clusters(
    call_ids: Sequence[str], embeddings: np.ndarray, threshold: float
) -> List[List[int]]:
    """Greedily merge clusters only when every cross-pair meets the threshold."""
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("Similarity threshold must be between -1 and 1.")

    similarity = embeddings @ embeddings.T
    clusters = [[index] for index in range(len(call_ids))]

    while True:
        best_pair: Optional[Tuple[int, int]] = None
        best_similarity = threshold
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                cross_similarities = similarity[
                    np.ix_(clusters[left], clusters[right])
                ]
                cluster_similarity = float(np.min(cross_similarities))
                if cluster_similarity >= best_similarity:
                    best_similarity = cluster_similarity
                    best_pair = (left, right)

        if best_pair is None:
            break
        left, right = best_pair
        clusters[left] = sorted(clusters[left] + clusters[right])
        del clusters[right]

    return sorted(clusters, key=lambda cluster: min(cluster))


def write_npz_checkpoint(
    path: Path,
    audio_paths: Sequence[str],
    embeddings: np.ndarray,
) -> None:
    """Atomically save completed clip embeddings so an interrupted run can resume."""
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("wb") as output_file:
        np.savez_compressed(
            output_file,
            audio_paths=np.asarray(audio_paths, dtype=np.str_),
            embeddings=embeddings,
        )
    temporary_path.replace(path)


def load_manifest(path: Path) -> List[dict]:
    """Read and validate the JSON training manifest."""
    with path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest, list):
        raise ValueError(f"Expected a JSON list in {path}")

    seen_audio = set()
    for item in manifest:
        if not isinstance(item, dict):
            raise ValueError("Every manifest entry must be a JSON object.")
        if not isinstance(item.get("audio"), str) or not item["audio"]:
            raise ValueError("Every manifest entry needs a non-empty 'audio' path.")
        if not isinstance(item.get("call_id"), str) or not item["call_id"]:
            raise ValueError(
                "Every manifest entry needs a non-empty string 'call_id'."
            )
        if item["audio"] in seen_audio:
            raise ValueError(f"Duplicate audio path in manifest: {item['audio']}")
        seen_audio.add(item["audio"])
    return manifest


def generate_call_review(
    manifest: List[dict],
    data_root: Path,
    output_dir: Path,
    threshold: float,
    device: str,
) -> None:
    """Embed each clip, aggregate by call, cluster calls, and write review CSVs."""
    import torch
    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier

    device = resolve_device(device, torch.cuda.is_available())
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "clip_embeddings.npz"
    audio_paths = [item["audio"] for item in manifest]
    vectors: List[Optional[np.ndarray]] = [None] * len(manifest)

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            cached_paths = cache["audio_paths"].tolist()
            cached_embeddings = cache["embeddings"]
        if cached_embeddings.shape[0] != len(cached_paths):
            raise ValueError(f"Invalid embedding cache: {cache_path}")
        cached_by_path = {
            audio: cached_embeddings[index]
            for index, audio in enumerate(cached_paths)
        }
        for index, audio in enumerate(audio_paths):
            if audio in cached_by_path:
                vectors[index] = cached_by_path[audio]
        print(
            f"Resuming with {sum(vector is not None for vector in vectors)} "
            "cached clip embeddings."
        )

    print(f"Loading SpeechBrain speaker-embedding model on {device}: {MODEL_SOURCE}")
    encoder = EncoderClassifier.from_hparams(
        source=MODEL_SOURCE,
        savedir=str(output_dir / "pretrained_model"),
        run_opts={"device": device},
    )

    for index, item in enumerate(manifest):
        if vectors[index] is not None:
            continue

        audio_path = (data_root / item["audio"]).resolve()
        try:
            audio_path.relative_to(data_root.resolve())
        except ValueError as exc:
            raise ValueError(f"Audio path escapes data root: {item['audio']}") from exc
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform, sample_rate = torchaudio.load(str(audio_path))
        if waveform.numel() == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=SAMPLE_RATE
            )
        waveform = waveform.to(device)

        with torch.no_grad():
            embedding = encoder.encode_batch(waveform)
        vector = embedding.detach().cpu().numpy().reshape(-1).astype(np.float32)
        vectors[index] = l2_normalize(vector)

        completed = index + 1
        if completed % CHECKPOINT_EVERY == 0 or completed == len(manifest):
            completed_indices = [
                i for i, vector in enumerate(vectors) if vector is not None
            ]
            complete_vectors = np.stack([vectors[i] for i in completed_indices])
            write_npz_checkpoint(
                cache_path,
                [audio_paths[i] for i in completed_indices],
                complete_vectors[completed_indices],
            )
            print(f"Embedded {len(completed_indices)}/{len(manifest)} clips.")

    if any(vector is None for vector in vectors):
        raise RuntimeError("Embedding extraction ended with incomplete clip results.")
    clip_embeddings = np.stack(vectors)
    call_to_indices: Dict[str, List[int]] = {}
    for index, item in enumerate(manifest):
        call_to_indices.setdefault(item["call_id"], []).append(index)

    call_ids = sorted(call_to_indices)
    call_embeddings = []
    for call_id in call_ids:
        per_clip = clip_embeddings[call_to_indices[call_id]]
        call_embeddings.append(l2_normalize(per_clip.mean(axis=0)))
    call_embeddings_array = np.stack(call_embeddings)

    clusters = complete_link_clusters(call_ids, call_embeddings_array, threshold)
    proposed_by_call = {}
    for cluster_number, cluster in enumerate(clusters, start=1):
        proposed_id = f"proposed_speaker_{cluster_number:04d}"
        for index in cluster:
            proposed_by_call[call_ids[index]] = proposed_id

    review_path = output_dir / "speaker_review.csv"
    with review_path.open("w", encoding="utf-8", newline="") as review_file:
        writer = csv.DictWriter(
            review_file,
            fieldnames=[
                "call_id",
                "current_speaker_id",
                "clip_count",
                "proposed_speaker_id",
                "reviewed_speaker_id",
                "notes",
            ],
        )
        writer.writeheader()
        for call_id in call_ids:
            call_items = [manifest[i] for i in call_to_indices[call_id]]
            existing_ids = sorted(
                {str(item.get("speaker_id", "")) for item in call_items}
            )
            proposed_id = proposed_by_call[call_id]
            writer.writerow(
                {
                    "call_id": call_id,
                    "current_speaker_id": ";".join(existing_ids),
                    "clip_count": len(call_items),
                    "proposed_speaker_id": proposed_id,
                    "reviewed_speaker_id": proposed_id,
                    "notes": "",
                }
            )

    similarity_rows = []
    for left in range(len(call_ids)):
        for right in range(left + 1, len(call_ids)):
            score = float(
                np.dot(call_embeddings_array[left], call_embeddings_array[right])
            )
            similarity_rows.append((score, call_ids[left], call_ids[right]))
    similarity_rows.sort(reverse=True)

    similarity_path = output_dir / "call_similarity.csv"
    with similarity_path.open("w", encoding="utf-8", newline="") as similarity_file:
        writer = csv.DictWriter(
            similarity_file,
            fieldnames=[
                "call_id_1",
                "call_id_2",
                "cosine_similarity",
                "same_proposed_cluster",
            ],
        )
        writer.writeheader()
        for score, first_call, second_call in similarity_rows:
            writer.writerow(
                {
                    "call_id_1": first_call,
                    "call_id_2": second_call,
                    "cosine_similarity": f"{score:.6f}",
                    "same_proposed_cluster": str(
                        proposed_by_call[first_call]
                        == proposed_by_call[second_call]
                    ).lower(),
                }
            )

    np.savez_compressed(
        output_dir / "call_embeddings.npz",
        call_ids=np.asarray(call_ids, dtype=np.str_),
        embeddings=call_embeddings_array,
    )
    print(f"Proposed clusters: {len(clusters)}")
    print(f"Review call labels in: {review_path}")
    print(f"Review pair similarities in: {similarity_path}")
    print(
        "Suggested groups use cosine threshold "
        f"{threshold:.2f}; manually verify before using them."
    )


def apply_reviewed_labels(
    manifest_path: Path, review_path: Path, output_path: Path
) -> None:
    """Apply reviewed speaker groups and write a separate manifest."""
    manifest = load_manifest(manifest_path)
    labels: Dict[str, str] = {}
    with review_path.open("r", encoding="utf-8-sig", newline="") as review_file:
        reader = csv.DictReader(review_file)
        required = {"call_id", "reviewed_speaker_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Speaker review CSV needs call_id and reviewed_speaker_id columns."
            )
        for row in reader:
            call_id = (row.get("call_id") or "").strip()
            speaker_id = (row.get("reviewed_speaker_id") or "").strip()
            if not call_id:
                raise ValueError("Speaker review CSV contains a blank call_id.")
            if not speaker_id:
                raise ValueError(
                    f"reviewed_speaker_id is blank for call_id {call_id}."
                )
            if call_id in labels:
                raise ValueError(f"Duplicate call_id in speaker review CSV: {call_id}")
            labels[call_id] = speaker_id

    manifest_call_ids = {item["call_id"] for item in manifest}
    missing_labels = manifest_call_ids - set(labels)
    extra_labels = set(labels) - manifest_call_ids
    if missing_labels or extra_labels:
        raise ValueError(
            f"Speaker review IDs do not match manifest; "
            f"missing={sorted(missing_labels)}, extra={sorted(extra_labels)}"
        )

    for item in manifest:
        item["speaker_id"] = labels[item["call_id"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, ensure_ascii=False, indent=2)
    print(f"Reviewed-speaker-label manifest saved to: {output_path}")
    print(f"Examples updated: {len(manifest)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Suggest speaker groups from audio and apply reviewed labels."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Exploratory cosine-similarity threshold for proposed call clusters.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--apply-reviewed",
        "--apply-proposed",
        dest="apply_reviewed",
        action="store_true",
        help="Apply reviewed_speaker_id values from speaker_review.csv.",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        help="Review CSV to use when applying speaker labels.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        help="Output manifest path to use when applying speaker labels.",
    )
    args = parser.parse_args()

    if args.apply_reviewed:
        review_csv = args.review_csv or args.output_dir / "speaker_review.csv"
        output_manifest = (
            args.output_manifest
            or ROOT / "tts_training_manifest_speaker_reviewed.json"
        )
        apply_reviewed_labels(args.manifest, review_csv, output_manifest)
        return

    manifest = load_manifest(args.manifest)
    if not manifest:
        raise ValueError(f"Manifest is empty: {args.manifest}")
    generate_call_review(
        manifest=manifest,
        data_root=args.data_root,
        output_dir=args.output_dir,
        threshold=args.threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
