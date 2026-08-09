#!/usr/bin/env python3
"""Extract table-verified grasp records from raw GRAB sequences."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import smplx
import torch
import trimesh
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grab.tabletop_grasps import (  # noqa: E402
    calibrated_table_distance_threshold,
    compute_tabletop_metrics,
    detect_contact_intervals,
    finger_group_contacts,
    pose_matrix,
    select_candidate_frame,
    tabletop_rejection_reasons,
)
from tools.utils import contact_ids as CONTACT_IDS  # noqa: E402
from tools.utils import hand_contact_ids as HAND_CONTACT_IDS  # noqa: E402


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "tabletop_grasp_extraction.yaml"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--grab-root", type=Path, help="Directory containing s1 through s10")
    parser.add_argument("--model-root", type=Path, help="SMPL-X/MANO model root")
    parser.add_argument("--output-root", type=Path, help="New, non-existing output directory")
    parser.add_argument("--sequence", action="append", help="Relative sequence path; may be repeated")
    parser.add_argument("--subject", action="append", help="Subject ID filter; may be repeated")
    parser.add_argument("--object", dest="object_name", action="append", help="Object filter; may be repeated")
    parser.add_argument("--intent", action="append", help="Motion-intent filter; may be repeated")
    parser.add_argument("--limit", type=int, help="Process the first N filtered sequences")
    parser.add_argument(
        "--validate-only", action="store_true", help="Validate inventory and references without writing"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--contact-sample-size", type=int)
    parser.add_argument("--contact-sample-seed", type=int)
    parser.add_argument("--min-contact-frames", type=int)
    parser.add_argument("--object-translation-threshold-m", type=float)
    parser.add_argument("--max-table-distance-m", type=float)
    parser.add_argument("--anchor-margin-m", type=float)
    parser.add_argument("--max-calibrated-table-distance-m", type=float)
    parser.add_argument("--footprint-margin-m", type=float)
    parser.add_argument("--min-footprint-overlap-fraction", type=float)
    parser.add_argument("--near-surface-band-m", type=float)
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Resolve YAML defaults and command-line overrides."""

    parser = build_parser()
    preliminary, _ = parser.parse_known_args(argv)
    config_path = preliminary.config
    if config_path is not None:
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file does not exist: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        valid_destinations = {action.dest for action in parser._actions}
        unknown = sorted(set(config) - valid_destinations)
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        parser.set_defaults(**config)

    args = parser.parse_args(argv)
    if args.grab_root is None:
        parser.error("--grab-root is required, either on the CLI or in --config")
    if args.model_root is None:
        parser.error("--model-root is required, either on the CLI or in --config")
    if not args.validate_only and args.output_root is None:
        parser.error("--output-root is required unless --validate-only is used")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.contact_sample_size < 1:
        parser.error("--contact-sample-size must be positive")
    if args.min_contact_frames < 1:
        parser.error("--min-contact-frames must be positive")
    return args


def json_default(value: object) -> object:
    """Convert common scientific Python values to JSON-compatible objects."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value: object) -> None:
    """Write deterministic, human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, default=json_default, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Write deterministic JSON Lines records."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=json_default, sort_keys=True, allow_nan=False))
            handle.write("\n")


def file_sha256(path: Path) -> str:
    """Compute a file SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> Dict[str, object]:
    """Return the current producer commit and dirty state."""

    def run(*arguments: str) -> str:
        return subprocess.check_output(arguments, cwd=REPO_ROOT, text=True).strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def load_mesh(path: Path) -> trimesh.Trimesh:
    """Load one mesh without topology processing."""

    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected a mesh at {path}, got {type(loaded).__name__}")
    return loaded


def discover_inventory(grab_root: Path) -> List[Dict[str, object]]:
    """Discover all raw sequences and their lightweight object/table references."""

    rows = []
    extraction_root = grab_root.parent
    for path in sorted(grab_root.glob("s*/*.npz")):
        with np.load(path, allow_pickle=True) as archive:
            object_data = archive["object"].item()
            table_data = archive["table"].item()
            object_mesh = extraction_root / str(object_data["object_mesh"])
            table_mesh = extraction_root / str(table_data["table_mesh"])
            rows.append(
                {
                    "source_sequence": path.relative_to(grab_root).as_posix(),
                    "subject_id": str(archive["sbj_id"].item()),
                    "object_name": str(archive["obj_name"].item()),
                    "motion_intent": str(archive["motion_intent"].item()),
                    "n_frames": int(archive["n_frames"].item()),
                    "framerate": float(archive["framerate"].item()),
                    "source_size_bytes": path.stat().st_size,
                    "object_mesh": str(object_mesh),
                    "object_mesh_exists": object_mesh.is_file(),
                    "table_mesh": str(table_mesh),
                    "table_mesh_exists": table_mesh.is_file(),
                }
            )
    return rows


def normalize_sequence_filter(values: Optional[Sequence[str]], grab_root: Path) -> Optional[set]:
    """Normalize explicit absolute or relative sequence filters."""

    if not values:
        return None
    normalized = set()
    for value in values:
        path = Path(value)
        if path.is_absolute():
            path = path.relative_to(grab_root)
        normalized.add(path.as_posix())
    return normalized


def select_inventory_rows(inventory: List[Dict[str, object]], args: argparse.Namespace) -> List[Dict[str, object]]:
    """Select deterministic pilot inputs from the complete inventory."""

    explicit_sequences = normalize_sequence_filter(args.sequence, args.grab_root)
    known_sequences = {str(row["source_sequence"]) for row in inventory}
    if explicit_sequences:
        missing = sorted(explicit_sequences - known_sequences)
        if missing:
            raise FileNotFoundError(f"Requested sequences are absent from GRAB_RAW_ROOT: {missing}")

    subjects = set(args.subject or [])
    objects = set(args.object_name or [])
    intents = set(args.intent or [])
    selected = []
    for row in inventory:
        include = True
        if explicit_sequences is not None:
            include &= str(row["source_sequence"]) in explicit_sequences
        if subjects:
            include &= str(row["subject_id"]) in subjects
        if objects:
            include &= str(row["object_name"]) in objects
        if intents:
            include &= str(row["motion_intent"]) in intents
        row["selected_for_run"] = bool(include)
        if include:
            selected.append(row)

    if args.limit is not None:
        selected = selected[: args.limit]
        selected_paths = {str(row["source_sequence"]) for row in selected}
        for row in inventory:
            row["selected_for_run"] = str(row["source_sequence"]) in selected_paths
    return selected


def validate_inputs(
    args: argparse.Namespace, inventory: List[Dict[str, object]], selected: List[Dict[str, object]]
) -> Dict[str, object]:
    """Validate inventory, asset references, and model roots without writing."""

    subjects = sorted({str(row["subject_id"]) for row in inventory}, key=lambda value: int(value[1:]))
    expected_subjects = [f"s{index}" for index in range(1, 11)]
    missing_assets = [
        str(row["source_sequence"])
        for row in inventory
        if not row["object_mesh_exists"] or not row["table_mesh_exists"]
    ]
    mano_root = args.model_root / "mano"
    result = {
        "grab_root": str(args.grab_root.resolve()),
        "model_root": str(args.model_root.resolve()),
        "sequence_count": len(inventory),
        "selected_sequence_count": len(selected),
        "subjects": subjects,
        "missing_asset_sequence_count": len(missing_assets),
        "missing_asset_sequences": missing_assets[:20],
        "mano_root_exists": mano_root.is_dir(),
        "expected_inventory": len(inventory) == 1335 and subjects == expected_subjects,
    }
    if not args.grab_root.is_dir():
        raise FileNotFoundError(f"GRAB root does not exist: {args.grab_root}")
    if not args.model_root.is_dir() or not mano_root.is_dir():
        raise FileNotFoundError(f"MANO model directory does not exist: {mano_root}")
    if missing_assets:
        raise FileNotFoundError(f"Missing object/table assets for {len(missing_assets)} sequences")
    return result


def resolve_device(requested: str) -> torch.device:
    """Resolve a deterministic execution device."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    return torch.device(requested)


def deterministic_sample_indices(vertex_count: int, sample_size: int, seed: int) -> np.ndarray:
    """Reproduce the legacy per-object NumPy vertex sample."""

    if vertex_count <= sample_size:
        return np.arange(vertex_count)
    random_state = np.random.RandomState(seed)
    return random_state.choice(vertex_count, sample_size, replace=False)


def mano_root_joints(
    hand_data: Mapping[str, object],
    frame_indices: Sequence[int],
    model_root: Path,
    extraction_root: Path,
    n_comps: int,
    is_right: bool,
    device: torch.device,
) -> np.ndarray:
    """Evaluate personalized MANO wrist joints only for selected frames."""

    if not frame_indices:
        return np.empty((0, 0, 3), dtype=np.float32)
    template_path = extraction_root / str(hand_data["vtemp"])
    if not template_path.is_file():
        raise FileNotFoundError(f"Hand template does not exist: {template_path}")
    template_vertices = np.asarray(load_mesh(template_path).vertices, dtype=np.float32)
    model = smplx.create(
        model_path=str(model_root),
        model_type="mano",
        is_rhand=is_right,
        v_template=template_vertices,
        num_pca_comps=n_comps,
        flat_hand_mean=True,
        batch_size=len(frame_indices),
    ).to(device)
    parameters = hand_data["params"]
    model_parameters = {
        key: torch.as_tensor(np.asarray(parameters[key])[frame_indices], dtype=torch.float32, device=device)
        for key in ("global_orient", "hand_pose", "transl")
    }
    with torch.no_grad():
        output = model(**model_parameters)
    return output.joints.detach().cpu().numpy()


def process_sequence(
    sequence_path: Path,
    args: argparse.Namespace,
    output_root: Path,
    object_cache: Dict[str, Tuple[trimesh.Trimesh, np.ndarray]],
    table_cache: Dict[str, trimesh.Trimesh],
    device: torch.device,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    """Process one raw sequence and return candidate/selection/rejection rows."""

    extraction_root = args.grab_root.parent
    relative_sequence = sequence_path.relative_to(args.grab_root).as_posix()
    with np.load(sequence_path, allow_pickle=True) as archive:
        subject_id = str(archive["sbj_id"].item())
        object_name = str(archive["obj_name"].item())
        motion_intent = str(archive["motion_intent"].item())
        n_comps = int(archive["n_comps"].item())
        object_data = archive["object"].item()
        table_data = archive["table"].item()
        contact_data = archive["contact"].item()
        left_hand_data = archive["lhand"].item()
        right_hand_data = archive["rhand"].item()

    object_mesh_path = extraction_root / str(object_data["object_mesh"])
    table_mesh_path = extraction_root / str(table_data["table_mesh"])
    object_cache_key = str(object_mesh_path.resolve())
    if object_cache_key not in object_cache:
        mesh = load_mesh(object_mesh_path)
        sample_indices = deterministic_sample_indices(
            len(mesh.vertices), args.contact_sample_size, args.contact_sample_seed
        )
        object_cache[object_cache_key] = (mesh, sample_indices)
    object_mesh, sample_indices = object_cache[object_cache_key]

    table_cache_key = str(table_mesh_path.resolve())
    if table_cache_key not in table_cache:
        table_cache[table_cache_key] = load_mesh(table_mesh_path)
    table_mesh = table_cache[table_cache_key]

    all_contact_labels = np.asarray(contact_data["object"])
    if all_contact_labels.shape[1] != len(object_mesh.vertices):
        raise ValueError(
            f"Contact/object mesh vertex mismatch for {relative_sequence}: "
            f"{all_contact_labels.shape[1]} != {len(object_mesh.vertices)}"
        )
    sampled_contact_labels = all_contact_labels[:, sample_indices]
    hand_values = tuple(HAND_CONTACT_IDS.values())
    intervals, contact_counts = detect_contact_intervals(sampled_contact_labels, hand_values)
    contact_id_to_name = {value: name for name, value in CONTACT_IDS.items()}

    object_params = object_data["params"]
    table_params = table_data["params"]
    object_vertices = np.asarray(object_mesh.vertices)
    table_vertices = np.asarray(table_mesh.vertices)
    anchor_metrics = compute_tabletop_metrics(
        object_vertices,
        object_params["global_orient"][0],
        object_params["transl"][0],
        table_vertices,
        table_params["global_orient"][0],
        table_params["transl"][0],
        args.footprint_margin_m,
        args.near_surface_band_m,
    )
    effective_table_threshold = calibrated_table_distance_threshold(
        args.max_table_distance_m,
        anchor_metrics.surface_distance_m,
        args.anchor_margin_m,
        args.max_calibrated_table_distance_m,
    )

    candidate_rows = []
    rejected_rows = []
    accepted = []
    for interval in intervals:
        row = {
            "source_sequence": relative_sequence,
            "subject_id": subject_id,
            "object_name": object_name,
            "motion_intent": motion_intent,
            "contact_segment_index": interval.segment_index,
            "contact_segment_start_frame": interval.start_frame,
            "contact_segment_end_frame": interval.end_frame,
            "contact_duration_frames": interval.duration_frames,
            "contact_sample_size": int(len(sample_indices)),
            "contact_sample_seed": args.contact_sample_seed,
        }
        if interval.duration_frames < args.min_contact_frames:
            row.update(status="rejected", rejection_reasons=["contact_interval_too_short"])
            candidate_rows.append(row)
            rejected_rows.append(dict(row))
            continue

        candidate = select_candidate_frame(
            object_params["transl"],
            object_params["global_orient"],
            contact_counts,
            interval,
            args.object_translation_threshold_m,
        )
        frame = candidate.frame_index
        hand_only_labels = np.where(
            np.isin(sampled_contact_labels[frame], hand_values), sampled_contact_labels[frame], 0
        )
        hand_contacts = finger_group_contacts(hand_only_labels, contact_id_to_name)
        finger_group_count = int(sum(sum(flags) for flags in hand_contacts.values()))
        row.update(
            source_frame_index=frame,
            frame_selection_reason=candidate.selection_reason,
            object_translation_from_interval_start_m=candidate.translation_from_interval_start_m,
            object_rotation_from_interval_start_rad=candidate.rotation_from_interval_start_rad,
            contact_vertex_count=candidate.contact_vertex_count,
            finger_group_count=finger_group_count,
            finger_contacts=hand_contacts,
        )
        if finger_group_count < 2:
            row.update(status="rejected", rejection_reasons=["fewer_than_two_finger_groups"])
            candidate_rows.append(row)
            rejected_rows.append(dict(row))
            continue

        metrics = compute_tabletop_metrics(
            object_vertices,
            object_params["global_orient"][frame],
            object_params["transl"][frame],
            table_vertices,
            table_params["global_orient"][frame],
            table_params["transl"][frame],
            args.footprint_margin_m,
            args.near_surface_band_m,
        )
        rejection_reasons = tabletop_rejection_reasons(
            metrics,
            effective_table_threshold,
            args.min_footprint_overlap_fraction,
        )
        row.update(
            tabletop_metrics=metrics.to_dict(),
            start_anchor_tabletop_metrics=anchor_metrics.to_dict(),
            effective_max_table_distance_m=effective_table_threshold,
        )
        if rejection_reasons:
            row.update(status="rejected", rejection_reasons=rejection_reasons)
            candidate_rows.append(row)
            rejected_rows.append(dict(row))
            continue

        row.update(status="selected", rejection_reasons=[])
        candidate_rows.append(row)
        accepted.append(
            {
                "manifest_row": row,
                "candidate": candidate,
                "interval": interval,
                "hand_contacts": hand_contacts,
                "metrics": metrics,
            }
        )

    selected_rows = []
    if accepted:
        frame_indices = [entry["candidate"].frame_index for entry in accepted]
        left_joints = mano_root_joints(
            left_hand_data, frame_indices, args.model_root, extraction_root, n_comps, False, device
        )
        right_joints = mano_root_joints(
            right_hand_data, frame_indices, args.model_root, extraction_root, n_comps, True, device
        )

        object_output_path = output_root / "object" / object_name / f"{object_name}.obj"
        if not object_output_path.exists():
            object_output_path.parent.mkdir(parents=True, exist_ok=True)
            object_mesh.export(object_output_path)

        for accepted_index, entry in enumerate(accepted):
            frame = entry["candidate"].frame_index
            interval = entry["interval"]
            hand_contacts = entry["hand_contacts"]
            table_pose = pose_matrix(
                table_params["global_orient"][frame],
                table_params["transl"][frame],
            )
            record = {
                "object": {
                    "name": object_name,
                    "path": object_output_path.relative_to(output_root).as_posix(),
                    "rel_scale": 1.0,
                    "pose": pose_matrix(
                        object_params["global_orient"][frame],
                        object_params["transl"][frame],
                    ),
                },
                "hand": {"left": {}, "right": {}},
                "extra": {
                    "dataset": "GRAB",
                    "source_sequence": relative_sequence,
                    "source_frame_index": frame,
                    "motion_intent": motion_intent,
                    "contact_segment_index": interval.segment_index,
                    "contact_segment_start_frame": interval.start_frame,
                    "contact_segment_end_frame": interval.end_frame,
                    "contact_duration_frames": interval.duration_frames,
                    "selection_policy": "segment_grasps_2_pre_motion_then_table_geometry_v1",
                    "frame_selection_reason": entry["candidate"].selection_reason,
                    "contact_sampling_policy": "legacy_seeded_object_vertex_sample",
                    "contact_sample_size": int(len(sample_indices)),
                    "contact_sample_seed": args.contact_sample_seed,
                    "object_translation_threshold_m": args.object_translation_threshold_m,
                    "object_translation_from_interval_start_m": entry["candidate"].translation_from_interval_start_m,
                    "object_rotation_from_interval_start_rad": entry["candidate"].rotation_from_interval_start_rad,
                    "tabletop_verified": True,
                    "coordinate_frame": "grab_world",
                    "table_pose": table_pose,
                    "tabletop_metrics": entry["metrics"].to_dict(),
                    "start_anchor_tabletop_metrics": anchor_metrics.to_dict(),
                    "effective_max_table_distance_m": effective_table_threshold,
                },
            }
            for side, hand_data, joints in (
                ("left", left_hand_data, left_joints),
                ("right", right_hand_data, right_joints),
            ):
                if not any(hand_contacts[side]):
                    continue
                parameters = hand_data["params"]
                record["hand"][side] = {
                    "trans": joints[accepted_index, 0],
                    "rot": np.asarray(parameters["global_orient"])[frame],
                    "mano_pose": np.asarray(parameters["hand_pose"])[frame],
                    "mano_betas": np.zeros((1, 10), dtype=np.float32),
                    "scale": 1000.0,
                    "contacts": hand_contacts[side],
                }

            base_name = sequence_path.stem
            output_path = (
                output_root / "grasp" / object_name / (f"{subject_id}_{base_name}_seg{interval.segment_index}.npy")
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, record)
            selection_row = dict(entry["manifest_row"])
            selection_row.update(
                output_relative_path=output_path.relative_to(output_root).as_posix(),
                output_sha256=file_sha256(output_path),
            )
            selected_rows.append(selection_row)

    sequence_summary = {
        "source_sequence": relative_sequence,
        "subject_id": subject_id,
        "object_name": object_name,
        "motion_intent": motion_intent,
        "status": "processed",
        "detected_interval_count": len(intervals),
        "selected_record_count": len(selected_rows),
        "rejected_candidate_count": len(rejected_rows),
    }
    return candidate_rows, selected_rows, rejected_rows, sequence_summary


def resolved_config(args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    """Build the config and environment record saved with an extraction."""

    values = vars(args).copy()
    values["config"] = str(args.config.resolve()) if args.config else None
    values["grab_root"] = str(args.grab_root.resolve())
    values["model_root"] = str(args.model_root.resolve())
    values["output_root"] = str(args.output_root.resolve()) if args.output_root else None
    values["resolved_device"] = str(device)
    values["producer_git"] = git_state()
    values["environment"] = {
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "smplx": str(getattr(smplx, "__version__", "unknown")),
        "trimesh": str(trimesh.__version__),
        "cuda_available": torch.cuda.is_available(),
    }
    return values


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run validation or extraction."""

    args = parse_args(argv)
    args.grab_root = args.grab_root.resolve()
    args.model_root = args.model_root.resolve()
    if args.output_root is not None:
        args.output_root = args.output_root.resolve()

    inventory = discover_inventory(args.grab_root)
    selected_inventory = select_inventory_rows(inventory, args)
    validation = validate_inputs(args, inventory, selected_inventory)
    if args.validate_only:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0
    if not selected_inventory:
        raise ValueError("No sequences matched the requested filters")
    if args.output_root.exists():
        raise FileExistsError(f"Output root already exists and will not be overwritten: {args.output_root}")

    device = resolve_device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "configs").mkdir()
    (args.output_root / "manifest").mkdir()
    config_record = resolved_config(args, device)
    with (args.output_root / "configs" / "resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_record, handle, sort_keys=True)
    write_jsonl(args.output_root / "manifest" / "source_inventory.jsonl", inventory)

    object_cache: Dict[str, Tuple[trimesh.Trimesh, np.ndarray]] = {}
    table_cache: Dict[str, trimesh.Trimesh] = {}
    candidate_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []
    rejected_rows: List[Dict[str, object]] = []
    sequence_summaries = {
        str(row["source_sequence"]): {
            "source_sequence": row["source_sequence"],
            "subject_id": row["subject_id"],
            "object_name": row["object_name"],
            "motion_intent": row["motion_intent"],
            "status": "not_selected_for_run",
            "detected_interval_count": 0,
            "selected_record_count": 0,
            "rejected_candidate_count": 0,
        }
        for row in inventory
    }
    errors = []
    for index, inventory_row in enumerate(selected_inventory, start=1):
        source_sequence = str(inventory_row["source_sequence"])
        print(f"[{index}/{len(selected_inventory)}] {source_sequence}")
        try:
            candidates, selections, rejections, summary = process_sequence(
                args.grab_root / source_sequence,
                args,
                args.output_root,
                object_cache,
                table_cache,
                device,
            )
            candidate_rows.extend(candidates)
            selected_rows.extend(selections)
            rejected_rows.extend(rejections)
            sequence_summaries[source_sequence] = summary
        except Exception as error:  # Preserve per-sequence failure evidence before returning non-zero.
            errors.append({"source_sequence": source_sequence, "error": repr(error)})
            sequence_summaries[source_sequence].update(status="error", error=repr(error))

    ordered_summaries = [sequence_summaries[str(row["source_sequence"])] for row in inventory]
    selected_candidate_count = sum(row.get("status") == "selected" for row in candidate_rows)
    rejected_candidate_count = sum(row.get("status") == "rejected" for row in candidate_rows)
    output_relative_paths = [str(row["output_relative_path"]) for row in selected_rows]
    if selected_candidate_count != len(selected_rows):
        raise RuntimeError(f"Selected candidate/output mismatch: {selected_candidate_count} != {len(selected_rows)}")
    if rejected_candidate_count != len(rejected_rows):
        raise RuntimeError(f"Rejected candidate/manifest mismatch: {rejected_candidate_count} != {len(rejected_rows)}")
    if len(candidate_rows) != len(selected_rows) + len(rejected_rows):
        raise RuntimeError("Every detected interval must resolve to selected or rejected")
    if len(output_relative_paths) != len(set(output_relative_paths)):
        raise RuntimeError("Selected candidates produced duplicate output paths")
    if len(ordered_summaries) != len(inventory):
        raise RuntimeError("Sequence summary does not cover the complete inventory")

    write_jsonl(args.output_root / "manifest" / "candidate_segments.jsonl", candidate_rows)
    write_jsonl(args.output_root / "manifest" / "selection.jsonl", selected_rows)
    write_jsonl(args.output_root / "manifest" / "rejected_candidates.jsonl", rejected_rows)
    write_jsonl(args.output_root / "manifest" / "sequence_summary.jsonl", ordered_summaries)

    rejection_counts = Counter(reason for row in rejected_rows for reason in row.get("rejection_reasons", ["unknown"]))
    summary = {
        "inventory_sequence_count": len(inventory),
        "selected_for_run_sequence_count": len(selected_inventory),
        "processed_sequence_count": sum(row["status"] == "processed" for row in ordered_summaries),
        "error_sequence_count": len(errors),
        "detected_interval_count": len(candidate_rows),
        "selected_record_count": len(selected_rows),
        "selected_sequence_count": sum(row["selected_record_count"] > 0 for row in ordered_summaries),
        "rejected_candidate_count": len(rejected_rows),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "errors": errors,
        "validation": validation,
    }
    write_json(args.output_root / "manifest" / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
