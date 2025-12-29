# Merge adamu pnd robot dataset w/ diverse grasptype

```
python merge_pickplace_final.py \
  --root /mnt/pfs/scalelab/yiqing/openpi \
  --sources PickPlaceBottle_6 PickPlaceBottle_7 PickPlaceBottle_8 PickPlaceBottle_9 PickPlaceBottle_10 \
  --output-name PickPlaceBottle_Merged_v3 \
  --camera-ids 0 1 \
  --overwrite
```

# Merge PickPlaceBottle Datasets

This repository now ships `merge_pickplace_pairs.py`, a self-contained script to merge
multiple fully materialized PickPlaceBottle datasets (each with `data/`, `videos/`,
and `meta/` folders) into a single dataset that is ready for OpenPI training.

## Quick Start

```bash
cd /mnt/pfs/scalelab/yiqing/openpi
python merge_pickplace_pairs.py \
  --inputs \
    /mnt/pfs/scalelab/yiqing/openpi/PickPlaceBottle/PickPlaceBottle_Merged_v2 \
    /mnt/pfs/scalelab/yiqing/openpi/PickPlaceBottle/PickPlaceBottle_Merged_v3 \
  --output-dir /mnt/pfs/scalelab/yiqing/openpi/PickPlaceBottle/PickPlaceBottle_MegaMerge \
  --overwrite
```

## Script Behavior

- Copies every parquet episode file and all requested camera videos.
- Rebuilds `meta/episodes.jsonl`, `meta/tasks.jsonl`, `meta/info.json` with the new episode counts.
- Filters camera features based on `--camera-ids` (defaults to all RGB cameras detected in the
  first dataset).
- Recomputes normalization statistics (`meta/stats.json` and `norm_stats.json`).
- Optionally copies preview images when `--copy-images` is provided.

Refer to `python merge_pickplace_pairs.py --help` for all flags such as chunk-size overrides,
custom camera subsets, and non-default input/output locations.

## From episodes.jsonl to tasks.json

```
python annotate/generate_tasks_jsonl.py --dataset-path /mnt/pfs/scalelab/yiqing/openpi/PickPlaceBottle/PickPlaceBottle_MegaMerge
```

## Compute norms
```
source /mnt/pfs/scalelab/yiqing/openpi/.venv/bin/activate
PYTHONPATH=$PWD/src python scripts/compute_norm_stats.py --config-name pi05_adamu
```
