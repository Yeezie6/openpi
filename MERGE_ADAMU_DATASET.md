# Merge adamu pnd robot dataset w/ diverse grasptype

```
python merge_pickplace_final.py \
  --root /mnt/pfs/scalelab/yiqing/openpi \
  --sources PickPlaceBottle_6 PickPlaceBottle_7 PickPlaceBottle_8 PickPlaceBottle_9 PickPlaceBottle_10 \
  --output-name PickPlaceBottle_Merged_v3 \
  --camera-ids 0 1 \
  --overwrite


python /mnt/pfs/scalelab/yiqing/openpi/annotate/merge_annotations.py


python annotate/update_annotations.py
```

#

## From episodes.jsonl to tasks.json

```
python annotate/generate_tasks_jsonl.py --dataset-path /mnt/pfs/scalelab/yiqing/openpi/PickPlaceBottle/PickPlaceBottle_Merged_v4

python annotate/update_parquet_task_index.py --dataset-path /mnt/pfs/scalelab/yiqing/openpi/PickPlaceBottle/PickPlaceBottle_Merged_v4
```

## Compute norms
```
rm -rf /mnt/pfs/scalelab/hf_cache/datasets/parquet

source /mnt/pfs/scalelab/yiqing/openpi/.venv/bin/activate
PYTHONPATH=$PWD/src python scripts/compute_norm_stats.py --config-name pi05_adamu
```
