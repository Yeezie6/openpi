import json
import os
import re

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def get_episode_index(key):
    match = re.search(r'episode_(\d+)', key)
    if match:
        return int(match.group(1))
    return -1

def merge_annotations():
    file1_path = "/mnt/pfs/scalelab/yiqing/openpi/annotate/annotations_robot.json"
    file2_path = "/mnt/pfs/scalelab/yiqing/openpi/annotate/intermidiate_annot_result/annotations_robot_v3.json"
    output_path = "/mnt/pfs/scalelab/yiqing/openpi/annotate/annotations_robot_v4.json"

    print(f"Loading {file1_path}...")
    data1 = load_json(file1_path)
    print(f"Loading {file2_path}...")
    data2 = load_json(file2_path)

    results1 = data1.get("results", {})
    results2 = data2.get("results", {})

    # Find the max index in data1
    max_idx = -1
    for key in results1.keys():
        idx = get_episode_index(key)
        if idx > max_idx:
            max_idx = idx
    
    print(f"Max episode index in file 1: {max_idx}")

    new_results = results1.copy()
    
    # Sort keys in data2 to maintain order if possible, though dicts are insertion ordered in recent python
    # It's better to sort by index to be deterministic
    keys2 = sorted(results2.keys(), key=get_episode_index)

    current_idx = max_idx + 1
    count_added = 0

    for key in keys2:
        old_items = results2[key]
        new_key = f"episode_{current_idx:06d}"
        
        # Deep copy items to modify base_name
        new_items = []
        for item in old_items:
            new_item = item.copy()
            new_item["base_name"] = new_key
            new_items.append(new_item)
        
        new_results[new_key] = new_items
        current_idx += 1
        count_added += 1

    total_base_names = len(new_results)
    
    merged_data = {
        "total_base_names": total_base_names,
        "results": new_results
    }

    print(f"Added {count_added} episodes from file 2.")
    print(f"Total episodes in merged file: {total_base_names}")
    print(f"Saving to {output_path}...")
    save_json(merged_data, output_path)
    print("Done.")

if __name__ == "__main__":
    merge_annotations()
