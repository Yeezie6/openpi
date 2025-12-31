import json
import re

# Paths
annotations_path = '/mnt/pfs/scalelab/yiqing/openpi/annotate/naive_1_annot_result/annotations_robot_naive_1.json'
# annotations_path = '/mnt/pfs/scalelab/yiqing/openpi/annotate/annotations_robot_v4.json'
episodes_path = '/mnt/pfs/scalelab/yiqing/openpi/PickPlaceBottle/PickPlaceBottle_Merged_v4/meta/episodes.jsonl'

# Load annotations_robot.json
annotations_map = {}
with open(annotations_path, 'r', encoding='utf-8') as f:
    annotations_data = json.load(f)
    for base_name, segments in annotations_data.get('results', {}).items():
        # Extract index from base_name (e.g., "episode_000264" -> 264)
        match = re.search(r'episode_(\d+)', base_name)
        if match:
            episode_idx = int(match.group(1))
            # Use the first segment's task description as the source
            if segments:
                annotations_map[episode_idx] = segments[0].get('task_description', '')

# Update episodes.jsonl
updated_count = 0
updated_lines = []
with open(episodes_path, 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        episode_idx = entry['episode_index']
        
        if episode_idx in annotations_map:
            # Replace tasks list with the description from annotations
            entry['tasks'] = [annotations_map[episode_idx]]
            updated_count += 1
        
        updated_lines.append(json.dumps(entry))

# Save updated episodes.jsonl
with open(episodes_path, 'w', encoding='utf-8') as f:
    for line in updated_lines:
        f.write(line + '\n')

print(f"Updated {updated_count} episodes in {episodes_path}")
