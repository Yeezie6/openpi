import os, sys, shutil
from dataset import LerobotDatasetWriter, LerobotDatasetReader
import json

def main(dataset_names, output_name, camera_ids=[0,1], data_type="rgb", num_actions=31, image_shape=(256, 256, 3),
         chunk_size=1000):           
    dataset_writer = LerobotDatasetWriter(
        output_path=output_name,
        camera_ids=[], #camera_ids,
        data_type="", #data_type,
        action_dim=num_actions,
        state_dim=num_actions,
        image_shape=image_shape,
    )

    i_chunk = 0
    i_episode = 0

    for name in dataset_names:
        source_dataset = LerobotDatasetReader(name)
        episode_end_ids = source_dataset.episode_ends
        episode_end_ids = [0] + [i+1 for i in episode_end_ids]
        print(episode_end_ids)

        episodes_info = []
        with open(os.path.join("datasets", name, "meta/episodes.jsonl"), 'r', encoding='utf-8') as f:
            for line in f:
                # 每行都是一个JSON对象
                item = json.loads(line.strip())
                episodes_info.append(item)

        for ep in range(1, len(episode_end_ids)):
            i_episode_source = episodes_info[ep-1]['episode_index']
            i_chunk_source = i_episode_source // chunk_size

            # move videos
            if "rgb" in data_type:
                for cid in camera_ids:
                    source_path = f"datasets/{name}/videos/chunk-{i_chunk_source:03d}/observation.camera_{cid}.rgb"
                    target_path = f"datasets/{output_name}/videos/chunk-{i_chunk:03d}/observation.camera_{cid}.rgb"
                    os.makedirs(target_path, exist_ok=True)
                    src_video = os.path.join(source_path, f"episode_{i_episode_source:06d}.mp4")
                    shutil.copyfile(src_video, 
                                    os.path.join(target_path, f"episode_{i_episode:06d}.mp4"))

                i_episode += 1
                if i_episode % chunk_size == 0:
                    i_chunk += 1

            for t in range(episode_end_ids[ep-1], episode_end_ids[ep]):
                data_dict = source_dataset.__getitem__(t, without_rgb=True)
                #print(data_dict)
                data_dict = {k: v.reshape(1, *v.shape) if hasattr(v, 'shape') else [v] for k,v in data_dict.items()}
                print([(k,v.shape if hasattr(v,'shape') else v) for k,v in data_dict.items()])
                dataset_writer.append_step(data_dict, episode_end=(t==episode_end_ids[ep]-1))
        
        source_dataset.close()

    dataset_writer.close()

if __name__=='__main__':
    # dataset_names = ['dataset_1028', 'dataset_1029', 
    #                  'dataset_1030_1', 'dataset_1030_2', 'dataset_1030_3', 'dataset_1030_4', 'dataset_1030_5',
    #                  'dataset_1103_1', 'dataset_1103_2', 
    #                  'dataset_1104_1', 'dataset_1104_2', 'dataset_1104_3', 'dataset_1104_4',
    #                  'dataset_1105_1', 'dataset_1105_2', 'dataset_1105_3', 'dataset_1105_4', 'dataset_1105_5',
    #                  'dataset_1106_1', 'dataset_1106_2', 'dataset_1106_3', 'dataset_1106_4', 'dataset_1106_5', 'dataset_1106_6',
    #                  'dataset_1107_1', 'dataset_1107_2', 'dataset_1107_3', 'dataset_1107_4', 'dataset_1107_5', 'dataset_1107_6',
    #                  'dataset_1110_1', 'dataset_1110_2', 'dataset_1111', 
    #                  'dataset_1112_1', 'dataset_1112_2', 'dataset_1114_1', 'dataset_1114_2', 'dataset_1114_3', 'dataset_1114_4',
    #                  'dataset_1117_1', 'dataset_1117_2', 'dataset_1117_3', 'dataset_1117_4',
    #                  'dataset_1118_1', 'dataset_1118_2', 'dataset_1118_3', 'dataset_1118_4', 'dataset_1118_5', 'dataset_1118_6',
    #                  'dataset_1119_1', 'dataset_1119_2', 'dataset_1119_3', 'dataset_1119_4', 'dataset_1119_5', 
    #                  'dataset_1120_1', 'dataset_1120_2', 'dataset_1120_3', 'dataset_1120_4', 'dataset_1120_5',
    #                  'dataset_1121_1', 'dataset_1121_2', 'dataset_1121_3', 'dataset_1121_4', 'dataset_1121_5',
    #                  'dataset_1124_1', 'dataset_1124_2', 'dataset_1124_3', 
    #                  ]
    # main(dataset_names, "adamu_grasp_place")

    # dataset_names = ['dataset_1112_1', 'dataset_1112_2', 'dataset_1114_1', 'dataset_1114_2',
    #                  'dataset_1114_3', 'dataset_1114_4', 'dataset_1117_1', 'dataset_1117_2',
    #                  'dataset_1117_3', 'dataset_1117_4']
    # main(dataset_names, "adamu_grasp_place_many_objects")

    # dataset_names = ["picksimple_1", "picksimple_2", "picksimple_3", "picksimple_4",
    #                  "picksimple_5", "picksimple_6", "picksimple_7", "picksimple_8",
    #                  "picksimple_9", "picksimple_10", "picksimple_11", "picksimple_12",
    #                  "picksimple_13", "picksimple_14", "picksimple_15", "picksimple_16",]
    # main(dataset_names, "adamu_pick_simple")

    # dataset_names = ["picksimple_v2/pickclutterv2_1", "picksimple_v2/pickclutterv2_2",
    #                  "picksimple_v2/pickclutterv2_3", "picksimple_v2/pickclutterv2_4",
    #                  "picksimple_v2/pickclutterv2_5", "picksimple_v2/pickclutterv2_6",
    #                  "picksimple_v2/pickclutterv2_7", "picksimple_v2/pickclutterv2_8",
    #                  "picksimple_v2/pickclutterv2_9", "picksimple_v2/pickclutterv2_10",
    #                  "picksimple_v2/pickclutterv2_11",
    #                  "picksimple_v2/picksimplev2_1", "picksimple_v2/picksimplev2_2",
    #                  "picksimple_v2/picksimplev2_3", "picksimple_v2/picksimplev2_4",
    #                  "picksimple_v2/picksimplev2_5", "picksimple_v2/picksimplev2_6",
    #                  "picksimple_v2/picksimplev2_7", "picksimple_v2/picksimplev2_8",
    #                  "picksimple_v2/picksimplev2_9", "picksimple_v2/picksimplev2_10",
    #                  "picksimple_v2/picksimplev2_11", "picksimple_v2/picksimplev2_12",
    #                  "picksimple_v2/picksimplev2_13", "picksimple_v2/picksimplev2_14",
    #                  "picksimple_v2/picksimplev2_15", "picksimple_v2/picksimplev2_16",
    #                  "picksimple_v2/picksimplev2_17"]
    # main(dataset_names, "adamu_pick_simple_v2")


    # dataset_names = ["picksimple_v2_cornercase/pickclutterv2_12", "picksimple_v2_cornercase/pickclutterv2_13",
    #                  "picksimple_v2_cornercase/pickclutterv2_14",
    #                  "picksimple_v2_cornercase/picksimplev2_18", "picksimple_v2_cornercase/picksimplev2_19", 
    #                  "picksimple_v2_cornercase/picksimplev2_20", "picksimple_v2_cornercase/picksimplev2_21",
    #                  "picksimple_v2_cornercase/picksimplev2_22", "picksimple_v2_cornercase/picksimplev2_23",
    #                  "picksimple_v2_cornercase/picksimplev2_24", "picksimple_v2_cornercase/picksimplev2_25",
    #                  "picksimple_v2_cornercase/picksimplev2_26"]
    # main(dataset_names, "debug")

    # dataset_names = ["picksimple_v3/pickclutterv3_1", "picksimple_v3/pickclutterv3_2",
    #                  "picksimple_v3/pickclutterv3_3", "picksimple_v3/pickclutterv3_4",
    #                  "picksimple_v3/pickclutterv3_5", "picksimple_v3/pickclutterv3_6",
    #                  "picksimple_v3/pickclutterv3_7", "picksimple_v3/pickclutterv3_8",
    #                  "picksimple_v3/pickclutterv3_9", "picksimple_v3/pickclutterv3_10",
    #                  "picksimple_v3/pickclutterv3_11", "picksimple_v3/pickclutterv3_12",
    #                  "picksimple_v3/pickclutterv3_13", "picksimple_v3/pickclutterv3_14",
    #                  "picksimple_v3/pickclutterv3_15",
    #                  "picksimple_v3/picksimplev3_1", "picksimple_v3/picksimplev3_2",
    #                  "picksimple_v3/picksimplev3_3", "picksimple_v3/picksimplev3_4"]
    # main(dataset_names, "adamu_pick_simple_v3")

    # dataset_names = ["adamu_pick_simple_cornercase_1k", "adamu_pick_simple_v2_2.1k", "adamu_pick_simple_v3_1.2k"]
    # main(dataset_names, "adamu_combined")

    dataset_names = ["dataset_other_tasks/wheel_7", "dataset_other_tasks/wheel_8",
                     "dataset_other_tasks/wheel_9", "dataset_other_tasks/wheel_10",
                     "dataset_other_tasks/wheel_11", "dataset_other_tasks/wheel_12",
                     "dataset_other_tasks/wheel_13", "dataset_other_tasks/wheel_14",
                     "dataset_other_tasks/wheel_15", "dataset_other_tasks/wheel_16",
                     "dataset_other_tasks/wheel_17", "dataset_other_tasks/wheel_18",
                     "dataset_other_tasks/wheel_19", "dataset_other_tasks/wheel_20",
                     "dataset_other_tasks/wheel_21", "dataset_other_tasks/wheel_22",
                     "dataset_other_tasks/wheel_23", "dataset_other_tasks/wheel_24",]
    main(dataset_names, "adamu_wheel")