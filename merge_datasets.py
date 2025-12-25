import os, sys
from dataset import LerobotDatasetWriter, LerobotDatasetReader

def main(dataset_names, output_name, camera_ids=[0,1], data_type=["rgb"], num_actions=31, image_shape=(256, 256, 3)):           
    dataset_writer = LerobotDatasetWriter(
        output_path=output_name,
        camera_ids=camera_ids,
        data_type=data_type,
        action_dim=num_actions,
        state_dim=num_actions,
        image_shape=image_shape,
    )

    for name in dataset_names:
        source_dataset = LerobotDatasetReader(name)
        episode_end_ids = source_dataset.episode_ends
        episode_end_ids = [0] + [i+1 for i in episode_end_ids]
        print(episode_end_ids)

        for ep in range(1, len(episode_end_ids)):
            for t in range(episode_end_ids[ep-1], episode_end_ids[ep]):
                data_dict = source_dataset[t]
                #print(data_dict)
                data_dict = {k: v.reshape(1, *v.shape) if hasattr(v, 'shape') else [v] for k,v in data_dict.items()}
                print([(k,v.shape if hasattr(v,'shape') else v) for k,v in data_dict.items()])
                dataset_writer.append_step(data_dict, episode_end=(t==episode_end_ids[ep]-1))
        
        source_dataset.close()

    dataset_writer.close()

if __name__=='__main__':
    dataset_names = ['datasets_1028/test_1', 'datasets_1028/test_2', 'datasets_1028/test_3', 'datasets_1028/test_4', 'datasets_1028/test_5']
    main(dataset_names, "adamu_grasp_place")
