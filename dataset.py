import shutil
import numpy as np
from tqdm import tqdm
import os, sys
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, compute_stats, serialize_dict, write_json, STATS_PATH

HF_LEROBOT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), 'datasets'))


class LerobotDatasetWriter:
    def __init__(self, 
                 output_path: str, 
                 camera_ids: list = [1],
                 data_type: str = "rgb",
                 action_dim = 31,
                 state_dim = 31, 
                 image_shape = (256,256,3), 
                 #depth_shape = (256,256),
                 fps = 20,
                 #depth_dmin_m: float = 0.15,  # 全局固定量程（很重要，别 per-frame minmax）
                 #depth_dmax_m: float = 1.00,
                ):
        repo_id = output_path
        output_path = os.path.join(HF_LEROBOT_HOME, output_path)
        print(output_path)
        if os.path.exists(output_path):
            print(f"警告：输出路径 {output_path} 已存在，将被删除。")
            shutil.rmtree(output_path)
        
        self.camera_ids = camera_ids
        self.data_type = data_type
        self.image_shape = image_shape
        #self.depth_shape = depth_shape
        self.fps = fps
        #self.depth_dmin_m = depth_dmin_m
        #self.depth_dmax_m = depth_dmax_m

        features = {
            # 关键帧名称要符合 LeRobot 约定或自定义
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": [f"qpos_{i}" for i in range(state_dim)],
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": [f"action_{i}" for i in range(action_dim)],
            },
        }

        for cid in camera_ids:
            if "rgb" in data_type:
                features[f"observation.camera_{cid}.rgb"] = {
                    "dtype": "video",
                    "shape": image_shape,
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": fps,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False
                    }
                }
            # if "depth" in data_type:
            #     features[f"observation.camera_{cid}.depth"] = {
            #         "dtype": "video",
            #         "shape": depth_shape,
            #         "names": ["height", "width", "channel"],
            #         "video_info": {
            #             "video.fps": fps,
            #             "video.codec": "h264",
            #             "video.pix_fmt": "yuv420p",
            #             "video.is_depth_map": False,
            #             "has_audio": False
            #         }
            #     }
            # if "pcl" in data_type:
            #     raise NotImplementedError

        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=output_path,
            robot_type="MyDexHand",  # 自定义机器人类型
            fps=fps,
            features=features,
            # # 可以根据机器性能调整线程和进程数以加速视频写入
            # image_writer_threads=10,
            # image_writer_processes=5,
        )
        self.push_to_hub = False
    
    def append_step(self, data: Dict[str, np.ndarray], episode_end: bool = False):
        # 拼接 state
        state = np.concatenate([data['body_qpos'], data['hand_qpos']], axis=-1).reshape(-1)
        action = data['action'].reshape(-1)
        frame_data = {
            'observation.state': state.astype(np.float32),
            'action': action.astype(np.float32),
        }
        for cid in self.camera_ids:
            if "rgb" in self.data_type:
                frame_data[f'observation.camera_{cid}.rgb'] = data[f'camera_{cid}.rgb'].reshape(*self.image_shape).astype(np.uint8)
            # if "depth" in self.data_type:
            #     depth_01 = np.clip(
            #         (data[f'camera_{cid}.depth'] - self.depth_dmin_m) / (self.depth_dmax_m - self.depth_dmin_m),
            #         0,
            #         1
            #     )
            #     depth_uint8 = (depth_01 * 255).round().astype(np.uint8)
            #     frame_data[f'observation.camera_{cid}.depth'] = depth_uint8.reshape(*self.depth_shape)
            # if "pcl" in self.data_type:
            #     raise NotImplementedError

        if 'instruction' in data:
            if isinstance(data['instruction'], str):
                self.text_des = data['instruction']
            elif isinstance(data['instruction'], list):
                self.text_des = data['instruction'][0]
                assert isinstance(self.text_des, str)
            else:
                raise ValueError(f"Unsupported instruction type: {type(data['instruction'])}")
        else:
            self.text_des = "Do the task."
        
        # 使用 add_frame 添加一帧数据到缓冲区
        #print(frame_data)
        self.dataset.add_frame(frame_data)

        if episode_end:
            self.dataset.save_episode(task=self.text_des)
    
    def close(self):
        # (可选) 推送到 Hugging Face Hub
        if self.push_to_hub:
            print("正在将数据集推送到 Hugging Face Hub...")
            self.dataset.push_to_hub(
                tags=["dexhand", "manipulation"], # 添加合适的标签
                private=False, # 或 True
                push_videos=True,
                license="apache-2.0", # 或其他许可证
            )
            print("推送完成！")


class LerobotDatasetReader:
    def __init__(self, repo_id: str):
        pth = os.path.join(HF_LEROBOT_HOME, repo_id)
        self.dataset = LeRobotDataset(repo_id, root=pth, local_files_only=True)
        self.episode_ends = [x.item()-1 for x in self.dataset.episode_data_index["to"]]
        # print(self.episode_ends)
        # exit(0)

    def compute_stats(self):
        self.dataset.stop_image_writer()
        self.dataset.meta.stats = compute_stats(self.dataset)
        serialized_stats = serialize_dict(self.dataset.meta.stats)
        write_json(serialized_stats, self.dataset.root / STATS_PATH)
    
    def _get_total_steps(self):
        return len(self.dataset)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, t, without_rgb=False):
        if not without_rgb:
            data = self.dataset[t]
        else:
            item = self.dataset.hf_dataset[t]
            ep_idx = item["episode_index"].item()
            query_indices = None
            if self.dataset.delta_indices is not None:
                current_ep_idx = self.dataset.episodes.index(ep_idx) if self.dataset.episodes is not None else ep_idx
                query_indices, padding = self.dataset._get_query_indices(t, current_ep_idx)
                query_result = self.dataset._query_hf_dataset(query_indices)
                item = {**item, **padding}
                for key, val in query_result.items():
                    item[key] = val
            # Add task as a string
            task_idx = item["task_index"].item()
            item["task"] = self.dataset.meta.tasks[task_idx]
            data = item

        ret = {}
        for k in data.keys():
            if "rgb" in k:
                ret[k[12:]] = (np.asarray(data[k]).transpose(1,2,0) * 255).astype(np.uint8) 
        ret["instruction"] = data["task"]
        ret["body_qpos"] = data["observation.state"][:19].cpu().numpy()
        ret["hand_qpos"] = data["observation.state"][19:31].cpu().numpy()
        ret["action"] = data["action"].cpu().numpy()
        return ret
    

    def close(self):
        pass



if __name__ == "__main__":
    # writer = LerobotDatasetWriter(output_path="example", camera_ids=[1], data_type="rgb")
    # for ep in range(5):
    #     for t in range(40):
    #         sample_data = {
    #             'right_arm_eef_pose': np.random.rand(1, 7).astype(np.float32),
    #             'right_hand_qpos': np.random.rand(1, 6).astype(np.float32),
    #             'action': np.random.rand(1, 13).astype(np.float32),
    #             'camera_1.rgb': (np.random.rand(1, 256, 256, 3)*255).astype(np.uint8),
    #             'instruction': f"Episode {ep} instruction."
    #         }
    #         writer.append_step(sample_data, episode_end=(t == 39))
    # writer.close()
    
    import matplotlib.pyplot as plt
    import cv2
    reader = LerobotDatasetReader(repo_id="adamu_combined")
    print(f"Total steps: {len(reader)}")
    print(reader.episode_ends)
    reader.compute_stats()
    exit(0)

    fig, ax = plt.subplots()
    #frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    #plt_img = ax.imshow(frame)
    
    for t in range(0, len(reader)):
        data = reader[t]
        print([(k, v.shape if hasattr(v, "shape") else type(v)) for k, v in data.items()])
        img = data['camera_1.rgb']
        print(img.shape, img.dtype, data['instruction'])
        #print(data['action'])
        #plt_img.set_data(img)
        #plt.pause(0.001)
        
        # cv2.imshow("img", img[:,:,::-1])
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        ax.clear()
        ax.imshow(img)
        plt.draw()
        plt.pause(0.01)
