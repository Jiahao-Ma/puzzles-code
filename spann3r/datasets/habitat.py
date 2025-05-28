import os
import io
import cv2
import json
import lmdb
import numpy as np
import os.path as osp
from PIL import Image
from collections import deque
from dust3r.utils.image import imread_cv2
from .base_many_view_dataset import BaseManyViewDataset


class habitat(BaseManyViewDataset):
    def __init__(self, num_seq=200, num_frames=5, load_lmdb=True, train_ratio=None, json_file=None, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.num_frames = num_frames
        self.load_lmdb = load_lmdb
        self.train_ratio = train_ratio
        if self.load_lmdb:
            if json_file is None:
                json_file = osp.join(self.ROOT, 'habitat.json')
            else:
                json_file = osp.join(self.ROOT, json_file)
            with open(json_file, 'r') as f: 
                self.habitat_json = json.load(f)
            lmdb_path = osp.join(self.ROOT, 'habitat.mdb')   
            map_size = int(os.path.getsize(lmdb_path) * 1.5)
            self.data_mdb = lmdb.open(lmdb_path,
                                            readonly=True,
                                            subdir=False,
                                            lock=False,
                                            readahead=False,
                                            map_size=map_size)
            self.data_prefix = 'habitat'
        # load all scenes
        self.load_all_scenes(ROOT, num_seq)
    
    def __len__(self):
        return len(self.scene_list) * self.num_seq
    
    def load_all_scenes(self, base_dir, num_seq=200):
        if not self.load_lmdb:
            self.scenes = {}
            
            data_all = os.listdir(base_dir)
            print('All datasets in Habitat:', data_all)
            
            for data in data_all:
                scenes = os.listdir(osp.join(base_dir, data))
                self.scenes[data] = scenes
            
            self.scenes = {(k, v2): list(range(num_seq)) for k, v in self.scenes.items() 
                            for v2 in v}
            self.scene_list = list(self.scenes.keys())
        else:
            self.scenes = {}
            self.scene_list = []
            data_all = [scene for scene in self.habitat_json.keys() if scene != 'total_num']
            print('All datasets in Habitat:', data_all)
            self.scenes['gibson'] = data_all[:-3]
            self.scenes['habitat_test'] = data_all[-3:]
            
            self.scenes = {(k, v2): list(range(num_seq)) for k, v in self.scenes.items()
                            for v2 in v}
            self.scene_list = list(self.scenes.keys())
    
    def _get_views(self, idx, resolution, rng, attempts=0): 
        data, scene = self.scene_list[idx // self.num_seq]
        seq_id = idx % self.num_seq
        

        imgs_idxs_ = list(range(1, self.num_frames+1))
        rng.shuffle(imgs_idxs_)
        imgs_idxs = deque(imgs_idxs_)

        views = []

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()
            if not self.load_lmdb:
                impath = osp.join(self.ROOT, data, scene, f"{seq_id:08}_{im_idx}.jpeg")
                depthpath = osp.join(self.ROOT, data, scene, f"{seq_id:08}_{im_idx}_depth.exr")
                cam_params_path = osp.join(self.ROOT, data, scene, f"{seq_id:08}_{im_idx}_camera_params.json")
                if not osp.exists(impath):
                    new_idx = rng.integers(0, self.__len__()-1)
                    return self._get_views(new_idx, resolution, rng)
                rgb_image = imread_cv2(impath)
                depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)

                cam_params = json.load(open(cam_params_path, 'r'))
            else:
                impath = osp.join('habitat', scene, f"{seq_id:08}_{im_idx}.jpeg")
                depthpath = osp.join('habitat', scene, f"{seq_id:08}_{im_idx}_depth.exr")
                cam_params_path = osp.join('habitat', scene, f"{seq_id:08}_{im_idx}_camera_params.json")
                with self.data_mdb.begin(write=False) as txn:
                    try:
                        rgb_image = np.array(Image.open(io.BytesIO(txn.get(impath.encode('utf-8')))).convert('RGB'))
                        depthmap = np.array(cv2.imdecode(np.frombuffer(txn.get(depthpath.encode('utf-8')), np.uint8), cv2.IMREAD_UNCHANGED))
                        cam_params = json.load(io.StringIO(txn.get(cam_params_path.encode('utf-8')).decode('utf-8')))
                    except Exception as e:
                        print(f"[Habitat] Error loading {impath} or {depthpath} or {cam_params_path}: {e}")
                        rgb_image = None
                        depthmap = None
                        cam_params = None
                    if rgb_image is None or depthmap is None or cam_params is None:
                        new_idx = rng.integers(0, self.__len__()-1)
                        return self._get_views(new_idx, resolution, rng)

            intrinsics_ = np.array(cam_params['camera_intrinsics'], dtype=np.float32)
            # cam_r: [3, 3], cam_t: [3, ]
            cam_r = np.array(cam_params['R_cam2world'], dtype=np.float32)
            cam_t = np.array(cam_params['t_cam2world'], dtype=np.float32)
            
            # camera_pose: [4, 4]
            camera_pose = np.eye(4).astype(np.float32)
            camera_pose[:3, :3] = cam_r
            camera_pose[:3, 3] = cam_t
            
            if rgb_image.shape[:2] != depthmap.shape[:2]: 
                print(f"[Habitat] RGB and depthmap shapes do not match: {rgb_image.shape} vs {depthmap.shape}")
                new_idx = rng.integers(0, self.__len__()-1)
                return self._get_views(new_idx, resolution, rng)

            rz_rgb_image, rz_depthmap, rz_intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics_, resolution, rng=rng, info=impath)
            
            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0 or (not np.isfinite(camera_pose).all()):
                if attempts >= 5:
                    new_idx = rng.integers(0, self.__len__()-1)
                    return self._get_views(new_idx, resolution, rng)
                return self._get_views(idx, resolution, rng, attempts+1)
            
            views.append(dict(
                img=rz_rgb_image,
                orig_img=rgb_image,
                depthmap=rz_depthmap,
                orig_depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=rz_intrinsics,
                orig_camera_intrinsics=intrinsics_,
                dataset='habitat',
                label=osp.join(data, scene),
                instance=osp.split(impath)[1],
                impath=impath,
            ))
        return views

