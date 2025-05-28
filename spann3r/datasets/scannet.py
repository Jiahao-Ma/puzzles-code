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


class Scannet(BaseManyViewDataset):
    def __init__(self, num_seq=100, num_frames=5, 
                 min_thresh=10, max_thresh=100, 
                 test_id=None, 
                 full_video=False, 
                 kf_every=1, load_lmdb=True, train_ratio=None, json_file=None, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.num_frames = num_frames
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self.train_ratio = train_ratio
        
        self.load_lmdb = load_lmdb
        if self.load_lmdb:
            if json_file is None:
                json_file = osp.join(self.ROOT, 'ScanNet.json')
            else:
                json_file = osp.join(self.ROOT, json_file)
            with open(json_file, 'r') as f:
                self.scannet_json = json.load(f)
            lmdb_path = osp.join(self.ROOT, 'ScanNet.mdb')   
            map_size = int(os.path.getsize(lmdb_path) * 1.5)
            self.data_mdb = lmdb.open(lmdb_path,
                                            readonly=True,
                                            subdir=False,
                                            lock=False,
                                            readahead=False,
                                            map_size=map_size)
            self.data_prefix = 'data/scans'
        else:
            self.scannet_json = None
            self.data_mdb = None

         # load all scenes
        self.load_all_scenes(ROOT)

        self.short_scene_list_ratio = 0.1

    def __len__(self):
        if self.split == 'train':
            return int(len(self.scene_list) * self.num_seq)
        else:
            if self.train_ratio is None or self.train_ratio < 0.9:
                return int(len(self.scene_list) * self.num_seq * self.short_scene_list_ratio)
            else:
                return int(len(self.scene_list) * self.num_seq)

    def load_all_scenes(self, base_dir):
        
        self.folder = {'train': 'scans', 'val': 'scans', 'test': 'scans_test'}[self.split]
        
        if self.test_id is None:
            meta_split = osp.join(base_dir, 'splits', f'scannetv2_{self.split}.txt')
            
            if not osp.exists(meta_split):
                raise FileNotFoundError(f"Split file {meta_split} not found")
            
            with open(meta_split) as f:
                self.scene_list = f.read().splitlines()

            valid_scene_id = list(self.scannet_json.keys())
            self.scene_list = [scene_id for scene_id in self.scene_list if scene_id in valid_scene_id]
                
            print(f"Found {len(self.scene_list)} scenes in split {self.split}")
            
        else:
            if isinstance(self.test_id, list):
                self.scene_list = self.test_id
            else:
                self.scene_list = [self.test_id]
                
            print(f"Test_id: {self.test_id}")
    
    def _get_views(self, idx, resolution, rng, attempts=0): 
        scene_id = self.scene_list[idx // self.num_seq]

        if not self.load_lmdb:
            # Load metadata
            intri_path = osp.join(self.ROOT, self.folder, scene_id, 'intrinsic/intrinsic_depth.txt')
            intri = np.loadtxt(intri_path).astype(np.float32)[:3, :3]

            # Load image data
            data_path = osp.join(self.ROOT, self.folder, scene_id, 'sensor_data')
            num_files = len([name for name in os.listdir(data_path) if 'color' in name])  
        else:
            intri_path = osp.join(self.data_prefix, scene_id, 'intrinsic/intrinsic_depth.txt')
            try:                
                with self.data_mdb.begin(write=False) as txn:
                    intri = np.loadtxt(io.BytesIO(txn.get(intri_path.encode("utf-8"))), dtype=np.float32)[:3, :3]
                    
            except:
                print(f"Error loading intrinsic file {intri_path}, reload")
                new_idx = rng.integers(0, self.__len__()-1)
                return self._get_views(new_idx, resolution, rng)
                
            
            # depth_path = osp.join(self.data_prefix, scene_id, 'sensor_data')
            num_files = len([name for name in self.scannet_json[scene_id]['sensor_data'] if 'color' in name])

        img_idxs_ = [f'{i:06d}' for i in range(num_files)]
        imgs_idxs = self.sample_frame_idx(img_idxs_, rng, full_video=self.full_video)
        imgs_idxs = deque(imgs_idxs)

        views = []

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()
            if not self.load_lmdb:
                # Load image data
                impath = osp.join(self.ROOT, self.folder, scene_id, 'sensor_data', f'frame-{im_idx}.color.jpg')
                depthpath = osp.join(self.ROOT, self.folder, scene_id, 'sensor_data', f'frame-{im_idx}.depth.png')
                posepath = osp.join(self.ROOT, self.folder, scene_id, 'sensor_data', f'frame-{im_idx}.pose.txt')

                rgb_image = imread_cv2(impath)
                depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
                rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

                depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0            
                camera_pose = np.loadtxt(posepath).astype(np.float32)
            else:
                impath = osp.join(self.data_prefix, scene_id, 'sensor_data', f'frame-{im_idx}.color.jpg')
                depthpath = osp.join(self.data_prefix, scene_id, 'sensor_data', f'frame-{im_idx}.depth.png')
                posepath = osp.join(self.data_prefix, scene_id, 'sensor_data', f'frame-{im_idx}.pose.txt')
                with self.data_mdb.begin(write=False) as txn:
                    try:
                        camera_pose = np.loadtxt(io.BytesIO(txn.get(posepath.encode("utf-8"))), dtype=np.float32).reshape(4, 4)
                        rgb_image = np.array(Image.open(io.BytesIO(txn.get(impath.encode('utf-8')))).convert('RGB'))
                        depthmap = np.array(Image.open(io.BytesIO(txn.get(depthpath.encode('utf-8')))))
                    except Exception as e:
                        print(f"[Scannet] Error loading {impath} or {depthpath} or {posepath}: {e}")
                        rgb_image = None
                        depthmap = None
                        camera_pose = None
                        
                    if rgb_image is None or depthmap is None or camera_pose is None:
                        with open('/scratch3/ma040/dataset/LMDB/incomplete.txt', 'a') as f:
                            f.write(f"scannet {scene_id} {impath} {depthpath} {posepath}\n")
                        new_idx = rng.integers(0, self.__len__()-1)
                        return self._get_views(new_idx, resolution, rng)
                    
                    rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))
                    depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0

            if rgb_image.shape[:2] != depthmap.shape[:2]: 
                print(f"[Scannet] RGB and depthmap shapes do not match: {rgb_image.shape} vs {depthmap.shape}")
                new_idx = rng.integers(0, self.__len__()-1)
                return self._get_views(new_idx, resolution, rng)
            rz_rgb_image, rz_depthmap, rz_intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intri, resolution, rng=rng, info=impath)
            
            # Check if the image is valid
            num_valid = (rz_depthmap > 0.0).sum()
            if num_valid == 0 or (not np.isfinite(camera_pose).all()):
                if self.full_video:
                    print(f"Warning: No valid depthmap found for {impath}")
                    continue
                else:
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
                orig_camera_intrinsics=intri,
                dataset='scannet',
                label=osp.join(scene_id, im_idx),
                instance=osp.split(impath)[1],
                impath=impath,
            ))
        
        return views

if __name__ == "__main__":

    num_frames=5
    print('loading dataset')

    dataset = Scannet(split='train', ROOT="./data/scannet_simple", resolution=224, num_seq=100, max_thresh=100)







        





    

