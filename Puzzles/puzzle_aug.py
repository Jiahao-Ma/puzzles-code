import cv2
import PIL
import copy
import numpy as np
from PIL import Image


from Puzzles.puzzles_utils import estimate_camera_pose_ransac, image_plane_uv, random_rotate, generate_overlapping_boxes
from Puzzles.puzzles_utils2d import RandomAffinePointCloud
from Puzzles.visualization import rasterization

from dust3r.utils.image import ImgInvNorm
from dust3r.utils.geometry import depthmap_to_camera_coordinates
from dust3r.datasets.utils.cropping import rescale_image_depthmap



def _default_box(width, height):    
    min_edge = min(width, height)
    left = (width - min_edge) // 2
    top = (height - min_edge) // 2
    right = left + min_edge
    bottom = top + min_edge
    return (left, top, right, bottom)


def random_partition_indices(N, K, rng):
    ids = np.ones(N, dtype=np.int32)
    if K > 1:
        cuts = sorted(rng.choice(np.arange(1, N), size=K-1, replace=False))
        groups = np.split(ids, cuts)
    else:
        groups = [ids]
    return groups

def depth2points(K, depth):
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    z = depth 
    x = (i - K[0, 2]) * z / K[0, 0]
    y = (j - K[1, 2]) * z / K[1, 1]
    points3d = np.stack((x, y, z), axis=-1)
    return points3d

def calibrate_puzzles(depthmap, camera_pose, camera_intrinsics, image, boxes, template_box, bk_view, rng, calib_num_pts=2048):
    pts3d_cam, _ = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)
    R_cam2world = camera_pose[:3, :3]
    t_cam2world = camera_pose[:3, 3]
    
    pts3d_world = np.einsum("ik, vuk -> vui", R_cam2world, pts3d_cam) + t_cam2world[None, None, :]

    '''
        Reference: 
            [1] https://github.com/pablospe/render_depthmap_example/issues/4
            [2] https://github.com/isl-org/Open3D/issues/3079
        Open3D headless rendering only support that the principal point is at the center of the image,
        where cx = width / 2.0 - 0.5 and cy = height / 2.0 - 0.5. For implementation simplicity, we use 
        the original focal length, which is slightly different from the formula in the paper. But basically, it is the same.
    '''
    templ_box_h, templ_box_w = template_box[3] - template_box[1], template_box[2] - template_box[0]
    boxK = camera_intrinsics.copy()
    boxK[0, 2] = templ_box_w / 2.0 - 0.5
    boxK[1, 2] = templ_box_h / 2.0 - 0.5

    batch_images, batch_points, batch_depths, batch_intrinsics, batch_camera_poses, batch_h, batch_w, batch_mask, batch_boxes = [], [], [], [], [], [], [], [], []
    for bbox in boxes:
        box_h, box_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        box_points = pts3d_world[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        box_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]] 
        box_depth = depthmap[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if box_h != templ_box_h or box_w != templ_box_w:
            box_image = cv2.resize(box_image, (templ_box_w, templ_box_h), interpolation=cv2.INTER_LINEAR)
            box_points = cv2.resize(box_points, (templ_box_w, templ_box_h), interpolation=cv2.INTER_NEAREST)
            box_depth = cv2.resize(box_depth, (templ_box_w, templ_box_h), interpolation=cv2.INTER_NEAREST)  
        box_mask = box_depth > 0
        # calibrate poses
        box_uv = image_plane_uv(templ_box_w, templ_box_h, normalize=False)
        box_uv_ds = box_uv.reshape(-1, 2)
        mask_ds = box_mask.reshape(-1)
        box_points_ds = box_points.reshape(-1, 3) 
        if np.sum(mask_ds) > calib_num_pts:
            idx = rng.choice(np.arange(mask_ds.shape[0])[mask_ds], calib_num_pts, replace=False)
            box_points_ds_sm = box_points_ds[idx]
            box_uv_ds_sm = box_uv_ds[idx]
            mask_sm = mask_ds[idx]
        calibrate_success = False
        try:
            w2c, _, _ = estimate_camera_pose_ransac(box_points_ds_sm[mask_sm], box_uv_ds_sm[mask_sm], boxK)
            calibrate_success = True
        except:
            calibrate_success = False

        if calibrate_success:
            batch_images.append(box_image)
            batch_points.append(box_points)
            batch_depths.append(box_depth)
            batch_intrinsics.append(boxK)
            batch_camera_poses.append(np.linalg.inv(w2c))
            batch_h.append(templ_box_h)
            batch_w.append(templ_box_w)
            batch_mask.append(box_mask)
            batch_boxes.append(bbox)
        else:
            batch_images.append(ImgInvNorm(bk_view['img'])*255)
            batch_points.append(bk_view['pts3d'])
            batch_depths.append(bk_view['depthmap'])
            batch_intrinsics.append(bk_view['camera_intrinsics'])
            batch_camera_poses.append(bk_view['camera_pose'])
            batch_h.append(bk_view['true_shape'][0])
            batch_w.append(bk_view['true_shape'][1])
            batch_mask.append(bk_view['valid_mask'])
            h,w=bk_view['orig_img'].shape[:2]
            batch_boxes.append(_default_box(w, h))

    return batch_images, batch_points, batch_depths, batch_intrinsics, batch_camera_poses, batch_h, batch_w, batch_mask, batch_boxes


class PuzzleAugment(object):
    def __init__(self, num_views, resolution, overlap_depth_threshold, puzzle_min_overlap, puzzle_max_overlap, transform3d = False, transform2d = True, img2clip=False, debug=False):
        self.num_views = num_views
        self.resolution = resolution if isinstance(resolution, (tuple, list)) else (resolution, resolution)
        self.img2clip = img2clip # clip2clip

        # Augmentation parameters
        self.min_rotation_angle = 30
        self.max_rotation_angle = 90

        self.min_coverage = 0.3
        self.max_coverage = 0.9
        self.jitter_std = 5.0
        self.bg_color = (0, 0, 0)
        self.rotate3D_r = 0.3
        self.bbox_min_overlap = 0.1
        self.bbox_max_overlap = 0.4

        self.minVisibility = 0.1 # the minimum visibility in the sequence (for visibility check)
        self.debug = debug

        self.overlap_depth_threshold = overlap_depth_threshold
        self.puzzle_min_overlap = puzzle_min_overlap
        self.puzzle_max_overlap = puzzle_max_overlap

        self.transform3d = transform3d
        if transform2d:
            self.transform2d = [
                    ("RandomAffine", lambda: RandomAffinePointCloud(degrees=(-45, 45), translate=(0, 0.2), scale=(0.8, 1.))),
                ]
        else:
            self.transform2d = False

    def random_param_gen(self, height, width, Nbox):
        min_box_size = int(min(np.sqrt((height * width) / (Nbox + 1)), min(height, width)))
        max_box_size = int(min(np.sqrt((height * width) / max(Nbox - 1, 1)), min(height, width)))

        return min_box_size, max_box_size
    
    def rescale(self, resolution, batch_images, batch_points, batch_depths, batch_masks, batch_intrinsics, batch_h, batch_w):
        for i in range(len(batch_images)):
            if not isinstance(batch_images[i], PIL.Image.Image):
                image = PIL.Image.fromarray(batch_images[i])
            else:
                image = batch_images[i]
            image_pil, batch_depths[i], batch_intrinsics[i] = rescale_image_depthmap(
                image, batch_depths[i], batch_intrinsics[i], resolution)
            batch_images[i] = np.array(image_pil)
            batch_points[i] = cv2.resize(batch_points[i], resolution, interpolation=cv2.INTER_NEAREST)
            batch_masks[i] = cv2.resize(batch_masks[i].astype(np.float32), resolution, interpolation=cv2.INTER_NEAREST).astype(np.bool_)
            batch_h[i], batch_w[i] = resolution
        return batch_images, batch_points, batch_depths, batch_masks, batch_intrinsics, batch_h, batch_w
    
    def RandomRotate(self, rng):
        return rng.random() < 0.3
    
    def _calculate_overlap(self, view1, view2, depth_threshold=1):
        pts3d_v1 = view1['pts3d'] # (H, W, 3)
        valid_mask_v1 = view1['valid_mask'] # (H, W)
        if np.sum(valid_mask_v1) == 0:
            return 0, None
        depthmap_v2 = view2['depthmap'] # (H, W)
        c2w_v2 = view2['camera_pose'] # (4, 4)
        K_v2 = view2['camera_intrinsics'] # (3, 3)
        h, w = view2['true_shape'] # (H, W)
        w2c_v2 = np.linalg.inv(c2w_v2)
        pts3d_v1 = np.concatenate([pts3d_v1, np.ones_like(pts3d_v1[..., :1])], axis=-1) # (H, W, 4)
        pts3d_v12 = (pts3d_v1 @ w2c_v2.T)[..., :3] # (H, W, 3)
        valid_z = pts3d_v12[..., 2] > 0

        pts2d = pts3d_v12 @ K_v2.T # (H, W, 3)
        pts2d = pts2d[..., :2] / np.maximum(pts2d[..., 2:], 1e-6) # (H, W, 2)
        x, y = pts2d[..., 0], pts2d[..., 1]
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h) & valid_z
        
        if depth_threshold is not None:
            y_v1, x_v1 = np.where(valid)

            x_v2 = np.clip(np.round(x[valid]).astype(np.int32), 0, w - 1)
            y_v2 = np.clip(np.round(y[valid]).astype(np.int32), 0, h - 1)

            depth_diff = np.abs(depthmap_v2[y_v2, x_v2] - pts3d_v12[valid, 2])
            valid = (depth_diff < depth_threshold)
            overlap = np.sum(valid) / (np.sum(valid_mask_v1) if np.sum(valid_mask_v1) > 0 else 1)
            overlap_pixl_v2 = (x_v2[valid], y_v2[valid])
            overlap_pixl_v1 = (x_v1[valid], y_v1[valid])
        else:
            y_v1, x_v1 = np.where(valid)

            overlap = np.sum(valid[valid_mask_v1]) / (np.sum(valid_mask_v1) if np.sum(valid_mask_v1) > 0 else 1)
            x_v2 = np.clip(np.round(x[valid]).astype(np.int32), 0, w - 1)
            y_v2 = np.clip(np.round(y[valid]).astype(np.int32), 0, h - 1)

            overlap_pixl_v2 = (x_v2, y_v2)
            overlap_pixl_v1 = (x_v1, y_v1)
        return overlap, overlap_pixl_v1, overlap_pixl_v2


    def overlapMatrix(self, views, depth_threshold=None):
        concates = dict() # for debug
        Nviews = len(views)
        overlap_matrix = np.zeros((Nviews, Nviews), dtype=np.float32)
        for i in range(Nviews):
            for j in range(i + 1, Nviews):
                overlap_matrix[i, j], overlap_pixl_v1, overlap_pixl_v2 = self._calculate_overlap(views[i], views[j], depth_threshold=depth_threshold)
                concates['{}_{}'.format(i, j)] = [overlap_matrix[i, j], overlap_pixl_v1, overlap_pixl_v2]

                overlap_matrix[j, i], overlap_pixl_v1, overlap_pixl_v2 = self._calculate_overlap(views[j], views[i], depth_threshold=depth_threshold)
                concates['{}_{}'.format(j, i)] = [overlap_matrix[j, i], overlap_pixl_v1, overlap_pixl_v2]
        return overlap_matrix, concates

    def sequence_visibility_check(self, views, depth_threshold=None):
        """
        Check if the sequence of views is visible in the overlap matrix.
        """
        Nviews = len(views)
        visibility = np.zeros((Nviews,), dtype=np.bool_)
        for i in range(0, Nviews):
            for j in range(0, i):
                if i == 0:
                    visibility[i] = True
                else:
                    overlap, _, _ = self._calculate_overlap(views[i], views[j], depth_threshold=depth_threshold)
                    if overlap > self.minVisibility:
                        visibility[i] = True
                        break
        return visibility


    def _debug_overlap(self, views, concates):
        purple = np.array([255, 0, 255], dtype=np.uint8)
        alpha_v1 = 0.3
        alpha_v2 = 0.3

        Nviews = len(views)
        canvas = None
        for i in range(Nviews): 
            for j in range(Nviews):
                if i == j:
                    continue
                results = concates['{}_{}'.format(i, j)]
                overlap = results[0]
                overlap_pixl_v1 = results[1]
                overlap_pixl_v2 = results[2]

                img_v1 = (ImgInvNorm(views[i]['img'])*255).astype(np.uint8)
                img_v2 = (ImgInvNorm(views[j]['img'])*255).astype(np.uint8)
                # highlight the overlapping pixels on img_v2
                H, W, _ = img_v2.shape
                rgba_v1 = np.zeros((H, W, 4), dtype=np.uint8)
                rgba_v1[..., :3] = img_v1
                rgba_v1[..., 3] = 255
                ys, xs = overlap_pixl_v1[1], overlap_pixl_v1[0]
                original_colors = img_v1[ys, xs].astype(np.float32)
                blended_colors = (1 - alpha_v1) * original_colors + alpha_v1 * purple
                rgba_v1[ys, xs, :3] = blended_colors.astype(np.uint8)

                rgba_v2 = np.zeros((H, W, 4), dtype=np.uint8)
                rgba_v2[..., :3] = img_v2
                rgba_v2[..., 3] = 255
                ys, xs = overlap_pixl_v2[1], overlap_pixl_v2[0]
                original_colors = img_v2[ys, xs].astype(np.float32)
                blended_colors = (1 - alpha_v2) * original_colors + alpha_v2 * purple
                rgba_v2[ys, xs, :3] = blended_colors.astype(np.uint8)

                concat = np.concatenate([rgba_v1, rgba_v2], axis=1)
                nH, nW = concat.shape[:2]
                # cv2.putText(concat, f'overlap: {overlap:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
                concat = cv2.cvtColor(concat, cv2.COLOR_RGBA2BGRA)
                if canvas is None:
                    canvas = np.zeros((nH*Nviews, nW * Nviews, 4), dtype=np.uint8)
                canvas[i*nH:(i+1)*nH, j*nW:(j+1)*nW] = concat
                
                cv2.putText(canvas, f'{overlap*100:.2f}%', (10 + j*nW, 30 + i*nH), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255, 255), thickness=2)
                

        cv2.imwrite('vis/overlap.png', canvas)
        return canvas
    

    def longest_true_sequence_indices(self, O: np.ndarray, T: float, redundantT: float, minT:float=0.1, rng=None):
        n = O.shape[0]

        # step1: find the longest sequence of indices
        valid = [i for i in range(n) if np.max(O[i, np.arange(n) != i]) > minT]
        if len(valid) <= 1:
            return list(range(n))
        cover_sets = {
                        i: {j for j in valid if O[i, j] >= T or i == j}
                        for i in valid
                    }
        seed_idx = valid[int(np.argmax([len(s) for s in cover_sets.values()]))]
        chain = list(cover_sets[seed_idx])

        # step2: remove redundant indices
        pruned = []
        for idx in chain:                               
            if any(
                O[idx, kept] >= redundantT
                and O[kept, idx] >= redundantT          
                for kept in pruned
            ):
                continue                                
            pruned.append(idx)

        if len(pruned) <= 1:
            O_sum = np.sum(O, axis=1)
            random_num = rng.integers(2, n-1)
            pruned = np.argsort(O_sum)[::-1][:random_num].tolist()
            return pruned
        
        # step3: find the longest sequence of indices again
        second_cover_sets = {
            i: {j for j in pruned if O[i, j] >= T or i == j}
            for i in pruned
        }
        second_seed = pruned[
            int(np.argmax([len(s) for s in second_cover_sets.values()]))
        ]
        second_chain = list(second_cover_sets[second_seed])
        
        # step4: reorder the chain
        final_chain = self.reorder_with_overlap(O, second_chain)
        return final_chain
        
        
    def select_reps(self, Overlap: np.ndarray, frame_idxs: list[int], rng) -> tuple[list[int], list[int]]:
        """
        Greedy selection of representative images based on the overlap matrix.
        Parameters:
            Overlap : np.ndarray
                Symmetric overlap matrix (shape: (n, n)), diagonal elements = 0
            T : float
                Threshold (0~1). If overlap >= T, images are considered "covering" each other,
                otherwise (< T), they are "non-overlapping".

        Returns:
            reps : list[int]
                List of representative image indices.
        """
        NViews = Overlap.shape[0]
        KViews = len(frame_idxs)
        
        if KViews == NViews:
            Nbox_per_view = [1] * len(frame_idxs)
        else:
            group = random_partition_indices(NViews, KViews, rng)
            Nbox_per_view = [len(g) for g in group]

        return Nbox_per_view
    
    
    def reorder_with_overlap(self, O: np.ndarray, reps: list[int]) -> list[int]:
        if len(reps) <= 1:
            return reps
        ordered = [reps[0]]
        remaining = set(reps[1:])
        while remaining:
            last = ordered[-1]
            candidates = [j for j in remaining if O[last, j] > 0]
            next_img = (max(candidates, key=lambda j: O[last, j])
                        if candidates else remaining.pop())
            remaining.discard(next_img)
            ordered.append(next_img)
        return ordered
    
    def remap_mask(self, views, concates, cur_frame_id, all_frame_id):
        overlap_id = [f'{cur_frame_id}_{idx}' for idx in all_frame_id if idx != cur_frame_id]
        h, w = views[cur_frame_id]['img'].shape[1:]
        th, tw = views[cur_frame_id]['orig_img'].shape[:2]
        mask = np.zeros((h, w), dtype=np.bool_)
        if len(overlap_id) == 0:    
            return np.ones((th, tw), dtype=np.bool_)
        for ovp_id in overlap_id:
            x = concates[ovp_id][1][0]
            y = concates[ovp_id][1][1]
            mask[y, x] = True
        # scale h, w to th, tw, and the ratio is the same
        scale = min(th / h, tw / w)
        nh, nw = int(h * scale), int(w * scale)
        mask = cv2.resize(mask.astype(np.float32), (nw, nh), interpolation=cv2.INTER_NEAREST).astype(np.bool_)
        # allocate the mask to the original image
        boarder_x = (tw - nw) // 2
        boarder_y = (th - nh) // 2
        mask = np.pad(mask, ((boarder_y, boarder_y), (boarder_x, boarder_x)), mode='constant', constant_values=0)
        return mask


    def puzzle_augment(self, views, rng):
        # Step1: calculate the overlap matrix
        overlap_matrix, concates = self.overlapMatrix(views, depth_threshold=self.overlap_depth_threshold)

        if self.debug:
            self._debug_overlap(views, concates)
        
        # Step2: select the keyframe based on overlap matrix
        frame_indices = self.longest_true_sequence_indices(overlap_matrix, self.minVisibility, self.puzzle_min_overlap, rng=rng)
        Nbox_per_view = self.select_reps(overlap_matrix, frame_indices, rng)
        if self.debug:
            print('Frame indices: ', [int(fi) for fi in frame_indices])
            print('Nbox per view: ', Nbox_per_view)

        new_views = []
        comb_idx = 0
        
        # Step3: Puzzle generation
        for frame_id, n_boxes in zip(frame_indices, Nbox_per_view):

            orig_img = views[frame_id]['orig_img']
            orig_depthmap = views[frame_id]['orig_depthmap']
            height, width = orig_img.shape[:2]
            mask = self.remap_mask(views, concates, frame_id, frame_indices)
            min_box_size, max_box_size = self.random_param_gen(height, width, n_boxes)
            boxes, template_box = generate_overlapping_boxes(
                                                            image=orig_img,
                                                            mask=mask,
                                                            n_boxes=n_boxes,
                                                            min_box_size=min_box_size,
                                                            max_box_size=max_box_size,
                                                            min_overlap=self.bbox_min_overlap,
                                                            max_overlap=self.bbox_max_overlap,
                                                            rng=rng,
                                                            )
            if boxes is None:
                # no valid region found, skip this view
                for i in range(n_boxes):
                    new_views.append(views[frame_id].copy())
                    new_views[-1]['img'] = (ImgInvNorm(views[frame_id]['img'])*255).astype(np.uint8) # (3, H, W) -> (H, W, 3) pixel value: 0~255
                    h, w = views[frame_id]['orig_img'].shape[:2]
                    new_views[-1]['bbox'] = _default_box(w, h)
                    new_views[-1]['n_boxes'] = n_boxes
                    new_views[-1]['comb'] = (ImgInvNorm(views[comb_idx]['img'])*255).astype(np.uint8)  
                    new_views[-1]['overlap_matrix'] = overlap_matrix
                    new_views[-1]['frame_id'] = frame_id
                    comb_idx += 1
                continue
            
            batch_images, batch_points, batch_depths, batch_intrinsics, batch_camera_poses, batch_h, batch_w, batch_masks, batch_boxes = \
                                                                                          calibrate_puzzles(depthmap=orig_depthmap,
                                                                                                            camera_pose=views[frame_id]['camera_pose'],
                                                                                                            camera_intrinsics=views[frame_id]['orig_camera_intrinsics'],
                                                                                                            image=orig_img,
                                                                                                            boxes=boxes,
                                                                                                            template_box=template_box,
                                                                                                            bk_view=views[frame_id],
                                                                                                            rng=rng,
                                                                                                            )
   

            batch_images, batch_points, batch_depths, batch_masks, batch_intrinsics, batch_h, batch_w = self.rescale(self.resolution, batch_images, batch_points, batch_depths, batch_masks, batch_intrinsics, batch_h, batch_w)

            

            for img, pts3d, depthmap, intrinsics, camera_pose, h, w, mask, bbox in zip(batch_images, batch_points, batch_depths, batch_intrinsics, batch_camera_poses, batch_h, batch_w, batch_masks, batch_boxes):
                new_view = views[frame_id].copy()
                new_view['img'] = img
                new_view['pts3d'] = pts3d
                new_view['depthmap'] = depthmap
                new_view['camera_intrinsics'] = intrinsics
                new_view['camera_pose'] = camera_pose   
                new_view['true_shape'] = np.int32((h, w))
                new_view['valid_mask'] = mask
                new_view['bbox'] = bbox
                new_view['n_boxes'] = n_boxes
                new_view['comb'] = (ImgInvNorm(views[comb_idx]['img'])*255).astype(np.uint8) 
                new_view['overlap_matrix'] = overlap_matrix
                new_view['frame_id'] = frame_id
                comb_idx += 1
                
                new_views.append(new_view)

    
        return new_views
    
    def rotation3D_augment(self, views, all_pts3d, all_pts3d_rgb, rng):
        # Randomly Rotation
        rotate_idx = np.where(np.array( [self.RandomRotate(rng) for i in range(1, len(views))] ) == True)[0] + 1
        
        # deleted first frame
        rotated_extrinsics, rotated_intrinsics = [], []
        if self.debug:
            print('Rotate indices: ', rotate_idx)
        for box_idx in rotate_idx:
            max_rotation_angle = rng.uniform(self.min_rotation_angle, self.max_rotation_angle)
            rotated_intrinsic = views[box_idx]['camera_intrinsics'].copy()
            new_extrinsic = random_rotate(points=views[box_idx]['pts3d'], 
                                          masks=views[box_idx]['valid_mask'], 
                                          intrinsic=rotated_intrinsic, 
                                          extrinsic=np.linalg.inv(views[box_idx]['camera_pose']), 
                                          width=views[box_idx]['true_shape'][1],
                                          height=views[box_idx]['true_shape'][0], 
                                          max_rotation_angle=max_rotation_angle,
                                          min_coverage=self.min_coverage,
                                          max_coverage=self.max_coverage,
                                          jitter_std=self.jitter_std,
                                          rng=rng,
                                         )
            rotated_extrinsics.append(new_extrinsic)
            rotated_intrinsics.append(rotated_intrinsic)


        render_image, render_depth = rasterization(points=all_pts3d, 
                                                   colors=all_pts3d_rgb, 
                                                   mask=np.ones(all_pts3d.shape[0], dtype=np.bool_),
                                                   intrinsics=rotated_intrinsics,
                                                   extrinsics=rotated_extrinsics,
                                                   height=self.resolution[0],
                                                   width=self.resolution[1], 
                                                   bg_color=self.bg_color
                                                   )

        for box_idx, r_im, r_depth, extr, intr in zip(rotate_idx, render_image, render_depth, rotated_extrinsics, rotated_intrinsics):
            transformed_points, transformed_mask = depthmap_to_camera_coordinates(r_depth, intr)
            camera_pose = np.linalg.inv(extr)
            R_cam2world = camera_pose[:3, :3]
            t_cam2world = camera_pose[:3, 3]
            transformed_points = np.einsum("ik, vuk -> vui", R_cam2world, transformed_points) + t_cam2world[None, None, :]
            views[box_idx]["img"] = r_im
            views[box_idx]["depthmap"] = r_depth
            views[box_idx]["camera_pose"] = camera_pose
            views[box_idx]["pts3d"] = transformed_points
            views[box_idx]["valid_mask"] = transformed_mask
        return views
    
    def rotation2D_augment(self, views, rng, p=0.5, min_overlap=0.4):
        to_pil = lambda arr: Image.fromarray(arr.astype(np.uint8)) if arr.dtype != np.uint8 else Image.fromarray(arr)
        # assert isinstance(self.resolution, tuple), f"resolution should be a tuple, but got {type(self.resolution)}"
        # resize_pil = lambda arr: transforms_v2.Resize(size=self.resolution)(arr)
        # resize_np = lambda arr: cv2.resize(arr, self.resolution, interpolation=cv2.INTER_NEAREST)
        
        # Keep the first one unchanged
        augmented_views = copy.deepcopy(views)
        augmented_views[0]['img'] = to_pil(augmented_views[0]['img'])
        
        for idx in range(1, len(views)):
            transformed_img = to_pil(augmented_views[idx]['img'])
            transformed_pts = augmented_views[idx]['pts3d']
            transformed_mask = to_pil(augmented_views[idx]['valid_mask'])
            transformed_depth = augmented_views[idx]['depthmap']
            used_transforms = []
            for name, func in self.transform2d:
                if rng.random() < p:
                    transform = func()
                    if name in ["RandomPerspective", "RandomAffine"]:
                        transformed_results = transform(image=transformed_img, pointcloud=transformed_pts, mask=transformed_mask, overlap=to_pil(np.ones_like(transformed_mask, dtype=np.bool_)), depth=transformed_depth)
                        
                        if np.mean(np.array(transformed_results['overlap'])) >= min_overlap:
                            transformed_img   = transformed_results['image']
                            transformed_pts   = transformed_results['pointcloud']
                            transformed_mask  = transformed_results['mask']
                            transformed_depth = transformed_results['depth']
                    else:
                        transformed_img = transform(transformed_img)
                    used_transforms.append(name)
            
            augmented_views[idx]['img'] = transformed_img
            augmented_views[idx]['pts3d'] = transformed_pts.astype(np.float32)
            augmented_views[idx]['valid_mask'] = np.array(transformed_mask, dtype=np.bool_)
            augmented_views[idx]['depthmap'] = transformed_depth.astype(np.float32)
            augmented_views[idx]['true_shape'] = views[idx]['true_shape']

        return augmented_views

    def __call__(self, views, rng):
        all_pts3d = np.concatenate([view['pts3d'][view['valid_mask']] for view in views])
        all_pts3d_rgb = np.concatenate([ImgInvNorm(view['img'])[view['valid_mask']] for view in views])
        
        # 1. Puzzle generation
        new_views = self.puzzle_augment(views.copy(), rng)
        
        # 2. Puzzle augmentation
        # 3D Rotation augmentation
        if self.transform3d:
            new_views = self.rotation3D_augment(new_views, all_pts3d, all_pts3d_rgb, rng)

        # 2D Rotation augmentation
        if self.transform2d:
            new_views = self.rotation2D_augment(new_views, rng)

        return new_views

            
        

