import os, sys; sys.path.append(os.path.join(os.getcwd(), 'MoGe'))
import cv2
import time
import random
import numpy as np
import open3d as o3d
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt 

from Puzzles.puzzles_utils import generate_overlapping_boxes, estimate_camera_pose_ransac, \
                                  random_rotate, calibrate_intrinsics_by_pts
from Puzzles.visualization import rasterization, puzzles_visibility_on_canvas, visualize_boxes, \
                                  visualize_puzzles_regions, visualize_puzzles_sequence
from MoGe import utils3d

def main(args):
    
    print('[1] Start predicting the depth maps and camera poses using MoGe...')
    os.system(f"python MoGe/infer.py --input {args.input} --output {args.output} --threshold {args.threshold} --maps")
    
    root = Path(args.input.split('.')[0])
    image_path = root / 'image.jpg'
    mask_path = root / 'mask.png'
    ori_intrinsics_path = root / 'intrinsics.npy'
    pcd_path = root / 'points.ply'
    I2C_bbox_path = root / 'I2C' / 'Image2Clips_wo_Rot_bbox.png'
    I2C_woRot_path = root / 'I2C' / 'Image2Clips_wo_Rot_region.png'
    I2C_wRot_path = root / 'I2C' / 'Image2Clips_w_Rot_region.png'
    
    random_seed = np.random.randint(0, 1000)
    rng = np.random.default_rng(random_seed)
    
    assert image_path.exists(), f"Image not found at {image_path}"
    assert mask_path.exists(), f"Mask not found at {mask_path}"
    assert ori_intrinsics_path.exists(), f"Intrinsics not found at {ori_intrinsics_path}"
    assert pcd_path.exists(), f"Point cloud not found at {pcd_path}"
    
    print(f'[2] Loading data {image_path}...')
    image = cv2.imread(str(image_path))
    height, width, _ = image.shape
    point_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED).astype(np.bool_)
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    points = np.array(pcd.points).reshape(height, width, 3)
    # remove the invalid points
    points[~point_mask] = 0.0

    # 2. Generate ordered overlapping patches
    boxes, template_box = generate_overlapping_boxes(
        image=image,
        mask=point_mask,
        n_boxes=args.NPuzzles,
        min_box_size_scale=args.min_box_size_scale,
        max_box_size_scale=args.max_box_size_scale,
        min_overlap=args.min_overlap,
        max_overlap=args.max_overlap,
        rng=rng,
    )
    bbox_colors = visualize_boxes(image, boxes, I2C_bbox_path)
    bbox_colors_rgb = [color[::-1] for color in bbox_colors]
    visualize_puzzles_regions(image=image[..., ::-1], boxes=boxes, puzzle_colors=bbox_colors_rgb, save_path=I2C_woRot_path, alpha_ratio=0.4, save_per_frame=True)
    
    # 3. Calibation
    '''
    # All patches use the same camera intrinsics.
    # Note: We use the template box and points to calibrate the camera intrinsics, which is not perfect but enough for visualization.
    # For the properly corrected intrinsics, please refer to the training code.
    '''
    print('[3] Calibrating camera (puzzles) intrinsics...')
    templ_box_h, templ_box_w = template_box[3] - template_box[1], template_box[2] - template_box[0]
    boxK, _, _ = calibrate_intrinsics_by_pts(template_box, points)
    # Extrinsics
    puzzles = []
    puzzles_boxes = []
    puzzles_masks = []
    puzzles_points = []
    puzzles_depths = []
    puzzles_intrinsics = [] 
    puzzles_extrinsics = []
    calib_num_pts = 1024
    for b_i, bbox in enumerate(boxes):
        box_h, box_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        box_points = points[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        box_mask = point_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        box_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        box_depths = box_points[..., 2]
        
        if box_h != templ_box_h or box_w != templ_box_w:
            # resize box_points to the same size as the template box
            box_points = cv2.resize(box_points, (templ_box_w, templ_box_h), interpolation=cv2.INTER_NEAREST)
            box_mask = cv2.resize(box_mask.astype(np.uint8), (templ_box_w, templ_box_h), interpolation=cv2.INTER_NEAREST).astype(np.bool_)
            box_image = cv2.resize(box_image, (templ_box_w, templ_box_h), interpolation=cv2.INTER_NEAREST)
            box_depths = cv2.resize(box_depths, (templ_box_w, templ_box_h), interpolation=cv2.INTER_NEAREST)
    
        box_uv = utils3d.numpy.image_pixel_center(templ_box_w, templ_box_h)
        box_uv_ds = box_uv.reshape(-1, 2)
        mask_ds = box_mask.reshape(-1)
        box_points_ds = box_points.reshape(-1, 3) 
        if np.sum(mask_ds) > calib_num_pts:
            idx = rng.choice(np.arange(mask_ds.shape[0])[mask_ds], calib_num_pts, replace=False)
            box_points_ds_sm = box_points_ds[idx]
            box_uv_ds_sm = box_uv_ds[idx]
            mask_sm = mask_ds[idx]
        try:
            w2c, _, _ = estimate_camera_pose_ransac(box_points_ds_sm[mask_sm], box_uv_ds_sm[mask_sm], boxK)
            
        except:
            print(f"Failed to estimate camera pose for box {b_i}")
        
        puzzles.append(box_image)
        puzzles_boxes.append(bbox)
        puzzles_masks.append(box_mask)
        puzzles_points.append(box_points)
        puzzles_depths.append(box_depths)
        puzzles_intrinsics.append(boxK)
        puzzles_extrinsics.append(w2c)
        
        cv2.imwrite(root / 'I2C' / f'puzzle_{b_i}.png', box_image)
    
    puzzles_vis = np.concatenate(puzzles, axis=1)
    cv2.imwrite(root / 'I2C' / 'puzzle_all.png', puzzles_vis)
        
    if args.Augment:
        print(f'[4] Puzzles augmentation.')
        rot_puzzles_extrinsics = []
        rot_or_not = []
        for pi, (intrinsic, extrinsic) in enumerate(zip(puzzles_intrinsics, puzzles_extrinsics)):
            if np.random.rand() > args.rotate_ratio:
            
                # Randomly rotate the puzzle
                rot_extrinsic = random_rotate(points=puzzles_points[pi],
                        masks=puzzles_masks[pi],
                        intrinsic=intrinsic,
                        extrinsic=extrinsic,
                        width=templ_box_w,
                        height=templ_box_h,
                        max_rotation_angle=args.max_rotation_angle,
                        min_coverage=args.rotate_min_coverage,
                        max_coverage=args.rotate_max_coverage,
                        front_back_ratio_thresh=args.front_back_ratio_thresh,
                        rng=rng,
                        max_tries=3,
                        vis_canvas=False
                        )
                
                rot_puzzles_extrinsics.append(rot_extrinsic)
                rot_or_not.append(True)
            else:
                rot_extrinsic = extrinsic
                rot_puzzles_extrinsics.append(rot_extrinsic)
                rot_or_not.append(False)

        render_images, render_depths = rasterization(points=points.reshape(-1,3),
                                                colors=image.reshape(-1,3),
                                                mask=point_mask.reshape(-1),
                                                intrinsics=puzzles_intrinsics,
                                                extrinsics=rot_puzzles_extrinsics,
                                                width=templ_box_w,
                                                height=templ_box_h,
                                                bg_color=(1, 1, 1))
        puzzles_extrinsics = rot_puzzles_extrinsics
        # Update the puzzles
        for pi, rot_ in enumerate(rot_or_not):
            if rot_:
                depth = render_depths[pi]
                puzzle_mask = depth > 0
                puzzle_points = utils3d.numpy.unproject_cv(utils3d.numpy.image_uv(width=templ_box_w, height=templ_box_h, dtype=depth.dtype), depth, extrinsics=puzzles_extrinsics[pi], intrinsics=puzzles_intrinsics[pi])
                puzzles_points[pi] = puzzle_points.reshape(templ_box_h, templ_box_w, 3)
                puzzles_masks[pi] = puzzle_mask.reshape(templ_box_h, templ_box_w)
                puzzles[pi] = render_images[pi]
            
            cv2.imwrite(root / 'I2C' / f'rot_puzzle_{pi}.png', puzzles[pi])
        
        puzzles_vis = np.concatenate(puzzles, axis=1)
        cv2.imwrite(root / 'I2C' / 'rot_puzzle_all.png', puzzles_vis)
        
        canvas_masks = puzzles_visibility_on_canvas(points = points, 
                                masks=point_mask,
                                intrinsics=puzzles_intrinsics,
                                extrinsics=puzzles_extrinsics,
                                height=templ_box_h,
                                width=templ_box_w,
                                )
            
        visualize_puzzles_regions(image=image[..., ::-1], masks=canvas_masks, puzzle_colors=bbox_colors_rgb, save_path=I2C_wRot_path, alpha_ratio=0.4, save_per_frame=True)
    else:
        print("[4] Puzzles augmentation is skipped.")
        
    # visualize the puzzles    
    puzzles_rgb = [cv2.cvtColor(puzzle, cv2.COLOR_BGR2RGB) for puzzle in puzzles]
    visualize_puzzles_sequence(puzzles=puzzles_rgb, save_path= root / 'I2C' / 'puzzles_video.mp4')
    
    print('Finalized the puzzles generation!')
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Image to Clips")
    parser.add_argument('--input', type=str, default='examples/I2C/indoor.jpg', help='The path to input images')
    parser.add_argument('--output', type=str, default='examples/I2C', help='Directory to save output clips')
    parser.add_argument('--threshold', type=float, default=0.01, help='Threshold for the edge of depth map')

    parser.add_argument('--NPuzzles', type=int, default=4, help='Number of puzzles to generate')
    parser.add_argument('--min_box_size_scale', type=float, default=0.3, help='Minimum patch size scale in percentage')
    parser.add_argument('--max_box_size_scale', type=float, default=0.6, help='Maximum patch size scale in percentage')
    parser.add_argument('--min_overlap', type=float, default=0.1, help='Minimum overlap between patches in percentage')
    parser.add_argument('--max_overlap', type=float, default=0.3, help='Maximum overlap between patches in percentage')
    parser.add_argument('--Augment', action="store_true", help='Whether to apply augmentations to puzzles')
    parser.add_argument('--rotate_ratio', type=float, default=0.2, help='Probability of rotating the patch')
    parser.add_argument('--rotate_min_coverage', type=float, default=0.2, help='Minimum coverage of the patch after rotation')
    parser.add_argument('--rotate_max_coverage', type=float, default=0.8, help='Maximum coverage of the patch after rotation')
    parser.add_argument('--max_rotation_angle', type=float, default=60.0, help='Rotation angle in degrees')
    parser.add_argument('--front_back_ratio_thresh', type=float, default=0.6, help='Threshold for front-back ratio')
    
    # parser.add_argument('--random_seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    main(args)
    '''
    example usage:
    python Image2Clips.py --input examples/I2C/alien.jpg --output examples/I2C --threshold 0.01 \
                          --NPuzzles 8 --min_box_size_scale 0.4 --max_box_size_scale 0.8 \
                          --min_overlap 0.1 --max_overlap 0.3 
                          
    python Image2Clips.py --input examples/I2C/lamp.jpg --output examples/I2C --threshold 0.01 \
                          --NPuzzles 6 --min_box_size_scale 0.2 --max_box_size_scale 0.5 \
                          --min_overlap 0.1 --max_overlap 0.3 --Augment \
                          --rotate_ratio 0.2 --rotate_min_coverage 0.2 --rotate_max_coverage 0.8 \
                          --max_rotation_angle 60.0 --front_back_ratio_thresh 0.6
                          
    python Image2Clips.py --input examples/I2C/boys.jpg --output examples/I2C --threshold 0.01 \
                          --NPuzzles 6 --min_box_size_scale 0.2 --max_box_size_scale 0.5 \
                          --min_overlap 0.1 --max_overlap 0.3 --Augment \
                          --rotate_ratio 0.2 --rotate_min_coverage 0.2 --rotate_max_coverage 0.8 \
                          --max_rotation_angle 60.0 --front_back_ratio_thresh 0.6
                          
    python Image2Clips.py --input examples/I2C/living_room.jpg --output examples/I2C --threshold 0.01 \
                          --NPuzzles 10 --min_box_size_scale 0.2 --max_box_size_scale 0.6 \
                          --min_overlap 0.1 --max_overlap 0.3 
                          
    python Image2Clips.py --input examples/I2C/mountain2.jpg --output examples/I2C --threshold 0.01 \
                          --NPuzzles 10 --min_box_size_scale 0.2 --max_box_size_scale 0.5 \
                          --min_overlap 0.1 --max_overlap 0.3 
                          
    python Image2Clips.py --input examples/I2C/rabbit.jpg --output examples/I2C --threshold 0.01 \
                          --NPuzzles 6 --min_box_size_scale 0.2 --max_box_size_scale 0.5 \
                          --min_overlap 0.1 --max_overlap 0.3 --Augment \
                          --rotate_ratio 0.2 --rotate_min_coverage 0.2 --rotate_max_coverage 0.8 \
                          --max_rotation_angle 60.0 --front_back_ratio_thresh 0.6
                          
    python Image2Clips.py --input examples/I2C/studio_room.jpg --output examples/I2C --threshold 0.01 \
                          --NPuzzles 8 --min_box_size_scale 0.2 --max_box_size_scale 0.5 \
                          --min_overlap 0.1 --max_overlap 0.3 --Augment \
                          --rotate_ratio 0.2 --rotate_min_coverage 0.2 --rotate_max_coverage 0.8 \
                          --max_rotation_angle 30.0 --front_back_ratio_thresh 0.6
                          
    python Image2Clips.py --input examples/I2C/wall.jpg --output examples/I2C --threshold 0.01 \
                          --NPuzzles 8 --min_box_size_scale 0.2 --max_box_size_scale 0.5 \
                          --min_overlap 0.1 --max_overlap 0.3 --Augment \
                          --rotate_ratio 0.2 --rotate_min_coverage 0.2 --rotate_max_coverage 0.8 \
                          --max_rotation_angle 60.0 --front_back_ratio_thresh 0.6
    '''