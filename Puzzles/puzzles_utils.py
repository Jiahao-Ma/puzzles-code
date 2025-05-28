import random
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from typing import List, Tuple

def calculate_iou(box1: Tuple[int, int, int, int], 
                 box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes.
    
    Args:
        box1: First box coordinates (x1, y1, x2, y2)
        box2: Second box coordinates (x1, y1, x2, y2)
    
    Returns:
        float: IoU value
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area

def image_plane_uv(width: int, height: int, dtype=None, normalize=False) -> np.ndarray:
    u = np.linspace(0.5, width - 0.5, width, dtype=dtype)
    v = np.linspace(0.5, height - 0.5, height, dtype=dtype)
    if normalize:
        u = (u - width / 2) / (width / 2)
        v = (v - height / 2) / (height / 2)
    u, v = np.meshgrid(u, v, indexing='xy')
    uv = np.stack([u, v], axis=-1)
    return uv


def generate_overlapping_boxes(
    image: np.ndarray,
    mask: np.ndarray,
    n_boxes: int,
    min_box_size_scale: float = None,
    max_box_size_scale: float = None,
    min_box_size: float = None,
    max_box_size: float = None,
    min_overlap: float = 0.2,
    max_overlap: float = 0.4,
    max_attempts: int = 1000,
    minTrue:float=0.6,
    rng=None,
    ) -> List[Tuple[int, int, int, int]]:
    def _default_box(width, height):    
        min_edge = min(width, height)
        left = (width - min_edge) // 2
        top = (height - min_edge) // 2
        right = left + min_edge
        bottom = top + min_edge
        return (left, top, right, bottom)
    """
    Generate N random boxes with overlap percentage within specified range.
    
    Args:
        image: Input image as numpy array (H, W, C)
        n_boxes: Number of boxes to generate
        min_box_size_scale: Minimum box size as a fraction of image size
        max_box_size_scale: Maximum box size as a fraction of image size
        min_overlap: Minimum required overlap between boxes (0-1)
        max_overlap: Maximum allowed overlap between boxes (0-1)
        max_attempts: Maximum attempts to place a new box
    
    Returns:
        List of boxes coordinates as (x1, y1, x2, y2)
    """
    if not min_box_size_scale and not max_box_size_scale and not min_box_size and not max_box_size:
        raise ValueError("At least one of min_box_size_scale, max_box_size_scale, min_box_size, or max_box_size must be provided.")
    # find the bbox for the true mask
    true_region = np.where(mask)
    if len(true_region[0]) == 0:
        return None, None

    h, w = image.shape[:2]
    cover_region = np.zeros((h, w), dtype=np.bool_)
    
    # Initialize list to store box coordinates
    boxes = []
    
    def is_valid_box(new_box: Tuple[int, int, int, int]) -> bool:
        """
        Check if the new box satisfies overlap constraints with at least one existing box
        and doesn't overlap too much with any box.
        """
        if not boxes:
            return True
        
        # Calculate IoU with all existing boxes
        ious = [calculate_iou(new_box, existing_box) for existing_box in boxes]
        
        # For the first box, any position is valid
        # if len(boxes) == 1:
        #     return max(ious) >= min_overlap and max(ious) <= max_overlap
            
        # For subsequent boxes:
        # 1. At least one box should have overlap in the desired range
        has_valid_overlap = any(min_overlap <= iou <= max_overlap for iou in ious)
        # 2. No box should have overlap more than max_overlap
        no_excessive_overlap = all(iou <= max_overlap for iou in ious)
        # 3. The new box should have at least minTrue overlap with the true mask
        temp_cover_region = cover_region.copy()
        temp_cover_region[new_box[1]:new_box[3], new_box[0]:new_box[2]] = True
        overlap = (temp_cover_region & mask).sum() / mask.sum()
        true_overlap = overlap >= min((cover_region.mean()*1.2), 0.7)
        
        return has_valid_overlap and no_excessive_overlap and true_overlap

    def generate_box_with_overlap(box_size, max_tries=50) -> Tuple[int, int, int, int]:
        """
        Generate a new box with position adjusted to achieve desired overlap.
        """
        # If no existing boxes, generate a random position
        if not boxes:
            best_cover = 0.0
            box_candidate = _default_box(w, h)
            while max_tries > 0:
                x1 = int(rng.integers(0, w - box_size))
                y1 = int(rng.integers(0, h - box_size))
                x2 = int(x1 + box_size)
                y2 = int(y1 + box_size)
                valid_cover = mask[y1:y2, x1:x2].mean()
                if valid_cover > best_cover:
                    best_cover = valid_cover
                    box_candidate = (x1, y1, x2, y2)
                if valid_cover >= minTrue:
                    return (x1, y1, x2, y2)
                max_tries -= 1
            return box_candidate
    
        # Select a random existing box to overlap with
        reference_box = rng.choice(boxes)
        rx1, ry1, _, _ = reference_box
        
        # Calculate maximum allowed shift based on desired overlap
        max_shift = int(box_size * (1 - min_overlap))
        
        # Generate random shift
        x_shift = rng.integers(-max_shift, max_shift)
        y_shift = rng.integers(-max_shift, max_shift)
        
        # Calculate new box coordinates
        x1 = max(0, min(w - box_size, rx1 + x_shift))
        y1 = max(0, min(h - box_size, ry1 + y_shift))
        
        return (int(x1), int(y1), int(x1 + box_size), int(y1 + box_size))

    # Generate boxes
    attempts = 0
    while len(boxes) < n_boxes and attempts < max_attempts:
        # generate random box size within the specified range
            
        if max_box_size is None and min_box_size is None:
            box_size = rng.integers(int(min_box_size_scale * min(h, w)), int(max_box_size_scale * min(h, w)))
        elif max_box_size_scale is None and min_box_size_scale is None:
            box_size = rng.integers(min_box_size, max_box_size)
        
        new_box = generate_box_with_overlap(box_size)
        
        if is_valid_box(new_box):
            cover_region[new_box[1]:new_box[3], new_box[0]:new_box[2]] = True
            boxes.append(new_box)
        
        attempts += 1
    
    # Print warning if couldn't generate all boxes
    if len(boxes) < n_boxes:
        # print(f"Warning: Could only generate {len(boxes)} boxes out of {n_boxes} requested")
        sup_boxes = rng.choice(boxes, n_boxes-len(boxes), replace=True)
        # add slightly jittered boxes
        for sbox in sup_boxes:
            x1, y1, x2, y2 = sbox
            supbox_w = x2 - x1
            supbox_h = y2 - y1

            offset = int(0.2 * min(supbox_w, supbox_h))
            random_offset_x = int(rng.integers(-offset, offset + 1))
            random_offset_y = int(rng.integers(-offset, offset + 1))

            new_x1 = max(0, min(w - supbox_w, x1 + random_offset_x))
            new_y1 = max(0, min(h - supbox_h, y1 + random_offset_y))
            new_x2 = new_x1 + supbox_w
            new_y2 = new_y1 + supbox_h
            
            boxes.append((int(new_x1), int(new_y1), int(new_x2), int(new_y2)))
            
    # the first box is the template box
    box_w=boxes[0][2]-boxes[0][0]
    box_h=boxes[0][3]-boxes[0][1]
    templ_x1 = w//2 - box_w//2
    templ_y1 = h//2 - box_h//2
    template_box = (templ_x1, templ_y1, templ_x1 + box_w, templ_y1 + box_h)
    return boxes, template_box

def estimate_camera_pose_ransac(points_3d, points_2d, K):
    """
    Estimate camera pose (rotation and translation) using OpenCV's solvePnPRansac.

    Args:
        points_3d (np.ndarray): shape (N, 3)
        points_2d (np.ndarray): shape (N, 2)
        K (np.ndarray): camera intrinsic matrix (3, 3)

    Returns:
        w2c: (4, 4) world-to-camera matrix
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d, points_2d,
        K, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        raise RuntimeError("solvePnPRansac failed to find a solution")

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(-1)

    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = t

    return w2c.astype(np.float32), R.astype(np.float32), t.astype(np.float32)


def random_axis_angle_rotation(angle_deg, center, rng=None):
    """
    Generate a rotation matrix for a given angle around a random axis passing through the origin.
    This function ensures that the rotation keeps the point on the surface of a sphere centered at 'center'.
    
    Args:
    angle_deg (float): The rotation angle in degrees.
    center (array): The center of the sphere (and rotation).
    
    Returns:
    numpy.ndarray: The 4x4 rotation matrix.
    """
    angle_rad = np.deg2rad(angle_deg)
    # Random axis of rotation, uniformly distributed over the sphere
    # axis = np.random.normal(0, 1, 3)
    axis = rng.normal(0, 1, 3)
    axis /= np.linalg.norm(axis)  # Normalize to make it a unit vector

    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    one_minus_cos = 1 - cos_angle

    # Rodrigues' rotation formula for rotation matrix
    rotation_matrix = np.array([
        [cos_angle + axis[0]**2 * one_minus_cos,
         axis[0] * axis[1] * one_minus_cos - axis[2] * sin_angle,
         axis[0] * axis[2] * one_minus_cos + axis[1] * sin_angle],
        [axis[1] * axis[0] * one_minus_cos + axis[2] * sin_angle,
         cos_angle + axis[1]**2 * one_minus_cos,
         axis[1] * axis[2] * one_minus_cos - axis[0] * sin_angle],
        [axis[2] * axis[0] * one_minus_cos - axis[1] * sin_angle,
         axis[2] * axis[1] * one_minus_cos + axis[0] * sin_angle,
         cos_angle + axis[2]**2 * one_minus_cos]
    ])

    # Create a homogeneous 4x4 rotation matrix
    full_rotation_matrix = np.eye(4)
    full_rotation_matrix[:3, :3] = rotation_matrix
    full_rotation_matrix[:3, 3] = center - rotation_matrix @ center

    return full_rotation_matrix

def random_rotate_around_pc_mean(extrinsic, point_cloud, max_angle, jitter_std, rng=None):
    """
    Adjust the camera orientation towards the centroid of the point cloud, 
    based on intrinsic and extrinsic parameters. Incorporates maximum rotation 
    angle and random jitter.

    Args:
        extrinsic: w2c, 4x4 numpy array, the camera extrinsic matrix.
        point_cloud: Nx3 numpy array, the point cloud data.
        max_angle: float, the maximum allowed rotation angle (in radians).
        jitter_std: float, standard deviation for random jitter (in radians).

    Returns:
        new_extrinsic: new w2c, 4x4 numpy array, the updated extrinsic matrix.
    """
    # Calculate centroid of the point cloud
    centroid = np.mean(point_cloud, axis=0)

    c2w = np.linalg.inv(extrinsic) # w2c to c2w

    # angle_to_rotate = np.random.uniform(0, max_angle) + np.random.normal(0, jitter_std)
    angle_to_rotate = rng.uniform(0, max_angle) + rng.normal(0, jitter_std)
    
    random_transform = random_axis_angle_rotation(angle_to_rotate, centroid, rng=rng)
    new_c2w = random_transform @ c2w
    new_extrinsic = np.linalg.inv(new_c2w)
        
    return new_extrinsic

def normal_estimation(points):
    '''
        Simple normal estimation using cross product.
        Notice that the normal is inward-pointing normal, pointing to the inside of the surface.
        points: (H, W, 3)
    '''
    p_up = points[:-2, 1:-1]
    p_down = points[2:, 1:-1]
    p_left = points[1:-1, :-2]
    p_right = points[1:-1, 2:]
    v_horizontal = p_right - p_left
    v_vertical = p_down - p_up
    normals = np.cross(v_horizontal, v_vertical, axis=2)
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / np.where(norms == 0, 1, norms)
    full_normals = np.zeros_like(points)
    full_normals[1:-1, 1:-1] = normals
    return full_normals

def fast_camera_coverage(camera_matrix, w2c, points_3d, image_width, image_height, vis_canvas=False, max_rotate=80):#, min_render_points=-1):
    """
    Optimized version of calculate_camera_covered_points using vectorized operations.
    Only samples a subset of points for faster computation.
    """
    # Estimate the normal of the point cloud
    points_3d_n3 = points_3d.reshape(-1, 3)
    normals = normal_estimation(points_3d).reshape(-1, 3)
    c2w = np.linalg.inv(w2c)
    v = c2w[:3, 3].reshape(1, 3) - points_3d_n3
    v_norm = np.linalg.norm(v, axis=1).reshape(-1, 1)
    v = v / np.where(v_norm == 0, 1, v_norm)
    dot_product = np.einsum('ij,ij->i', normals, v)
    # 90 degree is the threshold
    # front_canvas = np.zeros_like(dot_product)
    # front_canvas[dot_product > 0] = 0 # positive means that camera capture the back of the surface (Bad!)
    # front_canvas[dot_product < 0] = 1 # negative means that camera capture the front of the surface (Good!)
    
    # Notice the formula below is different from the paper, because the normal is inward-pointing normal, pointing to the inside of the surface,
    # while the paper is outward-pointing normal, pointing to the outside of the surface (camera). Basically, they are same. 
    front_canvas = np.zeros_like(dot_product)
    front_canvas[dot_product < np.cos(np.radians(180 - max_rotate))] = 1

    # Transform points to camera space (vectorized)
    points_cam = (w2c[:3, :3] @ points_3d_n3.T + w2c[:3, 3:]).T
    
    # Project valid points (vectorized)
    points_2d = (camera_matrix @ points_cam.T).T
    points_2d = points_2d[:, :2] / points_2d[:, 2:]
    
    # Check image bounds (vectorized)
    visible = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image_width) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image_height)
    
    visible_canvas = np.zeros((image_height, image_width), dtype=np.float32)
    visible_canvas[points_2d[visible, 1].astype(int), points_2d[visible, 0].astype(int)] = 1.0

    if vis_canvas:
    # if np.mean(front_canvas) < 0.9:
        normal_canvas = front_canvas.reshape(points_3d.shape[:2])
        plt.figure(figsize=(12, 9))
        plt.subplot(1, 2, 1)
        plt.imshow(normal_canvas)
        plt.axis('off') 
        plt.title(f'Normal Map. Front / Back : {np.mean(front_canvas):.5f}')
        plt.subplot(1, 2, 2)
        plt.imshow(visible_canvas.astype(np.uint8)*255)
        plt.title(f'Camera Coverage : {np.mean(visible_canvas):.5f}')
        plt.axis('off')
        plt.show()

    return np.mean(visible_canvas), np.mean(front_canvas), visible

def random_rotate(
    points, 
    masks,
    intrinsic,
    extrinsic,
    width,
    height,
    max_rotation_angle=30.0,
    min_coverage=0.2,
    max_coverage=0.8,
    jitter_std=30.0,
    max_tries=5,
    rng=None,
    front_back_ratio_thresh=0.8, # 0.8 means 80% of the points should be in front of the camera
    vis_canvas=False
):  
    '''
        Randomly rotate the camera around the point cloud mean.
        The camera should cover the point cloud and the front-back ratio should be larger than front_back_ratio_thresh.
    '''
    best_extrinsic = extrinsic
    best_front_back_ratio = 0.0
    for _ in range(max_tries):
        new_extrinsic = random_rotate_around_pc_mean(
                                               extrinsic, 
                                               points[masks].reshape(-1, 3), 
                                               max_rotation_angle, 
                                               jitter_std, rng=rng)
        coverage, front_back_ratio, _ = fast_camera_coverage(intrinsic, 
                                            new_extrinsic, 
                                            points,  
                                            width, height, 
                                            vis_canvas=vis_canvas)
        
        if front_back_ratio < front_back_ratio_thresh:
            continue
        if coverage > min_coverage and coverage < max_coverage:
            if front_back_ratio > best_front_back_ratio:
                best_extrinsic = new_extrinsic
                best_front_back_ratio = front_back_ratio
            
    return best_extrinsic

def estimate_camera_intrinsics(point_cloud: np.ndarray, uv_coords: np.ndarray) -> np.ndarray:
    """
    Optimized version of camera intrinsic matrix estimation using vectorized operations.
    
    Parameters:
        point_cloud (np.ndarray): 3D points in camera coordinate system (N, 3)
        uv_coords (np.ndarray): 2D pixel projections (N, 2)
    
    Returns:
        np.ndarray: Estimated 3x3 camera intrinsic matrix K
    """
    # Input validation
    if not (point_cloud.shape[1] == 3 and uv_coords.shape[1] == 2 and 
           point_cloud.shape[0] == uv_coords.shape[0]):
        raise ValueError("Invalid input shapes")
    
    # Extract coordinates
    X = point_cloud[:, 0]
    Y = point_cloud[:, 1]
    Z = point_cloud[:, 2]
    u = uv_coords[:, 0]
    v = uv_coords[:, 1]
    
    # Construct A matrix directly using block operations
    n = len(point_cloud)
    A = np.zeros((2*n, 4))
    A[0:2*n:2, 0] = X
    A[0:2*n:2, 2] = Z
    A[1:2*n:2, 1] = Y
    A[1:2*n:2, 3] = Z
    
    # Construct b vector directly
    b = np.zeros(2*n)
    b[0:2*n:2] = u * Z
    b[1:2*n:2] = v * Z
    
    # Solve using numpy's optimized least squares solver
    params = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Construct the intrinsic matrix
    K = np.array([
        [params[0], 0, params[2]],
        [0, params[1], params[3]],
        [0, 0, 1]
    ])
    
    return K

def calibrate_intrinsics_by_pts(template, points, downsample_size=12):
    x1, y1, x2, y2 = template
    box_w = x2 - x1
    box_h = y2 - y1
    box_points = points[y1:y2, x1:x2]
    box_uvs = image_plane_uv(box_w, box_h)

    steph = box_h // downsample_size
    stepw = box_w // downsample_size

    box_points_ds = box_points[::steph, ::stepw].reshape(-1, 3)
    box_uvs_ds = box_uvs[::steph, ::stepw].reshape(-1, 2)
    # remove the inf and nan values
    mask = np.all(np.isfinite(box_points_ds), axis=1)
    box_points_ds = box_points_ds[mask]
    box_uvs_ds = box_uvs_ds[mask]

    K = estimate_camera_intrinsics(box_points_ds, box_uvs_ds)
    return K, box_points_ds, box_uvs_ds