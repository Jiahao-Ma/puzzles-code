import cv2
import torch
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
import torchvision.transforms.v2 as transforms_v2
from torchvision.transforms import InterpolationMode
from torchvision.transforms import _functional_tensor as F_t 
from torchvision.transforms.functional import _get_inverse_affine_matrix

def _to_numpy(img):
    """
    Convert various image formats (Tensor, PIL, ndarray) to numpy.ndarray in HWC format.
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
        if img.ndim == 4:  # (B, C, H, W) -> list of images
            return [_to_numpy(i) for i in img]
        if img.ndim == 3:  # (C, H, W)
            img = img.numpy()
            if img.shape[0] == 1:
                img = img[0]  # Grayscale
            else:
                img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
        return np.clip(img, 0, 1)
    elif isinstance(img, Image.Image):
        return np.array(img) / 255.0
    elif isinstance(img, np.ndarray):
        if img.ndim == 3 or img.ndim == 2:
            return np.clip(img / 255.0 if img.dtype == np.uint8 else img, 0, 1)
        elif img.ndim == 4:  # Batch of images
            return [_to_numpy(i) for i in img]
    raise TypeError(f"Unsupported image type: {type(img)}")

def plot(images, titles=None, rows=1, sub_figure_size=2, save_pth=None):
    """
    Display images in a grid with optional titles.
    
    Args:
        images (Tensor | ndarray | PIL.Image | list): Input image(s).
        titles (list of str, optional): Titles for each image.
        rows (int): Number of rows.
    """
    if not isinstance(images, list):
        images = _to_numpy(images)
        if isinstance(images, list):
            pass
        else:
            images = [images]
    else:
        images = [_to_numpy(img) for img in images]

    num_images = len(images)
    cols = (num_images + rows - 1) // rows

    plt.figure(figsize=(cols * sub_figure_size, rows * sub_figure_size))
    for i, img in enumerate(images):
        ax = plt.subplot(rows, cols, i + 1)
        if img.ndim == 2:  # Grayscale
            if np.all(img == 255):
                ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            else:
                ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.axis('off')
        if titles and i < len(titles):
            ax.set_title(titles[i])
    plt.tight_layout()
    if save_pth is not None:
        plt.savefig(save_pth, bbox_inches='tight')
    else:
        plt.show()

def plot_mask(images, titles=None, rows=1, sub_figure_size=2, save_pth=None):
    """
    Display images in a grid with optional titles.
    
    Args:
        images (Tensor | ndarray | PIL.Image | list): Input image(s).
        titles (list of str, optional): Titles for each image.
        rows (int): Number of rows.
    """


    num_images = len(images)
    cols = (num_images + rows - 1) // rows

    plt.figure(figsize=(cols * sub_figure_size, rows * sub_figure_size))
    for i, img in enumerate(images):
        ax = plt.subplot(rows, cols, i + 1)
        if img.ndim == 2:  # Grayscale
            if np.all(img == 255):
                ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            else:
                ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.axis('off')
        if titles and i < len(titles):
            ax.set_title(titles[i])
    plt.tight_layout()
    if save_pth is not None:
        plt.savefig(save_pth, bbox_inches='tight')
    else:
        plt.show()

def calculate_overlapping(boxes, height, width):
    def _cal_overlap(box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        return x_left, y_top, x_right, y_bottom

    num_boxes = len(boxes)
    overlap_masks = []
    for cur_b in range(0, num_boxes):
        overlap_mask = np.zeros((height, width), dtype=np.bool_)
        if cur_b == 0:
            overlap = _cal_overlap(boxes[cur_b+1], boxes[cur_b])
            overlap_mask[overlap[1]:overlap[3], overlap[0]:overlap[2]] = True
        else:    
            for pre_b in range(cur_b):
                if pre_b == cur_b:
                    continue
                overlap = _cal_overlap(boxes[pre_b], boxes[cur_b])
                overlap_mask[overlap[1]:overlap[3], overlap[0]:overlap[2]] = True
        overlap_masks.append(overlap_mask[boxes[cur_b][1]:boxes[cur_b][3], boxes[cur_b][0]:boxes[cur_b][2]])

    return overlap_masks

class RandomAffinePointCloud(transforms_v2.RandomAffine):
    def __init__(self,
                 degrees,
                 translate=None,
                 scale=None,
                 shear=None,
                 interpolation=InterpolationMode.BILINEAR,
                 pts_interpolation=InterpolationMode.NEAREST,
                 fill=0,
                 center=None):
        super().__init__(degrees, translate, scale, shear, interpolation, fill, center)
        self.pts_interpolation = pts_interpolation  

    def forward(self, image: Image.Image, pointcloud: np.ndarray=None, mask: Image.Image = None, overlap: Image.Image = None, depth: np.ndarray=None):
        # Get parameters for affine transformation
        angle, translations, scale, shear = self.get_params(
            self.degrees, self.translate, self.scale, self.shear, image.size)

        transformed_image = transforms_v2.functional.affine(
            image,
            angle=angle,
            translate=translations,
            scale=scale,
            shear=shear,
            interpolation=self.interpolation,
            fill=self.fill,
            center=self.center
        )
        
        if pointcloud is not None:
            assert depth is not None, "Depth must be provided for pointcloud transformation"
            height, width = pointcloud.shape[:2]
            center_f = [0.0, 0.0]
            if self.center is not None:
                center_f = [1.0 * (c - s * 0.5) for c, s in zip(self.center, [width, height])]

            matrix = _get_inverse_affine_matrix(center_f, angle, translations, scale, shear)

            # Apply affine transformation to pointcloud
            pointcloud_t = torch.tensor(pointcloud).permute(2, 0, 1).unsqueeze(0)
            transformed_points = F_t.affine(
                pointcloud_t,
                matrix=matrix,
                interpolation=self.pts_interpolation.value,
                fill=self.fill
            ).squeeze(0).permute(1, 2, 0).numpy()

            depth_t = torch.tensor(depth).unsqueeze(0).unsqueeze(0)
            transformed_depth = F_t.affine(
                depth_t,
                matrix=matrix,
                interpolation=self.pts_interpolation.value,
                fill=self.fill
            ).squeeze(0).squeeze(0).numpy()
        else:
            transformed_points = None
            transformed_depth = None

        if mask is not None:
            transformed_mask = transforms_v2.functional.affine(
                mask,
                angle=angle,
                translate=translations,
                scale=scale,
                shear=shear,
                interpolation=self.pts_interpolation,
                fill=self.fill,
                center=self.center
            )
        else:
            transformed_mask = None
        
        if overlap is not None:
            transformed_overlap = transforms_v2.functional.affine(
                overlap,
                angle=angle,
                translate=translations,
                scale=scale,
                shear=shear,
                interpolation=self.pts_interpolation,
                fill=self.fill,
                center=self.center
            )
        else:
            transformed_overlap = None

        # transformed_depth[~np.array(transformed_mask)] = 0.0
        return {'image': transformed_image, 'pointcloud': transformed_points, 'mask': transformed_mask, 'overlap': transformed_overlap, 'depth': transformed_depth} 
        


class RandomPerspectivePointCloud(transforms_v2.RandomPerspective):
    def __init__(
        self,
        distortion_scale: float = 0.5,
        p: float = 0.5,
        interpolation=InterpolationMode.BILINEAR,
        pts_interpolation=InterpolationMode.NEAREST,
        fill=0,
    ):
        super().__init__(distortion_scale, p, interpolation, fill)
        self.pts_interpolation = pts_interpolation  # For point cloud interpolation

    def forward(self, image: Image.Image, pointcloud: np.ndarray=None, mask: Image.Image = None, overlap: Image.Image = None, depth: np.ndarray=None):
        if torch.rand(1) >= self.p:
            return {'image': image, 'pointcloud': pointcloud, 'mask': mask, 'overlap': overlap, 'depth': depth} 

        width, height = image.size
        half_height, half_width = height // 2, width // 2

        bound_height = int(self.distortion_scale * half_height) + 1
        bound_width = int(self.distortion_scale * half_width) + 1

        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [
            [int(torch.randint(0, bound_width, size=(1,))), int(torch.randint(0, bound_height, size=(1,)))],
            [int(torch.randint(width - bound_width, width, size=(1,))), int(torch.randint(0, bound_height, size=(1,)))],
            [int(torch.randint(width - bound_width, width, size=(1,))), int(torch.randint(height - bound_height, height, size=(1,)))],
            [int(torch.randint(0, bound_width, size=(1,))), int(torch.randint(height - bound_height, height, size=(1,)))],
        ]

        # perspective_coeffs = _get_perspective_coeffs(startpoints, endpoints)
        transformed_image = transforms_v2.functional.perspective(
            image,
            startpoints=startpoints,
            endpoints=endpoints,
            interpolation=self.interpolation,
            fill=self.fill
        )

        if pointcloud is not None:
            pointcloud_t = torch.tensor(pointcloud).permute(2, 0, 1).unsqueeze(0)
            transformed_pointcloud = transforms_v2.functional.perspective(
                pointcloud_t,
                startpoints=startpoints,
                endpoints=endpoints,
                interpolation=self.pts_interpolation,
                fill=self.fill
            ).squeeze(0).permute(1, 2, 0).numpy()
        else:
            transformed_pointcloud = None

        if mask is not None:
            transformed_mask = transforms_v2.functional.perspective(
                mask,
                startpoints=startpoints,
                endpoints=endpoints,
                interpolation=self.pts_interpolation,
                fill=self.fill
            )
        else:
            transformed_mask = None

        if overlap is not None:
            transformed_overlap = transforms_v2.functional.perspective(
                overlap,
                startpoints=startpoints,
                endpoints=endpoints,
                interpolation=self.pts_interpolation,
                fill=self.fill
            )
        else:
            transformed_overlap = None

        if depth is not None:
            depth_t = torch.tensor(depth).unsqueeze(0).unsqueeze(0)
            transformed_depth = transforms_v2.functional.perspective(
                depth_t,
                startpoints=startpoints,
                endpoints=endpoints,
                interpolation=self.pts_interpolation,
                fill=self.fill
            ).squeeze(0).squeeze(0).numpy()
        else:
            transformed_depth = None
        # transformed_depth[~np.array(transformed_mask)] = 0.0
        return {'image': transformed_image, 'pointcloud': transformed_pointcloud, 'mask': transformed_mask, 'overlap': transformed_overlap, 'depth': transformed_depth}

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

def generate_overlapping_boxes(
    image: np.ndarray,
    mask: np.ndarray,
    n_boxes: int,
    min_box_size_scale: float = None,
    max_box_size_scale: float = None,
    min_box_size: float = None,
    max_box_size: float = None,
    min_overlap: float = 0.2,
    max_overlap: float = 0.5,
    max_attempts: int = 50,
    rng=None,
    minTrue=0.5
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
    top, left = np.min(true_region, axis=1)
    bottom, right = np.max(true_region, axis=1)
    top, left, bottom, right = int(top), int(left), int(bottom), int(right)

    h, w = image.shape[:2]
    total_cover = np.zeros((h, w), dtype=np.bool_)
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
        if len(boxes) == 1:
            return max(ious) >= min_overlap #and max(ious) <= max_overlap
            
        # For subsequent boxes:
        # 1. At least one box should have overlap in the desired range
        has_valid_overlap = any(min_overlap <= iou <= max_overlap for iou in ious)
        has_valid_overlap = any(min_overlap <= iou for iou in ious)
        # 2. No box should have overlap more than max_overlap
        no_excessive_overlap = all(iou <= max_overlap for iou in ious)
        # 3. The new box should have at least minTrue overlap with the true mask
        true_overlap = mask[new_box[1]:new_box[3], new_box[0]:new_box[2]].mean() >= minTrue
        
        # return has_valid_overlap and no_excessive_overlap and true_overlap
        return has_valid_overlap and true_overlap

    def generate_box_with_overlap(box_size, init_max_tries=50, max_tries=50) -> Tuple[int, int, int, int]:
        """
        Generate a new box with position adjusted to achieve desired overlap.
        """
        # If no existing boxes, generate a random position
        if not boxes:
            best_cover = 0.0
            box_candidate = _default_box(w, h)
            while init_max_tries > 0:
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
                init_max_tries -= 1
            
            return box_candidate

        best_x1, best_y1, best_x2, best_y2 = rng.choice(boxes)
        max_cover_ratio = 0.0
        while max_tries > 0:
            tmp_max_cover = total_cover.copy()
            # Select a random existing box to overlap with
            reference_box = rng.choice(boxes)
            rx1, ry1, _, _ = reference_box
            
            # Calculate maximum allowed shift based on desired overlap
            max_shift = int(box_size * (1 - min_overlap))
            
            # Generate random shift
            x_shift = rng.integers(-max_shift, max_shift)
            y_shift = rng.integers(-max_shift, max_shift)
            
            # Calculate new box coordinates
            x1 = max(left, min(right - box_size, rx1 + x_shift))
            y1 = max(top, min(bottom - box_size, ry1 + y_shift))
            
            tmp_max_cover[y1:y1 + box_size, x1:x1 + box_size] = True
            cur_cover_ratio = tmp_max_cover.mean()
            cur_cover_ratio = cur_cover_ratio * mask[y1:y1 + box_size, x1:x1 + box_size].mean() # cover more valid area 
            if cur_cover_ratio > max_cover_ratio:
                max_cover_ratio = cur_cover_ratio
                best_x1, best_y1 = x1, y1
                best_x2, best_y2 = x1 + box_size, y1 + box_size
            max_tries -= 1

        return (int(best_x1), int(best_y1), int(best_x2), int(best_y2))

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
            total_cover[new_box[1]:new_box[3], new_box[0]:new_box[2]] = True
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

            offset = int(0.1 * min(supbox_w, supbox_h))
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
