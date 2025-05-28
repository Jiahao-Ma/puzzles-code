import os
import cv2
import random
import open3d 
import imageio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
from pathlib import Path
from .puzzles_utils import calculate_iou

def pad_to_multiple_of_16(img):
    h, w = img.shape[:2]
    new_w = ((w + 15) // 16) * 16
    new_h = ((h + 15) // 16) * 16
    canvas = np.ones((new_h, new_w, 3), dtype=img.dtype)*255
    canvas[:h, :w] = img
    return canvas

def visualize_puzzles_sequence(puzzles: List[np.ndarray],
                               save_path: str = 'puzzles_sequence.mp4'
                               ):
    """
    Visualize a sequence of puzzles as a video in 2xn grid format.
    This function takes a list of numpy arrays representing puzzle images,
    
    Args:
        puzzles (List[np.ndarray]): List of numpy arrays representing puzzle images.
    """
    if not puzzles:
        raise ValueError("The list of puzzles is empty.")
    
    num_puzzles = len(puzzles)   
    if num_puzzles % 2 == 0:
        grid_shape = (2, num_puzzles // 2)
    else:
        grid_shape = (1, num_puzzles)
    puzzle_size = puzzles[0].shape[:2]  # (height, width)
    grid_height = puzzle_size[0] * grid_shape[0]
    grid_width = puzzle_size[1] * grid_shape[1]
    frames = []
    grid_image = np.ones((grid_height, grid_width, 3), dtype=np.uint8)*255
    for i, puzzle in enumerate(puzzles):
        row = i // grid_shape[1]
        col = i % grid_shape[1]
        y_start = row * puzzle_size[0]
        y_end = y_start + puzzle_size[0]
        x_start = col * puzzle_size[1]
        x_end = x_start + puzzle_size[1]
        grid_image[y_start:y_end, x_start:x_end] = puzzle
        frames.append(grid_image.copy())
        
    # Convert each puzzle to an image
    frames = [pad_to_multiple_of_16(frame) for frame in frames]
    frames = [Image.fromarray(frame) for frame in frames]

    # Save the frames as a video
    imageio.mimsave(save_path, frames, fps=1, codec='libx264', quality=8)
    print(f"Video saved as '{save_path}'")


def visualize_puzzles_regions(image: np.ndarray,
                              boxes: List[Tuple[int, int, int, int]]=None,
                              masks: np.ndarray=None,
                              puzzle_colors: List[Tuple[int, int, int]]=None,
                              save_path: Path=None,  
                              alpha_ratio: float=0.5,
                              save_per_frame: bool=False,
                              ) -> None:
    height, width = image.shape[:2]
    image = Image.fromarray(image).convert('RGBA')
    
    if boxes is not None:
        masks = []
        for box in boxes:
            mask = np.zeros((height, width), dtype=np.uint8)
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = 255
            mask = Image.fromarray(mask).convert('L')
            masks.append(mask)
    elif masks is not None:
        new_masks = []
        for mask in masks:
            mask = mask.astype(np.uint8).reshape(height, width) * 255
            mask = Image.fromarray(mask).convert('L')
            new_masks.append(mask)
        masks = new_masks
    else:
        raise ValueError("Either boxes or masks should have values.")
    
    if save_per_frame:
        base_dir = Path(str(save_path).split('.')[0])
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
    
    video_frames = []
    for m_i, mask in enumerate(masks):
        mask_rgba = Image.new('RGBA', (image.size))
        mask_rgba.paste(puzzle_colors[m_i], mask=mask)
        alpha = int(255 * alpha_ratio)
        mask_rgba.putalpha(mask.point(lambda p: alpha if p > 0 else 0))
        image = Image.alpha_composite(image, mask_rgba)
        if save_per_frame:
            save_perframe_path = base_dir / f"puzzle_{m_i}.png"
            image.save(save_perframe_path, format='PNG')
            video_frames.append(image.copy())
    image.save(save_path, format='PNG')
    
    if save_per_frame:
        frames_np = [np.array(frame.convert('RGB')) for frame in video_frames]
        frames_np = [pad_to_multiple_of_16(frame) for frame in frames_np]
        # save video frames as a video
        imageio.mimsave(
            str(base_dir / "puzzles_video.mp4"),
            frames_np,
            fps=1,                
            codec='libx264',       
            quality=8,             
        )


def visualize_boxes(
    image: np.ndarray,
    boxes: List[Union[Tuple[int, int, int, int], np.ndarray]],
    save_path: str = None,
    bbox_colors: List[Tuple[int, int, int]] = None,
    box_type: str = 'rectangle'
) -> None:
    """
    Visualize boxes or polygons on the image.
    
    Args:
        image: Input image
        boxes: List of either:
               - box coordinates (x1, y1, x2, y2) for rectangle type
               - numpy array of points (N, 2) for polygon type
        save_path: Optional path to save the visualization
        bbox_colors: Optional list of colors for each box
        box_type: Type of box to draw ('rectangle' or 'polygon')
    """
    # Create a copy of the image for visualization
    vis_image = image.copy()
    
    # Generate random colors for each box if not provided
    if bbox_colors is None:
        colors = [(random.randint(0, 255), 
                random.randint(0, 255), 
                random.randint(0, 255)) for _ in boxes]
    else:
        colors = bbox_colors
    
    # Draw boxes and calculate overlaps
    for i, (box, color) in enumerate(zip(boxes, colors)):
        if box_type == 'rectangle':
            x1, y1, x2, y2 = box
            # Draw rectangle
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 7)
            # Add box number
            text_pos = (x1+5, y1+50)
            
            # Calculate overlaps
            print_ious = []
            for j, other_box in enumerate(boxes):
                if i != j:
                    iou = calculate_iou(box, other_box)
                    print_ious.append([j+1, iou])
            print_info = f"Overlap between B {i+1} and"
            for info in print_ious:
                print_info += f" B {info[0]}: {info[1]:.2f},"
            print(print_info[:-1])
            
        elif box_type == 'polygon':
            # Ensure points are in the correct format
            points = box.astype(np.int32)
            if len(points.shape) == 2 and points.shape[1] == 2:
                # Draw polygon
                cv2.polylines(vis_image, [points], True, color, 7)
                # Add box number at the mean position of points
                text_pos = tuple(points.mean(axis=0).astype(int))
            else:
                print(f"Warning: Invalid polygon points shape for box {i}")
                continue
        
        # Add box number
        cv2.putText(vis_image, f'#{i}', text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
    
    # save figure using opencv
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    cv2.imwrite(save_path, vis_image)
        
    return colors

class VisOpen3D:
    def __init__(self, width=1920, height=1080, visible=True):
        self.__vis = open3d.visualization.Visualizer()
        self.__vis.create_window(width=width, height=height, visible=visible)
        '''
            Run in linux server without displaying window, 
                    (1) create virtual framebuffer; [Recommended] 
                    (2) Use open3d offscreen rendering. [Discarded] 
            
            If warning : 
                [Open3D WARNING] GLFW Error: X11: The DISPLAY environment variable is missing
                [Open3D WARNING] Failed to initialize GLFW
            
            Solution:
                
                sudo apt-get install xvfb  # Install Xvfb if it's not already installed
                
                Xvfb :99 -screen 0 1024x768x16 &  # Start Xvfb on display :99
                export DISPLAY=:99  # Set the DISPLAY environment variable
        '''
        self.__width = width
        self.__height = height

        if visible:
            self.poll_events()
            self.update_renderer()

    def __del__(self):
        self.__vis.destroy_window()

    def render(self):
        self.__vis.poll_events()
        self.__vis.update_renderer()
        self.__vis.run()

    def poll_events(self):
        self.__vis.poll_events()

    def update_renderer(self):
        self.__vis.update_renderer()

    def run(self):
        self.__vis.run()

    def destroy_window(self):
        self.__vis.destroy_window()

    def add_geometry(self, data):
        self.__vis.add_geometry(data)

    def update_point_cloud_visualization(self, bg_color=np.array([1, 1, 1]), point_size=3.0, light_on=True):
        render_options = self.__vis.get_render_option()
        render_options.background_color = bg_color
        render_options.point_size = point_size
        render_options.light_on = light_on

    def update_view_point(self, intrinsic, extrinsic):
        ctr = self.__vis.get_view_control()
        param = self.convert_to_open3d_param(intrinsic, extrinsic)
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        self.__vis.update_renderer()

    def update_view_point_by_param(self, intrinsic, extrinsic):
        ctr = self.__vis.get_view_control()
        param = open3d.camera.PinholeCameraParameters()
        param.intrinsic = open3d.camera.PinholeCameraIntrinsic(
            width=self.__width,
            height=self.__height,
            fx=intrinsic[0,0],
            fy=intrinsic[1,1],
            cx=intrinsic[0,2],
            cy=intrinsic[1,2],
        )
        param.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        self.__vis.update_renderer()

    def get_view_point_intrinsics(self):
        ctr = self.__vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        intrinsic = param.intrinsic.intrinsic_matrix
        return intrinsic

    def get_view_point_extrinsics(self):
        ctr = self.__vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        extrinsic = param.extrinsic
        return extrinsic

    def get_view_control(self):
        return self.__vis.get_view_control()

    def save_view_point(self, filename):
        ctr = self.__vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        open3d.io.write_pinhole_camera_parameters(filename, param)

    def load_view_point(self, filename):
        param = open3d.io.read_pinhole_camera_parameters(filename)
        intrinsic = param.intrinsic.intrinsic_matrix
        extrinsic = param.extrinsic
        self.update_view_point(intrinsic, extrinsic)

    def convert_to_open3d_param(self, intrinsic, extrinsic):
        param = open3d.camera.PinholeCameraParameters()
        param.intrinsic = open3d.camera.PinholeCameraIntrinsic()
        param.intrinsic.intrinsic_matrix = intrinsic
        param.extrinsic = extrinsic
        return param

    def capture_screen_float_buffer(self, show=False):
        image = self.__vis.capture_screen_float_buffer(do_render=True)

        if show:
            plt.imshow(image)
            plt.show()

        return np.array(image)

    def save_camera_params(self, intrinsics, extrinsics, filename):
        np.savez(filename, intrinsics=intrinsics, extrinsics=extrinsics)

    def capture_screen_image(self, filename):
        self.__vis.capture_screen_image(filename, do_render=True)

    def capture_depth_float_buffer(self, show=False, save_name=None, save_by_opencv=True):
        depth = self.__vis.capture_depth_float_buffer(do_render=True)

        if show:
            plt.imshow(depth)
            plt.show()
        if save_name is not None:
            if save_by_opencv:
                # save by opencv
                depth_np = np.array(depth)
                depth_np = (depth_np - np.min(depth_np)) / (np.max(depth_np) - np.min(depth_np)) * 255
                cv2.imwrite(save_name, (depth_np).astype(np.uint8))
            else:
                plt.imshow(depth)
                plt.axis('off')
                plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
                plt.close()
            
        return np.array(depth)

    def capture_depth_image(self, filename):
        # self.__vis.capture_depth_image(filename, do_render=True)
        depth = self.__vis.capture_depth_float_buffer(do_render=True)
        np.save(filename, depth)
        # to read the saved depth image file use:
        # depth = open3d.io.read_image(filename)
        # plt.imshow(depth)
        # plt.show()

    def draw_camera(self, intrinsic, extrinsic, scale=1, color=None):
        # intrinsics
        K = intrinsic

        # convert extrinsics matrix to rotation and translation matrix
        extrinsic = np.linalg.inv(extrinsic)
        R = extrinsic[0:3,0:3]
        t = extrinsic[0:3,3]

        width = self.__width
        height = self.__height

        geometries = vis_camera(K, R, t, width, height, scale, color)
        for g in geometries:
            self.add_geometry(g)

    def draw_points3D(self, points3D, color=None):
        geometries = draw_points3D(points3D, color)
        for g in geometries:
            self.add_geometry(g)
            
            
def vis_camera(K, R, t, width, height, scale=1, color=None):
    """ Create axis, plane and pyramid geometries in Open3D format
    :   param K     : calibration matrix (camera intrinsics)
    :   param R     : rotation matrix
    :   param t     : translation
    :   param width : image width
    :   param height: image height
    :   param scale : camera model scale
    :   param color : color of the image plane and pyramid lines
    :   return      : camera model geometries (axis, plane and pyramid)
    """

    # default color
    if color is None:
        color = [0.8, 0.2, 0.8]

    # camera model scale
    s = 1 / scale

    # intrinsics
    Ks = np.array([[K[0, 0] * s,            0, K[0,2]],
                   [          0,  K[1, 1] * s, K[1,2]],
                   [          0,            0, K[2,2]]])
    Kinv = np.linalg.inv(Ks)

    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))

    # axis
    axis = create_coordinate_frame(T, scale=scale*0.1)

    # points in pixel
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [width, 0, 1],
        [0, height, 1],
        [width, height, 1],
    ]

    # pixel to camera coordinate system
    points = [scale * Kinv @ p for p in points_pixel]

    # image plane
    width = abs(points[1][0]) + abs(points[3][0])
    height = abs(points[1][1]) + abs(points[3][1])
    
    plane = open3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
    plane.paint_uniform_color(color)
    plane.transform(T)
    plane.translate(R @ [points[1][0], points[1][1], scale])

    # pyramid
    points_in_world = [(R @ p + t) for p in points]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]
    colors = [color for i in range(len(lines))]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points_in_world),
        lines=open3d.utility.Vector2iVector(lines))
    line_set.colors = open3d.utility.Vector3dVector(colors)

    # return as list in Open3D format
    return [axis, plane, line_set]

def draw_points3D(points3D, color=None):
    # color: default value
    if color is None:
        color = [0.8, 0.2, 0.8]

    geometries = []
    for pt in points3D:
        sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01,
                                                            resolution=20)
        sphere.translate(pt)
        sphere.paint_uniform_color(np.array(color))
        geometries.append(sphere)

    return geometries

def create_coordinate_frame(T, scale=0.25):
    frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
    frame.transform(T)
    return frame

def rasterization(points, colors, mask, intrinsics, extrinsics, height, width, bg_color=(0,0,0), window_visible=False, point_size=3):
    """
    Rasterization of point clouds with camera poses.
    """
    if colors.max() > 1:
        colors = colors / 255
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points[mask])
    pcd.colors = open3d.utility.Vector3dVector(colors[mask])
    vis = VisOpen3D(width=width, height=height, visible=window_visible)
    vis.update_point_cloud_visualization(bg_color=bg_color, point_size=point_size) 
    vis.add_geometry(pcd)
    if isinstance(intrinsics, list) and isinstance(extrinsics, list):
        render_images = []
        render_depths = []
        for intrinsic, extrinsic in zip(intrinsics, extrinsics):
            vis.update_view_point(intrinsic, extrinsic)
            render_image = vis.capture_screen_float_buffer(False) 
            render_image = (render_image * 255).astype(np.uint8)
            render_depth = vis.capture_depth_float_buffer(False)
            render_images.append(render_image)
            render_depths.append(render_depth)
        vis.__del__()
        return render_images, render_depths
    else:
        vis.update_view_point(intrinsic, extrinsic)
        render_image = vis.capture_screen_float_buffer(False) 
        render_image = (render_image * 255).astype(np.uint8)
        render_depth = vis.capture_depth_float_buffer(False)
        vis.__del__()
        return render_image, render_depth

def puzzles_visibility_on_canvas(points, masks, extrinsics, intrinsics, height, width):
    points = points.reshape(-1, 3).T # (3, N)
    canvas_masks = []
    for ext_mat, int_mat in zip(extrinsics, intrinsics):
        points_cam = ext_mat[:3, :3] @ points + ext_mat[:3, 3:4]
        points_cam = int_mat @ points_cam
        points_cam = points_cam[:2, :] / points_cam[2, :] # (2, N)
        canvas_mask = masks.reshape(-1) & (points_cam[0, :] >= 0) & (points_cam[0, :] < width) & \
                      (points_cam[1, :] >= 0) & (points_cam[1, :] < height)
        canvas_masks.append(canvas_mask)
    return canvas_masks