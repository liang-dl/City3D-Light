import os
import numpy as np
import shutil

def load_poses_bounds(npy_path):
    poses_bounds = np.load(npy_path)  # (N, 17)
    poses = poses_bounds[:, :-2].reshape([-1, 3, 5])
    bounds = poses_bounds[:, -2:]
    return poses, bounds

def extract_camera_positions(poses):
    # 相机中心位置：从位姿矩阵中提取
    cam_centers = poses[:, :3, 3]  # shape (N, 3)
    return cam_centers

def compute_grid_regions(cam_centers, grid_size=50.0):
    # 简单按 (x, y) 坐标分块
    min_xyz = np.min(cam_centers, axis=0)
    max_xyz = np.max(cam_centers, axis=0)
    print(f"Scene bounds: min {min_xyz}, max {max_xyz}")

    grid_min = min_xyz[:2]
    grid_max = max_xyz[:2]

    # 计算网格尺寸
    x_bins = np.arange(grid_min[0], grid_max[0] + grid_size, grid_size)
    y_bins = np.arange(grid_min[1], grid_max[1] + grid_size, grid_size)

    print(f"Grid X bins: {len(x_bins)-1}, Grid Y bins: {len(y_bins)-1}")

    return x_bins, y_bins

def assign_images_to_regions(cam_centers, x_bins, y_bins):
    region_indices = []
    for c in cam_centers:
        x_idx = np.digitize(c[0], x_bins) - 1
        y_idx = np.digitize(c[1], y_bins) - 1
        region_indices.append((x_idx, y_idx))
    return region_indices

def save_region_subsets(poses, bounds, region_indices, image_dir, output_root):
    region_to_indices = dict()
    for idx, reg in enumerate(region_indices):
        if reg not in region_to_indices:
            region_to_indices[reg] = []
        region_to_indices[reg].append(idx)

    for reg, indices in region_to_indices.items():
        region_name = f"region_{reg[0]}_{reg[1]}"
        region_path = os.path.join(output_root, region_name)
        os.makedirs(os.path.join(region_path, "images"), exist_ok=True)

        # 保存poses_bounds.npy
        reg_poses = poses[indices]
        reg_bounds = bounds[indices]
        reg_poses_bounds = np.concatenate([reg_poses.reshape([-1, 15]), reg_bounds], axis=-1)
        np.save(os.path.join(region_path, "poses_bounds.npy"), reg_poses_bounds)

        # 拷贝对应图片
        for idx in indices:
            src_img = os.path.join(image_dir, f"{idx:04d}.jpg")
            dst_img = os.path.join(region_path, "images", f"{idx:04d}.jpg")
            shutil.copy(src_img, dst_img)

        print(f"Saved region {region_name}: {len(indices)} images")

def main(data_root, output_root, grid_size=50.0):
    os.makedirs(output_root, exist_ok=True)
    npy_path = os.path.join(data_root, "poses_bounds.npy")
    image_dir = os.path.join(data_root, "images")

    poses, bounds = load_poses_bounds(npy_path)
    cam_centers = extract_camera_positions(poses)

    x_bins, y_bins = compute_grid_regions(cam_centers, grid_size=grid_size)
    region_indices = assign_images_to_regions(cam_centers, x_bins, y_bins)

    save_region_subsets(poses, bounds, region_indices, image_dir, output_root)

if __name__ == "__main__":
    # 示例
    data_root = "/path/to/your/full_dataset"    # 原始数据集路径
    output_root = "/path/to/your/output_regions" # 输出子区域路径
    grid_size = 100.0   # 每个区域块覆盖 100米 × 100米，可以根据需要调整

    main(data_root, output_root, grid_size)
