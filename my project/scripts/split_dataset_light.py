import os
import open3d as o3d
import numpy as np
from downsample_utils import downsample_ply

def split_pcd(input_ply, output_dir, tile_size=100.0, voxel_size=0.05, min_points=1000):
    os.makedirs(output_dir, exist_ok=True)
    pcd = o3d.io.read_point_cloud(input_ply)
    points = np.asarray(pcd.points)
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    tile_idx = 0
    for x in np.arange(min_bound[0], max_bound[0], tile_size):
        for y in np.arange(min_bound[1], max_bound[1], tile_size):
            mask = (points[:,0] >= x) & (points[:,0] < x+tile_size) & (points[:,1] >= y) & (points[:,1] < y+tile_size)
            tile_points = points[mask]
            if tile_points.shape[0] >= min_points:
                tile_pcd = o3d.geometry.PointCloud()
                tile_pcd.points = o3d.utility.Vector3dVector(tile_points)
                tile_dir = os.path.join(output_dir, f"tile_{tile_idx}")
                os.makedirs(tile_dir, exist_ok=True)
                tile_path = os.path.join(tile_dir, "input_raw.ply")
                o3d.io.write_point_cloud(tile_path, tile_pcd)

                # 下采样
                down_path = os.path.join(tile_dir, "input.ply")
                downsample_ply(tile_path, down_path, voxel_size)
                print(f"Tile {tile_idx} saved: {down_path}")
                tile_idx += 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_ply', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--tile_size', type=float, default=100.0)
    parser.add_argument('--voxel_size', type=float, default=0.05)
    args = parser.parse_args()

    split_pcd(args.input_ply, args.output_dir, args.tile_size, args.voxel_size)
