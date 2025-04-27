import open3d as o3d

def downsample_ply(input_ply, output_ply, voxel_size=0.05):
    pcd = o3d.io.read_point_cloud(input_ply)
    down_pcd = pcd.voxel_down_sample(voxel_size)
    o3d.io.write_point_cloud(output_ply, down_pcd)
    return down_pcd
