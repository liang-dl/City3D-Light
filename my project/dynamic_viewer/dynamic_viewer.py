# dynamic_viewer.py

import os
import numpy as np
import open3d as o3d
from utils import load_all_regions
from configs import LOAD_RADIUS, META_FILE_NAME, MESH_FILE_NAME

# 你的子块保存的总目录
OUTPUT_ROOT = "./output_root"

# 读取所有region meta信息
region_db = load_all_regions(OUTPUT_ROOT, META_FILE_NAME)

# 创建Viewer
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name='Dynamic City Viewer')

# 初始摄像机设置
def init_camera(vis):
    ctr = vis.get_view_control()
    ctr.set_zoom(0.5)

init_camera(vis)

# 加载函数
def load_region(region_info, vis):
    mesh_path = os.path.join(region_info["path"], MESH_FILE_NAME)
    if os.path.exists(mesh_path):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        region_info["mesh_obj"] = mesh
        vis.add_geometry(mesh)
        region_info["loaded"] = True
        print(f"[Load] {region_info['path']}")

# 卸载函数
def unload_region(region_info, vis):
    if region_info["mesh_obj"]:
        vis.remove_geometry(region_info["mesh_obj"])
        region_info["mesh_obj"] = None
    region_info["loaded"] = False
    print(f"[Unload] {region_info['path']}")

# 更新可见区域
def update_visible_regions(cam_center, vis):
    for region_name, region_info in region_db.items():
        center = np.array(region_info["center"])
        dist = np.linalg.norm(cam_center[:2] - center[:2])  # 只看xy平面距离

        if dist < LOAD_RADIUS:
            if not region_info["loaded"]:
                load_region(region_info, vis)
        else:
            if region_info["loaded"]:
                unload_region(region_info, vis)

# 每帧回调
def on_frame(vis):
    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    cam_center = cam_params.extrinsic[:3, 3]
    update_visible_regions(cam_center, vis)
    return False

# 绑定回调
vis.register_animation_callback(on_frame)

# 主循环
vis.run()
vis.destroy_window()
