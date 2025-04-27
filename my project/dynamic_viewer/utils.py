# utils.py

import os
import json

def load_all_regions(output_root, meta_file_name):
    region_db = {}

    for name in os.listdir(output_root):
        region_dir = os.path.join(output_root, name)
        meta_path = os.path.join(region_dir, meta_file_name)
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                region_db[name] = {
                    "center": meta["center"],
                    "size": meta["size"],
                    "loaded": False,
                    "path": region_dir,
                    "mesh_obj": None
                }
    return region_db
