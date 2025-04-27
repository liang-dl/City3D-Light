import os
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor

def train_tile(tile_dir, ngp_exec_path, config_path, n_threads=1):
    input_ply = os.path.join(tile_dir, "input.ply")
    if not os.path.exists(input_ply):
        return

    meta_path = os.path.join(tile_dir, "meta.json")
    cmd = [
        ngp_exec_path,
        "--mode", "train",
        "--scene", input_ply,
        "--config", config_path,
        "--save_mesh",
        "--n_steps", "1500",
        "--output", tile_dir
    ]
    print(f"Training tile: {tile_dir}")
    subprocess.run(cmd)

def batch_train(tiles_dir, ngp_exec_path, config_path, n_workers=2):
    tile_dirs = [os.path.join(tiles_dir, d) for d in os.listdir(tiles_dir) if os.path.isdir(os.path.join(tiles_dir, d))]
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(train_tile, tile_dir, ngp_exec_path, config_path) for tile_dir in tile_dirs]
        for f in futures:
            f.result()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiles_dir', type=str, required=True)
    parser.add_argument('--instant_ngp_exec', type=str, required=True)
    parser.add_argument('--config_path', type=str, default="configs/light_ngp_config.json")
    parser.add_argument('--n_workers', type=int, default=2)
    args = parser.parse_args()

    batch_train(args.tiles_dir, args.instant_ngp_exec, args.config_path, args.n_workers)
