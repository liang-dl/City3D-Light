import os
import subprocess
import time
import torch
from concurrent.futures import ThreadPoolExecutor

def get_free_gpus(threshold=3000):
    """检测哪些GPU空闲（剩余显存大于threshold MB）"""
    import pynvml
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    free_gpus = []
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mem_mb = meminfo.free / 1024 / 1024
        if free_mem_mb > threshold:
            free_gpus.append(i)
    pynvml.nvmlShutdown()
    return free_gpus

def train_one_tile(tile_dir, ngp_exec_path, config_path, gpu_id):
    input_ply = os.path.join(tile_dir, "input.ply")
    if not os.path.exists(input_ply):
        print(f"Tile {tile_dir} missing input.ply, skipped.")
        return

    cmd = [
        ngp_exec_path,
        "--mode", "train",
        "--scene", input_ply,
        "--config", config_path,
        "--save_mesh",
        "--n_steps", "1500",
        "--output", tile_dir,
        "--device", f"cuda:{gpu_id}"
    ]
    log_file = os.path.join(tile_dir, "train_log.txt")
    with open(log_file, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

def multi_gpu_train(tiles_dir, ngp_exec_path, config_path, n_workers=4, mem_threshold=3000):
    tile_dirs = [os.path.join(tiles_dir, d) for d in os.listdir(tiles_dir) if os.path.isdir(os.path.join(tiles_dir, d))]

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for tile_dir in tile_dirs:
            while True:
                free_gpus = get_free_gpus(mem_threshold)
                if free_gpus:
                    gpu_id = free_gpus[0]
                    print(f"Launching {tile_dir} on GPU {gpu_id}")
                    futures.append(executor.submit(train_one_tile, tile_dir, ngp_exec_path, config_path, gpu_id))
                    time.sleep(3)  # 稍微等待，防止过载
                    break
                else:
                    print("No free GPU available, waiting...")
                    time.sleep(10)

        for f in futures:
            f.result()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiles_dir', type=str, required=True)
    parser.add_argument('--instant_ngp_exec', type=str, required=True)
    parser.add_argument('--config_path', type=str, default="configs/light_ngp_config.json")
    parser.add_argument('--n_workers', type=int, default=4)
    args = parser.parse_args()

    multi_gpu_train(args.tiles_dir, args.instant_ngp_exec, args.config_path, args.n_workers)
