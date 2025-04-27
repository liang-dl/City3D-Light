# City3D - Instant-NGP 缩算力训练套件

## 使用步骤

1. 准备点云
    - 输入：城市或航拍图像 ➔ 用COLMAP或DepthAnything重建 ➔ 输出`full_scene.ply`

2. 切块并下采样
    ```bash
    python scripts/split_dataset_light.py --input_ply full_scene.ply --output_dir tiles/
    ```

3. 训练每个小块
    ```bash
    python scripts/train_small_ngp_light.py --tiles_dir tiles/ --instant_ngp_exec /path/to/instant-ngp/build/testbed --n_workers 2
    ```

4. 本地浏览训练结果
    ```bash
    python viewer/dynamic_viewer.py --tiles_dir tiles/
    ```

---
