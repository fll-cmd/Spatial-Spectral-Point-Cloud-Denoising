# -*- coding: utf-8 -*-
"""
ablation_study_final_batch_TESTPC1_fixed.py

逻辑：
1. 原始点云 points_orig 保持不变，用于最终保存
2. 工作点云 points_work = points_orig.copy(); points_work[:,2] *= 100
3. A/B/C 模块全部基于 points_work 计算
4. 保存结果时使用 points_orig[mask]
5. 可视化时单独复制数组，不修改原数组
"""

import os
import time
import numpy as np
import torch
from pytorch3d.ops import knn_points
import pyvista as pv


# ====================== 配置 ======================

CONFIG = {
    "input_dir": r"E:\MATLAB程序\1.3\1.3.2\TESTPC1",
    "output_dir": r"E:\MATLAB程序\1.3\1.3.2\TESTPC1\消融实验",

    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # 可视化参数
    "z_vis_scale": 100.0,       # 仅用于显示
    "max_points_plot": 400000,

    # 模块 A：频域异常度特征
    "k_neighbor_graph": 10,
    "cheb_order": 20,
    "auto_scale_K": 6,
    "auto_scale_alpha": 1.3,
    "sensitivity_factor": 2.5,
    "anomaly_threshold":4000,

    # 模块 B：邻域距离特征
    "k_neighbor_distance": 10,
    "distance_threshold_std": 1.0,

    # 模块 C：法向一致性特征
    "k_neighbor_normal": 20,
    "normal_threshold": 0.99,
}

device = CONFIG["device"]
os.makedirs(CONFIG["output_dir"], exist_ok=True)


# ====================== 读取点云 ======================

def load_pointcloud(file_path):
    try:
        data = np.loadtxt(file_path, delimiter="\t", dtype=np.float64)
    except Exception:
        try:
            data = np.loadtxt(file_path, delimiter=",", dtype=np.float64)
        except Exception:
            data = np.loadtxt(file_path, dtype=np.float64)

    data = np.atleast_2d(data)

    if data.shape[1] < 3:
        raise ValueError(f"文件列数不足 3 列: {file_path}")

    return data[:, :3]


# ====================== KNN ======================

def compute_knn(points, k):
    pts_t = torch.tensor(points, dtype=torch.float32, device=device)

    knn = knn_points(
        pts_t[None],
        pts_t[None],
        K=k + 1
    )

    idx = knn.idx[0, :, 1:]
    d2 = knn.dists[0, :, 1:]

    return idx.cpu().numpy(), d2.cpu().numpy()


# ====================== 图拉普拉斯 ======================

def build_graph_laplacian_from_knn(N, knn_idx, knn_d2):
    k = knn_idx.shape[1]

    row_idx = torch.arange(N, device=device).repeat_interleave(k)
    col_idx = torch.tensor(knn_idx, device=device).reshape(-1)

    weight = torch.exp(
        -torch.tensor(knn_d2, dtype=torch.float32, device=device).reshape(-1)
    )

    W = torch.sparse_coo_tensor(
        torch.stack([row_idx, col_idx]),
        weight,
        size=(N, N)
    )

    W = 0.5 * (W + W.transpose(0, 1)).coalesce()

    deg = torch.sparse.sum(W, dim=1).to_dense()
    deg_positive = deg[deg > 0]

    if deg_positive.numel() == 0:
        raise ValueError("图中所有节点度为 0，请检查点云或 k 邻域参数。")

    deg_safe = torch.where(deg > 0, deg, deg_positive.min())
    deg_inv_sqrt = 1.0 / torch.sqrt(deg_safe + 1e-12)

    def laplacian_fn(x):
        if x.dim() == 1:
            x = x.unsqueeze(1)

        y = deg_inv_sqrt.view(-1, 1) * x
        Wy = torch.sparse.mm(W, y)

        return torch.nan_to_num(
            x - deg_inv_sqrt.view(-1, 1) * Wy,
            nan=0.0
        )

    return laplacian_fn


# ====================== Chebyshev 热核滤波 ======================

def cheb_heat(laplacian_fn, signal, tau, K, lambda_max=2.0):
    if signal.dim() == 1:
        signal = signal.unsqueeze(1)

    quad_M = 50

    j = np.arange(quad_M)
    thetas = (j + 0.5) * np.pi / quad_M
    lam = 0.5 * lambda_max * (np.cos(thetas) + 1.0)

    g_vals = np.exp(-tau * lam)

    coeffs = np.zeros(K + 1, dtype=np.float32)
    base = np.pi / quad_M

    for k in range(K + 1):
        factor = 1.0 / np.pi if k == 0 else 2.0 / np.pi
        coeffs[k] = factor * base * np.sum(g_vals * np.cos(k * thetas))

    c_t = torch.tensor(coeffs, dtype=torch.float32, device=device)

    def apply_L_tilde(v):
        return (2.0 / lambda_max) * laplacian_fn(v) - v

    Tkm2 = signal
    out = c_t[0] * Tkm2

    if K >= 1:
        Tkm1 = apply_L_tilde(Tkm2)
        out += c_t[1] * Tkm1

        for k in range(2, K + 1):
            Tk = 2.0 * apply_L_tilde(Tkm1) - Tkm2
            out += c_t[k] * Tk
            Tkm2, Tkm1 = Tkm1, Tk

    return torch.nan_to_num(out, nan=0.0).squeeze()


# ====================== 模块 A：频域异常度 ======================

def get_wavelet_mask(points_work):
    N = points_work.shape[0]

    knn_idx, knn_d2 = compute_knn(
        points_work,
        CONFIG["k_neighbor_graph"]
    )

    laplacian_fn = build_graph_laplacian_from_knn(
        N,
        knn_idx,
        knn_d2
    )

    d = np.sqrt(knn_d2)
    d_mean = np.mean(d)
    d_med = np.median(d)

    s_min = 0.5 / (d_mean / (d_med + 1e-12))

    scales = np.array(
        [
            s_min * (CONFIG["auto_scale_alpha"] ** k)
            for k in range(CONFIG["auto_scale_K"])
        ],
        dtype=np.float32
    )

    signal = torch.tensor(
        points_work[:, 2],
        dtype=torch.float32,
        device=device
    )

    anomaly_scores = np.zeros(N, dtype=np.float32)

    for i, tau in enumerate(scales):
        c_np = cheb_heat(
            laplacian_fn,
            signal,
            tau,
            CONFIG["cheb_order"]
        ).detach().cpu().numpy()

        local_mean = c_np[knn_idx].mean(axis=1)

        local_std = np.sqrt(
            ((c_np[knn_idx] - local_mean[:, None]) ** 2).mean(axis=1)
        )

        anomaly_scores += (
            local_std
            * (i + 1)
            * CONFIG["sensitivity_factor"]
        )

    if CONFIG["anomaly_threshold"] is None:
        threshold = anomaly_scores.mean() + anomaly_scores.std()
    else:
        threshold = CONFIG["anomaly_threshold"]

    mask_keep = anomaly_scores <= threshold

    return mask_keep, anomaly_scores, threshold


# ====================== 模块 B：邻域距离异常度 ======================

def get_distance_mask(points_work):
    _, d2 = compute_knn(
        points_work,
        CONFIG["k_neighbor_distance"]
    )

    scores = np.sqrt(d2).mean(axis=1)

    mu = scores.mean()
    sigma = scores.std()

    mask_keep = (
        (scores >= mu - CONFIG["distance_threshold_std"] * sigma) &
        (scores <= mu + CONFIG["distance_threshold_std"] * sigma)
    )

    return mask_keep, scores, mu, sigma


# ====================== 模块 C：法向一致性 ======================

def get_normal_mask(points_work):
    N = points_work.shape[0]

    knn_idx, _ = compute_knn(
        points_work,
        CONFIG["k_neighbor_normal"]
    )

    normals = np.zeros((N, 3), dtype=np.float32)

    for i in range(N):
        neighbors = points_work[knn_idx[i]]
        cov = np.cov(neighbors.T)
        _, _, vh = np.linalg.svd(cov)

        n = vh[-1]
        normals[i] = n / (np.linalg.norm(n) + 1e-12)

    scores = np.zeros(N, dtype=np.float32)

    for i in range(N):
        scores[i] = np.abs(
            normals[knn_idx[i]] @ normals[i]
        ).mean()

    mask_keep = scores >= CONFIG["normal_threshold"]

    return mask_keep, scores


# ====================== 可视化与保存 ======================

def sample_for_plot(points, mask_keep):
    N = points.shape[0]
    max_n = CONFIG["max_points_plot"]

    if max_n is not None and N > max_n:
        idx = np.random.choice(N, max_n, replace=False)
        return points[idx], mask_keep[idx]

    return points, mask_keep


def show_and_save_ablation(
        points_orig,
        points_work,
        mask_keep,
        file_name,
        ablation_name):

    pv.close_all()

    # 保存结果：必须使用原始坐标
    denoised_pts = points_orig[mask_keep]

    txt_save_path = os.path.join(
        CONFIG["output_dir"],
        f"{file_name}_{ablation_name}.txt"
    )

    np.savetxt(txt_save_path, denoised_pts, fmt="%.6f")

    # 可视化：使用原始坐标，再仅显示时放大 Z
    pts_plot, mask_plot = sample_for_plot(points_orig, mask_keep)

    pts_vis = pts_plot.copy()
    pts_vis[:, 2] *= CONFIG["z_vis_scale"]

    plotter = pv.Plotter(
        shape=(1, 3),
        title=f"{file_name} - {ablation_name}",
        off_screen=True
    )

    plotter.subplot(0, 0)
    plotter.add_text("Original Cloud", font_size=10)
    plotter.add_points(
        pts_vis,
        color="deepskyblue",
        point_size=2
    )

    plotter.subplot(0, 1)
    plotter.add_text("Anomaly Detection", font_size=10)

    colors = np.full((pts_vis.shape[0], 3), 0.7)
    colors[~mask_plot] = [1.0, 0.0, 0.0]

    plotter.add_points(
        pts_vis,
        scalars=colors,
        rgb=True,
        point_size=3
    )

    plotter.subplot(0, 2)
    plotter.add_text("Denoised Cloud", font_size=10)
    plotter.add_points(
        pts_vis[mask_plot],
        color="lawngreen",
        point_size=2
    )

    plotter.link_views()
    plotter.camera_position = "yz"

    img_save_path = os.path.join(
        CONFIG["output_dir"],
        f"{file_name}_{ablation_name}_view.png"
    )

    plotter.show(screenshot=img_save_path)
    plotter.close()

    print(
        f"    {file_name}_{ablation_name}: "
        f"{points_orig.shape[0]} -> {denoised_pts.shape[0]} "
        f"| 删除 {points_orig.shape[0] - denoised_pts.shape[0]} 点"
    )


# ====================== 单文件处理 ======================

def process_one_file(input_path):
    start_time = time.time()

    file_name = os.path.splitext(
        os.path.basename(input_path)
    )[0]

    print("\n" + "=" * 70)
    print(f"正在处理: {input_path}")
    print(f"输出目录: {CONFIG['output_dir']}")

    # 原始坐标
    points_orig = load_pointcloud(input_path)

    # 工作坐标：仅用于算法处理
    points_work = points_orig.copy()
    points_work[:, 2] *= 200.0

    print(f"数据加载成功: {points_orig.shape[0]} 点")
    print("处理坐标: X, Y 保持不变，Z × 100")
    print("保存坐标: 使用原始 X, Y, Z")

    print("[Step 1/3] 计算模块 A：频域异常度特征...")
    mask_A, score_A, threshold_A = get_wavelet_mask(points_work)
    print(
        f"    A 阈值 = {threshold_A:.6f}, "
        f"保留 {np.sum(mask_A)} / {len(mask_A)}"
    )

    print("[Step 2/3] 计算模块 B：邻域距离特征...")
    mask_B, score_B, mu_B, sigma_B = get_distance_mask(points_work)
    print(
        f"    B 均值 = {mu_B:.6f}, 标准差 = {sigma_B:.6f}, "
        f"保留 {np.sum(mask_B)} / {len(mask_B)}"
    )

    print("[Step 3/3] 计算模块 C：法向一致性特征...")
    mask_C, score_C = get_normal_mask(points_work)
    print(
        f"    C 阈值 = {CONFIG['normal_threshold']:.6f}, "
        f"保留 {np.sum(mask_C)} / {len(mask_C)}"
    )

    ablations = {
        "1_A_only": mask_A,
        "2_B_only": mask_B,
        "3_C_only": mask_C,
        "4_BC_noA": mask_B & mask_C,
        "5_AC_noB": mask_A & mask_C,
        "6_AB_noC": mask_A & mask_B,
        "7_Full_ABC": mask_A & mask_B & mask_C,
    }

    print("正在生成消融实验结果...")

    for ablation_name, mask in ablations.items():
        show_and_save_ablation(
            points_orig,
            points_work,
            mask,
            file_name,
            ablation_name
        )

    print(f"单文件完成，耗时: {time.time() - start_time:.2f}s")


# ====================== 文件列表 ======================

def build_file_list():
    file_list = []

    for group_id in [1, 2, 3]:
        for sample_id in range(1, 11):
            file_name = f"{group_id}-{sample_id}.txt"

            file_path = os.path.join(
                CONFIG["input_dir"],
                file_name
            )

            file_list.append(file_path)

    return file_list


# ====================== 主函数 ======================

def main():
    total_start = time.time()

    file_list = build_file_list()

    existing_files = []
    missing_files = []

    for fp in file_list:
        if os.path.exists(fp):
            existing_files.append(fp)
        else:
            missing_files.append(fp)

    if missing_files:
        print("以下文件不存在，将跳过：")
        for fp in missing_files:
            print(fp)

    if not existing_files:
        raise FileNotFoundError(
            "没有找到任何待处理文件，请检查文件名是否为 1-1.txt 到 3-10.txt。"
        )

    print(f"共检测到 {len(existing_files)} 个待处理文件")
    print(f"统一输出目录: {CONFIG['output_dir']}")

    for i, fp in enumerate(existing_files, start=1):
        print(f"\n[{i}/{len(existing_files)}]")
        process_one_file(fp)

    print("\n" + "=" * 70)
    print("全部 TESTPC1 消融实验处理完成")
    print(f"总耗时: {time.time() - total_start:.2f}s")
    print(f"总输出目录: {CONFIG['output_dir']}")
    print("=" * 70)


if __name__ == "__main__":
    main()