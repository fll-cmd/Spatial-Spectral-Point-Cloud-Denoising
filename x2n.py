# -*- coding: utf-8 -*-
"""
ablation_study_cylinder_batch.py

圆柱面点云去噪消融实验批量版
输入:
    E:\MATLAB程序\1.3\1.3.2\TESTPC2
    1-1.txt 到 1-10.txt
    2-1.txt 到 2-10.txt
    3-1.txt 到 3-10.txt

输出:
    E:\MATLAB程序\1.3\1.3.2\TESTPC2\消融实验

输出文件示例:
    1-1_1_A_only.txt
    1-1_1_A_only_Anomalies.png
    1-1_1_A_only_Comparison.png
"""

import os
import time
import numpy as np
import torch
from pytorch3d.ops import knn_points
import pyvista as pv


# ====================== 配置 ======================

CONFIG = {
    "input_dir": r"E:\MATLAB程序\1.3\1.3.2\TESTPC2",
    "output_dir": r"E:\MATLAB程序\1.3\1.3.2\TESTPC2\消融实验",

    "cylinder_radius": 20.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "z_vis_scale": 1.0,
    "max_points_plot": 400000,

    # 模块 A：图小波
    "k_neighbor_graph": 3,
    "cheb_order": 20,
    "auto_scale_K": 10,
    "auto_scale_alpha": 1.05,
    "sensitivity_factor": 2.5,
    "anomaly_threshold": 670000000,

    # 模块 B：邻域距离
    "k_neighbor_distance": 20,
    "distance_threshold_std": 2.0,

    # 模块 C：法向一致性
    "k_neighbor_normal": 5,
    "normal_threshold": 0.98,
}

device = CONFIG["device"]
os.makedirs(CONFIG["output_dir"], exist_ok=True)


# ====================== 读取点云 ======================

def load_pointcloud(file_path):
    try:
        data = np.loadtxt(file_path, delimiter="\t", dtype=np.float64)
    except:
        try:
            data = np.loadtxt(file_path, delimiter=",", dtype=np.float64)
        except:
            data = np.loadtxt(file_path, dtype=np.float64)

    data = np.atleast_2d(data)

    if data.shape[1] < 3:
        raise ValueError(f"文件列数不足 3 列: {file_path}")

    return data[:, :3]


# ====================== 基础工具函数 ======================

def compute_knn(points, k):
    pts_t = torch.tensor(points, dtype=torch.float32, device=device)

    knn = knn_points(
        pts_t[None],
        pts_t[None],
        K=k + 1
    )

    return (
        knn.idx[0, :, 1:].cpu().numpy(),
        knn.dists[0, :, 1:].cpu().numpy()
    )


def compute_cylinder_surface_distance(points, R):
    r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    return r - R


def build_graph_laplacian(N, knn_idx, knn_d2):
    k = knn_idx.shape[1]

    row = torch.arange(N, device=device).repeat_interleave(k)
    col = torch.tensor(knn_idx, device=device).reshape(-1)

    w = torch.exp(
        -torch.tensor(knn_d2, device=device).reshape(-1)
    )

    W = torch.sparse_coo_tensor(
        torch.stack([row, col]),
        w,
        (N, N)
    )

    W = 0.5 * (W + W.T).coalesce()

    deg = torch.sparse.sum(W, dim=1).to_dense()
    deg_positive = deg[deg > 0]

    if deg_positive.numel() == 0:
        raise ValueError("图中所有节点度为 0，请检查点云或 k 邻域参数。")

    deg_safe = torch.where(deg > 0, deg, deg_positive.min())
    deg_inv = 1.0 / torch.sqrt(deg_safe + 1e-12)

    def lap(x):
        if x.ndim == 1:
            x = x[:, None]

        y = deg_inv[:, None] * x
        Wy = torch.sparse.mm(W, y)

        return torch.nan_to_num(
            x - deg_inv[:, None] * Wy,
            nan=0.0
        )

    return lap


def cheb_heat(lap, signal, tau, K):
    if signal.ndim == 1:
        signal = signal[:, None]

    T0 = signal.clone()
    out = T0.clone()

    if K > 0:
        T1 = lap(T0)
        T1 = torch.clamp(T1, -1e6, 1e6)

        out = out + T1

        for _ in range(2, K + 1):
            T2 = 2 * lap(T1) - T0
            T2 = torch.clamp(T2, -1e6, 1e6)

            out = out + T2
            T0, T1 = T1, T2

    return torch.nan_to_num(out.squeeze(), nan=0.0)


# ====================== Mask 计算 ======================

def get_wavelet_mask(points):
    print("    [A] 计算图小波...")

    knn_idx_A, knn_d2_A = compute_knn(
        points,
        CONFIG["k_neighbor_graph"]
    )

    lap = build_graph_laplacian(
        points.shape[0],
        knn_idx_A,
        knn_d2_A
    )

    d = np.sqrt(knn_d2_A)
    base = np.median(d)

    scales = np.array(
        [
            base * (CONFIG["auto_scale_alpha"] ** i)
            for i in range(CONFIG["auto_scale_K"])
        ],
        dtype=np.float32
    )

    dist_signal = compute_cylinder_surface_distance(
        points,
        CONFIG["cylinder_radius"]
    )

    signal = torch.tensor(
        dist_signal,
        dtype=torch.float32,
        device=device
    )

    wave_score = np.zeros(points.shape[0], np.float32)

    for i, s in enumerate(scales):
        c = cheb_heat(
            lap,
            signal,
            s,
            CONFIG["cheb_order"]
        )

        c_np = np.nan_to_num(c.cpu().numpy())

        local_mean = np.mean(c_np[knn_idx_A], axis=1)

        local_std = np.sqrt(
            np.mean(
                (c_np[knn_idx_A] - local_mean[:, None]) ** 2,
                axis=1
            )
        )

        wave_score += (
            local_std
            * (i + 1)
            * CONFIG["sensitivity_factor"]
        )

    return wave_score <= CONFIG["anomaly_threshold"]


def get_distance_mask(points):
    print("    [B] 计算邻域距离...")

    _, d2_B = compute_knn(
        points,
        CONFIG["k_neighbor_distance"]
    )

    dist_score = np.sqrt(d2_B).mean(axis=1)

    return (
        np.abs(dist_score - dist_score.mean())
        <= CONFIG["distance_threshold_std"] * dist_score.std()
    )


def get_normal_mask(points):
    print("    [C] 计算法向一致性...")

    N = points.shape[0]

    normals = np.zeros_like(points)

    knn_idx_C, _ = compute_knn(
        points,
        CONFIG["k_neighbor_normal"]
    )

    for i in range(N):
        nb = points[knn_idx_C[i]]
        _, _, v = np.linalg.svd(np.cov(nb.T))
        n = v[-1]
        normals[i] = n / (np.linalg.norm(n) + 1e-12)

    norm_score = np.zeros(N, np.float32)

    for i in range(N):
        norm_score[i] = np.abs(
            np.dot(normals[knn_idx_C[i]], normals[i])
        ).mean()

    return norm_score >= CONFIG["normal_threshold"]
def amplify_cylinder_radial_deviation(points, R, radial_scale=100.0):
    """
    沿圆柱径向放大点云偏差：
    原半径 r = sqrt(x^2 + y^2)
    原偏差 dr = r - R
    新半径 r_new = R + radial_scale * dr

    z 不变。
    """
    pts = points.copy()

    x = pts[:, 0]
    y = pts[:, 1]

    r = np.sqrt(x**2 + y**2)
    r_safe = np.where(r > 1e-12, r, 1e-12)

    dr = r - R
    r_new = R + radial_scale * dr

    pts[:, 0] = x / r_safe * r_new
    pts[:, 1] = y / r_safe * r_new

    return pts

# ====================== 可视化与保存 ======================

def sample_for_plot(points, mask):
    N = points.shape[0]
    max_n = CONFIG["max_points_plot"]

    if max_n is not None and N > max_n:
        idx = np.random.choice(N, max_n, replace=False)
        return points[idx], mask[idx]

    return points, mask


def show_anomalies(points, mask, save_path):
    pv.close_all()

    pts, mask_plot = sample_for_plot(points, mask)

    pts_vis = pts.copy()
    pts_vis[:, 2] *= CONFIG["z_vis_scale"]

    colors = np.tile(
        np.array([0.7, 0.7, 0.7]),
        (pts_vis.shape[0], 1)
    )

    colors[~mask_plot] = [1.0, 0.0, 0.0]

    pl = pv.Plotter(off_screen=True)
    pl.add_points(
        pts_vis,
        scalars=colors,
        rgb=True,
        point_size=3,
        render_points_as_spheres=True
    )

    pl.camera_position = "yz"
    pl.show(screenshot=save_path)
    pl.close()


def show_comparison(original_points, denoised_points, save_path):
    pv.close_all()

    if denoised_points.shape[0] == 0:
        return

    orig_plot, _ = sample_for_plot(
        original_points,
        np.ones(original_points.shape[0], dtype=bool)
    )

    clean_plot = denoised_points

    if (
        CONFIG["max_points_plot"] is not None
        and clean_plot.shape[0] > CONFIG["max_points_plot"]
    ):
        idx = np.random.choice(
            clean_plot.shape[0],
            CONFIG["max_points_plot"],
            replace=False
        )
        clean_plot = clean_plot[idx]

    orig_vis = orig_plot.copy()
    clean_vis = clean_plot.copy()

    orig_vis[:, 2] *= CONFIG["z_vis_scale"]
    clean_vis[:, 2] *= CONFIG["z_vis_scale"]

    pl = pv.Plotter(off_screen=True)

    pl.add_points(
        orig_vis,
        color=(0.6, 0.6, 0.6),
        point_size=2,
        opacity=0.5
    )

    pl.add_points(
        clean_vis,
        color=(0.0, 0.4, 1.0),
        point_size=3
    )

    pl.camera_position = "yz"
    pl.show(screenshot=save_path)
    pl.close()


def save_and_plot_result(points, mask, file_name, ablation_name):
    clean_points = points[mask]

    txt_path = os.path.join(
        CONFIG["output_dir"],
        f"{file_name}_{ablation_name}.txt"
    )

    np.savetxt(txt_path, clean_points, fmt="%.6f")

    anomaly_png = os.path.join(
        CONFIG["output_dir"],
        f"{file_name}_{ablation_name}_Anomalies.png"
    )

    comparison_png = os.path.join(
        CONFIG["output_dir"],
        f"{file_name}_{ablation_name}_Comparison.png"
    )

    show_anomalies(points, mask, anomaly_png)
    show_comparison(points, clean_points, comparison_png)

    print(
        f"    {file_name}_{ablation_name}: "
        f"{points.shape[0]} -> {clean_points.shape[0]}"
    )


# ====================== 单文件处理 ======================

def process_one_file(input_path):
    start = time.time()

    file_name = os.path.splitext(
        os.path.basename(input_path)
    )[0]

    print("\n" + "=" * 70)
    print(f"正在处理: {input_path}")
    print(f"输出目录: {CONFIG['output_dir']}")

    points = load_pointcloud(input_path)
    points = amplify_cylinder_radial_deviation(
        points,
        CONFIG["cylinder_radius"],
        radial_scale=100.0
    )
    #points_work = points.copy()
    #points_work[:, 2] *= 100.0
    print(f"原始点云数量: {points.shape[0]}")

    mask_A = get_wavelet_mask(points)
    mask_B = get_distance_mask(points)
    mask_C = get_normal_mask(points)

    ablations = {
        "1_A_only": mask_A,
        "2_B_only": mask_B,
        "3_C_only": mask_C,
        "4_BC_noA": mask_B & mask_C,
        "5_AC_noB": mask_A & mask_C,
        "6_AB_noC": mask_A & mask_B,
        "7_Full_ABC": mask_A & mask_B & mask_C,
    }
    # 在 mask_all = mask_wave & mask_dist & mask_norm 之前插入

    print(f"\n[诊断] mask_A 保留点数: {mask_A.sum()}")
    print(f"[诊断] mask_B 保留点数: {mask_B.sum()}")
    print(f"[诊断] mask_C 保留点数: {mask_C.sum()}")
    print(f"[诊断] points dtype: {points.dtype}")
    print(f"[诊断] points[:3]: {points[:3]}")
    for ablation_name, mask in ablations.items():
        save_and_plot_result(
            points,
            mask,
            file_name,
            ablation_name
        )

    print(f"单文件完成，耗时: {time.time() - start:.2f}s")


# ====================== 批处理入口 ======================

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
    print("全部 TESTPC2 消融实验处理完成")
    print(f"总耗时: {time.time() - total_start:.2f}s")
    print(f"总输出目录: {CONFIG['output_dir']}")
    print("=" * 70)


if __name__ == "__main__":
    main()