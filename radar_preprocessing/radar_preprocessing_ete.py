import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
from tqdm import tqdm
import json
from nuscenes.utils.data_classes import RadarPointCloud

# CONFIG
INPUT_DIR = "/home/omar_ben_emad/3d_object_detection_and_tracking/data/v1.0-mini/sweeps/RADAR_FRONT"
OUTPUT_DIR = "/home/omar_ben_emad/3d_object_detection_and_tracking/output/radar_vis_out"
POLAR_DIR = os.path.join(OUTPUT_DIR, "polar_plot_visualizations")
VEL_DIR = os.path.join(OUTPUT_DIR, "velocity_plot_visualizations")
DATA_DIR = os.path.join(OUTPUT_DIR, "polar_data")

os.makedirs(POLAR_DIR, exist_ok=True)
os.makedirs(VEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

FPS = 30


def load_radar_pcd(file_path):

    radar_pc = RadarPointCloud.from_file(file_path)
    points = radar_pc.points  # shape (18, N)

    if points.shape[1] == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    x = points[0, :]
    y = points[1, :]
    vx = points[8, :]   # vx_comp
    vy = points[9, :]   # vy_comp
    dyn_prop = points[3, :]
    valid = dyn_prop == 0   # moving targets only

    x = x[valid]
    y = y[valid]
    vx = vx[valid]
    vy = vy[valid]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    rho_safe = np.where(rho == 0, 1e-6, rho)

    phi_dot = (x * vy - y * vx) / (rho_safe**2)

    return rho, phi, phi_dot, x, y, vx, vy

def plot_polar(rho, phi, save_path):
    plt.figure()
    ax = plt.subplot(111, projection='polar')
    ax.scatter(phi, rho, s=5)
    ax.set_title("Polar Plot")
    plt.savefig(save_path)
    plt.close()


def plot_velocity(phi, phi_dot, save_path):
    plt.figure()
    plt.scatter(phi, phi_dot, s=5)
    plt.xlabel("phi")
    plt.ylabel("phi_dot")
    plt.title("Velocity Plot")
    plt.savefig(save_path)
    plt.close()

def create_gif(folder, images, output_gif, fps=2):

    list_file = os.path.join(folder, "images.txt")
    duration = 1.0 / fps

    with open(list_file, "w") as f:
        for img in images:
            f.write(f"file '{img}'\n")
            f.write(f"duration {duration}\n")
        f.write(f"file '{images[-1]}'\n")

    palette_path = os.path.join(folder, "palette.png")
    gif_path = os.path.join(folder, output_gif)

    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-vf", "palettegen",
        palette_path
    ], check=True)

    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-i", palette_path,
        "-lavfi", "paletteuse",
        gif_path
    ], check=True)

    print(f"GIF saved at: {gif_path}")

def process_all_sweeps():

    files = sorted([
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.endswith(".pcd")
    ])

    polar_images = []
    velocity_images = []
    all_frames = [] 

    for idx, file_path in enumerate(tqdm(files, desc="Processing radar sweeps")):

        rho, phi, phi_dot, x, y, vx, vy = load_radar_pcd(file_path)
        radar_points = []
        for i in range(len(rho)):
            radar_points.append({
                "rho": float(rho[i]),
                "phi": float(phi[i]),
                "phi_dot": float(phi_dot[i]),
                "x": float(x[i]),
                "y": float(y[i]),
                "vx": float(vx[i]),
                "vy": float(vy[i])
            })

        frame_data = {
            "frame_id": idx,
            "num_points": len(rho),
            "radar_points": radar_points
        }

        all_frames.append(frame_data)  
        polar_img = os.path.join(POLAR_DIR, f"polar_plot_{idx:04d}.png")
        velocity_img = os.path.join(VEL_DIR, f"velocity_plot_{idx:04d}.png")

        plot_polar(rho, phi, polar_img)
        plot_velocity(phi, phi_dot, velocity_img)

        polar_images.append(polar_img)
        velocity_images.append(velocity_img)
    output_json = os.path.join(DATA_DIR, "radar_all_frames.json")
    with open(output_json, "w") as f:
        json.dump({"frames": all_frames}, f, indent=4)

    print(f"Saved combined JSON: {output_json}")

    create_gif(POLAR_DIR, polar_images, "polar_plot.gif", FPS)
    create_gif(VEL_DIR, velocity_images, "velocity_plot.gif", FPS)
if __name__ == "__main__":
    process_all_sweeps()