import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from metanet_model import run_metanet


# python
def plot_scenario(res, lanes, title_suffix=""):
    import re
    import os

    time = res["time"]
    densities = res["densities"]
    speeds = res["speeds"]
    flows = res["flows"]
    queue_ramp = res["queue_ramp"]
    queue_main = res["queue_main"]

    # Build output filename base: extract scenario letter and detect ALINEA usage.
    title_up = (title_suffix or "").upper()
    m = re.search(r"SCENARIO\s*([A-Z])", title_up)
    scenario_letter = m.group(1) if m else (
        res.get("scenario", "unknown") if isinstance(res.get("scenario"), str) else "unknown")
    if "NO" in title_up:
        mode = ""
    elif "ALINEA" in title_up or "K_I" in title_up:
        mode = "alinea"
    base_name = f"scenario_{scenario_letter.lower()}_{mode}_metanet".replace("__", "_")

    # Ensure outputs directory exists
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    time_s = time * 3600.0
    cells = np.arange(1, len(lanes) + 1)
    T_mesh, C_mesh = np.meshgrid(time_s, cells)

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)

    for i in range(len(lanes)):
        axes[0].plot(time_s, densities[:, i], label=f"Cell {i + 1}")
    axes[0].set_ylabel("Density [veh/km/lane]")
    axes[0].set_title(f"Density {title_suffix}")
    axes[0].legend(loc="upper right")

    for i in range(len(lanes)):
        axes[1].plot(time_s, speeds[:, i], label=f"Cell {i + 1}")
    axes[1].set_ylabel("Speed [km/h]")
    axes[1].set_title("Speed")

    for i in range(len(lanes)):
        axes[2].plot(time_s, flows[:, i], label=f"Cell {i + 1}")
    axes[2].set_ylabel("Flow [veh/h]")
    axes[2].set_title("Flow")

    axes[3].plot(time_s, queue_ramp, label="Ramp queue")
    axes[3].plot(time_s, queue_main, label="Mainline queue")
    axes[3].set_xlabel("Time [s]")
    axes[3].set_ylabel("Queue [veh]")
    axes[3].set_title("Queues")
    axes[3].legend(loc="upper right")
    axes[3].set_ylim(bottom=0)

    plt.tight_layout()
    panels_path = f"outputs/{base_name}.png"
    plt.savefig(panels_path, dpi=300, bbox_inches="tight")
    plt.show()

    # 2D density space-time plot (cells vs time) - saved separately
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    # densities shape: (time_steps, num_cells) -> transpose for cells on y-axis
    pcm = ax.pcolormesh(time_s, cells, densities.T, shading="auto", cmap="turbo")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Cell")
    ax.set_title(f"Density (2D) {title_suffix}")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Density [veh/km/lane]")
    density2d_path = f"outputs/{base_name}_density2d.png"
    plt.savefig(density2d_path, dpi=300, bbox_inches="tight")
    plt.show()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(T_mesh, C_mesh, densities.T, cmap="turbo", edgecolor="none")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Cell")
    ax.set_zlabel("Density [veh/km/lane]")
    ax.set_title(f"3D Density {title_suffix}")
    fig.colorbar(surf, shrink=0.5, aspect=10)
    density3d_path = f"outputs/{base_name}_density3d.png"
    plt.savefig(density3d_path, dpi=300, bbox_inches="tight")
    plt.show()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(T_mesh, C_mesh, speeds.T, cmap="turbo", edgecolor="none")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Cell")
    ax.set_zlabel("Speed [km/h]")
    ax.set_title(f"3D Speed {title_suffix}")
    fig.colorbar(surf, shrink=0.5, aspect=10)
    speed3d_path = f"outputs/{base_name}_speed3d.png"
    plt.savefig(speed3d_path, dpi=300, bbox_inches="tight")
    plt.show()


def scan_K(d_main, d_ramp, lanes, measured_cell, K_min=0.5, K_max=20.0, n_K=1000):
    K_values = np.linspace(K_min, K_max, n_K)
    vht_values = np.zeros_like(K_values)
    avg_speed_values = np.zeros_like(K_values)

    for idx, K_I in enumerate(K_values):
        res = run_metanet(d_main, d_ramp, lanes, K_I=K_I, measured_cell=measured_cell)
        vht_values[idx] = res["vht"]
        avg_speed_values[idx] = res["avg_speed"]

    return K_values, vht_values, avg_speed_values


if __name__ == "__main__":
    lanes_A = np.full(6, 3.0)
    lanes_B = np.full(6, 3.0)
    lanes_C = np.array([3.0, 3.0, 3.0, 3.0, 1.0, 3.0])

    dA_main, dA_ramp = 4000.0, 2000.0
    dB_main, dB_ramp = 4000.0, 2500.0
    dC_main, dC_ramp = 1500.0, 1500.0

    res_A = run_metanet(dA_main, dA_ramp, lanes_A, K_I=0.0, measured_cell=None, lane_drop_cell=None)
    print("Scenario A (no ALINEA)")
    print(f"  VKT = {res_A['vkt']:.1f} veh·km")
    print(f"  VHT = {res_A['vht']:.1f} veh·h")
    print(f"  Avg speed = {res_A['avg_speed']:.1f} km/h")

    res_B = run_metanet(dB_main, dB_ramp, lanes_B, K_I=0.0, measured_cell=None, lane_drop_cell=None)
    print("Scenario B (no ALINEA)")
    print(f"  VKT = {res_B['vkt']:.1f} veh·km")
    print(f"  VHT = {res_B['vht']:.1f} veh·h")
    print(f"  Avg speed = {res_B['avg_speed']:.1f} km/h")

    res_C = run_metanet(dC_main, dC_ramp, lanes_C, K_I=0.0, measured_cell=None, lane_drop_cell=3)
    print("Scenario C (no ALINEA)")
    print(f"  VKT = {res_C['vkt']:.1f} veh·km")
    print(f"  VHT = {res_C['vht']:.1f} veh·h")
    print(f"  Avg speed = {res_C['avg_speed']:.1f} km/h")

    plot_scenario(res_A, lanes_A, title_suffix="– Scenario A, no ALINEA")
    plot_scenario(res_B, lanes_B, title_suffix="– Scenario B, no ALINEA")
    plot_scenario(res_C, lanes_C, title_suffix="– Scenario C, no ALINEA")

    K_B, vht_B, avg_B = scan_K(dB_main, dB_ramp, lanes_B, measured_cell=2)
    idx_B = np.argmin(vht_B)
    K_opt_B = K_B[idx_B]
    print(f"Scenario B: K_opt = {K_opt_B:.2f}, VHT_min = {vht_B[idx_B]:.1f} veh·h")

    plt.figure(figsize=(7, 4))
    plt.plot(K_B, vht_B)
    plt.xlabel("K_I")
    plt.ylabel("VHT [veh·h]")
    plt.title("Scenario B: VHT vs K_I")
    plt.grid(True)
    plt.show()

    best_B = run_metanet(dB_main, dB_ramp, lanes_B, K_I=K_opt_B, measured_cell=2, lane_drop_cell=None)
    print("Scenario B with ALINEA (K_opt)")
    print(f"  VKT = {best_B['vkt']:.1f} veh·km")
    print(f"  VHT = {best_B['vht']:.1f} veh·h")
    print(f"  Avg speed = {best_B['avg_speed']:.1f} km/h")
    plot_scenario(best_B, lanes_B, title_suffix=f"– Scenario B, K_I = {K_opt_B:.2f}")

    K_C, vht_C, avg_C = scan_K(dC_main, dC_ramp, lanes_C, measured_cell=4)
    idx_C = np.argmin(vht_C)
    K_opt_C = K_C[idx_C]
    print(f"Scenario C: K_opt = {K_opt_C:.2f}, VHT_min = {vht_C[idx_C]:.1f} veh·h")

    plt.figure(figsize=(7, 4))
    plt.plot(K_C, vht_C)
    plt.xlabel("K_I")
    plt.ylabel("VHT [veh·h]")
    plt.title("Scenario C: VHT vs K_I")
    plt.grid(True)
    plt.show()

    best_C = run_metanet(dC_main, dC_ramp, lanes_C, K_I=K_opt_C, measured_cell=4, lane_drop_cell=3)
    print("Scenario C with ALINEA (K_opt)")
    print(f"  VKT = {best_C['vkt']:.1f} veh·km")
    print(f"  VHT = {best_C['vht']:.1f} veh·h")
    print(f"  Avg speed = {best_C['avg_speed']:.1f} km/h")
    plot_scenario(best_C, lanes_C, title_suffix=f"– Scenario C, K_I = {K_opt_C:.2f}")
