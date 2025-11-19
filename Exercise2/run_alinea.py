import concurrent.futures
import os

import matplotlib.pyplot as plt
import numpy as np

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
    title_low = (title_suffix or "").lower()
    m = re.search(r"SCENARIO\s*([A-Z])", title_up)
    scenario_letter = m.group(1) if m else (
        res.get("scenario", "unknown") if isinstance(res.get("scenario"), str) else "unknown")

    # Detect ALINEA/K_I while handling negations like "no alinea", "without alinea", etc.
    has_alinea = bool(re.search(r"\balinea\b|\bk_i\b", title_low))
    negated = bool(
        re.search(r"\b(no|not|without)\b\s*(alinea|\bk_i\b)|\b(alinea|\bk_i\b)\s*(no|not|without)\b", title_low))
    mode = "alinea" if has_alinea and not negated else ""

    # Build base name and collapse multiple underscores
    if mode:
        base_name = f"scenario_{scenario_letter.lower()}_{mode}_metanet"
    else:
        base_name = f"scenario_{scenario_letter.lower()}_metanet"
    base_name = re.sub(r"_+", "_", base_name)

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


def _run_single(args):
    i, j, Ki, Kp, d_main, d_ramp, lanes, measured_cell = args
    res = run_metanet(d_main, d_ramp, lanes, K_I=Ki, K_P=Kp, measured_cell=measured_cell)
    return i, j, res["vht"], res["avg_speed"]


def scan_K(
        d_main, d_ramp, lanes, measured_cell,
        Ki_min=0.0, Ki_max=20.0, n_Ki=50,
        Kp_min=0.0, Kp_max=10.0, n_Kp=25,
        n_jobs=None,
        coarse_refine=False,
        coarse_factor=5,
        refine_frac=0.2
):
    """
    Efficient scan over K_I (rows) and K_P (cols).
    - Uses parallel execution (n_jobs default = number of CPUs).
    - If coarse_refine=True, does a coarse scan first (grid reduced by coarse_factor),
      then refines around the best coarse cell using refine_frac window.
    Returns:
      Ki_values, Kp_values, vht_grid, avg_speed_grid, best
    """
    n_jobs = n_jobs or os.cpu_count() or 1

    def _parallel_scan(Ki_values, Kp_values):
        vht_grid = np.full((len(Ki_values), len(Kp_values)), np.nan)
        avg_speed_grid = np.full((len(Ki_values), len(Kp_values)), np.nan)

        tasks = []
        for i, Ki in enumerate(Ki_values):
            for j, Kp in enumerate(Kp_values):
                tasks.append((i, j, Ki, Kp, d_main, d_ramp, lanes, measured_cell))

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for i, j, vht, avg in ex.map(_run_single, tasks):
                vht_grid[i, j] = vht
                avg_speed_grid[i, j] = avg

        return vht_grid, avg_speed_grid

    Ki_values = np.linspace(Ki_min, Ki_max, n_Ki)
    Kp_values = np.linspace(Kp_min, Kp_max, n_Kp)

    if not coarse_refine:
        vht_grid, avg_speed_grid = _parallel_scan(Ki_values, Kp_values)
    else:
        # coarse scan
        ncKi = max(2, n_Ki // coarse_factor)
        ncKp = max(2, n_Kp // coarse_factor)
        Ki_coarse = np.linspace(Ki_min, Ki_max, ncKi)
        Kp_coarse = np.linspace(Kp_min, Kp_max, ncKp)
        vht_coarse, avg_coarse = _parallel_scan(Ki_coarse, Kp_coarse)

        idx_flat = np.nanargmin(vht_coarse)
        i_c, j_c = np.unravel_index(idx_flat, vht_coarse.shape)
        Ki_best_c = Ki_coarse[i_c]
        Kp_best_c = Kp_coarse[j_c]

        # refine window around coarse best
        Ki_span = max((Ki_max - Ki_min) * refine_frac, (Ki_coarse[1] - Ki_coarse[0]))
        Kp_span = max((Kp_max - Kp_min) * refine_frac, (Kp_coarse[1] - Kp_coarse[0]))

        Ki_lo = max(Ki_min, Ki_best_c - Ki_span)
        Ki_hi = min(Ki_max, Ki_best_c + Ki_span)
        Kp_lo = max(Kp_min, Kp_best_c - Kp_span)
        Kp_hi = min(Kp_max, Kp_best_c + Kp_span)

        Ki_values = np.linspace(Ki_lo, Ki_hi, n_Ki)
        Kp_values = np.linspace(Kp_lo, Kp_hi, n_Kp)
        vht_grid, avg_speed_grid = _parallel_scan(Ki_values, Kp_values)

    idx_flat = np.nanargmin(vht_grid)
    i_best, j_best = np.unravel_index(idx_flat, vht_grid.shape)
    best = {
        "Ki": Ki_values[i_best],
        "Kp": Kp_values[j_best],
        "vht": vht_grid[i_best, j_best],
        "avg_speed": avg_speed_grid[i_best, j_best],
        "i_idx": i_best,
        "j_idx": j_best,
    }

    return Ki_values, Kp_values, vht_grid, avg_speed_grid, best


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

    Ki_vals, Kp_vals, vht_grid, avg_grid, best = scan_K(dB_main, dB_ramp, lanes_B, measured_cell=4)

    # If scan was effectively 1D (single K_P), take that column; otherwise take the column at best K_P
    if len(Kp_vals) == 1:
        K_B = Ki_vals
        vht_B = vht_grid[:, 0]
        avg_B = avg_grid[:, 0]
        j_best = 0
    else:
        j_best = best["j_idx"]
        K_B = Ki_vals
        vht_B = vht_grid[:, j_best]
        avg_B = avg_grid[:, j_best]

    # pick optimal K_I from the 1D slice
    idx_B = int(np.nanargmin(vht_B))
    K_opt_B = K_B[idx_B]
    print(f"Scenario B: K_opt = {K_opt_B:.2f}, VHT_min = {vht_B[idx_B]:.1f} veh·h")

    plt.figure(figsize=(7, 4))
    plt.plot(K_B, vht_B, marker='o')
    plt.xlabel("K_I")
    plt.ylabel("VHT [veh·h]")
    plt.title("Scenario B: VHT vs K_I")
    plt.grid(True)
    plt.show()

    # choose K_P to use in the final simulation (use the K_P from the best column or 0 if only one)
    Kp_for_sim = float(Kp_vals[j_best]) if len(Kp_vals) >= 1 else 0.0
    best_B = run_metanet(dB_main, dB_ramp, lanes_B, K_I=K_opt_B, K_P=Kp_for_sim, measured_cell=4, lane_drop_cell=None)
    print("Scenario B with ALINEA (K_opt)")
    print(f"  VKT = {best_B['vkt']:.1f} veh·km")
    print(f"  VHT = {best_B['vht']:.1f} veh·h")
    print(f"  Avg speed = {best_B['avg_speed']:.1f} km/h")
    plot_scenario(best_B, lanes_B, title_suffix=f"– Scenario B, K_I = {K_opt_B:.2f}, K_P = {Kp_for_sim:.2f}")

    # perform 2D scan (Ki x Kp)
    Ki_vals, Kp_vals, vht_grid, avg_grid, best = scan_K(dC_main, dC_ramp, lanes_C, measured_cell=5)

    # print best pair found
    print(f"Scenario C: K_I_opt = {best['Ki']:.4f}, K_P_opt = {best['Kp']:.4f}, VHT_min = {best['vht']:.1f} veh·h")

    # If you want a 2D heatmap of VHT over (K_I, K_P)
    plt.figure(figsize=(7, 5))
    # imshow expects [rows, cols] = [len(Ki), len(Kp)]
    extent = [Kp_vals[0], Kp_vals[-1], Ki_vals[-1], Ki_vals[0]]  # flip y for natural orientation
    plt.imshow(vht_grid, aspect='auto', extent=extent, cmap='viridis')
    plt.colorbar(label='VHT [veh·h]')
    plt.xlabel("K_P")
    plt.ylabel("K_I")
    plt.title("Scenario C: VHT (K_I vs K_P)")
    plt.scatter([best['Kp']], [best['Ki']], color='red', marker='x', label='best')
    plt.legend()
    plt.grid(False)
    plt.show()

    # Also plot VHT vs K_I for the best K_P (1D cross-section, similar to previous plot)
    j_best = best['j_idx']
    plt.figure(figsize=(7, 4))
    plt.plot(Ki_vals, vht_grid[:, j_best], marker='o')
    plt.xlabel("K_I")
    plt.ylabel("VHT [veh·h]")
    plt.title(f"Scenario C: VHT vs K_I (K_P = {Kp_vals[j_best]:.3f})")
    plt.grid(True)
    plt.show()

    # simulate using best controller pair and plot results
    best_C = run_metanet(dC_main, dC_ramp, lanes_C, K_I=best['Ki'], K_P=best['Kp'], measured_cell=5, lane_drop_cell=3)
    print("Scenario C with ALINEA (best K_I, K_P)")
    print(f"  VKT = {best_C['vkt']:.1f} veh·km")
    print(f"  VHT = {best_C['vht']:.1f} veh·h")
    print(f"  Avg speed = {best_C['avg_speed']:.1f} km/h")
    plot_scenario(best_C, lanes_C, title_suffix=f"– Scenario C, K_I = {best['Ki']:.2f}, K_P = {best['Kp']:.2f}")
