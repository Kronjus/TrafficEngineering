########################################################################
# Imports
########################################################################
import math
import numpy as np
########################################################################
# Creating demands
########################################################################
def create_demands(time, d_main_peak, d_ramp_peak):
    time_h = time * 3600.0

    mainline = np.piecewise(
        time_h,
        [time_h < 450,
         (time_h >= 450) & (time_h < 3150),
         (time_h >= 3150) & (time_h < 3600),
         time_h >= 3600],
        [
            lambda t: (d_main_peak / 450.0) * t,
            lambda t: d_main_peak,
            lambda t: d_main_peak * (1.0 - (t - 3150.0) / (3600.0 - 3150.0)),
            0.0,
        ]
    )

    onramp = np.piecewise(
        time_h,
        [time_h < 900,
         (time_h >= 900) & (time_h < 2700),
         (time_h >= 2700) & (time_h < 3600),
         time_h >= 3600],
        [
            lambda t: (d_ramp_peak / 900.0) * t,
            lambda t: d_ramp_peak,
            lambda t: d_ramp_peak * (1.0 - (t - 2700.0) / (3600.0 - 2700.0)),
            0.0,
        ]
    )

    return np.stack((mainline, onramp), axis=1)
########################################################################
# METANET Simulation Loop
########################################################################
def run_metanet(
    d_main_peak,
    d_ramp_peak,
    lanes,
    K_I=0.0,
    measured_cell=None,
    lane_drop_cell=None,
):
    T_step = 10.0 / 3600.0
    T_final = 5000.0 / 3600.0
    time = np.arange(0.0, T_final, T_step)
    demands = create_demands(time, d_main_peak, d_ramp_peak)

    lanes = np.asarray(lanes, dtype=float)
    n_cells = len(lanes)
    merge_cell = 2  # on-ramp merges into cell 3 (index 2)

    # METANET parameters
    L = 0.5
    v_free = 100.0
    Q_lane = 2000.0
    rho_max = 180.0
    rho_crit = 32.97
    tau = 22.0 / 3600.0
    eta = 15.0
    kappa = 10.0
    delta = 1.4
    a = 2.0
    phi = 10.0
    w_back = Q_lane / (rho_max - rho_crit)

    # initialize state variables
    density = np.zeros(n_cells)
    speed = np.full(n_cells, v_free)
    flow = np.zeros(n_cells)

    queue_ramp = 0.0
    queue_main = 0.0
    r_prev = 0.0  # previous ramp flow for ALINEA

    # storing results
    densities = []
    speeds = []
    flows = []
    queue_r = []
    queue_m = []

    for step, t in enumerate(time):
        d_main = demands[step, 0]
        d_ramp = demands[step, 1]

        flow = density * speed * lanes # current flow

        # mainline origin
        arrivals_main = d_main + queue_main / T_step
        supply_main = w_back * (rho_max - density[0]) * lanes[0]
        q_in = min(arrivals_main, Q_lane * lanes[0], max(0.0, supply_main))
        queue_main = max(0.0, queue_main + T_step * (d_main - q_in))

        # ramp with ALINEA
        arrivals_ramp = d_ramp + queue_ramp / T_step
        supply_ramp = w_back * (rho_max - density[merge_cell]) * lanes[merge_cell]
        q_supply = max(0.0, supply_ramp)
        q_ramp_max = Q_lane

        if K_I > 0.0 and measured_cell is not None:
            rho_meas = density[measured_cell]
            r_cmd = r_prev + K_I * (rho_crit - rho_meas)
            r_cmd = max(0.0, r_cmd)
        else:
            r_cmd = arrivals_ramp

        q_ramp = min(r_cmd, arrivals_ramp, q_ramp_max, q_supply)
        queue_ramp = max(0.0, queue_ramp + T_step * (d_ramp - q_ramp))

        # current density
        new_density = density.copy()
        for i in range(n_cells):
            if i == 0:
                new_density[i] = density[i] + (T_step / (L * lanes[i])) * (q_in - flow[i])
            elif i == merge_cell:
                new_density[i] = density[i] + (T_step / (L * lanes[i])) * (flow[i - 1] + q_ramp - flow[i])
            else:
                new_density[i] = density[i] + (T_step / (L * lanes[i])) * (flow[i - 1] - flow[i])
            new_density[i] = min(rho_max, max(0.0, new_density[i]))

        # current speed
        new_speed = speed.copy()
        for i in range(n_cells):
            rho_i = density[i]
            rho_down = density[i + 1] if i < n_cells - 1 else density[i]
            V_eq = v_free * math.exp(-1.0 / a * (rho_i / rho_crit) ** a)

            if i == 0:
                new_speed[i] = (
                        speed[i]
                        + (T_step / tau) * (V_eq - speed[i])
                        - (eta * T_step / (tau * L)) * (rho_down - density[i]) / (density[i] + kappa)
                )
            else:
                new_speed[i] = (
                        speed[i]
                        + (T_step / tau) * (V_eq - speed[i])
                        + (T_step / L) * speed[i] * (speed[i - 1] - speed[i])
                        - (eta * T_step / (tau * L)) * (rho_down - density[i]) / (density[i] + kappa)
                )

            if i == merge_cell:
                merge_term = (delta * T_step / (L * lanes[i])) * (r_prev * speed[i] / (rho_i + kappa))
                new_speed[i] -= merge_term

            if lane_drop_cell is not None and i == lane_drop_cell:
                delta_lambda = lanes[i] - lanes[i + 1]
                lane_drop_term = (
                        phi * T_step * delta_lambda * rho_i * speed[i] ** 2
                        / (L * lanes[i] * rho_crit)
                )
                new_speed[i] -= lane_drop_term

            new_speed[i] = min(v_free, max(0.0, new_speed[i]))

        # update states for next iteration
        density = new_density
        speed = new_speed
        r_prev = q_ramp

        # store results
        densities.append(density.copy())
        speeds.append(speed.copy())
        flows.append(flow.copy())
        queue_r.append(queue_ramp)
        queue_m.append(queue_main)

    densities = np.array(densities)
    speeds = np.array(speeds)
    flows = np.array(flows)
    queue_r = np.array(queue_r)
    queue_m = np.array(queue_m)

    vkt = np.sum(flows * L * T_step)
    vht_main = np.sum(densities * lanes * L * T_step)
    vht_queue = np.sum(queue_r * T_step) + np.sum(queue_m * T_step)
    vht = vht_main + vht_queue
    avg_speed = vkt / vht

    return {
        "time": time,
        "densities": densities,
        "speeds": speeds,
        "flows": flows,
        "queue_ramp": queue_r,
        "queue_main": queue_m,
        "vkt": vkt,
        "vht": vht,
        "avg_speed": avg_speed,
    }