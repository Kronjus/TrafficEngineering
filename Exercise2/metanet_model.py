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

    # validate indices for measured_cell and lane_drop_cell
    if measured_cell is not None:
        if not (0 <= measured_cell < n_cells):
            print(f"Warning: measured_cell={measured_cell} out of range, ignoring ALINEA measurement.")
            measured_cell = None
    if lane_drop_cell is not None:
        if not (0 <= lane_drop_cell < n_cells - 1):
            print(f"Warning: lane_drop_cell={lane_drop_cell} invalid (must be 0..{n_cells - 2}), ignoring lane drop.")
            lane_drop_cell = None

    # initialize state variables
    density = np.zeros(n_cells)
    speed = np.full(n_cells, v_free)

    queue_ramp = 0.0
    queue_main = 0.0
    r_prev = 0.0  # previous achieved ramp flow

    # initialize ALINEA integrator/command state
    r_prev_cmd = min(Q_lane, d_ramp_peak) if K_I > 0.0 and measured_cell is not None else 0.0

    # storing results
    densities = []
    speeds = []
    flows = []
    queue_r = []
    queue_m = []

    # ramp geometry (assume single-lane ramp unless specified otherwise)
    ramp_lanes = 1.0
    q_ramp_max_const = Q_lane * ramp_lanes

    for step, t in enumerate(time):
        d_main = demands[step, 0]
        d_ramp = demands[step, 1]

        # raw flow estimate (density * speed * lanes)
        flow_est = density * speed * lanes

        # per-cell demand and supply
        demand = np.minimum(flow_est, Q_lane * lanes)
        supply = np.maximum(0.0, w_back * (rho_max - density) * lanes)

        # mainline origin
        arrivals_main = d_main + queue_main / T_step
        q_in = min(arrivals_main, Q_lane * lanes[0], supply[0])
        queue_main = max(0.0, queue_main + T_step * (d_main - q_in))

        # ramp arrivals / ALINEA (integrate using T_step)
        arrivals_ramp = d_ramp + queue_ramp / T_step
        if K_I > 0.0 and measured_cell is not None:
            rho_meas = density[measured_cell]
            r_cmd_candidate = r_prev_cmd + K_I * T_step * (rho_crit - rho_meas)
            r_cmd_candidate = max(0.0, r_cmd_candidate)
        else:
            r_cmd_candidate = arrivals_ramp

        # actuator / supply bounds for ramp: need q_ramp_max and q_supply at merge cell
        q_ramp_max = q_ramp_max_const
        q_supply = supply[merge_cell] if 0 <= merge_cell < n_cells else 0.0

        # desired ramp and applied q_ramp will be decided during merge allocation
        ramp_desired = r_cmd_candidate
        ramp_desired = min(ramp_desired, arrivals_ramp, q_ramp_max)

        # compute inter-cell flows f and allocate merge supply
        if n_cells >= 2:
            f = np.zeros(n_cells - 1)
            # normal links (except the upstream link feeding the merge cell)
            for i in range(n_cells - 1):
                if i == merge_cell - 1:
                    # skip merge upstream link here; handle below
                    continue
                f[i] = min(demand[i], supply[i + 1])

            # handle merge
            if 0 < merge_cell < n_cells:
                main_demand = demand[merge_cell - 1]
                supply_merge = supply[merge_cell]
                total_alloc = min(main_demand + ramp_desired, supply_merge)
                main_flow = min(main_demand, total_alloc)
                ramp_flow = min(ramp_desired, max(0.0, total_alloc - main_flow))
                f[merge_cell - 1] = main_flow
                q_ramp = ramp_flow
            else:
                # unreachable or no merge
                q_ramp = 0.0

            # outflow from last cell (vehicles leaving network)
            outflow_last = demand[-1]
        else:
            # single cell domain: no inter-cell flows
            f = np.zeros(0)
            q_ramp = 0.0
            outflow_last = demand[-1] if n_cells == 1 else 0.0

        # Anti-windup: update integrator to applied q_ramp
        if K_I > 0.0 and measured_cell is not None:
            if abs(q_ramp - r_cmd_candidate) < 1e-6:
                r_prev_cmd = r_cmd_candidate
            else:
                r_prev_cmd = q_ramp

        queue_ramp = max(0.0, queue_ramp + T_step * (d_ramp - q_ramp))

        # update densities using computed inter-cell flows and ramp contribution
        new_density = density.copy()
        for i in range(n_cells):
            if i == 0:
                f_out = f[0] if n_cells > 1 else outflow_last
                new_density[i] = density[i] + (T_step / (L * lanes[i])) * (q_in - f_out)
            elif i == merge_cell:
                f_in = f[i - 1] if i - 1 >= 0 else 0.0
                f_out = f[i] if i < n_cells - 1 else outflow_last
                new_density[i] = density[i] + (T_step / (L * lanes[i])) * (f_in + q_ramp - f_out)
            else:
                f_in = f[i - 1] if i - 1 >= 0 else 0.0
                f_out = f[i] if i < n_cells - 1 else outflow_last
                new_density[i] = density[i] + (T_step / (L * lanes[i])) * (f_in - f_out)
            new_density[i] = min(rho_max, max(0.0, new_density[i]))

        # update speeds (unchanged logic)
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
                if delta_lambda > 0:
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

        # Build applied outflow per cell (outflow from each cell)
        applied_flow = np.zeros(n_cells)
        if n_cells > 1:
            applied_flow[: n_cells - 1] = f.copy()
        applied_flow[-1] = outflow_last

        # store results
        densities.append(density.copy())
        speeds.append(speed.copy())
        flows.append(applied_flow.copy())
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
    avg_speed = vkt / vht if vht > 0 else 0.0

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
