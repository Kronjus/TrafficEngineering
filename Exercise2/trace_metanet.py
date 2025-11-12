"""Detailed step-by-step trace of METANET simulation to find the bug."""

from metanet_simulation import (
    METANETSimulation,
    METANETCellConfig,
    exponential_equilibrium_speed,
)


# Create simple 1-cell simulation for easier debugging
cell = METANETCellConfig(
    name="cell_0",
    length_km=0.5,
    lanes=3,
    free_flow_speed_kmh=100.0,
    jam_density_veh_per_km_per_lane=160.0,
    critical_density_veh_per_km_per_lane=33.5,
    tau_s=18.0,
    nu=60.0,
    kappa=40.0,
    delta=1.0,
    initial_density_veh_per_km_per_lane=20.0,
    initial_speed_kmh=0.0,  # Will use equilibrium
)

print(f"Cell configuration:")
print(f"  Length: {cell.length_km} km")
print(f"  Lanes: {cell.lanes}")
print(f"  Initial density: {cell.initial_density_veh_per_km_per_lane} veh/km/lane")

# Initial equilibrium speed
v_eq = exponential_equilibrium_speed(20.0, 100.0, 33.5)
print(f"  Initial equilibrium speed: {v_eq:.2f} km/h")

# Simulate manually for first 3 steps
dt = 0.1  # hours
L = 0.5  # km
lanes = 3
demand = 3000.0  # veh/h

print(f"\nSimulation parameters:")
print(f"  Time step: {dt} hours ({dt*60:.1f} minutes)")
print(f"  Upstream demand: {demand} veh/h")
print(f"  Downstream supply: 5000 veh/h")

# Step 0 (initial state)
rho_0 = 20.0
v_0 = v_eq  # Will be initialized to this
print(f"\n--- Step 0 (Initial) ---")
print(f"  Density: {rho_0:.2f} veh/km/lane")
print(f"  Speed: {v_0:.2f} km/h")
print(f"  Flow out: rho * v * lanes = {rho_0} * {v_0:.2f} * {lanes} = {rho_0 * v_0 * lanes:.2f} veh/h")

# Step 1
print(f"\n--- Step 1 ---")
# Inflow
inflow = min(demand, 5000.0)  # From boundary, limited by receiving (which should be large initially)
print(f"  Inflow: {inflow:.2f} veh/h")

# Outflow (from previous step's state)
outflow = rho_0 * v_0 * lanes
print(f"  Outflow: {outflow:.2f} veh/h")

# Density change
d_rho = (dt / (L * lanes)) * (inflow - outflow)
rho_1 = rho_0 + d_rho
print(f"  d_rho = ({dt} / ({L} * {lanes})) * ({inflow:.2f} - {outflow:.2f})")
print(f"        = {dt / (L * lanes):.4f} * {inflow - outflow:.2f}")
print(f"        = {d_rho:.2f}")
print(f"  New density: {rho_0:.2f} + {d_rho:.2f} = {rho_1:.2f} veh/km/lane")

# Speed update (simplified - just relaxation term for now)
V_s = exponential_equilibrium_speed(rho_0, 100.0, 33.5)
tau_hours = 18.0 / 3600.0
d_v_relax = (dt / tau_hours) * (V_s - v_0)
v_1 = v_0 + d_v_relax
print(f"  V_s(rho_0) = {V_s:.2f} km/h")
print(f"  d_v (relax) = ({dt} / {tau_hours:.6f}) * ({V_s:.2f} - {v_0:.2f})")
print(f"              = {dt / tau_hours:.2f} * {V_s - v_0:.2f}")
print(f"              = {d_v_relax:.2f} km/h")
print(f"  New speed: {v_0:.2f} + {d_v_relax:.2f} = {v_1:.2f} km/h")

# Step 2
print(f"\n--- Step 2 ---")
inflow_2 = min(demand, 5000.0)
outflow_2 = rho_1 * v_1 * lanes
print(f"  Inflow: {inflow_2:.2f} veh/h")
print(f"  Outflow: rho * v * lanes = {rho_1:.2f} * {v_1:.2f} * {lanes} = {outflow_2:.2f} veh/h")

d_rho_2 = (dt / (L * lanes)) * (inflow_2 - outflow_2)
rho_2 = rho_1 + d_rho_2
print(f"  d_rho = {d_rho_2:.2f}")
print(f"  New density: {rho_1:.2f} + {d_rho_2:.2f} = {rho_2:.2f} veh/km/lane")

print("\n" + "="*60)
print("Now run actual simulation and compare:")

sim = METANETSimulation(
    cells=[cell],
    time_step_hours=dt,
    upstream_demand_profile=demand,
    downstream_supply_profile=5000.0,
)
result = sim.run(steps=3)

print(f"\nActual densities: {[f'{d:.2f}' for d in result.densities['cell_0'][:4]]}")
print(f"Actual flows: {[f'{f:.2f}' for f in result.flows['cell_0'][:3]]}")
print(f"\nExpected density progression: {rho_0:.2f} -> {rho_1:.2f} -> {rho_2:.2f}")
print(f"Actual density progression: {' -> '.join([f'{d:.2f}' for d in result.densities['cell_0'][:3]])}")
