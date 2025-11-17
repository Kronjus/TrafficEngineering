# METANET Sending Flow Fix - Complete Summary

## Problem Statement

The METANET simulation incorrectly computed sending flows by including same-step inflow in the "available vehicles" calculation. This violated the fundamental METANET/CTM temporal discretization principle:

**Correct Discretization**: Flows at time step t should be computed from the state at time t, then used to update the state to time t+1.

**Bug**: The code added same-step inflow to available vehicles when computing sending flows, allowing vehicles to enter and exit a cell in the same discrete time step.

## Root Cause Analysis

### Bug Location 1: Internal Cell Sending (Lines 772-784)

**Buggy Code:**
```python
if inflows[idx-1] > 0:
    max_sending = (prev_current_vehicles + inflows[idx-1] * self.dt) / self.dt
else:
    max_sending = prev_current_vehicles / self.dt
```

**Problem**: When computing how much cell `idx-1` can send to cell `idx`, the code added `inflows[idx-1]` (vehicles entering cell `idx-1` in the same time step) to the available vehicles.

### Bug Location 2: Last Cell Outflow (Lines 851-863)

**Buggy Code:**
```python
current_vehicles = densities[last_idx] * self.cells[last_idx].length_km * self.cells[last_idx].lanes
inflow_vehicles = inflows[last_idx] * self.dt
max_available_flow = (current_vehicles + inflow_vehicles) / self.dt
```

**Problem**: The last cell's outflow calculation included vehicles that entered during the same time step.

## The Fix

### Fix 1: Internal Cell Sending

**Corrected Code:**
```python
# What can cell idx-1 send? Limited by vehicles at START of time step
# IMPORTANT: Do NOT include inflows[idx-1] here as that would allow
# vehicles to enter and exit cell idx-1 in the same time step,
# violating METANET/CTM temporal discretization where flows are
# computed from state at time t, then used to update to time t+1.
prev_current_vehicles = densities[idx-1] * self.cells[idx-1].length_km * self.cells[idx-1].lanes
max_sending = prev_current_vehicles / self.dt
```

### Fix 2: Last Cell Outflow

**Corrected Code:**
```python
# CRITICAL FIX: Constrain outflow to vehicles available at START of time step
# IMPORTANT: Do NOT include inflow_vehicles here as that would allow
# vehicles to enter and exit the last cell in the same time step,
# violating METANET/CTM temporal discretization where flows are
# computed from state at time t, then used to update to time t+1.
current_vehicles = densities[last_idx] * self.cells[last_idx].length_km * self.cells[last_idx].lanes
max_available_flow = current_vehicles / self.dt
```

## Validation

### New Tests Added

1. **test_no_same_step_exit_of_entering_vehicles**: 
   - Verifies that outflow from a cell is constrained by vehicles present at the start of the time step
   - Checks conservation holds across all time steps

2. **test_global_vehicle_conservation**: 
   - Verifies global vehicle conservation: `Total(t+1) = Total(t) + Inflow(t) - Outflow(t)`
   - Tests conservation over multiple time steps with varying demand

### Test Results

```
✅ CTM tests: 39/39 passed
✅ METANET tests: 63/63 passed (including 2 new conservation tests)
✅ Unit conversion tests: 11/11 passed
✅ VKT comparison test: 1/1 passed
✅ Security scan: 0 vulnerabilities

Total: 114/114 tests passed ✓
```

### VKT Comparison

| Metric | CTM | METANET (After Fix) | Difference |
|--------|-----|---------------------|------------|
| VKT | 9180.00 | 9170.93 | **0.10%** ✓ |

The 0.10% difference is within acceptable numerical precision for different discretization schemes.

### Conservation Verification

Perfect conservation is maintained:
- No vehicles created or destroyed
- Total vehicles = Initial + Cumulative_Inflow - Cumulative_Outflow
- Error: 0.00% (within numerical precision)

## Impact and Benefits

### Before Fix
- ❌ Vehicles could enter and exit cells in the same time step
- ❌ Violated METANET/CTM temporal discretization
- ❌ Potential for inflated VKT due to double-counting
- ❌ Conservation could be violated in edge cases

### After Fix
- ✅ Flows computed strictly from state at time t
- ✅ Matches CTM and METANET temporal discretization
- ✅ Perfect vehicle conservation
- ✅ VKT matches CTM within 0.10%
- ✅ Prevents same-step entry and exit of vehicles

## Technical Details

### Temporal Discretization Principle

The correct METANET/CTM time-stepping order is:

1. **At time t**: Read current densities ρ(t) and speeds v(t)
2. **Compute flows**: Calculate sending and receiving based on state at t
3. **Resolve flows**: Determine actual flows respecting capacity and conservation
4. **Update state**: Compute ρ(t+1) and v(t+1) using flows
5. **Repeat**: Move to next time step

The bug violated step 2 by using a mix of state at t and flows computed during step t.

### Mathematical Formulation

**Density Update (Correct)**:
```
ρᵢ(k+1) = ρᵢ(k) + (T/LᵢλᵢΣ) × (qᵢ₋₁(k) - qᵢ(k) + rᵢ(k))
```

**Sending Constraint (Corrected)**:
```
qᵢ(k) ≤ ρᵢ(k) × Lᵢ × λᵢ / T
```

Where:
- ρᵢ(k) = density at time k (state at START of time step)
- qᵢ(k) = flow during time step k
- T = time step duration
- Lᵢ = cell length
- λᵢ = number of lanes

**Key Point**: The sending constraint uses ρᵢ(k), NOT ρᵢ(k) plus any inflow during step k.

## Files Modified

1. **Exercise2/metanet_simulation.py**
   - Lines 772-783: Fixed internal cell sending calculation
   - Lines 851-863: Fixed last cell outflow calculation

2. **Exercise2/test_metanet_simulation.py**
   - Added `test_no_same_step_exit_of_entering_vehicles`
   - Added `test_global_vehicle_conservation`

## Backward Compatibility

The fix maintains backward compatibility with existing code:
- All existing tests pass
- API unchanged
- No changes to function signatures
- Only internal flow calculation logic modified

## Conclusion

This fix restores the correct METANET/CTM temporal discretization by ensuring:
1. All flows are computed from the state at the beginning of each time step
2. Vehicles cannot enter and exit a cell in the same discrete time step
3. Perfect vehicle conservation is maintained
4. VKT calculations match CTM within numerical precision

The implementation now correctly matches the mathematical formulation of METANET and CTM models.
