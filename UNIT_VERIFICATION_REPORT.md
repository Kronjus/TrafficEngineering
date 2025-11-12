# Unit Consistency Verification Report for metanet_simulation.py

## Executive Summary

**Status:** ✅ **NO UNIT CONVERSION ERRORS FOUND**

A comprehensive analysis of the METANET simulation code has verified that:
1. All unit conversions are mathematically correct
2. Lane drops and lane increases are handled properly
3. Vehicle conservation is maintained across all scenarios
4. All 72 tests pass (61 existing + 11 new unit-specific tests)

## Analysis Performed

### 1. Unit Verification

Every variable and equation was analyzed for dimensional consistency:

| Variable | Units | Verification |
|----------|-------|--------------|
| `dt` (time_step_hours) | hours | ✅ Correct |
| `tau_s` | seconds → hours | ✅ Converted correctly (÷3600) |
| `rho` (density) | veh/km/lane | ✅ Per-lane, not total |
| `v` (speed) | km/h | ✅ Correct |
| `q` (flow) | veh/h | ✅ Total flow (includes lanes) |
| `N` (queue) | vehicles | ✅ Count, not rate |
| `nu` | km²/h | ✅ Correct |
| `kappa` | veh/km/lane | ✅ Correct |
| `delta` | dimensionless | ✅ Correct |
| `ALINEA_gain` | (veh/h)/(veh/km/lane) | ✅ Correct |

### 2. Critical Equations Verified

#### Density Update (Line 890)
```python
d_rho = (self.dt / (L_i * lambda_i)) * (inflow_total - outflow_total - s_i)
```
**Unit Analysis:**
- `dt`: hours
- `L_i`: km
- `lambda_i`: lanes (dimensionless)
- `inflow_total`, `outflow_total`: veh/h
- Result: `hours / (km × lanes) × veh/h = veh/(km × lanes) = veh/km/lane` ✅

**Why This is Correct for Lane Changes:**
- `inflow_total` and `outflow_total` are total flows (veh/h) across all lanes
- Dividing by `(L_i × lambda_i)` converts to per-lane density
- Each cell uses its own `lambda_i`, so lane changes are handled correctly

#### Flow Calculation (Line 611, 664)
```python
mainline_flow_in = densities[idx-1] * speeds[idx-1] * self.cells[idx-1].lanes
```
**Unit Analysis:**
- `densities`: veh/km/lane
- `speeds`: km/h
- `lanes`: dimensionless
- Result: `(veh/km/lane) × (km/h) × lanes = veh/h` ✅

#### Ramp Queue Update (Line 642, 660)
```python
ramp.queue_veh += d_i * self.dt                    # Add arrivals
ramp.queue_veh -= ramp_flow * self.dt              # Subtract departures
```
**Unit Analysis:**
- `d_i`, `ramp_flow`: veh/h
- `dt`: hours
- Result: `vehicles + (veh/h × hours) = vehicles` ✅

### 3. Lane Drop/Increase Analysis

**Test Scenarios:**
- 3 lanes → 2 lanes (lane drop) ✅
- 2 lanes → 3 lanes (lane increase) ✅
- 3 → 2 → 3 → 2 (multiple changes) ✅
- Lane drop with on-ramp merging ✅

**Key Finding:**
The implementation correctly maintains per-lane density while computing total flows. The factor `1/lambda_i` in the density update ensures conservation:

**Example:** 3 lanes → 2 lanes
- Cell 0: 40 veh/km/lane, 3 lanes → flow = 40 × 80 × 3 = 9600 veh/h
- Cell 1: Receives 9600 veh/h, has 2 lanes
- Density change in Cell 1: `d_rho = (0.1 / (0.5 × 2)) × 9600 = 960 veh/km/lane`
- The division by 2 (Cell 1's lanes) is correct ✅

### 4. Profile Unit Expectations

**Critical Finding:**
Profiles MUST return veh/h, not vehicles per time step.

**Common Error:**
If a profile returns 100 (meaning 100 vehicles per 6-minute step with dt=0.1h):
- Code interprets as 100 veh/h
- Actual vehicles per step: 100 × 0.1 = 10 vehicles
- Error factor: 1/10 (off by factor of dt)

**Mitigation Added:**
- `_diagnose_profile_units()` method warns about suspicious values
- Automatic checking at simulation initialization
- User-facing diagnostic tools provided

### 5. Ramp Flow and Queue Consistency

**Verification Method:**
Queue reconstruction from arrivals and departures:
```
N(k+1) = N(k) + T × d(k) - T × r(k)
```

**Test Results:**
All queue reconstructions matched reported queues to within numerical precision (< 0.01 vehicles).

### 6. ALINEA Control Unit Analysis

**Control Law:**
```python
rate_adjustment = ramp.alinea_gain * density_error
```

**Unit Analysis:**
- `alinea_gain`: (veh/h) per (veh/km/lane)
- `density_error`: veh/km/lane
- Result: `((veh/h)/(veh/km/lane)) × (veh/km/lane) = veh/h` ✅

## Changes Made

### 1. Enhanced Documentation (metanet_simulation.py)

**Module-level Documentation:**
- Comprehensive unit conventions section
- Detailed explanation of lane drop/increase handling
- Profile unit expectations
- Common error patterns and solutions

**Inline Comments:**
- Unit derivations for all critical equations
- Clarification of per-lane vs. total quantities
- Explanation of conservation across lane changes

### 2. Diagnostic Tools Added

**`_diagnose_profile_units()` Method:**
- Automatically checks profiles at initialization
- Warns if values appear to be in wrong units
- Heuristic detection of common errors

**Unit Validation:**
- Warning if `time_step_hours > 1.0` (likely wrong units)
- Automatic profile checking for all inputs

### 3. User-Facing Diagnostic Tools

**diagnose_metanet_units.py:**
- `reconstruct_queue()`: Rebuild queue from arrivals/flows
- `compare_queues()`: Compare reconstructed to reported
- `check_profile_units()`: Detect wrong profile units
- `print_unit_summary()`: Reference for all units

**example_unit_diagnostics.py:**
- Complete working examples
- Demonstration of all diagnostic tools
- Examples of correct and incorrect usage

### 4. Comprehensive Test Suite

**test_unit_conversions.py (11 new tests):**
- `TestUnitConversions`: 3 tests for basic unit handling
- `TestLaneChanges`: 4 tests for lane drops/increases
- `TestProfileUnitExpectations`: 2 tests for profile units
- `TestRampFlowAndQueueConsistency`: 2 tests for queue reconstruction

**All Tests Pass:**
- 61 original tests ✅
- 11 new unit-specific tests ✅
- **Total: 72 tests, 0 failures**

## Conclusions

### Main Findings

1. **No Unit Conversion Errors:** All equations use consistent units throughout.

2. **Lane Changes Handled Correctly:** The factor `1/(L × lanes)` in density update ensures proper conservation when lane counts change.

3. **Profile Units Critical:** The most likely source of discrepancies between models is profiles returning wrong units (per-step instead of per-hour).

4. **Queue Tracking Correct:** Ramp queues are properly maintained in vehicles (not rates).

5. **ALINEA Units Correct:** Control gain has proper dimensions for density-based feedback.

### Recommendations for Users

1. **Always use veh/h for profiles**, not vehicles per time step.
   - If you have per-step values, multiply by `1/dt`

2. **Use diagnostic tools** when comparing METANET to other models:
   ```python
   from diagnose_metanet_units import reconstruct_queue, compare_queues
   reconstructed = reconstruct_queue(arrival_profile, ramp_flows, initial_queue, dt)
   compare_queues(reconstructed, reported_queue)
   ```

3. **Check for warnings** at simulation initialization about profile units.

4. **Verify dt is in hours**, not seconds or minutes.

5. **When debugging lane drops**, remember:
   - Density is per-lane (veh/km/lane)
   - Flow is total (veh/h)
   - Both are correct; they serve different purposes

### Likely Causes of METANET vs. CTM Discrepancies

If METANET reports very different queue values than CTM:

1. **Profile unit mismatch** (most likely)
   - Solution: Use diagnostic tools to verify

2. **Different model dynamics**
   - METANET uses continuous speed dynamics
   - CTM uses discrete supply/demand
   - Expected to have some differences

3. **Different parameter values**
   - Capacity, jam density, critical density
   - Ensure parameters are calibrated consistently

4. **Different time steps**
   - METANET may be more sensitive to dt
   - Try smaller time steps if unstable

## Security Analysis

**CodeQL Analysis Result:** ✅ No security issues found

**Manual Review:**
- No hardcoded credentials
- No unsafe file operations
- No SQL injection risks
- No command injection risks
- Input validation appropriate
- No buffer overflow risks (Python)

## Files Modified

1. `Exercise2/metanet_simulation.py` - Enhanced with documentation and diagnostics
2. `Exercise2/test_unit_conversions.py` - New comprehensive test suite
3. `Exercise2/diagnose_metanet_units.py` - New diagnostic utilities
4. `Exercise2/example_unit_diagnostics.py` - New example demonstrating tools

## References

**METANET Specification:**
- Density update: `ρ(k+1) = ρ(k) + T/(L×λ) × (q_in + r - q_out - s)`
- Speed update: Includes relaxation, convection, anticipation, ramp coupling
- Flow: `q = ρ × v × λ`

**Unit Conventions:**
- All profiles: veh/h
- All densities: veh/km/lane
- All queues: vehicles (count)
- Time step: hours

---

**Report Date:** 2025-11-12

**Analysis Status:** Complete

**Verification Status:** ✅ All unit conversions correct

**Test Status:** ✅ All 72 tests passing

**Security Status:** ✅ No issues found
