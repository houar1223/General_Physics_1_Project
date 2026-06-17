import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Bernoulli Equation Simulation
# Article:
# "Measurement of Gravitational Acceleration Using Bernoulli's Equation"
#
# This code directly simulates:
# 1. The decrease of water level in a cylindrical tank.
# 2. The velocity of water ejected from a side hole.
# 3. The projectile trajectory of the water jet.
# 4. The estimation of gravitational acceleration g
#    using the two methods described in the article.
# ============================================================


# ============================================================
# 1. Experimental constants
# ============================================================

g_true = 9.7981          # Gravitational acceleration in Daegu, m/s^2

tank_diameter = 0.190    # Tank diameter, m
hole_diameter = 0.0058   # Hole diameter, m
hole_height = 0.193      # Height of the hole from the bottom, m

cd = 0.61                # Discharge coefficient
cv = 0.98                # Velocity coefficient

tank_area = np.pi * (tank_diameter / 2) ** 2
hole_area = np.pi * (hole_diameter / 2) ** 2

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

random_generator = np.random.default_rng(42)


# ============================================================
# 2. Governing equation for draining water
# ============================================================
#
# Let H(t) = h(t) - ha.
#
# H(t) is the water head above the hole.
# h(t) is the height of the water surface.
# ha is the height of the hole.
#
# From Bernoulli's equation and the discharge coefficient:
#
# dH/dt = - (cd * Ab / A0) * sqrt(2 * g * H)
#
# where:
# A0 = tank cross-sectional area
# Ab = hole cross-sectional area
# ============================================================

def water_head_derivative(H, g=g_true):
    """Return dH/dt for the draining tank."""
    H = max(H, 0.0)
    return - (cd * hole_area / tank_area) * np.sqrt(2 * g * H)


def rk4_step(H, dt):
    """Perform one fourth-order Runge-Kutta step."""
    k1 = water_head_derivative(H)
    k2 = water_head_derivative(H + 0.5 * dt * k1)
    k3 = water_head_derivative(H + 0.5 * dt * k2)
    k4 = water_head_derivative(H + dt * k3)

    H_next = H + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return max(H_next, 0.0)


def simulate_water_drainage(initial_head, dt=0.05):
    """
    Simulate the decrease of water head H(t).

    Parameters
    ----------
    initial_head : float
        Initial water head above the hole, in meters.
    dt : float
        Time step, in seconds.

    Returns
    -------
    time_array : numpy.ndarray
        Time values.
    head_array : numpy.ndarray
        Water head values H(t).
    """

    time_values = [0.0]
    head_values = [initial_head]

    while head_values[-1] > 1e-8:
        next_head = rk4_step(head_values[-1], dt)
        next_time = time_values[-1] + dt

        time_values.append(next_time)
        head_values.append(next_head)

        if next_time > 10000:
            raise RuntimeError("The simulation took too long. Check the input values.")

    return np.array(time_values), np.array(head_values)


# ============================================================
# 3. Jet velocity and horizontal range
# ============================================================
#
# Ideal Torricelli velocity:
#
# va = sqrt(2 * g * H)
#
# Corrected jet velocity:
#
# v = cv * sqrt(2 * g * H)
#
# Projectile falling time from the hole:
#
# t_fall = sqrt(2 * ha / g)
#
# Horizontal range:
#
# x_R = v * t_fall
#     = 2 * cv * sqrt(H * ha)
# ============================================================

def jet_velocity(H, g=g_true):
    """Return the corrected horizontal velocity of the water jet."""
    H = np.maximum(H, 0.0)
    return cv * np.sqrt(2 * g * H)


def horizontal_range(H, g=g_true):
    """Return the horizontal range of the water jet."""
    velocity = jet_velocity(H, g)
    falling_time = np.sqrt(2 * hole_height / g)
    return velocity * falling_time


def projectile_trajectory(H, number_of_points=200, g=g_true):
    """
    Return the projectile trajectory of the water jet for a given water head.

    Parameters
    ----------
    H : float
        Water head above the hole, in meters.
    number_of_points : int
        Number of points used to draw the trajectory.
    g : float
        Gravitational acceleration, in m/s^2.

    Returns
    -------
    x_values : numpy.ndarray
        Horizontal positions, in meters.
    y_values : numpy.ndarray
        Vertical positions, in meters.
    """

    velocity = jet_velocity(H, g)

    if velocity <= 0:
        return np.array([0.0]), np.array([hole_height])

    falling_time = np.sqrt(2 * hole_height / g)
    range_value = velocity * falling_time

    x_values = np.linspace(0, range_value, number_of_points)
    time_values = x_values / velocity
    y_values = hole_height - 0.5 * g * time_values ** 2

    y_values = np.maximum(y_values, 0.0)

    return x_values, y_values


# ============================================================
# 4. Method 1: Estimating g from the water-level decrease
# ============================================================
#
# The article uses the linear relation:
#
# sqrt(H(t)) = sqrt(H0) - [cd * Ab * sqrt(g) / (sqrt(2) * A0)] * t
#
# Therefore, if the slope of sqrt(H) versus time is k:
#
# g = [ -k * sqrt(2) * A0 / (cd * Ab) ]^2
# ============================================================

def estimate_g_from_water_level(time_sample, head_sample):
    """Estimate g from the slope of sqrt(H) versus time."""
    y_values = np.sqrt(np.maximum(head_sample, 0.0))

    slope, intercept = np.polyfit(time_sample, y_values, 1)

    estimated_g = ((-slope * np.sqrt(2) * tank_area) / (cd * hole_area)) ** 2

    return estimated_g, slope, intercept


def run_method_1():
    """Run Method 1 simulation and generate a graph."""
    initial_heads = [0.210, 0.240, 0.270, 0.300]

    height_noise_std = 0.0005
    results = []

    plt.figure(figsize=(9, 6))

    for initial_head in initial_heads:
        time_values, head_values = simulate_water_drainage(initial_head)

        sample_time = np.arange(0, time_values[-1], 10.0)
        true_sample_head = np.interp(sample_time, time_values, head_values)

        measured_sample_head = true_sample_head + random_generator.normal(
            0,
            height_noise_std,
            size=len(true_sample_head)
        )
        measured_sample_head = np.clip(measured_sample_head, 0.0, None)

        estimated_g, slope, intercept = estimate_g_from_water_level(
            sample_time,
            measured_sample_head
        )

        results.append((initial_head, estimated_g, slope))

        y_measured = np.sqrt(measured_sample_head)
        y_fit = slope * sample_time + intercept

        plt.scatter(
            sample_time,
            y_measured,
            s=18,
            label=f"H0 = {initial_head * 1000:.0f} mm, g = {estimated_g:.3f} m/s^2"
        )
        plt.plot(sample_time, y_fit)

    plt.xlabel("Time t (s)")
    plt.ylabel("sqrt(H) (m^0.5)")
    plt.title("Method 1: Estimation of g from Water-Level Decrease")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "method_1_water_level_fit.png", dpi=200)

    return results


# ============================================================
# 5. Method 2: Estimating g from horizontal range
# ============================================================
#
# The article uses the linear relation:
#
# x_R(t) = 2 * cv * sqrt(H0 * ha)
#          - [cv * cd * Ab / A0] * sqrt(2 * g * ha) * t
#
# Therefore, if the slope of x_R versus time is k:
#
# g = [ -k * A0 / (cv * cd * Ab) ]^2 / (2 * ha)
# ============================================================

def estimate_g_from_horizontal_range(time_sample, range_sample):
    """Estimate g from the slope of x_R versus time."""
    slope, intercept = np.polyfit(time_sample, range_sample, 1)

    estimated_g = ((-slope * tank_area) / (cv * cd * hole_area)) ** 2 / (2 * hole_height)

    return estimated_g, slope, intercept


def run_method_2():
    """Run Method 2 simulation and generate a graph."""
    initial_head = 0.200

    range_noise_std = 0.003

    time_values, head_values = simulate_water_drainage(initial_head)

    sample_time = np.arange(0, time_values[-1], 30.0)
    sample_head = np.interp(sample_time, time_values, head_values)

    true_range = horizontal_range(sample_head)

    measured_range = true_range + random_generator.normal(
        0,
        range_noise_std,
        size=len(true_range)
    )
    measured_range = np.clip(measured_range, 0.0, None)

    estimated_g, slope, intercept = estimate_g_from_horizontal_range(
        sample_time,
        measured_range
    )

    range_fit = slope * sample_time + intercept

    plt.figure(figsize=(9, 6))
    plt.scatter(sample_time, measured_range, s=25, label="Simulated measurement")
    plt.plot(sample_time, range_fit, label=f"Linear fit, g = {estimated_g:.3f} m/s^2")
    plt.xlabel("Time t (s)")
    plt.ylabel("Horizontal range x_R (m)")
    plt.title("Method 2: Estimation of g from Horizontal Range")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "method_2_horizontal_range_fit.png", dpi=200)

    return initial_head, estimated_g, slope


# ============================================================
# 6. Plot projectile trajectories
# ============================================================

def plot_projectile_snapshots():
    """Plot several water-jet trajectories at different times."""
    initial_head = 0.200

    time_values, head_values = simulate_water_drainage(initial_head)

    selected_times = [0, 60, 120, 180, 240, 300]

    plt.figure(figsize=(9, 6))

    for selected_time in selected_times:
        if selected_time > time_values[-1]:
            continue

        current_head = np.interp(selected_time, time_values, head_values)
        x_values, y_values = projectile_trajectory(current_head)

        plt.plot(
            x_values,
            y_values,
            label=f"t = {selected_time:.0f} s, H = {current_head * 1000:.1f} mm"
        )

    plt.axhline(0, linewidth=1)
    plt.axhline(hole_height, linestyle="--", linewidth=1, label="Hole height")
    plt.xlabel("Horizontal position x (m)")
    plt.ylabel("Vertical position y (m)")
    plt.title("Projectile Trajectories of the Water Jet")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "water_jet_projectile_snapshots.png", dpi=200)


# ============================================================
# 7. Plot full simulation results
# ============================================================

def plot_full_simulation():
    """Plot water head, jet velocity, and horizontal range over time."""
    initial_head = 0.200

    time_values, head_values = simulate_water_drainage(initial_head)

    velocity_values = jet_velocity(head_values)
    range_values = horizontal_range(head_values)

    plt.figure(figsize=(9, 6))
    plt.plot(time_values, head_values)
    plt.xlabel("Time t (s)")
    plt.ylabel("Water head H(t) (m)")
    plt.title("Water Head as a Function of Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "water_head_vs_time.png", dpi=200)

    plt.figure(figsize=(9, 6))
    plt.plot(time_values, velocity_values)
    plt.xlabel("Time t (s)")
    plt.ylabel("Jet velocity v (m/s)")
    plt.title("Jet Velocity as a Function of Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "jet_velocity_vs_time.png", dpi=200)

    plt.figure(figsize=(9, 6))
    plt.plot(time_values, range_values)
    plt.xlabel("Time t (s)")
    plt.ylabel("Horizontal range x_R (m)")
    plt.title("Horizontal Range as a Function of Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "horizontal_range_vs_time.png", dpi=200)


# ============================================================
# 8. Main execution
# ============================================================

if __name__ == "__main__":
    print("===== Experimental Parameters =====")
    print(f"Tank diameter = {tank_diameter} m")
    print(f"Hole diameter = {hole_diameter} m")
    print(f"Tank area = {tank_area:.6e} m^2")
    print(f"Hole area = {hole_area:.6e} m^2")
    print(f"Hole height = {hole_height} m")
    print(f"Discharge coefficient cd = {cd}")
    print(f"Velocity coefficient cv = {cv}")
    print(f"True gravitational acceleration = {g_true} m/s^2")
    print()

    print("===== Method 1: Estimating g from Water-Level Decrease =====")
    method_1_results = run_method_1()

    method_1_g_values = []

    for initial_head, estimated_g, slope in method_1_results:
        method_1_g_values.append(estimated_g)
        print(
            f"Initial head = {initial_head * 1000:.0f} mm, "
            f"slope = {slope:.6e}, "
            f"estimated g = {estimated_g:.4f} m/s^2"
        )

    print(f"Method 1 average estimated g = {np.mean(method_1_g_values):.4f} m/s^2")
    print()

    print("===== Method 2: Estimating g from Horizontal Range =====")
    initial_head, method_2_g, method_2_slope = run_method_2()

    print(
        f"Initial head = {initial_head * 1000:.0f} mm, "
        f"slope = {method_2_slope:.6e}, "
        f"estimated g = {method_2_g:.4f} m/s^2"
    )
    print()

    plot_projectile_snapshots()
    plot_full_simulation()

    print("===== Saved Figures =====")
    print("outputs/method_1_water_level_fit.png")
    print("outputs/method_2_horizontal_range_fit.png")
    print("outputs/water_jet_projectile_snapshots.png")
    print("outputs/water_head_vs_time.png")
    print("outputs/jet_velocity_vs_time.png")
    print("outputs/horizontal_range_vs_time.png")

    plt.show()