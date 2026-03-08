import math
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


G0 = 9.81
EARTH_RADIUS = 6_371_000.0


@dataclass
class TrajectoryResult:
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray

    @property
    def max_height(self) -> float:
        return float(np.max(self.y))

    @property
    def flight_time(self) -> float:
        return float(self.t[-1])

    @property
    def range(self) -> float:
        return float(self.x[-1])


def interpolate_crossing(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    t1: float,
    t2: float,
    target1: float,
    target2: float,
) -> Tuple[float, float]:
    d1 = y1 - target1
    d2 = y2 - target2
    if abs(d2 - d1) < 1e-12:
        return x2, t2
    frac = d1 / (d1 - d2)
    x_cross = x1 + frac * (x2 - x1)
    t_cross = t1 + frac * (t2 - t1)
    return x_cross, t_cross


def simulate_normal(
    v0: float,
    angle_deg: float,
    dt: float,
    max_time: float,
) -> TrajectoryResult:
    theta = math.radians(angle_deg)
    vx = v0 * math.cos(theta)
    vy = v0 * math.sin(theta)

    x_vals, y_vals, t_vals = [0.0], [0.0], [0.0]
    x, y, t = 0.0, 0.0, 0.0

    while t < max_time:
        x_prev, y_prev, t_prev = x, y, t
        vy_prev = vy

        x += vx * dt
        y += vy * dt
        vy -= G0 * dt
        t += dt

        if y < 0.0:
            x_cross, t_cross = interpolate_crossing(
                x_prev, y_prev, x, y, t_prev, t, 0.0, 0.0
            )
            x_vals.append(x_cross)
            y_vals.append(0.0)
            t_vals.append(t_cross)
            break

        x_vals.append(x)
        y_vals.append(y)
        t_vals.append(t)

        if vy_prev < 0 and y_vals[-1] < 0.0:
            break

    return TrajectoryResult(np.array(x_vals), np.array(y_vals), np.array(t_vals))


def simulate_air_resistance(
    v0: float,
    angle_deg: float,
    mass: float,
    cd: float,
    rho: float,
    area: float,
    dt: float,
    max_time: float,
) -> TrajectoryResult:
    theta = math.radians(angle_deg)
    vx = v0 * math.cos(theta)
    vy = v0 * math.sin(theta)

    x_vals, y_vals, t_vals = [0.0], [0.0], [0.0]
    x, y, t = 0.0, 0.0, 0.0

    k = 0.5 * cd * rho * area

    while t < max_time:
        x_prev, y_prev, t_prev = x, y, t

        v = math.hypot(vx, vy)
        if v > 1e-12:
            drag_ax = -(k / mass) * v * vx
            drag_ay = -(k / mass) * v * vy
        else:
            drag_ax = 0.0
            drag_ay = 0.0

        ax = drag_ax
        ay = drag_ay - G0

        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt

        if y < 0.0:
            x_cross, t_cross = interpolate_crossing(
                x_prev, y_prev, x, y, t_prev, t, 0.0, 0.0
            )
            x_vals.append(x_cross)
            y_vals.append(0.0)
            t_vals.append(t_cross)
            break

        x_vals.append(x)
        y_vals.append(y)
        t_vals.append(t)

    return TrajectoryResult(np.array(x_vals), np.array(y_vals), np.array(t_vals))


def simulate_inclined_plane(
    v0: float,
    launch_angle_deg: float,
    slope_angle_deg: float,
    dt: float,
    max_time: float,
) -> TrajectoryResult:
    theta = math.radians(launch_angle_deg)
    phi = math.radians(slope_angle_deg)

    vx = v0 * math.cos(theta)
    vy = v0 * math.sin(theta)

    x_vals, y_vals, t_vals = [0.0], [0.0], [0.0]
    x, y, t = 0.0, 0.0, 0.0

    def slope_height(xi: float) -> float:
        return math.tan(phi) * xi

    while t < max_time:
        x_prev, y_prev, t_prev = x, y, t
        slope_prev = slope_height(x_prev)

        x += vx * dt
        y += vy * dt
        vy -= G0 * dt
        t += dt

        slope_now = slope_height(x)

        if y < slope_now and t > dt:
            x_cross, t_cross = interpolate_crossing(
                x_prev,
                y_prev,
                x,
                y,
                t_prev,
                t,
                slope_prev,
                slope_now,
            )
            x_vals.append(x_cross)
            y_vals.append(slope_height(x_cross))
            t_vals.append(t_cross)
            break

        x_vals.append(x)
        y_vals.append(y)
        t_vals.append(t)

    return TrajectoryResult(np.array(x_vals), np.array(y_vals), np.array(t_vals))


def simulate_changing_gravity(
    v0: float,
    angle_deg: float,
    dt: float,
    max_time: float,
) -> TrajectoryResult:
    theta = math.radians(angle_deg)
    vx = v0 * math.cos(theta)
    vy = v0 * math.sin(theta)

    x_vals, y_vals, t_vals = [0.0], [0.0], [0.0]
    x, y, t = 0.0, 0.0, 0.0

    while t < max_time:
        x_prev, y_prev, t_prev = x, y, t

        g_h = G0 * (EARTH_RADIUS / (EARTH_RADIUS + max(y, 0.0))) ** 2

        vy -= g_h * dt
        x += vx * dt
        y += vy * dt
        t += dt

        if y < 0.0:
            x_cross, t_cross = interpolate_crossing(
                x_prev, y_prev, x, y, t_prev, t, 0.0, 0.0
            )
            x_vals.append(x_cross)
            y_vals.append(0.0)
            t_vals.append(t_cross)
            break

        x_vals.append(x)
        y_vals.append(y)
        t_vals.append(t)

    return TrajectoryResult(np.array(x_vals), np.array(y_vals), np.array(t_vals))


def simulate_decreasing_mass_rocket(
    v0: float,
    angle_deg: float,
    m0: float,
    mf: float,
    burn_time: float,
    thrust: float,
    dt: float,
    max_time: float,
) -> TrajectoryResult:
    theta = math.radians(angle_deg)
    vx = v0 * math.cos(theta)
    vy = v0 * math.sin(theta)

    x_vals, y_vals, t_vals = [0.0], [0.0], [0.0]
    x, y, t = 0.0, 0.0, 0.0

    mass_rate = (m0 - mf) / burn_time if burn_time > 1e-9 else 0.0

    while t < max_time:
        x_prev, y_prev, t_prev = x, y, t

        if t <= burn_time and burn_time > 1e-9:
            m = max(mf, m0 - mass_rate * t)
            a_thrust = thrust / m
            ax = a_thrust * math.cos(theta)
            ay = a_thrust * math.sin(theta) - G0
        else:
            ax = 0.0
            ay = -G0

        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt

        if y < 0.0:
            x_cross, t_cross = interpolate_crossing(
                x_prev, y_prev, x, y, t_prev, t, 0.0, 0.0
            )
            x_vals.append(x_cross)
            y_vals.append(0.0)
            t_vals.append(t_cross)
            break

        x_vals.append(x)
        y_vals.append(y)
        t_vals.append(t)

    return TrajectoryResult(np.array(x_vals), np.array(y_vals), np.array(t_vals))


def metrics_block(title: str, result: TrajectoryResult) -> None:
    st.markdown(f"### {title}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Max Height (m)", f"{result.max_height:.2f}")
    c2.metric("Range (m)", f"{result.range:.2f}")
    c3.metric("Flight Time (s)", f"{result.flight_time:.2f}")


def single_plot(title: str, result: TrajectoryResult, extra_line=None) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(result.x, result.y, label=title, linewidth=2)
    if extra_line is not None:
        extra_x, extra_y, lbl = extra_line
        ax.plot(extra_x, extra_y, "--", label=lbl, alpha=0.8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)


def main() -> None:
    st.set_page_config(page_title="Projectile Motion Simulator", layout="wide")
    st.title("Projectile Motion Simulator (5 Scenarios)")

    st.sidebar.header("Global Simulation Controls")
    dt = st.sidebar.number_input("Time Step dt (s)", min_value=0.001, max_value=0.2, value=0.01, step=0.001)
    max_time = st.sidebar.number_input(
        "Maximum Simulated Time (s)", min_value=1.0, max_value=2000.0, value=120.0, step=1.0
    )

    st.sidebar.markdown("## Scenario Parameters")

    with st.sidebar.expander("Normal (Ideal)", expanded=True):
        n_v0 = st.number_input("Initial Speed v0 (m/s) [Normal]", 0.0, 5000.0, 70.0, 1.0)
        n_angle = st.number_input("Launch Angle (deg) [Normal]", -89.0, 89.0, 45.0, 1.0)

    with st.sidebar.expander("Air Resistance", expanded=False):
        a_v0 = st.number_input("Initial Speed v0 (m/s) [Air]", 0.0, 5000.0, 70.0, 1.0)
        a_angle = st.number_input("Launch Angle (deg) [Air]", -89.0, 89.0, 45.0, 1.0)
        a_mass = st.number_input("Mass (kg) [Air]", 0.01, 10_000.0, 1.0, 0.1)
        a_cd = st.number_input("Drag Coefficient C_d", 0.0, 5.0, 0.47, 0.01)
        a_rho = st.number_input("Air Density rho (kg/m^3)", 0.1, 5.0, 1.225, 0.01)
        a_area = st.number_input("Cross-sectional Area A (m^2)", 0.0001, 100.0, 0.01, 0.001)

    with st.sidebar.expander("Inclined Plane", expanded=False):
        i_v0 = st.number_input("Initial Speed v0 (m/s) [Incline]", 0.0, 5000.0, 70.0, 1.0)
        i_launch_angle = st.number_input("Launch Angle from Horizontal (deg)", -89.0, 89.0, 50.0, 1.0)
        i_slope = st.number_input("Slope Angle of Plane (deg)", -45.0, 45.0, 15.0, 1.0)

    with st.sidebar.expander("Changing Gravity", expanded=False):
        g_v0 = st.number_input("Initial Speed v0 (m/s) [Variable g]", 0.0, 20_000.0, 700.0, 1.0)
        g_angle = st.number_input("Launch Angle (deg) [Variable g]", -89.0, 89.0, 60.0, 1.0)

    with st.sidebar.expander("Decreasing Mass (Rocket-style)", expanded=False):
        r_v0 = st.number_input("Initial Speed v0 (m/s) [Rocket]", 0.0, 10_000.0, 20.0, 1.0)
        r_angle = st.number_input("Launch Angle (deg) [Rocket]", -89.0, 89.0, 70.0, 1.0)
        r_m0 = st.number_input("Initial Mass m0 (kg)", 0.1, 100_000.0, 50.0, 0.1)
        r_mf = st.number_input("Final Mass mf (kg)", 0.05, 100_000.0, 30.0, 0.1)
        r_burn = st.number_input("Burn Time (s)", 0.0, 500.0, 8.0, 0.1)
        r_thrust = st.number_input("Thrust (N)", 0.0, 5_000_000.0, 1500.0, 10.0)

    if r_mf > r_m0:
        st.sidebar.error("For rocket scenario, final mass mf must be <= initial mass m0.")
        st.stop()

    normal = simulate_normal(n_v0, n_angle, dt, max_time)
    air = simulate_air_resistance(a_v0, a_angle, a_mass, a_cd, a_rho, a_area, dt, max_time)
    incline = simulate_inclined_plane(i_v0, i_launch_angle, i_slope, dt, max_time)
    variable_g = simulate_changing_gravity(g_v0, g_angle, dt, max_time)
    rocket = simulate_decreasing_mass_rocket(r_v0, r_angle, r_m0, r_mf, r_burn, r_thrust, dt, max_time)

    st.header("Combined Trajectory Comparison")
    fig_all, ax_all = plt.subplots(figsize=(10, 6))
    ax_all.plot(normal.x, normal.y, label="Normal (Ideal)")
    ax_all.plot(air.x, air.y, label="Air Resistance")
    ax_all.plot(incline.x, incline.y, label="Inclined Plane")
    ax_all.plot(variable_g.x, variable_g.y, label="Changing Gravity")
    ax_all.plot(rocket.x, rocket.y, label="Decreasing Mass (Rocket)")

    max_x = max(np.max(normal.x), np.max(air.x), np.max(incline.x), np.max(variable_g.x), np.max(rocket.x))
    slope_line_x = np.linspace(0, max(1.0, max_x), 250)
    slope_line_y = np.tan(math.radians(i_slope)) * slope_line_x
    ax_all.plot(slope_line_x, slope_line_y, "--", label="Inclined Plane Surface", alpha=0.7)

    ax_all.set_xlabel("x (m)")
    ax_all.set_ylabel("y (m)")
    ax_all.grid(True, alpha=0.3)
    ax_all.legend()
    st.pyplot(fig_all)

    st.header("Key Metrics")
    metrics = {
        "Normal (Ideal)": normal,
        "Air Resistance": air,
        "Inclined Plane": incline,
        "Changing Gravity": variable_g,
        "Decreasing Mass (Rocket)": rocket,
    }

    metric_rows = []
    for name, result in metrics.items():
        metric_rows.append(
            {
                "Scenario": name,
                "Max Height (m)": round(result.max_height, 3),
                "Range (m)": round(result.range, 3),
                "Flight Time (s)": round(result.flight_time, 3),
            }
        )
    st.dataframe(metric_rows, use_container_width=True)

    st.header("Individual Scenario Outputs")
    tabs = st.tabs(list(metrics.keys()))

    for tab, (name, result) in zip(tabs, metrics.items()):
        with tab:
            metrics_block(name, result)
            if name == "Inclined Plane":
                local_x = np.linspace(0, max(1.0, float(np.max(result.x))), 250)
                local_y = np.tan(math.radians(i_slope)) * local_x
                single_plot(name, result, extra_line=(local_x, local_y, "Inclined Surface"))
            else:
                single_plot(name, result)


if __name__ == "__main__":
    main()
