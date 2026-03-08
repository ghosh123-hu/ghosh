import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(page_title="Physics Motion Simulator", layout="wide")
st.title("Physics Motion Simulator")
st.caption("One real physics graph per section")


def truncate_at_ground(t, y, v):
    hit = np.where(y <= 0)[0]
    if len(hit) == 0:
        return t, y, v
    i = hit[0]
    return t[: i + 1], y[: i + 1], v[: i + 1]


def simulate_air_drag(mass, g, cd, area, rho, v0_down, y0, t_max, dt):
    # Downward-positive model:
    # m dv/dt = mg - 0.5*rho*Cd*A*v*|v|
    n = int(t_max / dt) + 1
    t = np.linspace(0, t_max, n)
    v = np.zeros(n)
    s = np.zeros(n)  # downward displacement
    y = np.zeros(n)

    v[0] = v0_down
    y[0] = y0

    for i in range(1, n):
        drag = 0.5 * rho * cd * area * v[i - 1] * abs(v[i - 1])
        a = g - drag / mass
        v[i] = v[i - 1] + a * dt
        s[i] = s[i - 1] + v[i] * dt
        y[i] = y0 - s[i]

        if y[i] <= 0:
            return t[: i + 1], y[: i + 1], v[: i + 1]

    return t, y, v


def simulate_rocket(m0, m_dry, mdot, ve, burn_time, g, angle_deg, dt, t_max):
    angle = np.deg2rad(angle_deg)
    x = y = 0.0
    vx = vy = 0.0
    m = m0

    t_vals = [0.0]
    x_vals = [x]
    y_vals = [y]
    speed_vals = [0.0]
    m_vals = [m]

    t = 0.0

    while t < t_max:
        burning = (t < burn_time) and (m > m_dry)

        if burning:
            thrust = mdot * ve
            ax = (thrust / m) * np.cos(angle)
            ay = (thrust / m) * np.sin(angle) - g
            m = max(m_dry, m - mdot * dt)
        else:
            ax = 0.0
            ay = -g

        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt

        t_vals.append(t)
        x_vals.append(x)
        y_vals.append(y)
        speed_vals.append(np.hypot(vx, vy))
        m_vals.append(m)

        if t > 0.5 and y < 0:
            break

    t_arr = np.array(t_vals)
    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)
    speed_arr = np.array(speed_vals)
    m_arr = np.array(m_vals)

    return t_arr, x_arr, y_arr, speed_arr, m_arr


def mpl_line(x, y, title, xlabel, ylabel, color="#1f77b4", extra=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, color=color, linewidth=2)
    if extra is not None:
        extra(ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


def mpl_multi(lines, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for x, y, label, color, lw in lines:
        ax.plot(x, y, label=label, color=color, linewidth=lw)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "1) Normal Gravity Motion",
        "2) Air Resistance Motion",
        "3) Inclined Plane Motion",
        "4) Changing Gravity Environment",
        "5) Rocket Motion + Projectile",
    ]
)

with tab1:
    st.subheader("Normal Gravity Motion")
    c1, c2 = st.columns(2)

    with c1:
        mass = st.number_input("Mass (kg)", min_value=0.1, value=5.0, step=0.1)
        g = st.number_input("Gravity g (m/s²)", min_value=0.1, value=9.81, step=0.1)
        y0 = st.number_input("Initial height (m)", min_value=0.0, value=50.0, step=1.0)
        u = st.number_input("Initial vertical velocity (+up) (m/s)", value=0.0, step=0.1)
        t_max = st.slider("Simulation time (s)", 1, 30, 10)
        n = st.slider("Points", 50, 1000, 300)

    t = np.linspace(0, t_max, n)
    y = y0 + u * t - 0.5 * g * t**2
    v = u - g * t
    t, y, v = truncate_at_ground(t, y, v)
    weight = mass * g

    with c2:
        st.metric("Weight (N)", f"{weight:.3f}")
        st.metric("Time to ground (s)", f"{t[-1]:.3f}")
        st.metric("Impact speed (m/s)", f"{abs(v[-1]):.3f}")
        st.caption("Model: y = y0 + ut - (1/2)gt^2")

    mpl_line(t, y, "Height vs Time", "Time (s)", "Height (m)", color="#0a84ff", extra=lambda ax: ax.axhline(0, color="black", linewidth=1))

with tab2:
    st.subheader("Air Resistance Motion")
    c1, c2 = st.columns(2)

    with c1:
        mass = st.number_input("Mass (kg)", min_value=0.01, value=80.0, step=0.1, key="drag_mass")
        cd = st.number_input("Drag coefficient C_d", min_value=0.01, value=0.8, step=0.01)
        area = st.number_input("Area (m²)", min_value=0.001, value=0.7, step=0.01)
        rho = st.number_input("Air density (kg/m³)", min_value=0.1, value=1.225, step=0.01)
        v0 = st.number_input("Initial downward velocity (m/s)", value=0.0, step=0.1)
        y0 = st.number_input("Initial height (m)", min_value=1.0, value=300.0, step=1.0, key="drag_y0")
        g = st.number_input("Gravity (m/s²)", min_value=0.1, value=9.81, step=0.1, key="drag_g")
        t_max = st.slider("Simulation time (s)", 1, 200, 60, key="drag_t")
        dt = st.slider("Time step dt (s)", 0.001, 0.2, 0.01, key="drag_dt")

    t, y, v = simulate_air_drag(mass, g, cd, area, rho, v0, y0, t_max, dt)
    terminal_v = np.sqrt((2 * mass * g) / (rho * cd * area))

    with c2:
        st.metric("Estimated terminal velocity (m/s)", f"{terminal_v:.3f}")
        st.metric("Fall time (s)", f"{t[-1]:.3f}")
        st.metric("Impact speed (m/s)", f"{abs(v[-1]):.3f}")
        st.caption("Model: m dv/dt = mg - 0.5 rho C_d A v|v|")

    mpl_line(t, v, "Velocity vs Time (With Quadratic Drag)", "Time (s)", "Downward Velocity (m/s)", color="#ff453a")

with tab3:
    st.subheader("Inclined Plane Motion")
    c1, c2 = st.columns(2)

    with c1:
        m = st.number_input("Mass (kg)", min_value=0.01, value=10.0, step=0.1)
        theta_deg = st.slider("Angle theta (degrees)", 1, 89, 30)
        g = st.number_input("Gravity g (m/s²)", min_value=0.1, value=9.81, step=0.1, key="incl_g")
        mu = st.number_input("Friction coefficient mu", min_value=0.0, value=0.2, step=0.01)
        u0 = st.number_input("Initial speed on slope (m/s)", min_value=0.0, value=0.0, step=0.1)
        t_max = st.slider("Simulation time (s)", 1, 60, 12, key="incl_t")
        n = st.slider("Points", 50, 1000, 300, key="incl_n")

    theta = np.deg2rad(theta_deg)
    f_parallel = m * g * np.sin(theta)
    normal = m * g * np.cos(theta)
    friction = mu * normal
    a = g * (np.sin(theta) - mu * np.cos(theta))

    t = np.linspace(0, t_max, n)

    if a >= 0:
        s = u0 * t + 0.5 * a * t**2
        v = u0 + a * t
    else:
        # Block slows down and can stop.
        t_stop = u0 / abs(a) if u0 > 0 else 0
        s_stop = u0 * t_stop + 0.5 * a * t_stop**2
        s = np.where(t <= t_stop, u0 * t + 0.5 * a * t**2, s_stop)
        v = np.where(t <= t_stop, u0 + a * t, 0.0)

    with c2:
        st.metric("Acceleration (m/s²)", f"{a:.3f}")
        st.metric("Parallel force mg sin(theta) (N)", f"{f_parallel:.3f}")
        st.metric("Normal force mg cos(theta) (N)", f"{normal:.3f}")
        st.metric("Friction force muN (N)", f"{friction:.3f}")

    mpl_line(t, s, "Displacement Along Incline vs Time", "Time (s)", "Displacement (m)", color="#30d158")

with tab4:
    st.subheader("Changing Gravity Environment")
    c1, c2 = st.columns(2)

    gravities = {"Earth": 9.81, "Moon": 1.62, "Mars": 3.71, "Custom": None}

    with c1:
        m = st.number_input("Mass (kg)", min_value=0.1, value=5.0, step=0.1, key="env_m")
        env = st.selectbox("Environment", list(gravities.keys()))
        custom_g = st.number_input("Custom gravity (m/s²)", min_value=0.01, value=5.0, step=0.1, disabled=(env != "Custom"))
        y0 = st.number_input("Initial height (m)", min_value=1.0, value=80.0, step=1.0, key="env_y0")
        u = st.number_input("Initial vertical velocity (+up) (m/s)", value=0.0, step=0.1, key="env_u")
        t_max = st.slider("Simulation time (s)", 1, 60, 12, key="env_t")
        n = st.slider("Points", 50, 1000, 300, key="env_n")

    g_sel = custom_g if env == "Custom" else gravities[env]
    weight = m * g_sel
    t = np.linspace(0, t_max, n)

    with c2:
        st.metric("Selected gravity (m/s²)", f"{g_sel:.3f}")
        st.metric("Weight (N)", f"{weight:.3f}")

    lines = []
    palette = {"Earth": "#0a84ff", "Moon": "#ffd60a", "Mars": "#ff453a"}
    for name in ["Earth", "Moon", "Mars"]:
        gv = gravities[name]
        y = y0 + u * t - 0.5 * gv * t**2
        lines.append((t, y, f"{name} ({gv} m/s²)", palette[name], 2))

    y_sel = y0 + u * t - 0.5 * g_sel * t**2
    lines.append((t, y_sel, f"Selected: {env}", "#34c759", 3))

    mpl_multi(lines, "Height vs Time for Different Gravities", "Time (s)", "Height (m)")

with tab5:
    st.subheader("Rocket Motion: Decreasing Mass -> Projectile")
    c1, c2 = st.columns(2)

    with c1:
        m0 = st.number_input("Initial mass (kg)", min_value=1.0, value=500.0, step=1.0)
        m_dry = st.number_input("Dry mass (kg)", min_value=0.1, max_value=float(m0), value=max(1.0, 0.3 * float(m0)), step=1.0)
        mdot = st.number_input("Fuel burn rate (kg/s)", min_value=0.01, value=2.0, step=0.1)
        ve = st.number_input("Exhaust velocity (m/s)", min_value=1.0, value=2200.0, step=10.0)
        burn_time = st.number_input("Burn time (s)", min_value=0.1, value=60.0, step=1.0)
        g = st.number_input("Gravity (m/s²)", min_value=0.1, value=9.81, step=0.1, key="rocket_g")
        angle = st.slider("Launch angle (degrees)", 5, 90, 80)
        dt = st.slider("Time step dt (s)", 0.001, 0.2, 0.02)
        t_max = st.slider("Max simulation time (s)", 5, 600, 220)

    if m_dry >= m0:
        st.warning("Dry mass must be lower than initial mass.")
    else:
        t, x, y, speed, m_curve = simulate_rocket(m0, m_dry, mdot, ve, burn_time, g, angle, dt, t_max)

        i_peak = int(np.argmax(y))
        peak_h = float(y[i_peak])
        t_peak = float(t[i_peak])
        t_total = float(t[-1])

        with c2:
            st.metric("Peak height (m)", f"{peak_h:.3f}")
            st.metric("Time to peak (s)", f"{t_peak:.3f}")
            st.metric("Total flight time (s)", f"{t_total:.3f}")
            st.metric("Final mass (kg)", f"{m_curve[-1]:.3f}")

        mpl_line(x, y, "Rocket Trajectory (Thrust + Ballistic)", "Horizontal distance x (m)", "Height y (m)", color="#bf5af2", extra=lambda ax: ax.axhline(0, color="black", linewidth=1))

st.markdown("---")
st.caption("Run locally with: streamlit run app.py")
