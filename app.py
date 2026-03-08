import numpy as np
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Physics Motion Simulator", layout="wide")
st.title("Physics Motion Simulator")
st.caption("Gravity, drag, inclined plane, gravity environments, and rocket + projectile motion")


def make_vertical_animation(y_values, title, y_label="Position (m)"):
    y_values = np.asarray(y_values)
    max_y = float(np.max(y_values)) if len(y_values) else 1.0
    min_y = float(np.min(y_values)) if len(y_values) else 0.0
    margin = max(1.0, 0.1 * (max_y - min_y + 1e-9))

    fig = go.Figure(
        data=[
            go.Scatter(
                x=[0],
                y=[y_values[0] if len(y_values) else 0],
                mode="markers",
                marker=dict(size=18, color="#ef553b"),
                name="Object",
            )
        ],
        layout=go.Layout(
            title=title,
            xaxis=dict(range=[-1, 1], title=""),
            yaxis=dict(range=[min_y - margin, max_y + margin], title=y_label),
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 30, "redraw": True}, "fromcurrent": True}],
                        }
                    ],
                }
            ],
        ),
        frames=[
            go.Frame(
                data=[
                    go.Scatter(
                        x=[0],
                        y=[y_values[i]],
                        mode="markers",
                        marker=dict(size=18, color="#ef553b"),
                    )
                ],
                name=str(i),
            )
            for i in range(len(y_values))
        ],
    )
    fig.update_xaxes(showticklabels=False)
    return fig


def make_slope_animation(x_values, y_values, slope_x, slope_y, title):
    fig = go.Figure(
        data=[
            go.Scatter(x=slope_x, y=slope_y, mode="lines", line=dict(width=5, color="#636efa"), name="Incline"),
            go.Scatter(
                x=[x_values[0] if len(x_values) else 0],
                y=[y_values[0] if len(y_values) else 0],
                mode="markers",
                marker=dict(size=18, color="#ef553b"),
                name="Block",
            ),
        ],
        layout=go.Layout(
            title=title,
            xaxis=dict(title="x (m)"),
            yaxis=dict(title="y (m)"),
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 30, "redraw": True}, "fromcurrent": True}],
                        }
                    ],
                }
            ],
        ),
        frames=[
            go.Frame(
                data=[
                    go.Scatter(x=slope_x, y=slope_y, mode="lines", line=dict(width=5, color="#636efa")),
                    go.Scatter(x=[x_values[i]], y=[y_values[i]], mode="markers", marker=dict(size=18, color="#ef553b")),
                ],
                name=str(i),
            )
            for i in range(len(x_values))
        ],
    )
    return fig


def make_trajectory_animation(x_values, y_values, title="Rocket Motion Animation"):
    fig = go.Figure(
        data=[
            go.Scatter(x=x_values, y=y_values, mode="lines", line=dict(color="#00cc96"), name="Path"),
            go.Scatter(
                x=[x_values[0] if len(x_values) else 0],
                y=[y_values[0] if len(y_values) else 0],
                mode="markers",
                marker=dict(size=14, color="#ef553b"),
                name="Rocket",
            ),
        ],
        layout=go.Layout(
            title=title,
            xaxis=dict(title="x (m)"),
            yaxis=dict(title="y (m)"),
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 25, "redraw": True}, "fromcurrent": True}],
                        }
                    ],
                }
            ],
        ),
        frames=[
            go.Frame(
                data=[
                    go.Scatter(x=x_values, y=y_values, mode="lines", line=dict(color="#00cc96")),
                    go.Scatter(
                        x=[x_values[i]],
                        y=[y_values[i]],
                        mode="markers",
                        marker=dict(size=14, color="#ef553b"),
                    ),
                ],
                name=str(i),
            )
            for i in range(len(x_values))
        ],
    )
    return fig


def simulate_drag(mass, g, cd, area, rho, v0, t_max, dt):
    n = int(t_max / dt) + 1
    t = np.linspace(0, t_max, n)
    v = np.zeros(n)
    y = np.zeros(n)
    v[0] = v0

    for i in range(1, n):
        drag = 0.5 * rho * cd * area * v[i - 1] * abs(v[i - 1])
        a = g - drag / mass
        v[i] = v[i - 1] + a * dt
        y[i] = y[i - 1] + v[i] * dt

    return t, y, v


def simulate_rocket_projectile(m0, m_dry, mdot, ve, burn_time, g, angle_deg, dt, t_max):
    angle = np.deg2rad(angle_deg)
    x, y = 0.0, 0.0
    vx, vy = 0.0, 0.0
    m = m0

    t_values = [0.0]
    x_values = [x]
    y_values = [y]
    vx_values = [vx]
    vy_values = [vy]
    m_values = [m]

    t = 0.0
    burn_end = burn_time

    while t < t_max:
        burning = (t < burn_time) and (m > m_dry)

        if burning:
            thrust = mdot * ve
            ax = (thrust * np.cos(angle)) / m
            ay = (thrust * np.sin(angle)) / m - g
            m = max(m_dry, m - mdot * dt)
            if m <= m_dry + 1e-9:
                burn_end = t
        else:
            ax = 0.0
            ay = -g

        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt

        t_values.append(t)
        x_values.append(x)
        y_values.append(y)
        vx_values.append(vx)
        vy_values.append(vy)
        m_values.append(m)

        if t > 0.5 and y < 0:
            break

    t_arr = np.array(t_values)
    x_arr = np.array(x_values)
    y_arr = np.array(y_values)
    vx_arr = np.array(vx_values)
    vy_arr = np.array(vy_values)
    v_arr = np.sqrt(vx_arr**2 + vy_arr**2)
    m_arr = np.array(m_values)

    return t_arr, x_arr, y_arr, vx_arr, vy_arr, v_arr, m_arr, burn_end


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
    col1, col2 = st.columns(2)

    with col1:
        mass = st.number_input("Mass (kg)", min_value=0.1, value=5.0, step=0.1)
        g = st.number_input("Gravity g (m/s²)", min_value=0.1, value=9.81, step=0.1)
        u = st.number_input("Initial velocity u (m/s)", value=0.0, step=0.1)
        t_max = st.slider("Simulation time (s)", 1, 30, 10)
        n_pts = st.slider("Number of timesteps", 50, 800, 250)

    t = np.linspace(0, t_max, n_pts)
    y = u * t + 0.5 * g * t**2
    v = u + g * t
    weight = mass * g

    with col2:
        st.metric("Weight (N)", f"{weight:.3f}")
        st.write("Equation used:  y = ut + 1/2 gt²,  v = u + gt")

    fig_pos = go.Figure()
    fig_pos.add_trace(go.Scatter(x=t, y=y, mode="lines", name="Position"))
    fig_pos.update_layout(title="Position vs Time", xaxis_title="Time (s)", yaxis_title="Position (m)")

    fig_vel = go.Figure()
    fig_vel.add_trace(go.Scatter(x=t, y=v, mode="lines", name="Velocity", line=dict(color="#ef553b")))
    fig_vel.update_layout(title="Velocity vs Time", xaxis_title="Time (s)", yaxis_title="Velocity (m/s)")

    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_pos, use_container_width=True)
    c2.plotly_chart(fig_vel, use_container_width=True)

    st.plotly_chart(make_vertical_animation(y, "Animated Motion (Gravity)"), use_container_width=True)


with tab2:
    st.subheader("Air Resistance Motion (Numerical Integration)")
    col1, col2 = st.columns(2)

    with col1:
        mass = st.number_input("Mass (kg)", min_value=0.01, value=80.0, step=0.1, key="drag_mass")
        cd = st.number_input("Drag coefficient C_d", min_value=0.01, value=0.8, step=0.01)
        area = st.number_input("Area (m²)", min_value=0.001, value=0.7, step=0.01)
        rho = st.number_input("Air density (kg/m³)", min_value=0.1, value=1.225, step=0.01)
        v0 = st.number_input("Initial velocity (m/s, downward +)", value=0.0, step=0.1)
        g = st.number_input("Gravity (m/s²)", min_value=0.1, value=9.81, step=0.1, key="drag_g")
        t_max = st.slider("Simulation time (s)", 1, 60, 20, key="drag_tmax")
        dt = st.slider("Time step dt (s)", 0.005, 0.2, 0.02, key="drag_dt")

    t, y, v = simulate_drag(mass, g, cd, area, rho, v0, t_max, dt)
    terminal_velocity = np.sqrt((2 * mass * g) / (rho * cd * area))

    with col2:
        st.metric("Estimated terminal velocity (m/s)", f"{terminal_velocity:.3f}")
        st.write("Drag force:  F_d = 1/2 rho C_d A v²")
        st.write("Net acceleration:  a = g - F_d/m")

    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(x=t, y=v, mode="lines", name="Velocity", line=dict(color="#ef553b")))
    fig_v.update_layout(title="Velocity vs Time", xaxis_title="Time (s)", yaxis_title="Velocity (m/s)")

    fig_y = go.Figure()
    fig_y.add_trace(go.Scatter(x=t, y=y, mode="lines", name="Position"))
    fig_y.update_layout(title="Position vs Time", xaxis_title="Time (s)", yaxis_title="Downward Position (m)")

    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_v, use_container_width=True)
    c2.plotly_chart(fig_y, use_container_width=True)

    st.plotly_chart(make_vertical_animation(y, "Animated Falling Object (With Drag)", "Downward Position (m)"), use_container_width=True)


with tab3:
    st.subheader("Inclined Plane Motion")
    col1, col2 = st.columns(2)

    with col1:
        m = st.number_input("Mass (kg)", min_value=0.01, value=10.0, step=0.1, key="incl_m")
        theta_deg = st.slider("Angle of incline (degrees)", 1, 89, 30)
        g = st.number_input("Gravity g (m/s²)", min_value=0.1, value=9.81, step=0.1, key="incl_g")
        use_friction = st.checkbox("Include friction", value=True)
        mu = st.number_input("Friction coefficient", min_value=0.0, value=0.2, step=0.01, disabled=not use_friction)
        u0 = st.number_input("Initial speed along slope (m/s)", value=0.0, step=0.1)
        t_max = st.slider("Simulation time (s)", 1, 30, 10, key="incl_tmax")
        n_pts = st.slider("Number of timesteps", 50, 800, 250, key="incl_n")

    theta = np.deg2rad(theta_deg)
    f_parallel = m * g * np.sin(theta)
    normal = m * g * np.cos(theta)
    friction = mu * normal if use_friction else 0.0
    f_net = f_parallel - friction
    if f_net < 0 and u0 <= 0:
        f_net = 0.0

    a = f_net / m

    t = np.linspace(0, t_max, n_pts)
    s = u0 * t + 0.5 * a * t**2
    v = u0 + a * t

    with col2:
        st.metric("Acceleration along slope (m/s²)", f"{a:.3f}")
        st.metric("Parallel force mg sin(theta) (N)", f"{f_parallel:.3f}")
        st.metric("Normal force mg cos(theta) (N)", f"{normal:.3f}")
        st.metric("Friction force (N)", f"{friction:.3f}")

    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(x=t, y=s, mode="lines", name="Displacement"))
    fig_s.update_layout(title="Displacement vs Time", xaxis_title="Time (s)", yaxis_title="Displacement along slope (m)")

    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(x=t, y=v, mode="lines", name="Velocity", line=dict(color="#ef553b")))
    fig_v.update_layout(title="Velocity vs Time", xaxis_title="Time (s)", yaxis_title="Velocity along slope (m/s)")

    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_s, use_container_width=True)
    c2.plotly_chart(fig_v, use_container_width=True)

    x_block = s * np.cos(theta)
    y_block = -s * np.sin(theta)
    slope_len = max(1.0, float(np.max(s)) + 2.0)
    slope_x = np.array([0.0, slope_len * np.cos(theta)])
    slope_y = np.array([0.0, -slope_len * np.sin(theta)])
    st.plotly_chart(
        make_slope_animation(x_block, y_block, slope_x, slope_y, "Animated Block on Inclined Plane"),
        use_container_width=True,
    )


with tab4:
    st.subheader("Changing Gravity Environment")
    col1, col2 = st.columns(2)

    gravity_options = {"Earth": 9.81, "Moon": 1.62, "Mars": 3.71, "Custom": None}

    with col1:
        m = st.number_input("Mass (kg)", min_value=0.1, value=5.0, step=0.1, key="env_m")
        env = st.selectbox("Select environment", list(gravity_options.keys()))
        custom_g = st.number_input("Custom gravity (m/s²)", min_value=0.01, value=5.0, step=0.1, disabled=(env != "Custom"))
        u = st.number_input("Initial velocity u (m/s)", value=0.0, step=0.1, key="env_u")
        t_max = st.slider("Simulation time (s)", 1, 30, 10, key="env_t")
        n_pts = st.slider("Number of timesteps", 50, 800, 250, key="env_n")

    g_selected = custom_g if env == "Custom" else gravity_options[env]
    weight = m * g_selected

    t = np.linspace(0, t_max, n_pts)
    y_selected = u * t + 0.5 * g_selected * t**2
    v_selected = u + g_selected * t

    with col2:
        st.metric("Selected gravity (m/s²)", f"{g_selected:.3f}")
        st.metric("Weight (N)", f"{weight:.3f}")

    fig_compare_pos = go.Figure()
    for name, gv in {"Earth": 9.81, "Moon": 1.62, "Mars": 3.71}.items():
        yy = u * t + 0.5 * gv * t**2
        fig_compare_pos.add_trace(go.Scatter(x=t, y=yy, mode="lines", name=f"{name} ({gv} m/s²)"))

    fig_compare_pos.add_trace(
        go.Scatter(x=t, y=y_selected, mode="lines", line=dict(width=4, dash="dot"), name=f"Selected: {env}")
    )
    fig_compare_pos.update_layout(title="Position vs Time under Different Gravity", xaxis_title="Time (s)", yaxis_title="Position (m)")

    fig_compare_vel = go.Figure()
    for name, gv in {"Earth": 9.81, "Moon": 1.62, "Mars": 3.71}.items():
        vv = u + gv * t
        fig_compare_vel.add_trace(go.Scatter(x=t, y=vv, mode="lines", name=f"{name} ({gv} m/s²)"))

    fig_compare_vel.add_trace(
        go.Scatter(x=t, y=v_selected, mode="lines", line=dict(width=4, dash="dot"), name=f"Selected: {env}")
    )
    fig_compare_vel.update_layout(title="Velocity vs Time under Different Gravity", xaxis_title="Time (s)", yaxis_title="Velocity (m/s)")

    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_compare_pos, use_container_width=True)
    c2.plotly_chart(fig_compare_vel, use_container_width=True)

    st.plotly_chart(make_vertical_animation(y_selected, f"Animated Motion in {env} Gravity"), use_container_width=True)


with tab5:
    st.subheader("Rocket Motion (Decreasing Mass -> Projectile Motion)")
    col1, col2 = st.columns(2)

    with col1:
        m0 = st.number_input("Initial mass (kg)", min_value=1.0, value=500.0, step=1.0)
        mdot = st.number_input("Fuel burn rate (kg/s)", min_value=0.01, value=2.0, step=0.1)
        ve = st.number_input("Exhaust velocity (m/s)", min_value=1.0, value=2000.0, step=10.0)
        burn_time = st.number_input("Burn time (s)", min_value=0.1, value=60.0, step=1.0)
        g = st.number_input("Gravity (m/s²)", min_value=0.1, value=9.81, step=0.1, key="rocket_g")

        st.markdown("Optional physical constraint")
        m_dry = st.number_input("Dry mass (kg)", min_value=0.1, max_value=float(m0), value=max(1.0, 0.3 * float(m0)), step=1.0)

        angle = st.slider("Launch angle (degrees)", 5, 90, 80)
        dt = st.slider("Time step dt (s)", 0.005, 0.2, 0.02, key="rocket_dt")
        t_max = st.slider("Max simulation time (s)", 5, 500, 200)

    if m_dry >= m0:
        st.warning("Dry mass must be lower than initial mass for fuel burn. Set dry mass < initial mass.")
    else:
        t, x, y, vx, vy, speed, mass_curve, burn_end = simulate_rocket_projectile(
            m0, m_dry, mdot, ve, burn_time, g, angle, dt, t_max
        )

        peak_idx = int(np.argmax(y))
        peak_height = float(y[peak_idx])
        time_to_peak = float(t[peak_idx])
        total_flight = float(t[-1])

        with col2:
            st.metric("Peak height (m)", f"{peak_height:.3f}")
            st.metric("Time to peak (s)", f"{time_to_peak:.3f}")
            st.metric("Total flight time (s)", f"{total_flight:.3f}")
            st.metric("Effective burn end (s)", f"{burn_end:.3f}")

        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=t, y=y, mode="lines", name="Height"))
        fig_h.update_layout(title="Height vs Time", xaxis_title="Time (s)", yaxis_title="Height (m)")

        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=t, y=speed, mode="lines", name="Speed", line=dict(color="#ef553b")))
        fig_v.add_trace(go.Scatter(x=t, y=vy, mode="lines", name="Vertical velocity", line=dict(dash="dash")))
        fig_v.update_layout(title="Velocity vs Time", xaxis_title="Time (s)", yaxis_title="Velocity (m/s)")

        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=t, y=mass_curve, mode="lines", name="Mass", line=dict(color="#ab63fa")))
        fig_m.update_layout(title="Mass vs Time", xaxis_title="Time (s)", yaxis_title="Mass (kg)")

        fig_traj = go.Figure()
        fig_traj.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Trajectory", line=dict(color="#00cc96")))
        fig_traj.update_layout(title="Rocket Trajectory", xaxis_title="x (m)", yaxis_title="y (m)")

        c1, c2 = st.columns(2)
        c1.plotly_chart(fig_h, use_container_width=True)
        c2.plotly_chart(fig_v, use_container_width=True)

        c3, c4 = st.columns(2)
        c3.plotly_chart(fig_m, use_container_width=True)
        c4.plotly_chart(fig_traj, use_container_width=True)

        st.plotly_chart(make_trajectory_animation(x, y), use_container_width=True)


st.markdown("---")
st.caption("Tip: run with `streamlit run app.py`")
