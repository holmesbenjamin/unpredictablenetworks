import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def update(val):
    a = slider_a.val
    b = slider_b.val
    b5 = slider_b5.val
    xbar = slider_xbar.val
    tau = int(slider_tau.val)
    initial_x = slider_initial_x.val
    N_agents = int(slider_N_agents.val)

    z, zdot_array, S_array = run_simulation(
        a, b, b5, initial_x, xbar, tau, N_agents
    )

    n = len(z)
    timestep_array = np.arange(1, n + 1)

    ax_z.cla()
    ax_z.plot(timestep_array, z, label='z over time')
    ax_z.set_title('z over Time')
    ax_z.set_xlabel('Time Steps')
    ax_z.set_ylabel('z')
    ax_z.grid(True)

    ax_zdot.cla()
    ax_zdot.plot(timestep_array, zdot_array, label='zdot over time')
    ax_zdot.set_title('zdot over Time')
    ax_zdot.set_xlabel('Time Steps')
    ax_zdot.set_ylabel('zdot')
    ax_zdot.grid(True)

    ax_S.cla()
    ax_S.plot(timestep_array, S_array, label='S over time')
    ax_S.set_title('Number of Agents in State 1 over Time')
    ax_S.set_xlabel('Time Steps')
    ax_S.set_ylabel('S')
    ax_S.grid(True)

    fig.canvas.draw_idle()

def run_simulation(a, b, b5, initial_x, xbar, tau, N_agents, T=5000.0, epsilon=1.0):
    N = int(T / epsilon)
    x = np.zeros((N_agents, N))
    z = np.zeros(N)
    zdot_array = np.zeros(N)
    S_array = np.zeros(N)

    x[:, 0] = initial_x

    delay_buffer_size = max(tau + 1, 1)
    delay_buffers = np.zeros((N_agents, delay_buffer_size))
    delay_buffers[:, 0] = x[:, 0]

    for n in range(1, N):
        idx_delay = (n - tau) % delay_buffer_size

        H_delayed = np.where(delay_buffers[:, idx_delay] - xbar >= 0, 1.0, 0.0)

        x_dot = a - b * H_delayed
        x[:, n] = x[:, n - 1] + epsilon * x_dot

        delay_buffers[:, n % delay_buffer_size] = x[:, n]

        S = np.sum(H_delayed)
        S_array[n] = S

        z_dot = -N_agents * a + b5 * S
        z[n] = z[n - 1] + epsilon * z_dot
        zdot_array[n] = z_dot

    return z, zdot_array, S_array

a_init = 1.0
b_init = 1.0
b5_init = 1.0
initial_x_init = 1.0
xbar_init = 1.0
tau_init = 0
N_agents_init = 2

z, zdot_array, S_array = run_simulation(
    a_init, b_init, b5_init, initial_x_init, xbar_init, tau_init, N_agents_init
)

fig, (ax_z, ax_zdot, ax_S) = plt.subplots(3, 1, figsize=(10, 12))
fig.subplots_adjust(left=0.1, bottom=0.35)

n = len(z)
timestep_array = np.arange(1, n + 1)

ax_z.plot(timestep_array, z, label='z over time')
ax_z.set_title('z over Time')
ax_z.set_xlabel('Time Steps')
ax_z.set_ylabel('z')
ax_z.grid(True)

ax_zdot.plot(timestep_array, zdot_array, label='zdot over time')
ax_zdot.set_title('zdot over Time')
ax_zdot.set_xlabel('Time Steps')
ax_zdot.set_ylabel('zdot')
ax_zdot.grid(True)

ax_S.plot(timestep_array, S_array, label='S over time')
ax_S.set_title('Number of Agents in State 1 over Time')
ax_S.set_xlabel('Time Steps')
ax_S.set_ylabel('S')
ax_S.grid(True)

axcolor = 'lightgoldenrodyellow'
ax_N_agents = plt.axes([0.15, 0.30, 0.75, 0.03], facecolor=axcolor)
ax_a = plt.axes([0.15, 0.25, 0.75, 0.03], facecolor=axcolor)
ax_b = plt.axes([0.15, 0.20, 0.75, 0.03], facecolor=axcolor)
ax_b5 = plt.axes([0.15, 0.15, 0.75, 0.03], facecolor=axcolor)
ax_xbar = plt.axes([0.15, 0.10, 0.75, 0.03], facecolor=axcolor)
ax_tau = plt.axes([0.15, 0.05, 0.75, 0.03], facecolor=axcolor)
ax_initial_x = plt.axes([0.15, 0.00, 0.75, 0.03], facecolor=axcolor)

slider_N_agents = Slider(ax_N_agents, 'Number of Agents', 1, 20, valinit=N_agents_init, valfmt='%0.0f', valstep=1)
slider_a = Slider(ax_a, 'a', -2000, 2000.0, valinit=a_init)
slider_b = Slider(ax_b, 'b', -2000, 2000.0, valinit=b_init)
slider_b5 = Slider(ax_b5, 'b5', -2000, 2000.0, valinit=b5_init)
slider_xbar = Slider(ax_xbar, 'xbar', 1, 2000, valinit=xbar_init)
slider_tau = Slider(ax_tau, 'tau', 0, 50, valinit=tau_init, valfmt='%0.0f', valstep=1)
slider_initial_x = Slider(ax_initial_x, 'initial x', 1, 2000, valinit=initial_x_init)

slider_N_agents.on_changed(update)
slider_a.on_changed(update)
slider_b.on_changed(update)
slider_b5.on_changed(update)
slider_xbar.on_changed(update)
slider_tau.on_changed(update)
slider_initial_x.on_changed(update)

plt.show()
