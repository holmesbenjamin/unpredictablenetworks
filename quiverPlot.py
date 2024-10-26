import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import matplotlib.cm as cm

class DraggablePlot:
    def __init__(self, ax):
        self.ax = ax
        self.press = None
        self.xlim = None
        self.ylim = None
        self.fig = ax.figure
        self.connect()

    def connect(self):
        self.cidpress = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.cidpress)
        self.fig.canvas.mpl_disconnect(self.cidrelease)
        self.fig.canvas.mpl_disconnect(self.cidmotion)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.press = event.xdata, event.ydata
        self.xlim = self.ax.get_xlim()
        self.ylim = self.ax.get_ylim()

    def on_release(self, event):
        self.press = None
        self.ax.figure.canvas.draw_idle()

    def on_motion(self, event):
        if self.press is None or event.inaxes != self.ax:
            return
        xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.ax.set_xlim(self.xlim[0] - dx, self.xlim[1] - dx)
        self.ax.set_ylim(self.ylim[0] - dy, self.ylim[1] - dy)
        self.ax.figure.canvas.draw_idle()

def zoom_factory(ax, base_scale=2.):
    def zoom_fun(event):
        if event.inaxes != ax:
            return
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata 
        ydata = event.ydata   

        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            return

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        ax.set_xlim([xdata - new_width / 2, xdata + new_width / 2])
        ax.set_ylim([ydata - new_height / 2, ydata + new_height / 2])
        ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect('scroll_event', zoom_fun)

def update(val):
    global cbar_phase  
    heaviside_method = radio_heaviside.value_selected
    integration_method = radio_integration.value_selected

    draggable_plot.disconnect()

    xlim = ax_phase.get_xlim()
    ylim = ax_phase.get_ylim()

    a = slider_a.val
    delta = 0.01  

    slider_b1.valmin = a + delta
    if slider_b1.val < slider_b1.valmin:
        slider_b1.set_val(slider_b1.valmin)
    b1 = slider_b1.val

    slider_b2.valmin = a + delta
    if slider_b2.val < slider_b2.valmin:
        slider_b2.set_val(slider_b2.valmin)
    b2 = slider_b2.val

    slider_b4.valmin = a + delta
    if slider_b4.val < slider_b4.valmin:
        slider_b4.set_val(slider_b4.valmin)
    b4 = slider_b4.val

    min_b3 = max(a - (b1 + b2) + delta, delta)
    slider_b3.valmin = min_b3
    if slider_b3.val < slider_b3.valmin:
        slider_b3.set_val(slider_b3.valmin)
    b3 = slider_b3.val

    b5 = slider_b5.val
    x1bar = slider_x1bar.val
    x2bar = slider_x2bar.val
    tau1 = slider_tau1.val
    tau2 = slider_tau2.val
    initial_x1 = slider_initial_x1.val
    initial_x2 = slider_initial_x2.val

    x1, x2, z, zdot_array, mode_array, zdot_av, norm_mu00_array, \
    norm_mu01_array, norm_mu10_array, norm_mu11_array = run_simulation(
        a, b1, b2, b3, b4, b5, initial_x1, initial_x2, x1bar, x2bar,
        int(tau1), int(tau2), integration_method=integration_method,
        heaviside_method=heaviside_method
    )

    ax_phase.cla()

    dx = x1[1:] - x1[:-1]
    dy = x2[1:] - x2[:-1]
    skip = 10  

    arrow_indices = np.arange(0, len(x1) - 1, skip)

    norm = plt.Normalize(0, len(x1) - 1)
    normalized_time = norm(arrow_indices)

    colors = cmap(normalized_time)

    quiver = ax_phase.quiver(
        x1[:-1][arrow_indices], x2[:-1][arrow_indices],
        dx[arrow_indices], dy[arrow_indices],
        scale_units='xy', angles='xy', scale=1, width=0.002,
        color=colors
    )

    ax_phase.set_xlim(xlim)
    ax_phase.set_ylim(ylim)

    ax_phase.set_xlabel('x1')
    ax_phase.set_ylabel('x2')
    ax_phase.set_title('Phase Space Trajectory with Direction')
    ax_phase.grid(True)

    sm.set_norm(norm)
    sm.set_array([])
    cbar_phase.update_normal(sm)

    ax_hist.cla()
    ax_hist.hist(
        mode_array, bins=[-0.5, 0.5, 1.5, 2.5, 3.5],
        rwidth=0.8, align='mid'
    )
    ax_hist.set_xticks([0, 1, 2, 3])
    ax_hist.set_xticklabels(['mu00', 'mu01', 'mu10', 'mu11'])
    ax_hist.set_title('State Histogram')
    ax_hist.set_xlabel('Modes')
    ax_hist.set_ylabel('Frequency')

    n = len(norm_mu11_array)
    timestep_array = np.arange(1, n + 1)

    ax_mu11.cla()
    ax_mu11.plot(timestep_array, norm_mu11_array, label='mu11')
    ax_mu11.set_title('mu11 Normalized over Time')
    ax_mu11.set_xlabel('Time Steps')
    ax_mu11.set_ylabel('Normalized mu11')

    ax_mu10.cla()
    ax_mu10.plot(
        timestep_array, norm_mu10_array, label='mu10', color='orange'
    )
    ax_mu10.set_title('mu10 Normalized over Time')
    ax_mu10.set_xlabel('Time Steps')
    ax_mu10.set_ylabel('Normalized mu10')

    ax_mu01.cla()
    ax_mu01.plot(
        timestep_array, norm_mu01_array, label='mu01', color='green'
    )
    ax_mu01.set_title('mu01 Normalized over Time')
    ax_mu01.set_xlabel('Time Steps')
    ax_mu01.set_ylabel('Normalized mu01')

    ax_mu00.cla()
    ax_mu00.plot(
        timestep_array, norm_mu00_array, label='mu00', color='red'
    )
    ax_mu00.set_title('mu00 Normalized over Time')
    ax_mu00.set_xlabel('Time Steps')
    ax_mu00.set_ylabel('Normalized mu00')

    ax_z.cla()
    ax_z.plot(timestep_array, z, label='z over time')
    ax_z.set_title('z over Time')
    ax_z.set_xlabel('Time Steps')
    ax_z.set_ylabel('z')

    ax_zdot.cla()
    ax_zdot.plot(timestep_array, zdot_array, label='zdot over time')
    ax_zdot.set_title('zdot over Time')
    ax_zdot.set_xlabel('Time Steps')
    ax_zdot.set_ylabel('zdot')

    zdot_av_text.set_text(f'zdot_av11 = {zdot_av:.2f}')

    draggable_plot.connect()

    fig1.canvas.draw_idle()
    fig3.canvas.draw_idle()
    fig4.canvas.draw_idle()

def run_simulation(
    a, b1, b2, b3, b4, b5, initial_x1, initial_x2,
    x1bar, x2bar, tau1, tau2, integration_method='Euler',
    heaviside_method='Standard', T=1000.0, epsilon=1.0
):
    N = int(T / epsilon)
    x1 = np.zeros(N)
    x2 = np.zeros(N)
    norm_mu00_array = np.zeros(N)
    norm_mu01_array = np.zeros(N)
    norm_mu10_array = np.zeros(N)
    norm_mu11_array = np.zeros(N)
    z = np.zeros(N)
    zdot_array = np.zeros(N)
    mode_array = np.zeros(N, dtype=int)

    x1[0] = initial_x1
    x2[0] = initial_x2
    z[0] = 0
    zdot_array[0] = 0

    delay_buffer_size1 = max(tau1 + 1, 1)
    delay_buffer_size2 = max(tau2 + 1, 1)
    delay_buffer1 = np.zeros(delay_buffer_size1)
    delay_buffer2 = np.zeros(delay_buffer_size2)
    delay_buffer1[0] = x1[0]
    delay_buffer2[0] = x2[0]

    time_mu00 = 0.0
    time_mu01 = 0.0
    time_mu10 = 0.0
    time_mu11 = 0.0

    H_function = get_heaviside_function(heaviside_method)

    for n in range(1, N):
        idx_delay1 = (n - tau1) % delay_buffer_size1
        idx_delay2 = (n - tau2) % delay_buffer_size2

        H1_delayed = H_function(delay_buffer1[idx_delay1] - x1bar)
        H2_delayed = H_function(delay_buffer2[idx_delay2] - x2bar)

        if integration_method == 'Euler':
            x1_dot = a - b1 * H1_delayed - b2 * H2_delayed - b3 * H1_delayed * H2_delayed
            x2_dot = a - b4 * H2_delayed
            x1[n] = x1[n - 1] + epsilon * x1_dot
            x2[n] = x2[n - 1] + epsilon * x2_dot
            z_dot = -2 * a + (b5 * H1_delayed * H2_delayed)
            z[n] = z[n - 1] + epsilon * z_dot
            zdot_array[n] = z_dot
        else:
            raise NotImplementedError(f"Integration method '{integration_method}' is not implemented.")

        delay_buffer1[n % delay_buffer_size1] = x1[n]
        delay_buffer2[n % delay_buffer_size2] = x2[n]

        if H1_delayed == 0.0 and H2_delayed == 0.0:
            time_mu00 += epsilon
            mode_array[n] = 0
        elif H1_delayed == 0.0 and H2_delayed == 1.0:
            time_mu01 += epsilon
            mode_array[n] = 1
        elif H1_delayed == 1.0 and H2_delayed == 0.0:
            time_mu10 += epsilon
            mode_array[n] = 2
        elif H1_delayed == 1.0 and H2_delayed == 1.0:
            time_mu11 += epsilon
            mode_array[n] = 3

        total_time = time_mu00 + time_mu01 + time_mu10 + time_mu11
        if total_time > 0:
            norm_mu00_array[n] = time_mu00 / total_time
            norm_mu01_array[n] = time_mu01 / total_time
            norm_mu10_array[n] = time_mu10 / total_time
            norm_mu11_array[n] = time_mu11 / total_time
        else:
            norm_mu00_array[n] = 0
            norm_mu01_array[n] = 0
            norm_mu10_array[n] = 0
            norm_mu11_array[n] = 0

    zdot_av = -2 * a + (b5 * norm_mu11_array[-1])
    return (
        x1, x2, z, zdot_array, mode_array, zdot_av,
        norm_mu00_array, norm_mu01_array, norm_mu10_array, norm_mu11_array
    )

def get_heaviside_function(method_name):
    if method_name == 'Standard':
        return standard_heaviside
    elif method_name == 'Smooth':
        return lambda x: smooth_heaviside(x, k=10)
    elif method_name == 'Regularized':
        return lambda x: regularized_heaviside(x, k=10)
    elif method_name == 'Polynomial':
        return lambda x: polynomial_heaviside(x, delta=0.1)
    elif method_name == 'Piecewise Linear':
        return lambda x: piecewise_linear_heaviside(x, delta=0.1)
    else:
        raise ValueError(f"Unknown Heaviside method: {method_name}")

def standard_heaviside(x):
    return 1.0 if x >= 0 else 0.0

def smooth_heaviside(x, k=10):
    return 1.0 / (1.0 + np.exp(-k * x))

def regularized_heaviside(x, k=10):
    return 0.5 * (1.0 + np.tanh(k * x))

def polynomial_heaviside(x, delta=0.1):
    if x < -delta:
        return 0.0
    elif x > delta:
        return 1.0
    else:
        return 0.5 + (x / (2 * delta)) + (1.0 / (2.0 * np.pi)) * np.sin(np.pi * x / delta)

def piecewise_linear_heaviside(x, delta=0.1):
    if x < -delta:
        return 0.0
    elif x > delta:
        return 1.0
    else:
        return (x + delta) / (2.0 * delta)

cbar_phase = None  
cmap = cm.viridis  

fig1, (ax_phase, ax_hist) = plt.subplots(2, 1, figsize=(10, 8))  
fig2 = plt.figure(figsize=(10, 6))  
fig3, ((ax_mu11, ax_mu10), (ax_mu01, ax_mu00)) = plt.subplots(2, 2, figsize=(12, 10)) 
fig4, (ax_z, ax_zdot) = plt.subplots(2, 1, figsize=(10, 8))
fig2.subplots_adjust(left=0.25, right=0.95)  

zdot_av_text = fig1.text(0.1, 0.9, '', transform=fig1.transFigure, fontsize=12)

a_init = 1.0
delta = 0.01  # Small delta to ensure strict inequality
b1_init = a_init + 1.0  # Must be greater than a_init
b2_init = a_init + 1.0  # Must be greater than a_init
b3_init = (a_init + 1.0)  # Must satisfy b1 + b2 + b3 > a
b4_init = a_init + 1.0  # Must be greater than a_init
b5_init = 1.0
initial_x1 = 1.0
initial_x2 = 1.0
x1bar_init = 1.0
x2bar_init = 1.0
tau1_init = 0
tau2_init = 0

x1, x2, z, zdot_array, mode_array, zdot_av, \
norm_mu00_array, norm_mu01_array, norm_mu10_array, norm_mu11_array = run_simulation(
    a_init, b1_init, b2_init, b3_init, b4_init, b5_init,
    initial_x1, initial_x2, x1bar_init, x2bar_init, tau1_init, tau2_init,
    integration_method='Euler', heaviside_method='Standard'
)

dx = x1[1:] - x1[:-1]
dy = x2[1:] - x2[:-1]
skip = 10  

arrow_indices = np.arange(0, len(x1) - 1, skip)

norm = plt.Normalize(0, len(x1) - 1)
normalized_time = norm(arrow_indices)

colors = cmap(normalized_time)

quiver = ax_phase.quiver(
    x1[:-1][arrow_indices], x2[:-1][arrow_indices],
    dx[arrow_indices], dy[arrow_indices],
    scale_units='xy', angles='xy', scale=1, width=0.002,
    color=colors
)

ax_phase.set_xlabel('x1')
ax_phase.set_ylabel('x2')
ax_phase.set_title('Phase Space Trajectory with Direction')
ax_phase.grid(True)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_phase = fig1.colorbar(sm, ax=ax_phase, label='Time Step')

n = len(norm_mu11_array)  
timestep_array = np.arange(1, n + 1)

ax_mu11.plot(timestep_array, norm_mu11_array, label='mu11')
ax_mu11.set_title('mu11 Normalized over Time')
ax_mu11.set_xlabel('Time Steps')
ax_mu11.set_ylabel('Normalized mu11')

ax_mu10.plot(timestep_array, norm_mu10_array, label='mu10', color='orange')
ax_mu10.set_title('mu10 Normalized over Time')
ax_mu10.set_xlabel('Time Steps')
ax_mu10.set_ylabel('Normalized mu10')

ax_mu01.plot(timestep_array, norm_mu01_array, label='mu01', color='green')
ax_mu01.set_title('mu01 Normalized over Time')
ax_mu01.set_xlabel('Time Steps')
ax_mu01.set_ylabel('Normalized mu01')

ax_mu00.plot(timestep_array, norm_mu00_array, label='mu00', color='red')
ax_mu00.set_title('mu00 Normalized over Time')
ax_mu00.set_xlabel('Time Steps')
ax_mu00.set_ylabel('Normalized mu00')

ax_hist.hist(
    mode_array, bins=[-0.5, 0.5, 1.5, 2.5, 3.5],
    rwidth=0.8, align='mid'
)
ax_hist.set_xticks([0, 1, 2, 3])
ax_hist.set_xticklabels(['mu00', 'mu01', 'mu10', 'mu11'])
ax_hist.set_title('State Histogram')
ax_hist.set_xlabel('Modes')
ax_hist.set_ylabel('Frequency')

ax_z.plot(timestep_array, z, label='z over time')
ax_z.set_title('z over Time')
ax_z.set_xlabel('Time Steps')
ax_z.set_ylabel('z')

ax_zdot.plot(timestep_array, zdot_array, label='zdot over time')
ax_zdot.set_title('zdot over Time')
ax_zdot.set_xlabel('Time Steps')
ax_zdot.set_ylabel('zdot')

ax_a = fig2.add_axes([0.25, 0.50, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_b1 = fig2.add_axes([0.25, 0.45, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_b2 = fig2.add_axes([0.25, 0.40, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_b3 = fig2.add_axes([0.25, 0.35, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_b4 = fig2.add_axes([0.25, 0.30, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_b5 = fig2.add_axes([0.25, 0.25, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_x1bar = fig2.add_axes([0.25, 0.20, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_x2bar = fig2.add_axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_tau1 = fig2.add_axes([0.25, 0.10, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_tau2 = fig2.add_axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_initial_x1 = fig2.add_axes([0.25, 0.65, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_initial_x2 = fig2.add_axes([0.25, 0.60, 0.65, 0.03], facecolor='lightgoldenrodyellow')

slider_a = Slider(ax_a, 'a', delta, 2000.0, valinit=a_init)
slider_b1 = Slider(ax_b1, 'b1', a_init + delta, 2000.0, valinit=b1_init)
slider_b2 = Slider(ax_b2, 'b2', a_init + delta, 2000.0, valinit=b2_init)
slider_b3 = Slider(ax_b3, 'b3', delta, 2000.0, valinit=b3_init)
slider_b4 = Slider(ax_b4, 'b4', a_init + delta, 2000.0, valinit=b4_init)
slider_b5 = Slider(ax_b5, 'b5', -2000.0, 2000.0, valinit=b5_init)
slider_x1bar = Slider(ax_x1bar, 'x1bar', 1, 2000, valinit=x1bar_init)
slider_x2bar = Slider(ax_x2bar, 'x2bar', 1, 2000, valinit=x2bar_init)
slider_tau1 = Slider(ax_tau1, 'tau1', 0, 50, valinit=tau1_init)
slider_tau2 = Slider(ax_tau2, 'tau2', 0, 50, valinit=tau2_init)
slider_initial_x1 = Slider(ax_initial_x1, 'initial x1', 1, 2000, valinit=initial_x1)
slider_initial_x2 = Slider(ax_initial_x2, 'initial x2', 1, 2000, valinit=initial_x2)

slider_a.on_changed(update)
slider_b1.on_changed(update)
slider_b2.on_changed(update)
slider_b3.on_changed(update)
slider_b4.on_changed(update)
slider_b5.on_changed(update)
slider_x1bar.on_changed(update)
slider_x2bar.on_changed(update)
slider_tau1.on_changed(update)
slider_tau2.on_changed(update)
slider_initial_x1.on_changed(update)
slider_initial_x2.on_changed(update)

ax_heaviside = fig2.add_axes([0.05, 0.25, 0.15, 0.6], facecolor='lightgoldenrodyellow')
ax_integration = fig2.add_axes([0.05, 0.05, 0.15, 0.15], facecolor='lightgoldenrodyellow')

heaviside_methods = ('Standard', 'Smooth', 'Regularized', 'Polynomial', 'Piecewise Linear')
radio_heaviside = RadioButtons(ax_heaviside, heaviside_methods)

integration_methods = ('Euler',)  # Only Euler implemented
radio_integration = RadioButtons(ax_integration, integration_methods)

radio_heaviside.on_clicked(update)
radio_integration.on_clicked(update)

zoom_factory(ax_phase)
draggable_plot = DraggablePlot(ax_phase)
plt.show()
