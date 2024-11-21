import numpy as np
import matplotlib.pyplot as plt
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

def run_simulation(
    a_x1, c_x1_1, c_x1_2, c_x1_12,
    a_x2, c_x2_1, c_x2_2, c_x2_12,
    a_y, c_y_12,
    initial_x1, initial_x2,
    x1bar, x2bar, tau1, tau2,
    integration_method='Euler',
    heaviside_method='Standard', T=10000.0, epsilon=1.0
):
    N = int(T / epsilon)
    x1 = np.zeros(N)
    x2 = np.zeros(N)
    y = np.zeros(N)
    ydot_array = np.zeros(N)
    mode_array = np.zeros(N, dtype=int)

    x1[0] = initial_x1
    x2[0] = initial_x2
    y[0] = 0
    ydot_array[0] = 0

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

    norm_mu00_array = np.zeros(N)
    norm_mu01_array = np.zeros(N)
    norm_mu10_array = np.zeros(N)
    norm_mu11_array = np.zeros(N)

    for n in range(1, N):
        if tau1 > 0:
            idx_delay1 = (n - tau1) % delay_buffer_size1
            H1_delayed = H_function(delay_buffer1[idx_delay1] - x1bar)
        else:
            H1_delayed = H_function(x1[n - 1] - x1bar)

        if tau2 > 0:
            idx_delay2 = (n - tau2) % delay_buffer_size2
            H2_delayed = H_function(delay_buffer2[idx_delay2] - x2bar)
        else:
            H2_delayed = H_function(x2[n - 1] - x2bar)

        if integration_method == 'Euler':
            x1_dot = a_x1 + c_x1_1 * H1_delayed + c_x1_2 * H2_delayed + c_x1_12 * H1_delayed * H2_delayed
            x2_dot = a_x2 + c_x2_1 * H1_delayed + c_x2_2 * H2_delayed + c_x2_12 * H1_delayed * H2_delayed
            y_dot = a_y + c_y_12 * H1_delayed * H2_delayed

            x1[n] = x1[n - 1] + epsilon * x1_dot
            x2[n] = x2[n - 1] + epsilon * x2_dot
            y[n] = y[n - 1] + epsilon * y_dot
            ydot_array[n] = y_dot
        else:
            raise NotImplementedError(f"Integration method '{integration_method}' is not implemented.")

        if tau1 > 0:
            delay_buffer1[n % delay_buffer_size1] = x1[n]
        if tau2 > 0:
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

    ydot_av = a_y + c_y_12 * norm_mu11_array[-1]

    return (
        x1, x2, y, ydot_array, mode_array, ydot_av,
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

def get_variant_parameters(variant):
    if variant == 'variant_i':
        return {
            'a_x1': 2,
            'c_x1_1': -3,
            'c_x1_2': 1,
            'c_x1_12': -3,
            'a_x2': 2,
            'c_x2_1': 0,
            'c_x2_2': -3,
            'c_x2_12': -3,
            'a_y': -1,
            'c_y_12': 4
        }
    elif variant == 'variant_ii':
        return {
            'a_x1': 1,
            'c_x1_1': -3,
            'c_x1_2': 1,
            'c_x1_12': 0,
            'a_x2': 3,
            'c_x2_1': -1,
            'c_x2_2': -6,
            'c_x2_12': 0,
            'a_y': -1,
            'c_y_12': 6
        }
    elif variant == 'variant_iii':
        return {
            'a_x1': 3,
            'c_x1_1': -5,
            'c_x1_2': 0,
            'c_x1_12': 0,
            'a_x2': 1,
            'c_x2_1': 0,
            'c_x2_2': -5,
            'c_x2_12': 0,
            'a_y': -1,
            'c_y_12': 6
        }
    else:
        raise ValueError(f"Unknown variant: {variant}")

variants = ['variant_i', 'variant_ii', 'variant_iii']

variant_colors = {
    'variant_i': 'blue',
    'variant_ii': 'green',
    'variant_iii': 'red'
}

heaviside_method = 'Standard'
integration_method = 'Euler'

for variant in variants:
    params = get_variant_parameters(variant)
    
    a_x1 = params['a_x1']
    c_x1_1 = params['c_x1_1']
    c_x1_2 = params['c_x1_2']
    c_x1_12 = params['c_x1_12']
    a_x2 = params['a_x2']
    c_x2_1 = params['c_x2_1']
    c_x2_2 = params['c_x2_2']
    c_x2_12 = params['c_x2_12']
    a_y = params['a_y']
    c_y_12 = params['c_y_12']
    
    initial_x1 = 1.0
    initial_x2 = 1.0
    x1bar = 1.0
    x2bar = 1.0
    tau1 = 0
    tau2 = 0
    
    x1, x2, y, ydot_array, mode_array, ydot_av, norm_mu00_array, \
    norm_mu01_array, norm_mu10_array, norm_mu11_array = run_simulation(
        a_x1, c_x1_1, c_x1_2, c_x1_12,
        a_x2, c_x2_1, c_x2_2, c_x2_12,
        a_y, c_y_12,
        initial_x1, initial_x2, x1bar, x2bar,
        tau1, tau2, integration_method=integration_method,
        heaviside_method=heaviside_method
    )
    
    N = len(x1)
    time_steps = np.arange(N)
    
    fig_phase_hist, axs_phase_hist = plt.subplots(2, 1, figsize=(15, 20))
    fig_phase_hist.suptitle(f'Dynamics for {variant}', fontsize=20)
    
    ax_phase = axs_phase_hist[0]
    scatter = ax_phase.scatter(x1, x2, c=time_steps, cmap='viridis', s=10, norm=plt.Normalize(vmin=0, vmax=N-1))  # Increased size to s=10
    ax_phase.set_xlabel('x1')
    ax_phase.set_ylabel('x2')
    ax_phase.set_title('Phase Space Trajectory (x1 vs x2)')
    ax_phase.grid(True)
    
    cbar_phase = fig_phase_hist.colorbar(scatter, ax=ax_phase)
    cbar_phase.set_label('Time Step')
    
    ax_phase.text(0.05, 0.95, f'ydot_av = {ydot_av:.2f}', transform=ax_phase.transAxes,
                  fontsize=14, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    zoom_factory(ax_phase)
    draggable_plot = DraggablePlot(ax_phase)
    
    ax_hist = axs_phase_hist[1]
    ax_hist.hist(mode_array, bins=[-0.5, 0.5, 1.5, 2.5, 3.5], 
                rwidth=0.8, align='mid', color=variant_colors[variant], alpha=0.7)
    ax_hist.set_xticks([0, 1, 2, 3])
    ax_hist.set_xticklabels(['mu00', 'mu01', 'mu10', 'mu11'])
    ax_hist.set_title('State Histogram')
    ax_hist.set_xlabel('Modes')
    ax_hist.set_ylabel('Frequency')
    ax_hist.grid(True)
    
    fig_phase_hist.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    fig_mu, axs_mu = plt.subplots(4, 1, figsize=(15, 25))
    fig_mu.suptitle(f'mu Normalized over Time for {variant}', fontsize=20)
    
    axs_mu[0].plot(time_steps, norm_mu00_array, label='mu00', color='red')
    axs_mu[0].set_title('mu00 Normalized over Time')
    axs_mu[0].set_xlabel('Time Steps')
    axs_mu[0].set_ylabel('Normalized mu00')
    axs_mu[0].grid(True)
    axs_mu[0].legend()
    
    axs_mu[1].plot(time_steps, norm_mu01_array, label='mu01', color='green')
    axs_mu[1].set_title('mu01 Normalized over Time')
    axs_mu[1].set_xlabel('Time Steps')
    axs_mu[1].set_ylabel('Normalized mu01')
    axs_mu[1].grid(True)
    axs_mu[1].legend()
    
    axs_mu[2].plot(time_steps, norm_mu10_array, label='mu10', color='orange')
    axs_mu[2].set_title('mu10 Normalized over Time')
    axs_mu[2].set_xlabel('Time Steps')
    axs_mu[2].set_ylabel('Normalized mu10')
    axs_mu[2].grid(True)
    axs_mu[2].legend()
    
    axs_mu[3].plot(time_steps, norm_mu11_array, label='mu11', color='purple')
    axs_mu[3].set_title('mu11 Normalized over Time')
    axs_mu[3].set_xlabel('Time Steps')
    axs_mu[3].set_ylabel('Normalized mu11')
    axs_mu[3].grid(True)
    axs_mu[3].legend()
    
    fig_mu.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    fig_y, ax_y = plt.subplots(1, 1, figsize=(15, 10))
    fig_y.suptitle(f'y over Time for {variant}', fontsize=20)
    
    ax_y.plot(time_steps, y, label='y over time', color='blue')
    ax_y.set_title('y over Time')
    ax_y.set_xlabel('Time Steps')
    ax_y.set_ylabel('y')
    ax_y.grid(True)
    ax_y.legend()
    
    fig_y.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.show()
