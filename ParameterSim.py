from matplotlib.text import Text
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class DraggablePlot:
    def __init__(self, ax):
        self.ax = ax
        self.press = None
        self.xlim = None
        self.ylim = None
        self.fig = ax.figure
        self.cidpress = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.press = event.xdata, event.ydata
        self.xlim = self.ax.get_xlim()
        self.ylim = self.ax.get_ylim()

    def on_release(self, event):
        self.press = None
        self.ax.figure.canvas.draw()

    def on_motion(self, event):
        if self.press is None or event.inaxes != self.ax:
            return

        xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress

        self.ax.set_xlim(self.xlim[0] - dx, self.xlim[1] - dx)
        self.ax.set_ylim(self.ylim[0] - dy, self.ylim[1] - dy)

        self.ax.figure.canvas.draw()

def zoom_factory(ax, base_scale=2.):
    def zoom_fun(event):
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
        plt.draw()

    fig1.canvas.mpl_connect('scroll_event', zoom_fun)

def update(val):
    xlim = ax_phase.get_xlim()
    ylim = ax_phase.get_ylim()

    b = slider_b.val
    c = slider_c.val
    d = slider_d.val
    e = slider_e.val
    f = slider_f.val
    x1bar = slider_x1bar.val
    x2bar = slider_x2bar.val
    tau1 = slider_tau1.val
    tau2 = slider_tau2.val
    initial_x1 = slider_initial_x1.val
    initial_x2 = slider_initial_x2.val

    x1, x2, z, mode_array, zdot_av, norm_mu11_array = run_simulation(b, c, d, e, f, initial_x1, initial_x2, x1bar, x2bar, int(tau1), int(tau2))

    ax_phase.cla()
    ax_phase.scatter(x1, x2, s=0.1, label='Phase Space x1 vs x2')

    ax_phase.set_xlim(xlim)
    ax_phase.set_ylim(ylim)

    ax_phase.set_xlabel('x1')
    ax_phase.set_ylabel('x2')
    ax_phase.set_title('Phase Space Trajectory (x1 vs x2)')
    ax_phase.grid(True)

    ax_hist.cla()
    ax_hist.hist(mode_array, bins=[-0.5, 0.5, 1.5, 2.5, 3.5], rwidth=0.8, align='mid')
    ax_hist.set_xticks([0, 1, 2, 3])
    ax_hist.set_xticklabels(['mu00', 'mu01', 'mu10', 'mu11'])
    ax_hist.set_title('State Histogram')
    ax_hist.set_xlabel('Modes')
    ax_hist.set_ylabel('Frequency')

    n = len(norm_mu11_array)  
    timestep_array = np.arange(1, n + 1)
    ax_mu11.cla()
    ax_mu11.plot(timestep_array, norm_mu11_array, label='Av time in mu11 over time steps')
    ax_mu11.set_title('Mu11 Normalized over Time')
    ax_mu11.set_xlabel('Time Steps')
    ax_mu11.set_ylabel('Normalized Mu11')

    #zdot_av_text.set_text(f'zdot_av11 = {zdot_av:.2f}')

    fig1.canvas.draw_idle()
    fig3.canvas.draw_idle()

def run_simulation(b, c, d, e, f, initial_x1, initial_x2, x1bar, x2bar, tau1, tau2, T=100000.0, epsilon=1.0):
    N = int(T / epsilon)
    #x1_dot = c - b * H1_delayed - e * H1_delayed * H2_delayed
    #x2_dot = c - d * H2_delayed
    #z_dot = -2 * c + f * H1_delayed * H2_delayed
    x1 = np.zeros(N)
    x2 = np.zeros(N)
    norm_mu11_array = np.zeros(N)
    z = np.zeros(N)
    mode_array = np.zeros(N, dtype=int)  

    x1[0] = initial_x1
    x2[0] = initial_x2
    z[0] = 0

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

    for n in range(1, N):
        idx_delay1 = (n - tau1) % delay_buffer_size1
        idx_delay2 = (n - tau2) % delay_buffer_size2

        H1_delayed = 1.0 if delay_buffer1[idx_delay1] - x1bar >= 0 else 0.0
        H2_delayed = 1.0 if delay_buffer2[idx_delay2] - x2bar >= 0 else 0.0

        x1_dot = c - b * H1_delayed - e * H1_delayed * H2_delayed
        x2_dot = c - d * H2_delayed

        x1[n] = x1[n - 1] + epsilon * x1_dot
        x2[n] = x2[n - 1] + epsilon * x2_dot

        delay_buffer1[n % delay_buffer_size1] = x1[n]
        delay_buffer2[n % delay_buffer_size2] = x2[n]

        
        z_dot = -2 * c + (f * H1_delayed * H2_delayed)
        z[n] = z[n - 1] + epsilon * z_dot

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
        norm_mu11_array[n] = time_mu11 / (time_mu00+time_mu01+time_mu10+time_mu11)
    print(f'norm_mu11 = {norm_mu11_array[-1]}')

    zdot_av = -2 * c + f * norm_mu11_array[-1]
    print(f'zdot_av = {zdot_av}')
    return x1, x2, z, mode_array, zdot_av, norm_mu11_array



fig1, (ax_phase, ax_hist) = plt.subplots(2, 1, figsize=(10, 8))  
fig2 = plt.figure(figsize=(8, 6))  
fig3, ax_mu11 = plt.subplots(1, 1, figsize=(10, 8)) 
plt.subplots_adjust(left=0.3, right=0.95)  
#zdot_av_text = fig1.text(0.1, 0.9, '', transform=fig1.transFigure, fontsize=12)
b_init = 1
c_init = 1
d_init = 1
e_init = 1
f_init = 1
initial_x1 = 1
initial_x2 = 1
x1bar_init = 1
x2bar_init = 1
tau1_init = 0
tau2_init = 0

x1, x2, z, mode_array, zdot_av, norm_mu11_array = run_simulation(b_init, c_init, d_init, e_init, f_init, initial_x1, initial_x2, x1bar_init, x2bar_init, tau1_init, tau2_init)
ax_phase.scatter(x1, x2, s=10, label='Phase Space x1 vs x2')
ax_phase.set_xlabel('x1')
ax_phase.set_ylabel('x2')
ax_phase.set_title('Phase Space Trajectory (x1 vs x2)')
ax_phase.grid(True)

n = len(norm_mu11_array)  
timestep_array = np.arange(1, n + 1)
ax_mu11.plot(timestep_array, norm_mu11_array, label='Av time in mu11 over time steps')
ax_mu11.set_title('Mu11 Normalized over Time')
ax_mu11.set_xlabel('Time Steps')
ax_mu11.set_ylabel('Normalized Mu11')



ax_hist.hist(mode_array, bins=[-0.5, 0.5, 1.5, 2.5, 3.5], rwidth=0.8, align='mid')
ax_hist.set_xticks([0, 1, 2, 3])
ax_hist.set_xticklabels(['mu00', 'mu01', 'mu10', 'mu11'])
ax_hist.set_title('State Histogram')
ax_hist.set_xlabel('Modes')
ax_hist.set_ylabel('Frequency')

ax_b = plt.axes([0.25, 0.45, 0.65, 0.03], facecolor='lightgoldenrodyellow', figure=fig2)
ax_c = plt.axes([0.25, 0.40, 0.65, 0.03], facecolor='lightgoldenrodyellow', figure=fig2)
ax_d = plt.axes([0.25, 0.35, 0.65, 0.03], facecolor='lightgoldenrodyellow', figure=fig2)
ax_e = plt.axes([0.25, 0.30, 0.65, 0.03], facecolor='lightgoldenrodyellow', figure=fig2)
ax_f = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor='lightgoldenrodyellow', figure=fig2)
ax_x1bar = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor='lightgoldenrodyellow', figure=fig2)
ax_x2bar = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow', figure=fig2)
ax_tau1 = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor='lightgoldenrodyellow', figure=fig2)
ax_tau2 = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow', figure=fig2)
ax_initial_x1 = plt.axes([0.25, 0.60, 0.65, 0.03], facecolor='lightgoldenrodyellow', figure=fig2)
ax_initial_x2 = plt.axes([0.25, 0.55, 0.65, 0.03], facecolor='lightgoldenrodyellow', figure=fig2)

slider_b = Slider(ax_b, 'b', 0.1, 2000.0, valinit=b_init)
slider_c = Slider(ax_c, 'c', 0.1, 2000.0, valinit=c_init)
slider_d = Slider(ax_d, 'd', 0.1, 2000.0, valinit=d_init)
slider_e = Slider(ax_e, 'e', 0.1, 2000.0, valinit=e_init)
slider_f = Slider(ax_f, 'f', 0.1, 2000.0, valinit=f_init)
slider_x1bar = Slider(ax_x1bar, 'x1bar', 1, 2000, valinit=x1bar_init)
slider_x2bar = Slider(ax_x2bar, 'x2bar', 1, 2000, valinit=x2bar_init)
slider_tau1 = Slider(ax_tau1, 'tau1', 0, 50, valinit=tau1_init)
slider_tau2 = Slider(ax_tau2, 'tau2', 0, 50, valinit=tau2_init)
slider_initial_x1 = Slider(ax_initial_x1, 'initial x1', 1, 2000, valinit=initial_x1)
slider_initial_x2 = Slider(ax_initial_x2, 'initial x2', 1, 2000, valinit=initial_x2)

slider_b.on_changed(update)
slider_c.on_changed(update)
slider_d.on_changed(update)
slider_e.on_changed(update)
slider_f.on_changed(update)
slider_x1bar.on_changed(update)
slider_x2bar.on_changed(update)
slider_tau1.on_changed(update)
slider_tau2.on_changed(update)
slider_initial_x1.on_changed(update)
slider_initial_x2.on_changed(update)
zoom_factory(ax_phase)
draggable_plot = DraggablePlot(ax_phase)
plt.show()
