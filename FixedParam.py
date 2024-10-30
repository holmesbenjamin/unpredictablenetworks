import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

max_tau = 50  
timesteps = 5000.0

def update(val):
    heaviside_method = radio_heaviside.value_selected
    integration_method = radio_integration.value_selected

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
    initial_x1 = slider_initial_x1.val
    initial_x2 = slider_initial_x2.val


    tau1_values = np.arange(0, max_tau + 1)
    mu00_tau1 = np.zeros(len(tau1_values))
    mu01_tau1 = np.zeros(len(tau1_values))
    mu10_tau1 = np.zeros(len(tau1_values))
    mu11_tau1 = np.zeros(len(tau1_values))

    for idx, tau1_val in enumerate(tau1_values):
        _, _, _, _, _, _, norm_mu00_array, norm_mu01_array, norm_mu10_array, norm_mu11_array = run_simulation(
            a, b1, b2, b3, b4, b5,
            initial_x1, initial_x2,
            x1bar, x2bar,
            int(tau1_val), 0,
            integration_method=integration_method,
            heaviside_method=heaviside_method,
            T=timesteps 
        )

        mu00_tau1[idx] = norm_mu00_array[-1]
        mu01_tau1[idx] = norm_mu01_array[-1]
        mu10_tau1[idx] = norm_mu10_array[-1]
        mu11_tau1[idx] = norm_mu11_array[-1]

    ax_tau1.cla()
    ax_tau1.plot(tau1_values, mu00_tau1, label='mu00')
    ax_tau1.plot(tau1_values, mu01_tau1, label='mu01')
    ax_tau1.plot(tau1_values, mu10_tau1, label='mu10')
    ax_tau1.plot(tau1_values, mu11_tau1, label='mu11')
    ax_tau1.set_xlabel('tau1')
    ax_tau1.set_ylabel('Final mu probabilities')
    ax_tau1.set_title('Mu probabilities vs tau1 (tau2=0)')
    ax_tau1.legend()

    tau2_values = np.arange(0, max_tau + 1)
    mu00_tau2 = np.zeros(len(tau2_values))
    mu01_tau2 = np.zeros(len(tau2_values))
    mu10_tau2 = np.zeros(len(tau2_values))
    mu11_tau2 = np.zeros(len(tau2_values))

    for idx, tau2_val in enumerate(tau2_values):
        _, _, _, _, _, _, norm_mu00_array, norm_mu01_array, norm_mu10_array, norm_mu11_array = run_simulation(
            a, b1, b2, b3, b4, b5,
            initial_x1, initial_x2,
            x1bar, x2bar,
            0, int(tau2_val),
            integration_method=integration_method,
            heaviside_method=heaviside_method,
            T=timesteps 
        )

        mu00_tau2[idx] = norm_mu00_array[-1]
        mu01_tau2[idx] = norm_mu01_array[-1]
        mu10_tau2[idx] = norm_mu10_array[-1]
        mu11_tau2[idx] = norm_mu11_array[-1]

    ax_tau2.cla()
    ax_tau2.plot(tau2_values, mu00_tau2, label='mu00')
    ax_tau2.plot(tau2_values, mu01_tau2, label='mu01')
    ax_tau2.plot(tau2_values, mu10_tau2, label='mu10')
    ax_tau2.plot(tau2_values, mu11_tau2, label='mu11')
    ax_tau2.set_xlabel('tau2')
    ax_tau2.set_ylabel('Final mu probabilities')
    ax_tau2.set_title('Mu probabilities vs tau2 (tau1=0)')
    ax_tau2.legend()

    fig_mu.canvas.draw_idle()

def run_simulation(
    a, b1, b2, b3, b4, b5, initial_x1, initial_x2,
    x1bar, x2bar, tau1, tau2, integration_method='Euler',
    heaviside_method='Standard', T=timesteps, epsilon=1.0
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



fig_sliders = plt.figure(figsize=(10, 6))
fig_sliders.subplots_adjust(left=0.25, right=0.95)

fig_mu, (ax_tau1, ax_tau2) = plt.subplots(1, 2, figsize=(14, 6))
fig_mu.subplots_adjust(wspace=0.3)

a_init = 1.0
delta = 0.01  
b1_init = a_init + 1.0  # greater than a_init
b2_init = a_init + 1.0  # greater than a_init
b3_init = (a_init + 1.0)  # must satisfy b1 + b2 + b3 > a
b4_init = a_init + 1.0  # greater than a_init
b5_init = 1.0
initial_x1 = 1.0
initial_x2 = 1.0
x1bar_init = 1.0
x2bar_init = 1.0

ax_a = fig_sliders.add_axes([0.25, 0.50, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_b1 = fig_sliders.add_axes([0.25, 0.45, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_b2 = fig_sliders.add_axes([0.25, 0.40, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_b3 = fig_sliders.add_axes([0.25, 0.35, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_b4 = fig_sliders.add_axes([0.25, 0.30, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_b5 = fig_sliders.add_axes([0.25, 0.25, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_x1bar = fig_sliders.add_axes([0.25, 0.20, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_x2bar = fig_sliders.add_axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_initial_x1 = fig_sliders.add_axes([0.25, 0.10, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_initial_x2 = fig_sliders.add_axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')

slider_a = Slider(ax_a, 'a', delta, 2000.0, valinit=a_init)
slider_b1 = Slider(ax_b1, 'b1', a_init + delta, 2000.0, valinit=b1_init)
slider_b2 = Slider(ax_b2, 'b2', a_init + delta, 2000.0, valinit=b2_init)
slider_b3 = Slider(ax_b3, 'b3', delta, 2000.0, valinit=b3_init)
slider_b4 = Slider(ax_b4, 'b4', a_init + delta, 2000.0, valinit=b4_init)
slider_b5 = Slider(ax_b5, 'b5', -2000.0, 2000.0, valinit=b5_init)
slider_x1bar = Slider(ax_x1bar, 'x1bar', 1, 2000, valinit=x1bar_init)
slider_x2bar = Slider(ax_x2bar, 'x2bar', 1, 2000, valinit=x2bar_init)
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
slider_initial_x1.on_changed(update)
slider_initial_x2.on_changed(update)

ax_heaviside = fig_sliders.add_axes([0.05, 0.25, 0.15, 0.6], facecolor='lightgoldenrodyellow')
ax_integration = fig_sliders.add_axes([0.05, 0.05, 0.15, 0.15], facecolor='lightgoldenrodyellow')

heaviside_methods = ('Standard', 'Smooth', 'Regularized', 'Polynomial', 'Piecewise Linear')
radio_heaviside = RadioButtons(ax_heaviside, heaviside_methods)

integration_methods = ('Euler',)  
radio_integration = RadioButtons(ax_integration, integration_methods)

radio_heaviside.on_clicked(update)
radio_integration.on_clicked(update)

update(None)

plt.show()
