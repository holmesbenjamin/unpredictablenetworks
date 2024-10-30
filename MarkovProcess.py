import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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
    else:
        raise ValueError(f"Unknown Heaviside method: {method_name}")

def standard_heaviside(x):
    return 1.0 if x >= 0 else 0.0

def calculate_transition_matrix(mode_sequence):
    num_modes = 4  # mu00, mu01, mu10, mu11
    transition_counts = np.zeros((num_modes, num_modes))

    for i in range(len(mode_sequence) - 1):
        current_mode = mode_sequence[i]
        next_mode = mode_sequence[i + 1]
        transition_counts[current_mode, next_mode] += 1

    # Add Laplace smoothing
    transition_counts += 1  # Adds 1 to all counts

    transition_matrix = np.zeros((num_modes, num_modes))
    for i in range(num_modes):
        total_transitions = np.sum(transition_counts[i, :])
        transition_matrix[i, :] = transition_counts[i, :] / total_transitions

    return transition_matrix

def run_markov_simulation_with_empirical_matrix(
    a, b1, b2, b3, b4, b5, initial_x1, initial_x2,
    x1bar, x2bar, tau1, tau2, transition_matrix, T=1000.0, epsilon=1.0
):
    N = int(T / epsilon)
    x1 = np.zeros(N)
    x2 = np.zeros(N)
    z = np.zeros(N)
    mode_array = np.zeros(N, dtype=int)

    x1[0] = initial_x1
    x2[0] = initial_x2
    z[0] = 0

    H1_initial = standard_heaviside(x1[0] - x1bar)
    H2_initial = standard_heaviside(x2[0] - x2bar)
    if H1_initial == 0 and H2_initial == 0:
        mode_array[0] = 0  # mu00
    elif H1_initial == 0 and H2_initial == 1:
        mode_array[0] = 1  # mu01
    elif H1_initial == 1 and H2_initial == 0:
        mode_array[0] = 2  # mu10
    else:
        mode_array[0] = 3  # mu11

    time_mu = np.zeros(4)

    for n in range(1, N):
        current_mode = mode_array[n - 1]

        next_mode = np.random.choice(4, p=transition_matrix[current_mode])

        mode_array[n] = next_mode

        time_mu[next_mode] += epsilon

        H1 = 1 if next_mode in [2, 3] else 0
        H2 = 1 if next_mode in [1, 3] else 0

        x1_dot = a - b1 * H1 - b2 * H2 - b3 * H1 * H2
        x2_dot = a - b4 * H2
        x1[n] = x1[n - 1] + epsilon * x1_dot
        x2[n] = x2[n - 1] + epsilon * x2_dot

        z_dot = -2 * a + b5 * H1 * H2
        z[n] = z[n - 1] + epsilon * z_dot

    total_time = time_mu.sum()
    norm_mu_arrays = [time_mu[i] / total_time for i in range(4)]

    return norm_mu_arrays

def update(val):
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

    _, _, _, _, mode_sequence, _, _, _, _, _ = run_simulation(
        a, b1, b2, b3, b4, b5, initial_x1, initial_x2,
        x1bar, x2bar, 0, 0, integration_method='Euler', T=1000.0
    )

    transition_matrix = calculate_transition_matrix(mode_sequence)
    print(transition_matrix)

    max_tau = 20

    mu_probs_tau1 = np.zeros((max_tau + 1, 4))
    mu_probs_tau2 = np.zeros((max_tau + 1, 4))

    for tau1_val in range(max_tau + 1):
        norm_mu_arrays = run_markov_simulation_with_empirical_matrix(
            a, b1, b2, b3, b4, b5, initial_x1, initial_x2,
            x1bar, x2bar, tau1_val, 0, transition_matrix, T=1000.0
        )
        mu_probs_tau1[tau1_val, :] = norm_mu_arrays

    ax_tau1.cla()
    tau1_values = np.arange(max_tau + 1)
    for i, label in enumerate(['mu00', 'mu01', 'mu10', 'mu11']):
        ax_tau1.plot(tau1_values, mu_probs_tau1[:, i], label=label)
    ax_tau1.set_xlabel('tau1')
    ax_tau1.set_ylabel('Final mu probabilities')
    ax_tau1.set_title('Mu probabilities vs tau1 (tau2=0)')
    ax_tau1.legend()

    for tau2_val in range(max_tau + 1):
        norm_mu_arrays = run_markov_simulation_with_empirical_matrix(
            a, b1, b2, b3, b4, b5, initial_x1, initial_x2,
            x1bar, x2bar, 0, tau2_val, transition_matrix, T=1000.0
        )
        mu_probs_tau2[tau2_val, :] = norm_mu_arrays

    ax_tau2.cla()
    tau2_values = np.arange(max_tau + 1)
    for i, label in enumerate(['mu00', 'mu01', 'mu10', 'mu11']):
        ax_tau2.plot(tau2_values, mu_probs_tau2[:, i], label=label)
    ax_tau2.set_xlabel('tau2')
    ax_tau2.set_ylabel('Final mu probabilities')
    ax_tau2.set_title('Mu probabilities vs tau2 (tau1=0)')
    ax_tau2.legend()

    fig_mu.canvas.draw_idle()

fig_sliders = plt.figure(figsize=(10, 6))
fig_sliders.subplots_adjust(left=0.25, right=0.95)

fig_mu, (ax_tau1, ax_tau2) = plt.subplots(1, 2, figsize=(14, 6))
fig_mu.subplots_adjust(wspace=0.3)

a_init = 1.0
delta = 0.01  
b1_init = a_init + 1.0
b2_init = a_init + 1.0
b3_init = (a_init + 1.0)
b4_init = a_init + 1.0
b5_init = 1.0
initial_x1 = 1.0
initial_x2 = 1.0
x1bar_init = 1.0
x2bar_init = 1.0

ax_a = fig_sliders.add_axes([0.25, 0.50, 0.65, 0.03])
ax_b1 = fig_sliders.add_axes([0.25, 0.45, 0.65, 0.03])
ax_b2 = fig_sliders.add_axes([0.25, 0.40, 0.65, 0.03])
ax_b3 = fig_sliders.add_axes([0.25, 0.35, 0.65, 0.03])
ax_b4 = fig_sliders.add_axes([0.25, 0.30, 0.65, 0.03])
ax_b5 = fig_sliders.add_axes([0.25, 0.25, 0.65, 0.03])
ax_x1bar = fig_sliders.add_axes([0.25, 0.20, 0.65, 0.03])
ax_x2bar = fig_sliders.add_axes([0.25, 0.15, 0.65, 0.03])
ax_initial_x1 = fig_sliders.add_axes([0.25, 0.10, 0.65, 0.03])
ax_initial_x2 = fig_sliders.add_axes([0.25, 0.05, 0.65, 0.03])

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

update(None)

plt.show()
