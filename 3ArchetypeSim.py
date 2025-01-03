import numpy as np
import matplotlib.pyplot as plt

def heaviside(x, epsilon=1.0):
    return 0.5 * (1.0 + np.tanh(x / epsilon))

def run_simulation(
    # == MARKET MAKER (i=1) ==
    a_x1,
    c_x1_1, c_x1_2, c_x1_3,
    c_x1_12, c_x1_13, c_x1_23,
    c_x1_123,

    # == FUNDAMENTAL INVESTOR (i=2) ==
    a_x2,
    c_x2_1, c_x2_2, c_x2_3,
    c_x2_12, c_x2_13, c_x2_23,
    c_x2_123,

    # == HIGH-FREQUENCY TRADER (i=3) ==
    a_x3,
    c_x3_1, c_x3_2, c_x3_3,
    c_x3_12, c_x3_13, c_x3_23,
    c_x3_123,
    tau1, tau2, tau3,
    initial_x1, initial_x2, initial_x3,
    x1bar, x2bar, x3bar,
    T=1000.0,
    epsilon=1.0
):
    N = int(T / epsilon)

    delay1 = int(np.floor(tau1))
    delay2 = int(np.floor(tau2))
    delay3 = int(np.floor(tau3))

    max_delay = max(delay1, delay2, delay3)
    X1 = np.ones(N + max_delay)*initial_x1
    X2 = np.ones(N + max_delay)*initial_x2
    X3 = np.ones(N + max_delay)*initial_x3

    for n in range(max_delay, N + max_delay - 1):
        i_mm  = n - delay1  #Market Maker
        i_fi  = n - delay2  #Fundamental Investor
        i_hft = n - delay3  #HFT

        H1 = heaviside(X1[i_mm] - x1bar)
        H2 = heaviside(X2[i_fi] - x2bar)
        H3 = heaviside(X3[i_hft] - x3bar)

        x1_dot = (
            a_x1
            + c_x1_1 * H1
            + c_x1_2 * H2
            + c_x1_3 * H3
            + c_x1_12 * (H1*H2)
            + c_x1_13 * (H1*H3)
            + c_x1_23 * (H2*H3)
            + c_x1_123 * (H1*H2*H3)
        )

        x2_dot = (
            a_x2
            + c_x2_1 * H1
            + c_x2_2 * H2
            + c_x2_3 * H3
            + c_x2_12 * (H1*H2)
            + c_x2_13 * (H1*H3)
            + c_x2_23 * (H2*H3)
            + c_x2_123 * (H1*H2*H3)
        )

        x3_dot = (
            a_x3
            + c_x3_1 * H1
            + c_x3_2 * H2
            + c_x3_3 * H3
            + c_x3_12 * (H1*H2)
            + c_x3_13 * (H1*H3)
            + c_x3_23 * (H2*H3)
            + c_x3_123 * (H1*H2*H3)
        )

        X1[n+1] = X1[n] + epsilon*x1_dot
        X2[n+1] = X2[n] + epsilon*x2_dot
        X3[n+1] = X3[n] + epsilon*x3_dot

    x1 = X1[max_delay:]
    x2 = X2[max_delay:]
    x3 = X3[max_delay:]

    H1_arr = np.array([heaviside(val - x1bar) for val in x1])
    H2_arr = np.array([heaviside(val - x2bar) for val in x2])
    H3_arr = np.array([heaviside(val - x3bar) for val in x3])

    mu000 = (1 - H1_arr)*(1 - H2_arr)*(1 - H3_arr)
    mu001 = (1 - H1_arr)*(1 - H2_arr)*H3_arr
    mu010 = (1 - H1_arr)*H2_arr*(1 - H3_arr)
    mu011 = (1 - H1_arr)*H2_arr*H3_arr
    mu100 = H1_arr*(1 - H2_arr)*(1 - H3_arr)
    mu101 = H1_arr*(1 - H2_arr)*H3_arr
    mu110 = H1_arr*H2_arr*(1 - H3_arr)
    mu111 = H1_arr*H2_arr*H3_arr

    mu_states = np.vstack([
        mu000, mu001, mu010, mu011,
        mu100, mu101, mu110, mu111
    ])
    return mu_states

def vary_and_plot(
    a_x1,
    c_x1_1, c_x1_2, c_x1_3,
    c_x1_12, c_x1_13, c_x1_23,
    c_x1_123,
    a_x2,
    c_x2_1, c_x2_2, c_x2_3,
    c_x2_12, c_x2_13, c_x2_23,
    c_x2_123,
    a_x3,
    c_x3_1, c_x3_2, c_x3_3,
    c_x3_12, c_x3_13, c_x3_23,
    c_x3_123,

    initial_x1, initial_x2, initial_x3,
    x1bar, x2bar, x3bar,
    T, epsilon,
    varying_tau, fixed_tau1, fixed_tau2, fixed_tau3
):
    mu_labels = [
        'mu000','mu001','mu010','mu011',
        'mu100','mu101','mu110','mu111'
    ]

    tau_values = np.arange(0, 201, 1)
    results = {}

    for tau in tau_values:
        if varying_tau == 'tau1':
            tau1 = tau
            tau2 = fixed_tau2
            tau3 = fixed_tau3
        elif varying_tau == 'tau2':
            tau1 = fixed_tau1
            tau2 = tau
            tau3 = fixed_tau3
        elif varying_tau == 'tau3':
            tau1 = fixed_tau1
            tau2 = fixed_tau2
            tau3 = tau
        else:
            raise ValueError("varying_tau must be 'tau1','tau2','tau3'.")

        mu_states = run_simulation(
            a_x1,
            c_x1_1, c_x1_2, c_x1_3,
            c_x1_12, c_x1_13, c_x1_23,
            c_x1_123,
            a_x2,
            c_x2_1, c_x2_2, c_x2_3,
            c_x2_12, c_x2_13, c_x2_23,
            c_x2_123,
            a_x3,
            c_x3_1, c_x3_2, c_x3_3,
            c_x3_12, c_x3_13, c_x3_23,
            c_x3_123,

            tau1, tau2, tau3,
            initial_x1, initial_x2, initial_x3,
            x1bar, x2bar, x3bar,
            T=T, epsilon=epsilon
        )

        num_steps = mu_states.shape[1]
        time_in_each = np.sum(mu_states, axis=1)
        probability = time_in_each / num_steps
        results[tau] = probability

    plt.figure(figsize=(10,6))
    for i, label in enumerate(mu_labels):
        y_vals = [results[t][i] for t in tau_values]
        plt.plot(tau_values, y_vals, marker=',', label=label)
    plt.title(f'Probability of Each mu State vs {varying_tau}')
    plt.xlabel(varying_tau)
    plt.ylabel('Probability')
    plt.grid(True)
    plt.legend()
    plt.show()

    mu111_vals = [results[t][7] for t in tau_values]
    plt.figure(figsize=(10,6))
    plt.plot(tau_values, mu111_vals, marker=',', color='black', label='mu111')
    plt.title(f'Probability of mu111 vs {varying_tau}')
    plt.xlabel(varying_tau)
    plt.ylabel('Probability')
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    # === MARKET MAKER (i=1) ===
    a_x1    =  3.0
    c_x1_1  =  0.0
    c_x1_2  =  2.0
    c_x1_3  = -6.0
    c_x1_12 = -2.0
    c_x1_13 =  0.5
    c_x1_23 = -1.0
    c_x1_123= -0.2

    # === FUNDAMENTAL INVESTOR (i=2) ===
    a_x2    =  0.0
    c_x2_1  =  2.0
    c_x2_2  =  0.3
    c_x2_3  = -4.0
    c_x2_12 = -0.1
    c_x2_13 =  0.2
    c_x2_23 = -0.3
    c_x2_123= -0.05

    # === HIGH-FREQUENCY TRADER (i=3) ===
    a_x3    = -1.0
    c_x3_1  =  1.0
    c_x3_2  =  0.5
    c_x3_3  =  0.0
    c_x3_12 =  3.0
    c_x3_13 = -0.5
    c_x3_23 =  0.4
    c_x3_123=  0.1

    initial_x1, initial_x2, initial_x3 = 2.5, 0.0, -1.5
    x1bar, x2bar, x3bar = 1.0, 1.0, 1.0

    T = 100000.0
    epsilon = 1.0

    vary_and_plot(
        a_x1, c_x1_1, c_x1_2, c_x1_3,
        c_x1_12, c_x1_13, c_x1_23, c_x1_123,

        a_x2, c_x2_1, c_x2_2, c_x2_3,
        c_x2_12, c_x2_13, c_x2_23, c_x2_123,

        a_x3, c_x3_1, c_x3_2, c_x3_3,
        c_x3_12, c_x3_13, c_x3_23, c_x3_123,

        initial_x1, initial_x2, initial_x3,
        x1bar, x2bar, x3bar,
        T, epsilon,
        varying_tau='tau1', fixed_tau1=None, fixed_tau2=15, fixed_tau3=2
    )

    vary_and_plot(
        a_x1, c_x1_1, c_x1_2, c_x1_3,
        c_x1_12, c_x1_13, c_x1_23, c_x1_123,

        a_x2, c_x2_1, c_x2_2, c_x2_3,
        c_x2_12, c_x2_13, c_x2_23, c_x2_123,

        a_x3, c_x3_1, c_x3_2, c_x3_3,
        c_x3_12, c_x3_13, c_x3_23, c_x3_123,

        initial_x1, initial_x2, initial_x3,
        x1bar, x2bar, x3bar,
        T, epsilon,
        varying_tau='tau2', fixed_tau1=4, fixed_tau2=None, fixed_tau3=2
    )

    vary_and_plot(
        a_x1, c_x1_1, c_x1_2, c_x1_3,
        c_x1_12, c_x1_13, c_x1_23, c_x1_123,

        a_x2, c_x2_1, c_x2_2, c_x2_3,
        c_x2_12, c_x2_13, c_x2_23, c_x2_123,

        a_x3, c_x3_1, c_x3_2, c_x3_3,
        c_x3_12, c_x3_13, c_x3_23, c_x3_123,

        initial_x1, initial_x2, initial_x3,
        x1bar, x2bar, x3bar,
        T, epsilon,
        varying_tau='tau3', fixed_tau1=4, fixed_tau2=15, fixed_tau3=None
    )

if __name__ == "__main__":
    main()
