import numpy as np
import matplotlib.pyplot as plt

def heaviside(x, epsilon=1.0):
    return 0.5 * (1.0 + np.tanh(x / epsilon))

def run_simulation_4agents(
    a_x1, 
    c_x1_1, c_x1_2, c_x1_3, c_x1_4,
    c_x1_12, c_x1_13, c_x1_14, c_x1_23, c_x1_24, c_x1_34,
    c_x1_123, c_x1_124, c_x1_134, c_x1_234,
    a_x2,
    c_x2_1, c_x2_2, c_x2_3, c_x2_4,
    c_x2_12, c_x2_13, c_x2_14, c_x2_23, c_x2_24, c_x2_34,
    c_x2_123, c_x2_124, c_x2_134, c_x2_234,
    a_x3,
    c_x3_1, c_x3_2, c_x3_3, c_x3_4,
    c_x3_12, c_x3_13, c_x3_14, c_x3_23, c_x3_24, c_x3_34,
    c_x3_123, c_x3_124, c_x3_134, c_x3_234,
    a_x4,
    c_x4_1, c_x4_2, c_x4_3, c_x4_4,
    c_x4_12, c_x4_13, c_x4_14, c_x4_23, c_x4_24, c_x4_34,
    c_x4_123, c_x4_124, c_x4_134, c_x4_234,
    tau1, tau2, tau3, tau4,
    initial_x1, initial_x2, initial_x3, initial_x4,
    x1bar, x2bar, x3bar, x4bar,

    T=1000.0,
    epsilon=1.0
):
    N = int(T / epsilon)
    delay1 = int(np.floor(tau1))
    delay2 = int(np.floor(tau2))
    delay3 = int(np.floor(tau3))
    delay4 = int(np.floor(tau4))

    max_delay = max(delay1, delay2, delay3, delay4)

    X1 = np.ones(N + max_delay)*initial_x1
    X2 = np.ones(N + max_delay)*initial_x2
    X3 = np.ones(N + max_delay)*initial_x3
    X4 = np.ones(N + max_delay)*initial_x4

    for n in range(max_delay, N + max_delay - 1):
        i_mm  = n - delay1
        i_fi  = n - delay2
        i_hft = n - delay3
        i_rt  = n - delay4  

        H1 = heaviside(X1[i_mm] - x1bar)
        H2 = heaviside(X2[i_fi] - x2bar)
        H3 = heaviside(X3[i_hft] - x3bar)
        H4 = heaviside(X4[i_rt]  - x4bar)

        x1_dot = (
            a_x1
            + c_x1_1*H1
            + c_x1_2*H2
            + c_x1_3*H3
            + c_x1_4*H4
            + c_x1_12*(H1*H2)
            + c_x1_13*(H1*H3)
            + c_x1_14*(H1*H4)
            + c_x1_23*(H2*H3)
            + c_x1_24*(H2*H4)
            + c_x1_34*(H3*H4)
            + c_x1_123*(H1*H2*H3)
            + c_x1_124*(H1*H2*H4)
            + c_x1_134*(H1*H3*H4)
            + c_x1_234*(H2*H3*H4)
        )

        x2_dot = (
            a_x2
            + c_x2_1*H1
            + c_x2_2*H2
            + c_x2_3*H3
            + c_x2_4*H4
            + c_x2_12*(H1*H2)
            + c_x2_13*(H1*H3)
            + c_x2_14*(H1*H4)
            + c_x2_23*(H2*H3)
            + c_x2_24*(H2*H4)
            + c_x2_34*(H3*H4)
            + c_x2_123*(H1*H2*H3)
            + c_x2_124*(H1*H2*H4)
            + c_x2_134*(H1*H3*H4)
            + c_x2_234*(H2*H3*H4)
        )

        x3_dot = (
            a_x3
            + c_x3_1*H1
            + c_x3_2*H2
            + c_x3_3*H3
            + c_x3_4*H4
            + c_x3_12*(H1*H2)
            + c_x3_13*(H1*H3)
            + c_x3_14*(H1*H4)
            + c_x3_23*(H2*H3)
            + c_x3_24*(H2*H4)
            + c_x3_34*(H3*H4)
            + c_x3_123*(H1*H2*H3)
            + c_x3_124*(H1*H2*H4)
            + c_x3_134*(H1*H3*H4)
            + c_x3_234*(H2*H3*H4)
        )

        x4_dot = (
            a_x4
            + c_x4_1*H1
            + c_x4_2*H2
            + c_x4_3*H3
            + c_x4_4*H4
            + c_x4_12*(H1*H2)
            + c_x4_13*(H1*H3)
            + c_x4_14*(H1*H4)
            + c_x4_23*(H2*H3)
            + c_x4_24*(H2*H4)
            + c_x4_34*(H3*H4)
            + c_x4_123*(H1*H2*H3)
            + c_x4_124*(H1*H2*H4)
            + c_x4_134*(H1*H3*H4)
            + c_x4_234*(H2*H3*H4)
        )

        X1[n+1] = X1[n] + epsilon*x1_dot
        X2[n+1] = X2[n] + epsilon*x2_dot
        X3[n+1] = X3[n] + epsilon*x3_dot
        X4[n+1] = X4[n] + epsilon*x4_dot

    x1 = X1[max_delay:]
    x2 = X2[max_delay:]
    x3 = X3[max_delay:]
    x4 = X4[max_delay:]

    H1_arr = np.array([heaviside(val - x1bar) for val in x1])
    H2_arr = np.array([heaviside(val - x2bar) for val in x2])
    H3_arr = np.array([heaviside(val - x3bar) for val in x3])
    H4_arr = np.array([heaviside(val - x4bar) for val in x4])

    mu0000 = (1-H1_arr)*(1-H2_arr)*(1-H3_arr)*(1-H4_arr)
    mu0001 = (1-H1_arr)*(1-H2_arr)*(1-H3_arr)*(H4_arr)
    mu0010 = (1-H1_arr)*(1-H2_arr)*(H3_arr)*(1-H4_arr)
    mu0011 = (1-H1_arr)*(1-H2_arr)*(H3_arr)*(H4_arr)
    mu0100 = (1-H1_arr)*(H2_arr)*(1-H3_arr)*(1-H4_arr)
    mu0101 = (1-H1_arr)*(H2_arr)*(1-H3_arr)*(H4_arr)
    mu0110 = (1-H1_arr)*(H2_arr)*(H3_arr)*(1-H4_arr)
    mu0111 = (1-H1_arr)*(H2_arr)*(H3_arr)*(H4_arr)
    mu1000 = (H1_arr)*(1-H2_arr)*(1-H3_arr)*(1-H4_arr)
    mu1001 = (H1_arr)*(1-H2_arr)*(1-H3_arr)*(H4_arr)
    mu1010 = (H1_arr)*(1-H2_arr)*(H3_arr)*(1-H4_arr)
    mu1011 = (H1_arr)*(1-H2_arr)*(H3_arr)*(H4_arr)
    mu1100 = (H1_arr)*(H2_arr)*(1-H3_arr)*(1-H4_arr)
    mu1101 = (H1_arr)*(H2_arr)*(1-H3_arr)*(H4_arr)
    mu1110 = (H1_arr)*(H2_arr)*(H3_arr)*(1-H4_arr)
    mu1111 = (H1_arr)*(H2_arr)*(H3_arr)*(H4_arr)

    mu_states = np.vstack([
        mu0000, mu0001, mu0010, mu0011,
        mu0100, mu0101, mu0110, mu0111,
        mu1000, mu1001, mu1010, mu1011,
        mu1100, mu1101, mu1110, mu1111
    ])
    return mu_states

def vary_and_plot_4agents(
    a_x1, c_x1_1, c_x1_2, c_x1_3, c_x1_4,
    c_x1_12, c_x1_13, c_x1_14, c_x1_23, c_x1_24, c_x1_34,
    c_x1_123, c_x1_124, c_x1_134, c_x1_234,

    a_x2, c_x2_1, c_x2_2, c_x2_3, c_x2_4,
    c_x2_12, c_x2_13, c_x2_14, c_x2_23, c_x2_24, c_x2_34,
    c_x2_123, c_x2_124, c_x2_134, c_x2_234,

    a_x3, c_x3_1, c_x3_2, c_x3_3, c_x3_4,
    c_x3_12, c_x3_13, c_x3_14, c_x3_23, c_x3_24, c_x3_34,
    c_x3_123, c_x3_124, c_x3_134, c_x3_234,

    a_x4, c_x4_1, c_x4_2, c_x4_3, c_x4_4,
    c_x4_12, c_x4_13, c_x4_14, c_x4_23, c_x4_24, c_x4_34,
    c_x4_123, c_x4_124, c_x4_134, c_x4_234,

    initial_x1, initial_x2, initial_x3, initial_x4,
    x1bar, x2bar, x3bar, x4bar,
    T, epsilon,
    varying_tau, 
    fixed_tau1=None, fixed_tau2=None, fixed_tau3=None, fixed_tau4=None
):

    mu_labels = [
        "mu0000","mu0001","mu0010","mu0011",
        "mu0100","mu0101","mu0110","mu0111",
        "mu1000","mu1001","mu1010","mu1011",
        "mu1100","mu1101","mu1110","mu1111"
    ]

    tau_values = np.arange(0, 201, 1)
    results = {}

    for tau in tau_values:
        if varying_tau == 'tau1':
            tau1 = tau
            tau2 = fixed_tau2
            tau3 = fixed_tau3
            tau4 = fixed_tau4
        elif varying_tau == 'tau2':
            tau1 = fixed_tau1
            tau2 = tau
            tau3 = fixed_tau3
            tau4 = fixed_tau4
        elif varying_tau == 'tau3':
            tau1 = fixed_tau1
            tau2 = fixed_tau2
            tau3 = tau
            tau4 = fixed_tau4
        elif varying_tau == 'tau4':
            tau1 = fixed_tau1
            tau2 = fixed_tau2
            tau3 = fixed_tau3
            tau4 = tau
        else:
            raise ValueError("varying_tau must be one of 'tau1','tau2','tau3','tau4'.")

        mu_states = run_simulation_4agents(
            a_x1, c_x1_1, c_x1_2, c_x1_3, c_x1_4,
            c_x1_12, c_x1_13, c_x1_14, c_x1_23, c_x1_24, c_x1_34,
            c_x1_123, c_x1_124, c_x1_134, c_x1_234,

            a_x2, c_x2_1, c_x2_2, c_x2_3, c_x2_4,
            c_x2_12, c_x2_13, c_x2_14, c_x2_23, c_x2_24, c_x2_34,
            c_x2_123, c_x2_124, c_x2_134, c_x2_234,

            a_x3, c_x3_1, c_x3_2, c_x3_3, c_x3_4,
            c_x3_12, c_x3_13, c_x3_14, c_x3_23, c_x3_24, c_x3_34,
            c_x3_123, c_x3_124, c_x3_134, c_x3_234,

            a_x4, c_x4_1, c_x4_2, c_x4_3, c_x4_4,
            c_x4_12, c_x4_13, c_x4_14, c_x4_23, c_x4_24, c_x4_34,
            c_x4_123, c_x4_124, c_x4_134, c_x4_234,

            tau1, tau2, tau3, tau4,
            initial_x1, initial_x2, initial_x3, initial_x4,
            x1bar, x2bar, x3bar, x4bar,
            T=T, epsilon=epsilon
        )

        num_steps = mu_states.shape[1]
        time_in_each = np.sum(mu_states, axis=1)
        prob_overall = time_in_each / num_steps

        transient_steps = int(0.1 * num_steps)
        time_in_each_transient = np.sum(mu_states[:, :transient_steps], axis=1)
        prob_transient = time_in_each_transient / transient_steps

        time_in_each_steady = np.sum(mu_states[:, transient_steps:], axis=1)
        prob_steady = time_in_each_steady / (num_steps - transient_steps)

        results[tau] = {
            'transient': prob_transient,
            'steady':    prob_steady,
            'overall':   prob_overall
        }

    plt.figure(figsize=(10,6))
    for i, label in enumerate(mu_labels):
        y_steady = [results[t]['steady'][i] for t in tau_values]
        plt.plot(tau_values, y_steady, label=f"{label} (steady)")

    plt.title(f"Probability of Each mu State vs {varying_tau} [Steady portion only]")
    plt.xlabel(varying_tau)
    plt.ylabel("Probability")
    plt.grid(True)
    plt.legend(ncol=3)
    plt.show()

    mu1111_tr = [results[t]['transient'][15] for t in tau_values]  
    mu1111_sd = [results[t]['steady'][15]    for t in tau_values]
    mu1111_ov = [results[t]['overall'][15]   for t in tau_values]

    plt.figure(figsize=(10,6))
    plt.plot(tau_values, mu1111_tr, '--', color='blue',  label='mu1111 (trans.)')
    plt.plot(tau_values, mu1111_sd, '-',  color='red',   label='mu1111 (steady)')
    plt.plot(tau_values, mu1111_ov, '-',  color='black', label='mu1111 (overall)')
    plt.title(f"mu1111 vs {varying_tau}: Transient, Steady, Overall")
    plt.xlabel(varying_tau)
    plt.ylabel("Probability")
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    a_x1 = 3.0
    c_x1_1=0.0; c_x1_2=2.0; c_x1_3=-6.0; c_x1_4=0.3
    c_x1_12=-2.0; c_x1_13=0.5;  c_x1_14=-0.1
    c_x1_23=-1.0; c_x1_24=-0.2; c_x1_34=-0.4
    c_x1_123=-0.2; c_x1_124=-0.1; c_x1_134=0.1; c_x1_234=-0.05

    a_x2=0.0
    c_x2_1=2.0; c_x2_2=0.3; c_x2_3=-4.0; c_x2_4=0.1
    c_x2_12=-0.1; c_x2_13=0.2; c_x2_14=0.05
    c_x2_23=-0.3; c_x2_24=-0.1; c_x2_34=-0.2
    c_x2_123=-0.05; c_x2_124=-0.02; c_x2_134=0.1; c_x2_234=-0.04

    a_x3=-1.0
    c_x3_1=1.0; c_x3_2=0.5; c_x3_3=0.0; c_x3_4=-0.2
    c_x3_12=3.0; c_x3_13=-0.5; c_x3_14=-0.2
    c_x3_23=0.4; c_x3_24=-0.3; c_x3_34=0.1
    c_x3_123=0.1; c_x3_124=-0.02; c_x3_134=-0.05; c_x3_234=0.0

    a_x4=0.1
    c_x4_1=0.4; c_x4_2=0.1; c_x4_3=-0.2; c_x4_4=0.2
    c_x4_12=0.1; c_x4_13=-0.1; c_x4_14=-0.1
    c_x4_23=-0.3; c_x4_24=-0.2; c_x4_34=-0.05
    c_x4_123=-0.05; c_x4_124=0.03; c_x4_134=-0.02; c_x4_234=-0.01

    fixed_tau2=15
    fixed_tau3=2
    fixed_tau4=10

    initial_x1=2.5; initial_x2=0.0; initial_x3=-1.5; initial_x4=0.3
    x1bar=1.0; x2bar=1.0; x3bar=1.0; x4bar=0.5

    T=50000.0
    epsilon=1.0

    vary_and_plot_4agents(
        a_x1, c_x1_1, c_x1_2, c_x1_3, c_x1_4,
        c_x1_12, c_x1_13, c_x1_14, c_x1_23, c_x1_24, c_x1_34,
        c_x1_123, c_x1_124, c_x1_134, c_x1_234,

        a_x2, c_x2_1, c_x2_2, c_x2_3, c_x2_4,
        c_x2_12, c_x2_13, c_x2_14, c_x2_23, c_x2_24, c_x2_34,
        c_x2_123, c_x2_124, c_x2_134, c_x2_234,

        a_x3, c_x3_1, c_x3_2, c_x3_3, c_x3_4,
        c_x3_12, c_x3_13, c_x3_14, c_x3_23, c_x3_24, c_x3_34,
        c_x3_123, c_x3_124, c_x3_134, c_x3_234,

        a_x4, c_x4_1, c_x4_2, c_x4_3, c_x4_4,
        c_x4_12, c_x4_13, c_x4_14, c_x4_23, c_x4_24, c_x4_34,
        c_x4_123, c_x4_124, c_x4_134, c_x4_234,

        initial_x1, initial_x2, initial_x3, initial_x4,
        x1bar, x2bar, x3bar, x4bar,
        T, epsilon,
        varying_tau='tau1',
        fixed_tau1=None,
        fixed_tau2=fixed_tau2,
        fixed_tau3=fixed_tau3,
        fixed_tau4=fixed_tau4
    )

    vary_and_plot_4agents(
        a_x1, c_x1_1, c_x1_2, c_x1_3, c_x1_4,
        c_x1_12, c_x1_13, c_x1_14, c_x1_23, c_x1_24, c_x1_34,
        c_x1_123, c_x1_124, c_x1_134, c_x1_234,

        a_x2, c_x2_1, c_x2_2, c_x2_3, c_x2_4,
        c_x2_12, c_x2_13, c_x2_14, c_x2_23, c_x2_24, c_x2_34,
        c_x2_123, c_x2_124, c_x2_134, c_x2_234,

        a_x3, c_x3_1, c_x3_2, c_x3_3, c_x3_4,
        c_x3_12, c_x3_13, c_x3_14, c_x3_23, c_x3_24, c_x3_34,
        c_x3_123, c_x3_124, c_x3_134, c_x3_234,

        a_x4, c_x4_1, c_x4_2, c_x4_3, c_x4_4,
        c_x4_12, c_x4_13, c_x4_14, c_x4_23, c_x4_24, c_x4_34,
        c_x4_123, c_x4_124, c_x4_134, c_x4_234,

        initial_x1, initial_x2, initial_x3, initial_x4,
        x1bar, x2bar, x3bar, x4bar,
        T, epsilon,
        varying_tau='tau2',
        fixed_tau1=4,
        fixed_tau2=None,
        fixed_tau3=2,
        fixed_tau4=10
    )

    vary_and_plot_4agents(
        a_x1, c_x1_1, c_x1_2, c_x1_3, c_x1_4,
        c_x1_12, c_x1_13, c_x1_14, c_x1_23, c_x1_24, c_x1_34,
        c_x1_123, c_x1_124, c_x1_134, c_x1_234,

        a_x2, c_x2_1, c_x2_2, c_x2_3, c_x2_4,
        c_x2_12, c_x2_13, c_x2_14, c_x2_23, c_x2_24, c_x2_34,
        c_x2_123, c_x2_124, c_x2_134, c_x2_234,

        a_x3, c_x3_1, c_x3_2, c_x3_3, c_x3_4,
        c_x3_12, c_x3_13, c_x3_14, c_x3_23, c_x3_24, c_x3_34,
        c_x3_123, c_x3_124, c_x3_134, c_x3_234,

        a_x4, c_x4_1, c_x4_2, c_x4_3, c_x4_4,
        c_x4_12, c_x4_13, c_x4_14, c_x4_23, c_x4_24, c_x4_34,
        c_x4_123, c_x4_124, c_x4_134, c_x4_234,

        initial_x1, initial_x2, initial_x3, initial_x4,
        x1bar, x2bar, x3bar, x4bar,
        T, epsilon,
        varying_tau='tau3',
        fixed_tau1=4,
        fixed_tau2=15,
        fixed_tau3=None,
        fixed_tau4=10
    )

    vary_and_plot_4agents(
        a_x1, c_x1_1, c_x1_2, c_x1_3, c_x1_4,
        c_x1_12, c_x1_13, c_x1_14, c_x1_23, c_x1_24, c_x1_34,
        c_x1_123, c_x1_124, c_x1_134, c_x1_234,

        a_x2, c_x2_1, c_x2_2, c_x2_3, c_x2_4,
        c_x2_12, c_x2_13, c_x2_14, c_x2_23, c_x2_24, c_x2_34,
        c_x2_123, c_x2_124, c_x2_134, c_x2_234,

        a_x3, c_x3_1, c_x3_2, c_x3_3, c_x3_4,
        c_x3_12, c_x3_13, c_x3_14, c_x3_23, c_x3_24, c_x3_34,
        c_x3_123, c_x3_124, c_x3_134, c_x3_234,

        a_x4, c_x4_1, c_x4_2, c_x4_3, c_x4_4,
        c_x4_12, c_x4_13, c_x4_14, c_x4_23, c_x4_24, c_x4_34,
        c_x4_123, c_x4_124, c_x4_134, c_x4_234,

        initial_x1, initial_x2, initial_x3, initial_x4,
        x1bar, x2bar, x3bar, x4bar,
        T, epsilon,
        varying_tau='tau4',
        fixed_tau1=4,
        fixed_tau2=15,
        fixed_tau3=2,
        fixed_tau4=None
    )

if __name__ == "__main__":
    main()
