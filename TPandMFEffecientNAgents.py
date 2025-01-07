import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import multiprocessing as mp
from functools import partial

@njit
def heaviside_vectorised(x, epsilon=1.0):
    return 0.5 * (1.0 + np.tanh(x / epsilon))

@njit
def ornstein_uhlenbeck_optimised(theta, mu, sigma, size, current):
    return current + theta * (mu - current) + sigma * size * np.random.normal()

class Agent:
    def __init__(self, a_x, 
                 c_x1, c_x2, c_x3, c_x4,
                 c_x12, c_x13, c_x14,
                 c_x23, c_x24, c_x34,
                 c_x123, c_x124, c_x134, c_x234,
                 pop_ratio=1,         
                 coupling_row=None  
                 ):
        self.a_x = a_x

        self.c_x = np.array([c_x1, c_x2, c_x3, c_x4], dtype=np.float64)

        self.c_x12 = c_x12
        self.c_x13 = c_x13
        self.c_x14 = c_x14
        self.c_x23 = c_x23
        self.c_x24 = c_x24
        self.c_x34 = c_x34

        self.c_x123 = c_x123
        self.c_x124 = c_x124
        self.c_x134 = c_x134
        self.c_x234 = c_x234

        self.pop_ratio = pop_ratio

        if coupling_row is None:
            coupling_row = np.ones(4, dtype=np.float64)
        self.coupling_row = coupling_row

def run_simulation_4agents(
    agents,
    delays,
    initial_conditions,
    xbar,
    T=1000.0,
    epsilon=1.0,
    lambda_params=None,
    sigma_params=None,
    N=20
):

    if lambda_params is None:
        lambda_params = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float64)
    if sigma_params is None:
        sigma_params = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float64)

    N = int(T / epsilon)
    delay_indices = [int(np.floor(tau)) for tau in delays]
    max_delay = max(delay_indices)

    X = np.ones((4, N + max_delay), dtype=np.float64) * initial_conditions[:, np.newaxis]

    phi = np.zeros(4, dtype=np.float64)

    for n in range(max_delay, N + max_delay - 1):
        idx_delays = [n - d for d in delay_indices]

        delayed_X = np.empty(4, dtype=np.float64)
        for i in range(4):
            delayed_X[i] = X[i, idx_delays[i]]

        H = heaviside_vectorised(delayed_X - xbar)

        for i, agent in enumerate(agents):
            phi[i] = ornstein_uhlenbeck_optimised(
                theta=lambda_params[i],
                mu=0.0,
                sigma=sigma_params[i],
                size=agent.pop_ratio,
                current=phi[i]
            )

        for agent_idx, agent in enumerate(agents):
            x_dot = agent.a_x
            x_dot += np.dot(agent.c_x, H)
            x_dot += agent.c_x12 * H[0] * H[1]
            x_dot += agent.c_x13 * H[0] * H[2]
            x_dot += agent.c_x14 * H[0] * H[3]
            x_dot += agent.c_x23 * H[1] * H[2]
            x_dot += agent.c_x24 * H[1] * H[3]
            x_dot += agent.c_x34 * H[2] * H[3]
            x_dot += agent.c_x123 * H[0] * H[1] * H[2]
            x_dot += agent.c_x124 * H[0] * H[1] * H[3]
            x_dot += agent.c_x134 * H[0] * H[2] * H[3]
            x_dot += agent.c_x234 * H[1] * H[2] * H[3]
            x_dot += np.dot(agent.coupling_row, phi)
            
            X[agent_idx, n + 1] = X[agent_idx, n] + epsilon * x_dot

    x_final = X[:, max_delay:]
    H_final = heaviside_vectorised(x_final - xbar[:, np.newaxis])
    H1_arr = H_final[0]
    H2_arr = H_final[1]
    H3_arr = H_final[2]
    H4_arr = H_final[3]

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

def simulate_tau(
    tau,
    varying_tau,
    fixed_delays,
    agents,
    initial_conditions,
    xbar,
    T,
    epsilon,
    lambda_params,
    sigma_params,
    N
):
    
    current_delays = dict(fixed_delays)
    current_delays[varying_tau] = tau

    delays_list = [
        current_delays['tau1'],
        current_delays['tau2'],
        current_delays['tau3'],
        current_delays['tau4']
    ]

    mu_states = run_simulation_4agents(
        agents=agents,
        delays=delays_list,
        initial_conditions=initial_conditions,
        xbar=xbar,
        T=T,
        epsilon=epsilon,
        lambda_params=lambda_params,
        sigma_params=sigma_params,
        N=N
    )

    num_steps = mu_states.shape[1]
    time_in_each = np.sum(mu_states, axis=1)
    prob_overall = time_in_each / num_steps

    transient_steps = int(0.1 * num_steps)
    time_in_each_transient = np.sum(mu_states[:, :transient_steps], axis=1)
    prob_transient = time_in_each_transient / transient_steps

    time_in_each_steady = np.sum(mu_states[:, transient_steps:], axis=1)
    prob_steady = time_in_each_steady / (num_steps - transient_steps)

    
    return tau, {
        'transient': prob_transient,
        'steady':    prob_steady,
        'overall':   prob_overall
    }

def vary_and_plot_4agents(params):
   
    (agents,
     initial_conditions,
     xbar,
     T,
     epsilon,
     varying_tau,
     fixed_delays,
     lambda_params,
     sigma_params,
     N) = params

    mu_labels = [
        "mu0000","mu0001","mu0010","mu0011",
        "mu0100","mu0101","mu0110","mu0111",
        "mu1000","mu1001","mu1010","mu1011",
        "mu1100","mu1101","mu1110","mu1111"
    ]

    tau_values = np.arange(0, 201, 1)
    results = {}

    
    partial_sim = partial(
        simulate_tau,
        varying_tau=varying_tau,
        fixed_delays=fixed_delays,
        agents=agents,
        initial_conditions=initial_conditions,
        xbar=xbar,
        T=T,
        epsilon=epsilon,
        lambda_params=lambda_params,
        sigma_params=sigma_params,
        N=N
    )

    with mp.Pool(mp.cpu_count()) as pool:
        results_list = pool.map(partial_sim, tau_values)

    results = {tau: data for (tau, data) in results_list}
    
    plt.figure(figsize=(14,8))
    for i, label in enumerate(mu_labels):
        steady_probs = [results[t]['steady'][i] for t in tau_values]
        plt.plot(tau_values, steady_probs, label=label)

    plt.title(f"Probability of Each Mode vs {varying_tau} [Steady Portion Only]")
    plt.xlabel(varying_tau)
    plt.ylabel("Probability")
    plt.grid(True)
    plt.legend(ncol=4, fontsize='small')
    plt.tight_layout()
    plt.show()

    mu_idx = mu_labels.index("mu1111")
    mu1111_tr = [results[t]['transient'][mu_idx] for t in tau_values]
    mu1111_sd = [results[t]['steady'][mu_idx] for t in tau_values]
    mu1111_ov = [results[t]['overall'][mu_idx] for t in tau_values]

    plt.figure(figsize=(14,6))
    plt.plot(tau_values, mu1111_tr, '--',  color='blue',  label='mu1111 (transient)')
    plt.plot(tau_values, mu1111_sd, '-',   color='red',   label='mu1111 (steady)')
    plt.plot(tau_values, mu1111_ov, '-',   color='black', label='mu1111 (overall)')
    plt.title(f"mu1111 vs {varying_tau}: Transient, Steady, Overall (Fraction of Total Pop)")
    plt.xlabel(varying_tau)
    plt.ylabel("Probability")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    
    population_ratio_array = np.array([1, 1, 1, 1], dtype=np.float64)
    N = population_ratio_array.sum()
    agents = [
        Agent(
            a_x=3.0,
            c_x1=0.0, c_x2=2.0, c_x3=-6.0, c_x4=0.3,
            c_x12=-2.0, c_x13=0.5, c_x14=-0.1,
            c_x23=-1.0, c_x24=-0.2, c_x34=-0.4,
            c_x123=-0.2, c_x124=-0.1, c_x134=0.1, c_x234=-0.05,
            pop_ratio=population_ratio_array[0],  
            coupling_row=np.array([0.3, 0.3, 1.0, 0.8], dtype=np.float64)
        ),
        Agent(
            a_x=0.0,
            c_x1=2.0, c_x2=0.3, c_x3=-4.0, c_x4=0.1,
            c_x12=-0.1, c_x13=0.2, c_x14=0.05,
            c_x23=-0.3, c_x24=-0.1, c_x34=-0.2,
            c_x123=-0.05, c_x124=-0.02, c_x134=0.1, c_x234=-0.04,
            pop_ratio=population_ratio_array[1],
            coupling_row=np.array([1.0, 1.0, 0.6, 0.2], dtype=np.float64)
        ),
        Agent(
            a_x=-1.0,
            c_x1=1.0, c_x2=0.5, c_x3=0.0, c_x4=-0.2,
            c_x12=3.0, c_x13=-0.5, c_x14=-0.2,
            c_x23=0.4, c_x24=-0.3, c_x34=0.1,
            c_x123=0.1, c_x124=-0.02, c_x134=-0.05, c_x234=0.0,
            pop_ratio=population_ratio_array[2],
            coupling_row=np.array([0.1, 0.4, 0.8, 1.0], dtype=np.float64)
        ),
        Agent(
            a_x=0.1,
            c_x1=0.4, c_x2=0.1, c_x3=-0.2, c_x4=0.2,
            c_x12=0.1, c_x13=-0.1, c_x14=-0.1,
            c_x23=-0.3, c_x24=-0.2, c_x34=-0.05,
            c_x123=-0.05, c_x124=0.03, c_x134=-0.02, c_x234=-0.01,
            pop_ratio=population_ratio_array[3],
            coupling_row=np.array([0.2, 0.4, 0.3, 1.0], dtype=np.float64)
        )
    ]

    fixed_delays_template = {
        'tau1': 4,
        'tau2': 15,
        'tau3': 2,
        'tau4': 10
    }

    initial_conditions = np.array([2.5, 0.0, -1.5, 0.3], dtype=np.float64)
    xbar = np.array([1.0, 1.0, 1.0, 0.5], dtype=np.float64)

    T = 10000.0    
    epsilon = 1.0 

    lambda_params = np.array([0.2, 0.6, 0.4, 0.9], dtype=np.float64)
    sigma_params  = np.array([0.03, 0.06, 0.05, 0.10], dtype=np.float64)
    

    sweeps = [
        ('tau1', {'tau1': fixed_delays_template['tau1'],
                  'tau2': fixed_delays_template['tau2'],
                  'tau3': fixed_delays_template['tau3'],
                  'tau4': fixed_delays_template['tau4']}),
        ('tau2', {'tau1': fixed_delays_template['tau1'],
                  'tau2': fixed_delays_template['tau2'],
                  'tau3': fixed_delays_template['tau3'],
                  'tau4': fixed_delays_template['tau4']}),
        ('tau3', {'tau1': fixed_delays_template['tau1'],
                  'tau2': fixed_delays_template['tau2'],
                  'tau3': fixed_delays_template['tau3'],
                  'tau4': fixed_delays_template['tau4']}),
        ('tau4', {'tau1': fixed_delays_template['tau1'],
                  'tau2': fixed_delays_template['tau2'],
                  'tau3': fixed_delays_template['tau3'],
                  'tau4': fixed_delays_template['tau4']})
    ]

    for varying_tau, fixed_tau_dict in sweeps:
        print(f"\n=== Varying {varying_tau} from 0..200 ===")
        params = (
            agents,
            initial_conditions,
            xbar,
            T,
            epsilon,
            varying_tau,
            fixed_tau_dict.copy(),  
            lambda_params,
            sigma_params,
            N
        )
        vary_and_plot_4agents(params)

if __name__ == "__main__":
    main()
