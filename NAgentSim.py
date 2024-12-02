import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, chain

def heaviside(x, method_name, epsilon_prime=1.0):
    if method_name == 'Standard':
        return standard_heaviside(x)
    elif method_name == 'Smooth':
        return smooth_heaviside(x, k=10)
    else:
        raise ValueError(f"Unknown Heaviside method: {method_name}")

def standard_heaviside(x):
    return 1.0 if x >= 0 else 0.0

def smooth_heaviside(x, k=10):
    return 1.0 / (1.0 + np.exp(-k * x))

def generate_subsets(agent_indices, max_order):
    return list(chain.from_iterable(combinations(agent_indices, r) for r in range(2, max_order+1)))

def get_variant_parameters(variant, N):
    if variant == 'variant_i' and N == 2:
        params = {
            'a_x': np.array([2, 2]),
            'c_x': {
                0: {0: -3, 1: 1, (0, 1): -3},
                1: {0: 0, 1: -3, (0, 1): -3}
            },
            'a_y': -1,
            'c_y': {(0, 1): 4},
            'xbar': np.array([1.0, 1.0])
        }
        return params
    elif variant == 'variant_ii' and N == 2:
        params = {
            'a_x': np.array([1, 3]),
            'c_x': {
                0: {0: -3, 1: 1, (0, 1): 0},
                1: {0: -1, 1: -6, (0, 1): 0}
            },
            'a_y': -1,
            'c_y': {(0, 1): 6},
            'xbar': np.array([1.0, 1.0])
        }
        return params
    elif variant == 'variant_iii' and N == 2:
        params = {
            'a_x': np.array([3, 1]),
            'c_x': {
                0: {0: -5, 1: 0, (0, 1): 0},
                1: {0: 0, 1: -5, (0, 1): 0}
            },
            'a_y': -1,
            'c_y': {(0, 1): 6},
            'xbar': np.array([1.0, 1.0])
        }
        return params
    else:
        raise ValueError(f"Unknown variant: {variant} for N = {N}")

def run_simulation(
    params,
    initial_x,
    xbar,
    tau_array,
    max_interaction_order=2,
    integration_method='Euler',
    heaviside_method='Standard',
    simulation_type=6,
    T=100000.0, epsilon=1.0
):
    N = len(initial_x)
    num_steps = int(T / epsilon)
    x = np.zeros((N, num_steps))
    x[:, 0] = initial_x
    y = np.zeros(num_steps)
    
    H = np.array([heaviside(x[i, 0] - xbar[i], heaviside_method) for i in range(N)])
    hj = H.copy()
    
    transient_steps = int(num_steps * 0.1) 
    sum_H = np.zeros(N)
    sum_H_product = 0.0
    count = 0
    
    agent_indices = list(range(N))
    higher_order_subsets = generate_subsets(agent_indices, max_interaction_order)
    
    for n in range(1, num_steps):
        if simulation_type == 6:
            for i in range(N):
                if n % tau_array[i] == 0:
                    H[i] = heaviside(x[i, n-1] - xbar[i], heaviside_method)
                    hj[i] = H[i]
                else:
                    H[i] = hj[i]
        else:
            raise ValueError(f"Unknown simulation type: {simulation_type}")
        
        for i in range(N):
            x_dot = params['a_x'][i]
            for j in agent_indices:
                c_ij = params['c_x'][i].get(j, 0.0)
                x_dot += c_ij * H[j]
            for subset in higher_order_subsets:
                c_ijs = params['c_x'][i].get(subset, 0.0)
                if c_ijs != 0.0:
                    H_product = np.prod([H[k] for k in subset])
                    x_dot += c_ijs * H_product
            x[i, n] = x[i, n-1] + epsilon * x_dot
        y_dot = params['a_y']
        for subset, c_ys in params['c_y'].items():
            H_product = np.prod([H[k] for k in subset])
            y_dot += c_ys * H_product
        y[n] = y[n-1] + epsilon * y_dot
        if n >= transient_steps:
            sum_H += H
            sum_H_product += np.prod(H)
            count += 1
    mu = sum_H / count
    mu_all = sum_H_product / count
    return mu, mu_all

def main():
    N = 2  
    variants = ['variant_i', 'variant_ii', 'variant_iii']

    heaviside_method = 'Standard' 
    integration_method = 'Euler' 

    initial_x = np.array([2.5, 2.5])
    xbar = np.array([1.0, 1.0])

    T = 100000.0  
    epsilon = 1.0  

    delta_values = np.arange(1, 101)

    for variant in variants:
        params = get_variant_parameters(variant, N)
        print(f"Running simulations for {variant}...")

        mu_all_values = []
        delta_values_list = []

        for delta in delta_values:
            tau_array = np.array([delta, 50])  # tau1 = delta, tau2 = 50

            mu, mu_all = run_simulation(
                params=params,
                initial_x=initial_x,
                xbar=xbar,
                tau_array=tau_array,
                integration_method=integration_method,
                heaviside_method=heaviside_method,
                simulation_type=6,
                T=T,
                epsilon=epsilon
            )

            mu_all_values.append(mu_all)
            delta_values_list.append(delta)

        mu_all_values = np.array(mu_all_values)
        delta_values_list = np.array(delta_values_list)

        plt.figure(figsize=(10, 6))
        plt.plot(delta_values_list, mu_all_values, label='μ_all', color='black')
        plt.title(f'Variant {variant}')
        plt.xlabel('δ (Agent 1 decision interval)')
        plt.ylabel('μ_all (Average of H1*H2)')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
