import numpy as np
import matplotlib.pyplot as plt

def heaviside(x, method_name, epsilon_prime=1.0):
    """
    Compute the Heaviside function based on the specified method.
    """
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

def get_variant_parameters(variant):
    """
    Retrieve parameters for the specified variant.
    """
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

def run_simulation(
    a_x1, c_x1_1, c_x1_2, c_x1_12,
    a_x2, c_x2_1, c_x2_2, c_x2_12,
    a_y, c_y_12,
    initial_x1, initial_x2,
    x1bar, x2bar,
    tau1, tau2,
    integration_method='Euler',
    heaviside_method='Standard',
    simulation_type=6,  
    T=100000.0, epsilon=1.0
):
    """
    Run the simulation based on the specified parameters and simulation type.
    
    Parameters:
    - All system parameters as specified
    - simulation_type: Integer (6) specifying the simulation method
    - T: Total simulation time
    - epsilon: Time-step
    
    Returns:
    - μ1, μ2, μ3: Averages of H1, H2, H1*H2
    """
    N = int(T / epsilon)
    x1 = np.zeros(N)
    x2 = np.zeros(N)
    y = np.zeros(N)

    x1[0] = initial_x1
    x2[0] = initial_x2
    y[0] = 0.0

    H1 = heaviside(x1[0] - x1bar, heaviside_method)
    H2 = heaviside(x2[0] - x2bar, heaviside_method)
    hj1 = H1
    hj2 = H2

    transient_steps = int(N * 0.1) 
    sum_H1 = 0.0
    sum_H2 = 0.0
    sum_H1H2 = 0.0
    count = 0

    for n in range(1, N):
        if simulation_type == 6:
            if n % tau1 == 0:
                H1 = heaviside(x1[n-1] - x1bar, heaviside_method)
                hj1 = H1
            else:
                H1 = hj1  

            if n % tau2 == 0:
                H2 = heaviside(x2[n-1] - x2bar, heaviside_method)
                hj2 = H2
            else:
                H2 = hj2 

        else:
            raise ValueError(f"Unknown simulation type: {simulation_type}")

        x1_dot = a_x1 + c_x1_1 * H1 + c_x1_2 * H2 + c_x1_12 * H1 * H2
        x2_dot = a_x2 + c_x2_1 * H1 + c_x2_2 * H2 + c_x2_12 * H1 * H2
        y_dot = a_y + c_y_12 * H1 * H2

        x1[n] = x1[n-1] + epsilon * x1_dot
        x2[n] = x2[n-1] + epsilon * x2_dot
        y[n] = y[n-1] + epsilon * y_dot

        if n >= transient_steps:
            sum_H1 += H1
            sum_H2 += H2
            sum_H1H2 += H1 * H2
            count += 1

    μ1 = sum_H1 / count
    μ2 = sum_H2 / count
    μ3 = sum_H1H2 / count

    return μ1, μ2, μ3

def main():
    variants = ['variant_i', 'variant_ii', 'variant_iii']

    heaviside_method = 'Standard' 
    integration_method = 'Euler' 

    initial_x1 = 4.0
    initial_x2 = 4.0
    x1bar = 1.0
    x2bar = 1.0

    T = 100000.0  
    epsilon = 1.0  

    delta_values = np.arange(1, 101)

    for variant in variants:
        params = get_variant_parameters(variant)
        print(f"Running simulations for {variant}...")

        μ3_values = []
        δ_values = []


        for δ in delta_values:
            tau1 = δ
            tau2 = 50

            μ1, μ2, μ3 = run_simulation(
                a_x1=params['a_x1'],
                c_x1_1=params['c_x1_1'],
                c_x1_2=params['c_x1_2'],
                c_x1_12=params['c_x1_12'],
                a_x2=params['a_x2'],
                c_x2_1=params['c_x2_1'],
                c_x2_2=params['c_x2_2'],
                c_x2_12=params['c_x2_12'],
                a_y=params['a_y'],
                c_y_12=params['c_y_12'],
                initial_x1=initial_x1,
                initial_x2=initial_x2,
                x1bar=x1bar,
                x2bar=x2bar,
                tau1=tau1,
                tau2=tau2,
                integration_method=integration_method,
                heaviside_method=heaviside_method,
                simulation_type=6,
                T=T,
                epsilon=epsilon
            )

            μ3_values.append(μ3)
            δ_values.append(δ)

        μ3_values = np.array(μ3_values)
        δ_values = np.array(δ_values)

        plt.figure(figsize=(10, 6))
        plt.plot(δ_values, μ3_values, label='μ3 (H1*H2)', color='black')
        plt.title(f'Variant {variant}')
        plt.xlabel('δ (Trader 1 decision interval)')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
