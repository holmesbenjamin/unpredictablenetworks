import numpy as np
import matplotlib.pyplot as plt
import random

def heaviside(x, method_name, epsilon_prime=1.0):
    """
    Compute the Heaviside function based on the specified method.
    
    Parameters:
    - x: Input value
    - method_name: Type of Heaviside function ('Standard', 'Smooth', etc.)
    - epsilon_prime: Parameter controlling the steepness of the sigmoid
    
    Returns:
    - Heaviside function value
    """
    if method_name == 'Standard':
        return standard_heaviside(x)
    elif method_name == 'Smooth':
        return smooth_heaviside(x, k=10)  
    elif method_name == 'Regularized':
        return regularized_heaviside(x, k=10)
    elif method_name == 'Polynomial':
        return polynomial_heaviside(x, delta=0.1)
    elif method_name == 'Piecewise Linear':
        return piecewise_linear_heaviside(x, delta=0.1)
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
    """
    Retrieve parameters for the specified variant.
    
    Parameters:
    - variant: String identifier for the variant ('variant_i', etc.)
    
    Returns:
    - Dictionary of parameters
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

simulation_types = {
    1: 'Discrete',
    2: 'Smoothed (ε\'=0.1)',
    3: 'Smoothed in Discrete Time (ε\'=2)',
    4: 'Shallower Sigmoid in Discrete Time (ε\'=1)',
    5: 'Intermediary Variable',
    6: 'Delay',
    7: 'Random Delay'
}

simulation_colors = {
    1: 'blue',
    2: 'green',
    3: 'red',
    4: 'purple',
    5: 'orange',
    6: 'cyan',
    7: 'magenta'
}

def run_simulation(
    a_x1, c_x1_1, c_x1_2, c_x1_12,
    a_x2, c_x2_1, c_x2_2, c_x2_12,
    a_y, c_y_12,
    initial_x1, initial_x2,
    x1bar, x2bar,
    tau1, tau2,
    integration_method='Euler',
    heaviside_method='Standard',
    simulation_type=1,  
    T=10000.0, epsilon=1.0
):
    """
    Run the simulation based on the specified parameters and simulation type.
    
    Parameters:
    - All system parameters as specified
    - simulation_type: Integer (1-7) specifying the simulation method
    - T: Total simulation time
    - epsilon: Time-step
    
    Returns:
    - y: Array of y over time
    """
    N = int(T / epsilon)
    x1 = np.zeros(N)
    x2 = np.zeros(N)
    y = np.zeros(N)
    mode_array = np.zeros(N, dtype=int)

    x1[0] = initial_x1
    x2[0] = initial_x2
    y[0] = 0.0

    if simulation_type in [6, 7]:
        hj1 = heaviside(x1[0] - x1bar, heaviside_method)
        hj2 = heaviside(x2[0] - x2bar, heaviside_method)
        delay_remaining_x1 = 0
        delay_remaining_x2 = 0

    if simulation_type == 5:
        zj1 = 0.0
        zj2 = 0.0

    for n in range(1, N):
        if simulation_type == 1:
            H1 = heaviside(x1[n-1] - x1bar, heaviside_method)
            H2 = heaviside(x2[n-1] - x2bar, heaviside_method)

        elif simulation_type == 2:
            H1 = (1 + np.tanh((x1[n-1] - x1bar)/0.1)) / 2
            H2 = (1 + np.tanh((x2[n-1] - x2bar)/0.1)) / 2

        elif simulation_type == 3:
            H1 = (1 + np.tanh((x1[n-1] - x1bar)/2.0)) / 2
            H2 = (1 + np.tanh((x2[n-1] - x2bar)/2.0)) / 2

        elif simulation_type == 4:
            H1 = (1 + np.tanh((x1[n-1] - x1bar)/1.0)) / 2
            H2 = (1 + np.tanh((x2[n-1] - x2bar)/1.0)) / 2

        elif simulation_type == 5:
            zj1 += epsilon * (x1[n-1] - zj1)
            zj2 += epsilon * (x2[n-1] - zj2)
            H1 = heaviside(zj1 - x1bar, heaviside_method)
            H2 = heaviside(zj2 - x2bar, heaviside_method)

        elif simulation_type in [6, 7]:
            if simulation_type == 6:
                if n % 2 == 0:
                    H1 = heaviside(x1[n-1] - x1bar, heaviside_method)
                    hj1 = H1
                else:
                    H1 = hj1  

                if n % 3 == 0:
                    H2 = heaviside(x2[n-1] - x2bar, heaviside_method)
                    hj2 = H2
                else:
                    H2 = hj2  

            elif simulation_type == 7:
                sign_change_x1 = (np.sign(x1[n-1] - x1bar) != np.sign(x1[n-2] - x1bar)) if n > 1 else False
                sign_change_x2 = (np.sign(x2[n-1] - x2bar) != np.sign(x2[n-2] - x2bar)) if n > 1 else False

                if sign_change_x1 and delay_remaining_x1 == 0:
                    delay_remaining_x1 = random.randint(0, 15)
                if sign_change_x2 and delay_remaining_x2 == 0:
                    delay_remaining_x2 = random.randint(0, 15)

                if delay_remaining_x1 > 0:
                    delay_remaining_x1 -= 1
                    H1 = hj1  
                else:
                    H1 = heaviside(x1[n-1] - x1bar, heaviside_method)
                    hj1 = H1  

                if delay_remaining_x2 > 0:
                    delay_remaining_x2 -= 1
                    H2 = hj2  
                else:
                    H2 = heaviside(x2[n-1] - x2bar, heaviside_method)
                    hj2 = H2  

        else:
            raise ValueError(f"Unknown simulation type: {simulation_type}")

        x1_dot = a_x1 + c_x1_1 * H1 + c_x1_2 * H2 + c_x1_12 * H1 * H2
        x2_dot = a_x2 + c_x2_1 * H1 + c_x2_2 * H2 + c_x2_12 * H1 * H2
        y_dot = a_y + c_y_12 * H1 * H2

        x1[n] = x1[n-1] + epsilon * x1_dot
        x2[n] = x2[n-1] + epsilon * x2_dot
        y[n] = y[n-1] + epsilon * y_dot

        if H1 == 0.0 and H2 == 0.0:
            mode_array[n] = 0
        elif H1 == 0.0 and H2 == 1.0:
            mode_array[n] = 1
        elif H1 == 1.0 and H2 == 0.0:
            mode_array[n] = 2
        elif H1 == 1.0 and H2 == 1.0:
            mode_array[n] = 3

    return y

def main():
    variants = ['variant_i', 'variant_ii', 'variant_iii']

    simulation_types = {
        1: 'Discrete',
        2: 'Smoothed (ε\'=0.1)',
        3: 'Smoothed in Discrete Time (ε\'=2)',
        4: 'Shallower Sigmoid in Discrete Time (ε\'=1)',
        5: 'Intermediary Variable',
        6: 'Delay',
        7: 'Random Delay'
    }

    simulation_colors = {
        1: 'purple',
        2: 'lime',
        3: 'pink',
        4: 'cyan',
        5: 'orange',
        6: 'red',
        7: 'midnightblue'
    }

    heaviside_method = 'Smooth'  
    integration_method = 'Euler'

    initial_x1 = 1.0
    initial_x2 = 1.0
    x1bar = 1.0
    x2bar = 1.0
    tau1 = 2  
    tau2 = 3

    T = 1000.0 
    epsilon = 1.0 

    y_results = {variant: {} for variant in variants}

    for variant in variants:
        params = get_variant_parameters(variant)
        print(f"Running simulations for {variant}...")

        for sim_type in simulation_types:
            sim_name = simulation_types[sim_type]
            print(f"  Simulation Type {sim_type}: {sim_name}")

            y = run_simulation(
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
                simulation_type=sim_type,
                T=T,
                epsilon=epsilon
            )

            y_results[variant][sim_type] = y

    for variant in variants:
        plt.figure(figsize=(15, 8))
        for sim_type in simulation_types:
            sim_name = simulation_types[sim_type]
            y = y_results[variant][sim_type]
            time_steps = np.arange(len(y))
            plt.plot(time_steps, y, label=sim_name, color=simulation_colors[sim_type])

        plt.title(f'y over Time for {variant}', fontsize=16)
        plt.xlabel('Time Steps', fontsize=14)
        plt.ylabel('y', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
