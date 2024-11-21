import sympy as sp
import numpy as np
from itertools import combinations, chain

def generate_subsets(s, max_order):
    return list(chain.from_iterable(combinations(s, r) for r in range(1, max_order+1)))

N = 3  

max_interaction_order = 2  

t = sp.symbols('t')

x_symbols = sp.symbols(f'x1:{N+1}')
x = [sp.Function(f'x{i+1}')(t) for i in range(N)]

H = [sp.Function(f'H{i+1}')(x[i]) for i in range(N)]

a_x = sp.symbols(f'a_x1:{N+1}')

dx_dt = []

agent_indices = list(range(N))

all_subsets = generate_subsets(agent_indices, max_interaction_order)

C_pairwise = sp.MatrixSymbol('C_pairwise', N, N)  

C_higher_order = {}

for order in range(2, max_interaction_order+1):
    subsets = combinations(agent_indices, order)
    for s in subsets:
        for i in agent_indices:
            coeff_symbol = sp.symbols(f'c_x{i+1}_' + ''.join([str(j+1) for j in s]))
            C_higher_order[(i, s)] = coeff_symbol

for i in agent_indices:
    equation = a_x[i]
    for j in agent_indices:
        coeff = C_pairwise[i, j]
        equation += coeff * H[j]
    for s in all_subsets:
        if len(s) >= 2:
            if (i, s) in C_higher_order:
                coeff = C_higher_order[(i, s)]
                H_product = sp.Mul(*(H[j] for j in s))
                equation += coeff * H_product
    dx_dt.append(sp.Eq(x[i].diff(t), equation))

#N
a_x_values = np.array([1.0, 2.0, 3.0])

#NxN
C_pairwise_values = np.array([
    [0.0, 0.5, 0.1],
    [-0.3, 0.0, 0.2],
    [0.4, -0.1, 0.0]
]) 

C_higher_order_values = {
    # (i, s): value
    # i: index of the agent being affected (0-based index)
    # s: tuple of agent indices involved in the interaction (0-based index)
    (0, (1, 2)): 0.05,  # c_{x1_23}
    (1, (0, 2)): -0.02,  # c_{x2_13}
}
coeff_values = {}

for i in agent_indices:
    coeff_values[a_x[i]] = a_x_values[i]

for i in agent_indices:
    for j in agent_indices:
        coeff_symbol = C_pairwise[i, j]
        coeff_values[coeff_symbol] = C_pairwise_values[i, j]

for key, coeff_symbol in C_higher_order.items():
    i, s = key
    if key in C_higher_order_values:
        coeff_values[coeff_symbol] = C_higher_order_values[key]
    else:
        coeff_values[coeff_symbol] = 0  

dx_dt_numeric = [eq.subs(coeff_values) for eq in dx_dt]

for i, eq in enumerate(dx_dt_numeric):
    print(f"Equation for x{i+1}:")
    sp.pprint(eq)
    print()
