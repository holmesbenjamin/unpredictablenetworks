import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

def heaviside(x, epsilon=1.0):
    return 0.5 * (1.0 + np.tanh(x / epsilon))

def run_simulation(
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
    T=1000.0, epsilon=1.0
):
    """
    Run the simulation of the 3-agent system (Market Maker, Fundamental Investor, HFT) 
    with all pairwise and three-way interaction coefficients. 
    Delays tau1, tau2, tau3 are floored to integers for indexing.

    Returns:
    x1, x2, x3 as numpy arrays of length ~ T/epsilon describing the system trajectory.
    """

    N = int(T / epsilon)
    delay1 = int(np.floor(tau1))
    delay2 = int(np.floor(tau2))
    delay3 = int(np.floor(tau3))
    max_delay = max(delay1, delay2, delay3)

    X1 = np.ones(N + max_delay) * initial_x1
    X2 = np.ones(N + max_delay) * initial_x2
    X3 = np.ones(N + max_delay) * initial_x3

    for n in range(max_delay, N + max_delay - 1):
        i_mm  = n - delay1  
        i_fi  = n - delay2  
        i_hft = n - delay3 

        H1 = heaviside(X1[i_mm] - x1bar)
        H2 = heaviside(X2[i_fi] - x2bar)
        H3 = heaviside(X3[i_hft] - x3bar)

        x1_dot = (
            a_x1
            + c_x1_1     * H1
            + c_x1_2     * H2
            + c_x1_3     * H3
            + c_x1_12    * (H1*H2)
            + c_x1_13    * (H1*H3)
            + c_x1_23    * (H2*H3)
            + c_x1_123   * (H1*H2*H3)
        )

        x2_dot = (
            a_x2
            + c_x2_1     * H1
            + c_x2_2     * H2
            + c_x2_3     * H3
            + c_x2_12    * (H1*H2)
            + c_x2_13    * (H1*H3)
            + c_x2_23    * (H2*H3)
            + c_x2_123   * (H1*H2*H3)
        )

        x3_dot = (
            a_x3
            + c_x3_1     * H1
            + c_x3_2     * H2
            + c_x3_3     * H3
            + c_x3_12    * (H1*H2)
            + c_x3_13    * (H1*H3)
            + c_x3_23    * (H2*H3)
            + c_x3_123   * (H1*H2*H3)
        )

        X1[n+1] = X1[n] + epsilon*x1_dot
        X2[n+1] = X2[n] + epsilon*x2_dot
        X3[n+1] = X3[n] + epsilon*x3_dot

    x1 = X1[max_delay:]
    x2 = X2[max_delay:]
    x3 = X3[max_delay:]

    return x1, x2, x3

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

    tau1 = 4
    tau2 = 15
    tau3 = 2

    initial_x1, initial_x2, initial_x3 = 2.5, 0.0, -1.5
    x1bar, x2bar, x3bar = 1.0, 1.0, 1.0

    T = 10000.0
    epsilon = 1.0

    x1, x2, x3 = run_simulation(
        a_x1,
        c_x1_1,  c_x1_2,  c_x1_3,
        c_x1_12, c_x1_13, c_x1_23,
        c_x1_123,

        a_x2,
        c_x2_1,  c_x2_2,  c_x2_3,
        c_x2_12, c_x2_13, c_x2_23,
        c_x2_123,

        a_x3,
        c_x3_1,  c_x3_2,  c_x3_3,
        c_x3_12, c_x3_13, c_x3_23,
        c_x3_123,

        tau1, tau2, tau3,
        initial_x1, initial_x2, initial_x3,
        x1bar, x2bar, x3bar,
        T=T,
        epsilon=epsilon
    )

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x1, x2, x3, color='blue', linewidth=1.0, label='Trajectory (line)')

    ax.set_title("3D Phase Space of (x1, x2, x3)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
