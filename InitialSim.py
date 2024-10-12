import numpy as np
import matplotlib.pyplot as plt


c = 7.0    
b = 70.0     
d = 114.0     
e = 10.0     
f = 1300.0     
#x1_dot = c - b * H1_delayed - e * H1_delayed * H2_delayed
#x2_dot = c - d * H2_delayed
#z_dot = -2 * c + f * H1_delayed * H2_delayed
x1bar = 2100
x2bar = 1999
epsilon = 1.0  #time step
tau1 = 1  #delay for x1 (time steps)
tau2 = 11  #delay for x2 (time steps)

T = 1000.0
N = int(T / epsilon)

x1 = np.zeros(N)
x2 = np.zeros(N)
z = np.zeros(N)
zdot = np.zeros(N)

x1[0] = 2000
x2[0] = 2000
z[0] = 0

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
mode_array = np.zeros(N, dtype=int)

for n in range(1, N):
    idx_delay1 = (n - tau1) % delay_buffer_size1
    idx_delay2 = (n - tau2) % delay_buffer_size2

    H1_delayed = 1.0 if delay_buffer1[idx_delay1] - x1bar >= 0 else 0.0
    H2_delayed = 1.0 if delay_buffer2[idx_delay2] - x2bar >= 0 else 0.0

    x1_dot = c - b * H1_delayed - e * H1_delayed * H2_delayed
    x2_dot = c - d * H2_delayed

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

    x1[n] = x1[n - 1] + epsilon * x1_dot
    x2[n] = x2[n - 1] + epsilon * x2_dot

    delay_buffer1[n % delay_buffer_size1] = x1[n]
    delay_buffer2[n % delay_buffer_size2] = x2[n]

    
    z_dot = -2 * c + (f * H1_delayed * H2_delayed)
    print(z_dot)
    zdot[n] = z_dot
    z[n] = z[n - 1] + epsilon * z_dot

total_time = time_mu00 + time_mu01 + time_mu10 + time_mu11

print(f"Time spent in mu00: {time_mu00:.2f} units ({(time_mu00/total_time)*100:.2f}%)")
print(f"Time spent in mu01: {time_mu01:.2f} units ({(time_mu01/total_time)*100:.2f}%)")
print(f"Time spent in mu10: {time_mu10:.2f} units ({(time_mu10/total_time)*100:.2f}%)")
print(f"Time spent in mu11: {time_mu11:.2f} units ({(time_mu11/total_time)*100:.2f}%)")

plt.figure(figsize=(12, 6))
plt.plot(np.arange(N) * epsilon, x1, label='x1(t)')
plt.plot(np.arange(N) * epsilon, x2, label='x2(t)')
plt.plot(np.arange(N) * epsilon, z, label='z(t)')
plt.xlabel('Time')
plt.ylabel('State Variables')
plt.title('Dynamics of x1, x2, and z over time')
plt.grid(True)
plt.legend()
plt.show()

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x1, x2, z)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('z')
ax.set_title('3D Trajectory of x1, x2, and z')
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(x1, x2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Phase Space Trajectory (x1 vs x2)')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(np.arange(N), zdot)
plt.xlabel('t')
plt.ylabel('z')
plt.title('z over time')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 3))
plt.plot(np.arange(N) * epsilon, mode_array, drawstyle='steps-post')
plt.xlabel('Time')
plt.ylabel('Mode')
plt.title('Mode over Time')
plt.yticks([0, 1, 2, 3], ['mu00', 'mu01', 'mu10', 'mu11'])
plt.grid(True)
plt.show()

