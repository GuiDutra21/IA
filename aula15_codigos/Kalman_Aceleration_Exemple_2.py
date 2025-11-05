import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

ti = 0
tf = 100
deltat = 1
t = np.linspace(ti, tf, int(tf/deltat))

A = np.array([[1, 0, deltat, 0],
              [0, 1, 0, deltat],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

B = np.array([[0.5 * deltat**2, 0],
              [0, 0.5 * deltat**2],
              [deltat, 0],
              [0, deltat]])

H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

sigmaest_x = 0.15
sigmaest_y = 0.05
sigmaobs_x = 6.0
sigmaobs_y = 1.5

Q = np.diag([sigmaest_x**2, sigmaest_y**2, sigmaest_x**2, sigmaest_y**2])
R = np.diag([sigmaobs_x**2, sigmaobs_y**2])

x0, y0, vx0, vy0 = 0, 0, 0, 0

# Acelerações: no eixo x acelera e desacelera; no eixo y aceleração diferente para deslocamento variado
u_x = np.zeros(t.size)
u_y = np.zeros(t.size)

# Accelera e desacelera no eixo x
u_x[:t.size // 2] = 2.0
u_x[t.size // 2:] = -2.0

# No eixo y aceleração menor e depois maior para um padrão diferente
u_y[:t.size // 3] = 1.0
u_y[t.size // 3: 2 * t.size // 3] = 3.0
u_y[2 * t.size // 3:] = 0.5

S = np.zeros((4, t.size))
S[:, 0] = [x0, y0, vx0, vy0]
for i in range(1, t.size):
    u = np.array([u_x[i - 1], u_y[i - 1]])
    S[:, i] = A @ S[:, i - 1] + B @ u

S_obs = S[:2, :] + np.vstack((sigmaobs_x * np.random.randn(t.size), sigmaobs_y * np.random.randn(t.size)))

x_hat = np.zeros((4, t.size))
P = np.eye(4) * 10
x_hat[:, 0] = np.array([S_obs[0, 0], S_obs[1, 0], 0, 0])

for i in range(1, t.size):
    u = np.array([u_x[i - 1], u_y[i - 1]])
    x_hat[:, i] = A @ x_hat[:, i - 1] + B @ u
    P = A @ P @ A.T + Q

    K = P @ H.T @ inv(H @ P @ H.T + R)
    y = S_obs[:, i] - H @ x_hat[:, i]
    x_hat[:, i] = x_hat[:, i] + K @ y
    P = P - K @ H @ P

res_pos_x = x_hat[0, :] - S[0, :]
res_pos_y = x_hat[1, :] - S[1, :]
res_v_x = x_hat[2, :] - S[2, :]
res_v_y = x_hat[3, :] - S[3, :]

plt.figure(figsize=(16, 12))

plt.subplot(3, 2, 1)
plt.title("Posição X - Real, Observada e Estimada")
plt.plot(t, S[0, :], 'b-', label='Posição X real')
plt.plot(t, S_obs[0, :], 'r.', label='Obs x (ruído)')
plt.plot(t, x_hat[0, :], 'g--', label='Estimada Kalman')
plt.xlabel('Tempo (s)')
plt.ylabel('Posição X (m)')
plt.legend()

plt.subplot(3, 2, 2)
plt.title("Posição Y - Real, Observada e Estimada")
plt.plot(t, S[1, :], 'b-', label='Posição Y real')
plt.plot(t, S_obs[1, :], 'r.', label='Obs y (ruído)')
plt.plot(t, x_hat[1, :], 'g--', label='Estimada Kalman')
plt.xlabel('Tempo (s)')
plt.ylabel('Posição Y (m)')
plt.legend()

plt.subplot(3, 2, 3)
plt.title("Velocidade X - Real e Estimada")
plt.plot(t, S[2, :], 'b-', label='Velocidade X real')
plt.plot(t, x_hat[2, :], 'g--', label='Estimada Kalman')
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade X (m/s)')
plt.legend()

plt.subplot(3, 2, 4)
plt.title("Velocidade Y - Real e Estimada")
plt.plot(t, S[3, :], 'b-', label='Velocidade Y real')
plt.plot(t, x_hat[3, :], 'g--', label='Estimada Kalman')
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade Y (m/s)')
plt.legend()

plt.subplot(3, 2, 5)
plt.title("Resíduo da Posição X (Estimativa - Real)")
plt.plot(t, res_pos_x, 'm-', label='Resíduo posição x')
plt.xlabel('Tempo (s)')
plt.ylabel('Erro posição X (m)')
plt.legend()

plt.subplot(3, 2, 6)
plt.title("Resíduo da Posição Y (Estimativa - Real)")
plt.plot(t, res_pos_y, 'c-', label='Resíduo posição y')
plt.xlabel('Tempo (s)')
plt.ylabel('Erro posição Y (m)')
plt.legend()

plt.tight_layout()
plt.show()
