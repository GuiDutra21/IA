import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_cov_ellipse(cov, mean, ax, n_std=1.0, facecolor='none', **kwargs):
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigvals)

    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      facecolor=facecolor, **kwargs)
    ax.add_patch(ellipse)

# Parâmetros do sistema
dt = 1.0
F = np.array([[1, dt],
              [0, 1]])          # Matriz de transição
H = np.array([[1, 0]])          # Medimos só a posição
Q = np.array([[1e-3, 0],
              [0, 5e-4]])        # Ruído do processo (aumentado)
R = np.array([[0.5]])           # Ruído da medição maior para discrepância visível

# Estado inicial
x = np.array([[0],
              [2]])             # posição e velocidade inicial (velocidade verdadeira 2 m/s)
P = np.eye(2)                    # Covariância inicial

# Simulação: movimento com velocidade constante e ruídos
true_positions = []
true_velocities = []
measurements = []
estimates = []
covariances = []

np.random.seed(42)

v_true = 2.0
for t in range(30):
    pos_true = v_true * t
    meas = pos_true + np.random.normal(0, np.sqrt(R[0, 0]))

    true_positions.append(pos_true)
    true_velocities.append(v_true)
    measurements.append(meas)

    # Predição
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q

    # Atualização
    y = np.array([[meas]]) - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x = x_pred + K @ y
    P = (np.eye(2) - K @ H) @ P_pred

    estimates.append(x.flatten())
    covariances.append(P)

# Plotar resultados
fig, ax = plt.subplots(figsize=(12, 7))

# Medições da posição
ax.scatter(range(len(measurements)), measurements, color='red', label='Medições (posição)', alpha=0.6)

# Posições reais e estimadas
ax.plot(true_positions, label='Posição Verdadeira', color='black', linewidth=2)
estimates = np.array(estimates)
ax.plot(estimates[:, 0], label='Posição Estimada', color='blue')

# Velocidades verdadeira e estimada
ax.plot(true_velocities, label='Velocidade Verdadeira', color='green', linestyle='dashed')
ax.plot(estimates[:, 1], label='Velocidade Estimada', color='orange', linestyle='dashed')

# Elipses de covariância em intervalos
for i in range(0, len(covariances), 5):
    cov = covariances[i][:2, :2]
    mean = estimates[i, :2]
    plot_cov_ellipse(cov, mean, ax, n_std=2, edgecolor='blue', alpha=0.3)

ax.set_xlabel('Tempo (s)')
ax.set_ylabel('Posição (m) / Velocidade (m/s)')
ax.set_title('Filtro de Kalman - Estimação para Foguete Subindo')
ax.legend(loc='upper left')
ax.grid(True)

ax.set_xlim(0, 29)  # Ajuste para mostrar de 0 a 29 segundos (30 pontos amostrados)
plt.show()
