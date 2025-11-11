import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

np.random.seed(42)  # Fixa os numeros aleatorios gerados

dt = 1.0 # Medicoes a cada 1 segundo
tempo_total = 300 # Tempo total (5 min)
num_passos = tempo_total
t = np.linspace(0, tempo_total - dt, num_passos)

# Temperatura real (mas com variacao periodoica)
deslocamento = 25.0
amplitude = 5.0
periodo = 100.0
omega = 1.4 * np.pi / periodo
temp_real = deslocamento + amplitude * np.sin(omega * t)

# Sensor ruidoso
sensor_noise_std = 1.5
medicao_sensor = temp_real + np.random.normal(0, sensor_noise_std, num_passos)

# Estado: [deslocamento (temperatura media), s (sen), c (cosseno)]
# Temos essas duas componentes a mais para linearizar a equacao
cosw, sinw = np.cos(omega * dt), np.sin(omega * dt)
A = np.array([ # Matriz de transicao de estados
    [1.0, 0.0, 0.0],
    [0.0, cosw, sinw],
    [0.0, -sinw, cosw]
])

H = np.array([[1.0, 1.0, 0.0]])  # z = deslocamento + s (sen)

Q = np.diag([1e-4, 0.01, 0.01]) # Ruido do processo
R = np.array([[sensor_noise_std**2]]) # Ruido das observacoes

x_hat = np.zeros((3, num_passos))
P = np.eye(3) * 10.0
x_hat[:, 0] = [medicao_sensor[0], 0.0, 0.0]

for k in range(1, num_passos):
    
    # Predicao
    x_hat[:, k] = A @ x_hat[:, k-1]
    P_pred = A @ P @ A.T + Q
    
    # Atualizacao
    z = np.array([medicao_sensor[k]])
    K = P_pred @ H.T @ inv(H @ P_pred @ H.T + R) # Ganho de Kalman
    x_hat[:, k] = x_hat[:, k] + (K @ (z - H @ x_hat[:, k])).ravel()
    P = (np.eye(3) - K @ H) @ P_pred
    
temp_kalman = (H @ x_hat).ravel()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

# Grafico da temperatura 
ax1.plot(t, temp_real, 'b-', label='Temperatura Real', linewidth=2)
ax1.plot(t, medicao_sensor, 'r.', alpha=0.3, label='Medições do Sensor')
ax1.plot(t, temp_kalman, 'g-', linewidth=2, label='Estimativa Kalman')
ax1.set_ylabel('Temperatura (°C)')
ax1.set_title('Filtro de Kalman - Estimativa da Temperatura')
ax1.legend()
ax1.grid(True)

# Grafico do erro  
erro = temp_real - temp_kalman
rmse = np.sqrt(np.mean(erro**2))
ax2.plot(t, erro, 'm-', linewidth=1.5, label='Erro (Real - Estimado)')
ax2.axhline(0, color='k', linestyle='--')
ax2.set_xlabel('Tempo (s)')
ax2.set_ylabel('Erro (°C)')
ax2.set_title(f'Erro de Estimativa (RMSE = {rmse:.3f} °C)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()