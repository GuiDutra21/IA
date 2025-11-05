import numpy as np
import matplotlib.pyplot as plt

# 1. Definir constantes para a simulação
dt = 0.1  # Passo de tempo (s)
g = 9.8  # Aceleração gravitacional (m/s^2)
total_time = 30  # Tempo total de simulação (s)
num_steps = int(total_time / dt)

# 2. Simular a trajetória real do foguete
true_altitude = []
true_velocity = []
pos = 0.0
vel = 0.0
for t in range(num_steps):
    if t * dt < 10:
        accel = 50.0 - g
    else:
        accel = -g
    
    pos += vel * dt + 0.5 * accel * dt**2
    vel += accel * dt
    
    true_altitude.append(pos)
    true_velocity.append(vel)

# 3. Simular medições ruidosas do sensor
# AUMENTANDO OS RUÍDOS para maior visibilidade
altimeter_noise_std = 50.0  # Ruído mais alto para o altímetro (m)
accel_noise_std = 2.0  # Ruído mais alto para o acelerômetro (m/s^2)
np.random.seed(42)

altimeter_measurements = np.array([h + np.random.normal(0, altimeter_noise_std) for h in true_altitude])

# 4. Inicializar o filtro de Kalman manualmente
x = np.array([[0.], [0.]])
P = np.array([[1000., 0.], [0., 1000.]])

F = np.array([[1., dt], [0., 1.]])
H = np.array([[1., 0.]])

# Atualizando as matrizes de covariância de ruído com os novos valores
q_var = accel_noise_std**2
Q = np.array([[0.25 * dt**4, 0.5 * dt**3], 
              [0.5 * dt**3, dt**2]]) * q_var

R = np.array([[altimeter_noise_std**2]])

estimated_state = []
for i in range(num_steps):
    # Entrada de controle do acelerômetro ruidoso
    if i * dt < 10:
        control_accel = 50.0 - g + np.random.normal(0, accel_noise_std)
    else:
        control_accel = -g + np.random.normal(0, accel_noise_std)
    u = np.array([[0.5 * control_accel * dt**2], [control_accel * dt]])
    
    # Etapa de PREDICAO
    x_pred = F @ x + u
    P_pred = F @ P @ F.T + Q
    
    # Etapa de ATUALIZACAO
    z = np.array([[altimeter_measurements[i]]])
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x = x_pred + K @ y
    P = (np.eye(2) - K @ H) @ P_pred
    
    estimated_state.append(x.flatten())

estimated_altitude = [state[0] for state in estimated_state]
estimated_velocity = [state[1] for state in estimated_state]

# 6. Plotar os resultados
time_points = np.arange(0, total_time, dt)

plt.figure(figsize=(12, 16))

# Plotar a Altitude
plt.subplot(2, 1, 1)
plt.plot(time_points, true_altitude, label='Altitude Verdadeira', linewidth=2, color='blue')
plt.plot(time_points, altimeter_measurements, label='Medições Ruidosas do Altímetro', alpha=0.6, color='red')
plt.plot(time_points, estimated_altitude, label='Estimativa do Filtro de Kalman', linewidth=2, linestyle='--', color='green')
plt.xlabel('Tempo (s)')
plt.ylabel('Altitude (m)')
plt.title('Filtro de Kalman para Estimativa de Altitude (Ruído Aumentado)')
plt.legend()
plt.grid(True)

# Plotar a Velocidade
plt.subplot(2, 1, 2)
plt.plot(time_points, true_velocity, label='Velocidade Verdadeira', linewidth=2, color='blue')
plt.plot(time_points, estimated_velocity, label='Estimativa do Filtro de Kalman', linewidth=2, linestyle='--', color='green')
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade (m/s)')
plt.title('Filtro de Kalman para Estimativa de Velocidade (Ruído Aumentado)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

