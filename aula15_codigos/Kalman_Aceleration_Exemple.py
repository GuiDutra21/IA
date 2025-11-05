import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Parâmetros iniciais
ti = 0
tf = 100     # quantidade de passos (duração do experimento)
deltat = 1
t = np.linspace(ti, tf, int(tf/deltat))

# Matriz de transição de estado (posição e velocidade)
A = np.array([[1, deltat],
              [0, 1]])

# Matriz de controle B para aceleração constante
B = np.array([[0.5 * deltat**2],
              [deltat]])

# Matriz de observação (só observando posição)
H = np.array([[1, 0]])

# Níveis de ruído ajustados
sigmaest = 1.0     # ruído do processo - mais alto para gerar dispersão inicial
sigmaobs = 50.0      # ruído da observação - nível significativo

Q = sigmaest**2 * np.eye(2)
R = sigmaobs**2

# Estado inicial estimado e matriz de covariância inicial
xhat = np.zeros((2, t.size))
P = np.eye(2) * 10

# Estado inicial verdadeiro (posição e velocidade)
s0 = 0
v0 = 0

# Simulando entrada de aceleração
aceleracao = 2.0
u = np.ones(t.size) * aceleracao

# Simulação da trajetória real
S = np.zeros(t.size)
v = np.zeros(t.size)
S[0] = s0
v[0] = v0
for i in range(1, t.size):
    processo_ruido = np.random.multivariate_normal([0, 0], Q)  # ruído do processo aplicado à trajetória real
    v[i] = v[i-1] + aceleracao * deltat + processo_ruido[1]
    S[i] = S[i-1] + v[i-1] * deltat + 0.5 * aceleracao * deltat**2 + processo_ruido[0]

# Observações ruidosas da posição
Sobs = S + sigmaobs * np.random.randn(t.size)

# Estado inicial baseado na observação inicial
xhat[:, 0] = np.array([Sobs[0], 0])

# Filtro de Kalman
for i in range(1, t.size):
    # Predição com entrada de controle
    xhat[:, i] = A @ xhat[:, i-1] + (B * u[i-1]).reshape(2)
    P = A @ P @ A.T + Q

    # Ganho de Kalman
    K = P @ H.T @ inv(H @ P @ H.T + R)

    # Atualização com observação
    innovation = Sobs[i] - H @ xhat[:, i]
    xhat[:, i] = xhat[:, i] + (K.flatten() * innovation)
    P = (np.eye(2) - K @ H) @ P

# Plotando resultado
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.plot(t, S, 'b-', label='Posição real')
plt.plot(t, Sobs, 'r.', label='Observação (posição)')
plt.plot(t, xhat[0, :], 'g--', label='Estimada (posição)')
plt.title('Filtro de Kalman - Posição')
plt.xlabel('Tempo (s)')
plt.ylabel('Posição (m)')
plt.legend()

plt.subplot(1,2,2)
plt.plot(t, v, 'b-', label='Velocidade real')
plt.plot(t, xhat[1, :], 'g--', label='Estimada (velocidade)')
plt.title('Filtro de Kalman - Velocidade')
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade (m/s)')
plt.legend()

plt.tight_layout()
plt.show()
