import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Initializing variables
ti = 0;
tf = 1000;
delta_t = 1;
t = np.linspace(ti, tf, int(tf/delta_t))
I = np.identity(4)

# Initial Conditions
vx = 5; # m/s - velocidade em X
vy = 2; # m/s - velocidade em Y
s0x = 0; # Posicoes iniciais
s0y = 0;
v0x = 0;
v0y = 0;
##t = 1  # Difference in time

# noise 
sigma_est_x = 0.15; sigma_est_y = 0.15; # ruído do processo
sigma_obs_x = 0.25; sigma_obs_y = 0.25; # ruído das observações

# Real model
S = np.zeros([2,t.size])
S[0,:] = s0x  + vx*t; # Posição real em X
S[1,:] = s0y  + vy*t; # Posição real em Y

S_obs = np.zeros([2,t.size])
S_obs[0,:] = S[0,:]+sigma_obs_x*np.random.randn(t.size)  # X com ruído
S_obs[1,:] = S[1,:]+sigma_obs_y*np.random.randn(t.size)  # Y com ruído

A = np.array([[1, 0, delta_t, 0],[0, 1, 0, delta_t],[0, 0, 1, 0],[0, 0, 0, 1]])

H = np.zeros([2,4])
H[0,0] = 1
H[1,1] = 1

Q = [10, 10, 25, 25] # Ruído do processo
Q = np.diag(Q) 

R = [sigma_obs_x, sigma_obs_y] # Ruído das observações
R = np.diag(R)

P0 = np.array([[sigma_obs_x, 0, 0, 0],[0, sigma_obs_y, 0, 0],[0, 0, 10**4, 0],[0, 0, 0, 10**4]])
x_hat = np.zeros([4,t.size])

# First step
x_hat[:,0] = np.array([S_obs[0,0], S_obs[1,0], v0x, v0y])
x_hat[:,1] = np.array([S_obs[0,1], S_obs[1,1],(S_obs[0,1]-S_obs[0,0])/delta_t, (S_obs[1,1]-S_obs[1,0])/delta_t])

P=A @ P0 @ A.T + Q

# Step 2 to end

for i in range(2, t.size):
    x_hat[:,i] = A @ x_hat[:,i-1] # Prediction
    P=A @ P @ A.T + Q  # Prediction
    K = P @ H.T @ inv(H @ P @ H.T + R) # Gain
    x_hat[:,i] = x_hat[:,i] + K @ (S_obs[:,i] - H @ x_hat[:,i]) # Actualization
    P = P - K @ H @ P # Actualization
    

# Position graphic x.
plt.plot(t, S[0,:], 'b-')
plt.plot(t, S_obs[0,:], 'r.')
plt.plot(t,x_hat[0,:],'g--')
plt.title('Line model Sx = %s + %s t and Sy = %s + %s t'%(s0x, vx, s0y, vy), fontsize=20, fontweight='bold')
plt.ylabel('Position (m)', fontsize=20)
plt.xlabel('time (s)', fontsize=20)

# Position graphic y.
plt.plot(t, S[1,:], 'b-')
plt.plot(t, S_obs[1,:], 'r.')
plt.plot(t,x_hat[1,:],'g--')


vx_r = np.ones(t.size)*vx
# velocity error
plt.figure()
plt.plot(t, vx_r, 'b-')
plt.plot(t, x_hat[2,:], 'r-')
plt.title('Line model Sx = %s + %s t and Sy = %s + %s t'%(s0x, vx, s0y, vy), fontsize=20, fontweight='bold')
plt.ylabel('Velocity (m/s)', fontsize=20)
plt.xlabel('time (s)', fontsize=20)

vy_r = np.ones(t.size)*vy
# velocity error
plt.plot(t, vy_r, 'b-')
plt.plot(t, x_hat[3,:], 'r-')

plt.tight_layout()
plt.show()