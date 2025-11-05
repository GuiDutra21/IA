import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Initializing variables
ti = 0;
tf = 1000; #1000 original, usar 200, 500 e 1000
delta_t = 1;
t = np.linspace(ti, tf, int(tf/delta_t))
I = np.identity(2)

# Initial Conditions
v = 5; # m/s
s0 = 0;
v0 = 0;
##t = 1  # Difference in time

# noise 
sigma_est = 0.15; # 0.05 é 26 dB SNR / 0.1 é 20 dB SNR / 0.5 é 6 dB SNR
sigma_obs = 5.0; 

# Real model
S = s0 + v*t;
S_obs = S+sigma_obs*np.random.randn(t.size)

A = np.array([[1, delta_t],[0, 1]])
H = np.zeros([1,2])
H[0,0] = 1
Q = (sigma_est**2)*I
R = (sigma_obs**2)
P0 = (sigma_est**2)*I
x_hat = np.zeros([2,t.size])

# First step
x_hat[:,0] = np.array([S_obs[0],v0])
x_hat[:,1] = np.array([S_obs[1],(S_obs[1]-S_obs[0])/delta_t])
P=A @ P0 @ A.T + Q

# Step 2 to end


for i in range(2, t.size):
    
    # Prediction
    x_hat[:,i] = A @ x_hat[:,i-1] 
    P=A @ P @ A.T + Q
    
    # Kalman Gain
    K = P @ H.T @ inv(H @ P @ H.T + R)
    
    # Actualization
    x_hat[:,i] = x_hat[:,i] + K @ (S_obs[i] - H @ x_hat[:,i])
    P = P - K @ H @ P 
    

# Position graphic.
plt.plot(t, S, 'b-')
plt.plot(t, S_obs, 'r.')
plt.plot(t,x_hat[0,:],'g--')
plt.title('Line model S = %s + %s t'%(s0, v), fontsize=20, fontweight='bold')
plt.ylabel('Position (m)', fontsize=20)
plt.xlabel('time (s)', fontsize=20)
plt.show()

v_r = np.ones(t.size)*v
# velocity error
plt.figure()
plt.plot(t, v_r, 'b-')
plt.plot(t, x_hat[1,:], 'r-')
plt.title('Line model S = %s + %s t'%(s0, v), fontsize=20, fontweight='bold')
plt.ylabel('Velocity (m/s)', fontsize=20)
plt.xlabel('time (s)', fontsize=20)
plt.show()