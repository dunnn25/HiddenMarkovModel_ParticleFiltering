import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 1000  # Number of particles
T = 50    # Number of time steps
v = 0.1   # Robot speed
landmark = np.array([5, 5])  # Landmark position
sigma_w = 0.1  # Process noise std
sigma_v = 0.2  # Observation noise std

# Nonlinear state transition function
def state_transition(x, v, sigma_w):
    theta = np.random.normal(0, 0.1)  # Random heading
    x_new = x + v * np.array([np.cos(theta), np.sin(theta)])
    x_new += np.random.normal(0, sigma_w, 2)
    return x_new

# Nonlinear observation function
def observation(x, landmark, sigma_v):
    distance = np.sqrt(np.sum((x - landmark)**2))
    return distance + np.random.normal(0, sigma_v)

# Likelihood function
def likelihood(z, x, landmark, sigma_v):
    expected_z = np.sqrt(np.sum((x - landmark)**2))
    return np.exp(-0.5 * (z - expected_z)**2 / sigma_v**2)

# Function to calculate MSE and RMSE
def calculate_metrics(true_states, estimated_states):
    # Remove the first state from true_states to match the length of estimated_states
    true_states = true_states[1:]  # true_states has T+1 elements, estimated_states has T elements
    
    # Calculate squared errors for each time step
    squared_errors = np.sum((true_states - estimated_states)**2, axis=1)  # (x_t - \hat{x}_t)^2 + (y_t - \hat{y}_t)^2
    
    # Calculate MSE
    mse = np.mean(squared_errors)
    
    # Calculate RMSE
    rmse = np.sqrt(mse)
    
    return mse, rmse

# Particle Filter
def particle_filter(N, T, v, landmark, sigma_w, sigma_v):
    # Initialize particles
    particles = np.random.normal(0, 1, (N, 2))
    weights = np.ones(N) / N
    
    # True state and observations
    true_states = [np.array([0.0, 0.0])]
    observations = []
    
    # Generate true trajectory and observations
    for t in range(T):
        true_state = state_transition(true_states[-1], v, sigma_w)
        true_states.append(true_state)
        z = observation(true_state, landmark, sigma_v)
        observations.append(z)
    
    # Store estimated states
    estimated_states = []
    
    for t in range(T):
        # Prediction
        for i in range(N):
            particles[i] = state_transition(particles[i], v, sigma_w)
        
        # Update weights
        z = observations[t]
        for i in range(N):
            weights[i] *= likelihood(z, particles[i], landmark, sigma_v)
        weights /= np.sum(weights)  # Normalize
        
        # Estimate state
        estimated_state = np.sum(particles * weights[:, np.newaxis], axis=0)
        estimated_states.append(estimated_state)
        
        # Resampling
        N_eff = 1 / np.sum(weights**2)
        if N_eff < 0.5 * N:
            indices = np.random.choice(N, N, p=weights)
            particles = particles[indices]
            weights = np.ones(N) / N
    
    return np.array(true_states), np.array(estimated_states), particles

# Run the filter
true_states, estimated_states, final_particles = particle_filter(N, T, v, landmark, sigma_w, sigma_v)

# Calculate MSE and RMSE
mse, rmse = calculate_metrics(true_states, estimated_states)
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# Plotting
plt.figure(figsize=(10, 8))
plt.plot(true_states[:, 0], true_states[:, 1], 'b-', label='True trajectory')
plt.plot(estimated_states[:, 0], estimated_states[:, 1], 'r--', label='Estimated trajectory')
plt.scatter(final_particles[:, 0], final_particles[:, 1], s=10, alpha=0.5, label='Particles')
plt.scatter(landmark[0], landmark[1], c='g', marker='x', s=200, label='Landmark')
plt.legend()
plt.title('Nonlinear HMM with Particle Filter')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.savefig('trajectory_with_metrics.png')
plt.show()