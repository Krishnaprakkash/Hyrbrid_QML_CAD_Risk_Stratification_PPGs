import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pennylane as qml

# Setup quantum device
dev = qml.device('default.qubit', wires=4, shots=None)

# Define variational quantum circuit
@qml.qnode(dev)
def circuit(features, params):
    # Amplitude encoding
    qml.AmplitudeEmbedding(features=features, wires=range(4), normalize=True)
    
    # Variational ansatz
    for i in range(4):
        qml.RY(params[i], wires=i)
        qml.RZ(params[4+i], wires=i)
    
    for i in range(3):
        qml.CNOT(wires=[i, i+1])
    
    return qml.expval(qml.Z(0) @ qml.Z(1))

# Function to compute gradient via parameter-shift rule
def compute_gradient(features, params):
    shift = np.pi / 2
    gradients = np.zeros_like(params)
    
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        
        params_plus[i] += shift
        params_minus[i] -= shift
        
        gradients[i] = (circuit(features, params_plus) - circuit(features, params_minus)) / (2 * np.sin(shift))
    
    return gradients

# Training simulation
np.random.seed(42)
n_params = 8
n_iterations = 100

# Dummy 16-dimensional PPG feature vector
dummy_features = np.random.randn(16)
dummy_features = dummy_features / np.linalg.norm(dummy_features)

# Initialize parameters
params = np.random.randn(n_params) * 0.1

# Storage for visualization
gradient_history = []
param_history = []
magnitude_history = []

# Adam optimizer
m = np.zeros(n_params)  # First moment
v = np.zeros(n_params)  # Second moment
beta1, beta2 = 0.9, 0.999
alpha = 0.01  # Learning rate

for iteration in range(n_iterations):
    # Compute gradients
    grads = compute_gradient(dummy_features, params)
    gradient_history.append(grads.copy())
    magnitude_history.append(np.linalg.norm(grads))
    
    # Adam update
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * (grads ** 2)
    
    m_hat = m / (1 - beta1 ** (iteration + 1))
    v_hat = v / (1 - beta2 ** (iteration + 1))
    
    params = params - alpha * m_hat / (np.sqrt(v_hat) + 1e-8)
    param_history.append(params.copy())

# Visualization
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

# 1. Gradient magnitude over time
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(magnitude_history, linewidth=2, color='#1f77b4')
ax1.fill_between(range(len(magnitude_history)), magnitude_history, alpha=0.3, color='#1f77b4')
ax1.set_xlabel('Training Iteration', fontsize=12)
ax1.set_ylabel('Gradient Magnitude ||∇L||', fontsize=12)
ax1.set_title('Gradient Magnitude Evolution During Training', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# 2. Heatmap of gradient values over iterations
ax2 = fig.add_subplot(gs[1, 0])
gradient_array = np.array(gradient_history).T
im = ax2.imshow(gradient_array, aspect='auto', cmap='RdBu', interpolation='bilinear')
ax2.set_xlabel('Training Iteration', fontsize=11)
ax2.set_ylabel('Parameter Index', fontsize=11)
ax2.set_title('Gradient Heatmap (Parameter-wise Evolution)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax2, label='Gradient Value')

# 3. Parameter trajectories
ax3 = fig.add_subplot(gs[1, 1])
param_array = np.array(param_history)
for i in range(n_params):
    ax3.plot(param_array[:, i], label=f'θ_{i}', alpha=0.7, linewidth=1.5)
ax3.set_xlabel('Training Iteration', fontsize=11)
ax3.set_ylabel('Parameter Value (radians)', fontsize=11)
ax3.set_title('Parameter Trajectories During Training', fontsize=12, fontweight='bold')
ax3.legend(loc='best', fontsize=8)
ax3.grid(True, alpha=0.3)

# 4. Per-parameter gradient magnitudes (boxplot across iterations)
ax4 = fig.add_subplot(gs[2, 0])
param_grad_mags = [np.abs(gradient_array[i, :]) for i in range(n_params)]
bp = ax4.boxplot(param_grad_mags, labels=[f'θ_{i}' for i in range(n_params)], patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('#ff7f0e')
ax4.set_ylabel('|Gradient| Magnitude', fontsize=11)
ax4.set_title('Gradient Magnitude Distribution per Parameter', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# 5. Gradient field scatter (final iteration snapshot)
ax5 = fig.add_subplot(gs[2, 1])
final_grads = gradient_history[-1]
colors = np.abs(final_grads)
scatter = ax5.scatter(range(n_params), final_grads, c=colors, s=150, cmap='viridis', 
                      edgecolors='black', linewidth=1.5, alpha=0.8)
ax5.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Gradient')
ax5.set_xlabel('Parameter Index', fontsize=11)
ax5.set_ylabel('Gradient Value (Final Iteration)', fontsize=11)
ax5.set_title('Final Gradient Snapshot (Iteration 100)', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax5, label='|Gradient|')

plt.suptitle('Quantum Circuit Gradient Dynamics: Adam Optimization Visualization', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('quantum_gradient_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✓ Visualization saved as 'quantum_gradient_visualization.png'")
print(f"Initial gradient magnitude: {magnitude_history[0]:.6f}")
print(f"Final gradient magnitude: {magnitude_history[-1]:.6f}")
print(f"Gradient reduction: {(1 - magnitude_history[-1]/magnitude_history[0])*100:.2f}%")
