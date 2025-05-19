import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters
U = 1.0        # Free stream velocity
a = 1.0        # Cylinder radius  
Gamma = 2 * np.pi  # Circulation
grid_size = 200    # Computational grid size (reduced for faster convergence)
x_range = 5.0      # Domain range [-x_range, x_range]
rho = 1.0          # Fluid density
max_iter = 5000    # Maximum iterations for SOR method
tol = 1e-6         # Convergence tolerance
omega = 1.8        # SOR relaxation parameter

# Create grid
x = np.linspace(-x_range, x_range, grid_size)
y = np.linspace(-x_range, x_range, grid_size)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)

# Distance from origin and angle in polar coordinates
R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)

# Initialize stream function (psi) and potential (phi) fields
psi = np.zeros((grid_size, grid_size))
phi = np.zeros((grid_size, grid_size))

# Create mask for points inside cylinder (True for inside)
mask_inside = R < a

# Initialize boundary conditions for psi based on analytical solution
# This helps accelerate convergence by providing a good initial guess
# For far-field boundary conditions at the domain edges
psi_analytical = U * (R - a**2 / R) * np.sin(Theta) - (Gamma / (2 * np.pi)) * np.log(R)

# Apply boundary conditions to psi
# 1. Far field boundaries (domain edges)
psi[0, :] = psi_analytical[0, :]          # Bottom edge
psi[-1, :] = psi_analytical[-1, :]        # Top edge
psi[:, 0] = psi_analytical[:, 0]          # Left edge
psi[:, -1] = psi_analytical[:, -1]        # Right edge

# 2. Cylinder boundary - for points just outside cylinder
# We'll use the analytical solution to set these values
mask_boundary = np.zeros_like(mask_inside, dtype=bool)
for i in range(1, grid_size-1):
    for j in range(1, grid_size-1):
        if not mask_inside[i, j] and np.any(mask_inside[i-1:i+2, j-1:j+2]):
            mask_boundary[i, j] = True

psi[mask_boundary] = psi_analytical[mask_boundary]

# Successive Over-Relaxation (SOR) method to solve Laplace's equation for stream function
# ∇²ψ = 0 (outside the cylinder)
def solve_laplace_sor(psi, mask_inside, omega, max_iter, tol):
    iter_count = 0
    error = 1.0
    
    while error > tol and iter_count < max_iter:
        psi_old = psi.copy()
        
        for i in range(1, grid_size-1):
            for j in range(1, grid_size-1):
                if not mask_inside[i, j] and not mask_boundary[i, j]:
                    # Finite difference discretization of Laplace equation
                    # (ψ_{i+1,j} + ψ_{i-1,j} + ψ_{i,j+1} + ψ_{i,j-1} - 4ψ_{i,j}) / (dx²) = 0
                    psi_new = 0.25 * (psi[i+1, j] + psi[i-1, j] + psi[i, j+1] + psi[i, j-1])
                    psi[i, j] = (1 - omega) * psi[i, j] + omega * psi_new
        
        # Calculate error (L2 norm of the difference)
        diff = psi - psi_old
        error = np.sqrt(np.sum(diff[~mask_inside]**2) / np.sum(~mask_inside))
        iter_count += 1
        
        if iter_count % 100 == 0:
            print(f"Iteration: {iter_count}, Error: {error:.6e}")
    
    return psi, iter_count, error

# Solve the system
print("Solving stream function using SOR method...")
psi, iterations, final_error = solve_laplace_sor(psi, mask_inside, omega, max_iter, tol)
print(f"Solution converged in {iterations} iterations with error {final_error:.6e}")

# Set inside cylinder values to NaN for visualization
psi[mask_inside] = np.nan

# Calculate velocity components by numerical differentiation of stream function
# u = ∂ψ/∂y, v = -∂ψ/∂x
u = np.zeros_like(psi)
v = np.zeros_like(psi)

for i in range(1, grid_size-1):
    for j in range(1, grid_size-1):
        if not mask_inside[i, j]:
            u[i, j] = (psi[i+1, j] - psi[i-1, j]) / (2 * dy)
            v[i, j] = -(psi[i, j+1] - psi[i, j-1]) / (2 * dx)

# Set inside cylinder values to NaN
u[mask_inside] = np.nan
v[mask_inside] = np.nan

# Calculate velocity magnitude
V_mag = np.sqrt(u**2 + v**2)

# Calculate pressure field using Bernoulli equation
P_inf = 0.0
P = P_inf + 0.5 * rho * (U**2 - V_mag**2)
P[mask_inside] = np.nan

# Create output directory
output_dir = "output_fd"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Plot stream function
plt.figure(figsize=(6, 6))
plt.contour(X, Y, psi, levels=50, colors='blue')
circle = plt.Circle((0, 0), a, color='black', fill=True)
plt.gca().add_patch(circle)
plt.title("Stream Function (Finite Differences)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.savefig(os.path.join(output_dir, 'stream_function_fd.png'))
plt.close()

# Plot velocity magnitude
plt.figure(figsize=(6, 6))
plt.contourf(X, Y, V_mag, levels=50, cmap='viridis')
circle = plt.Circle((0, 0), a, color='black', fill=True)
plt.gca().add_patch(circle)
plt.colorbar(label="Velocity Magnitude")
plt.title("Velocity Magnitude (Finite Differences)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.savefig(os.path.join(output_dir, 'velocity_magnitude_fd.png'))
plt.close()

# Plot pressure field
plt.figure(figsize=(6, 6))
plt.contourf(X, Y, P, levels=50, cmap='RdBu_r')
circle = plt.Circle((0, 0), a, color='black', fill=True)
plt.gca().add_patch(circle)
plt.colorbar(label="Pressure Field")
plt.title("Pressure Field (Finite Differences)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.savefig(os.path.join(output_dir, 'pressure_field_fd.png'))
plt.close()

# Plot velocity field
plt.figure(figsize=(6, 6))
# Use a coarser grid for the quiver plot (subsampling)
skip = 8  # Increased for better visualization
plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
           u[::skip, ::skip], v[::skip, ::skip], 
           color='green', scale=50)
circle = plt.Circle((0, 0), a, color='black', fill=True)
plt.gca().add_patch(circle)
plt.title("Velocity Field (Finite Differences)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.savefig(os.path.join(output_dir, 'velocity_field_fd.png'))
plt.close()

# Compare with analytical solution
# Calculate analytical solution for comparison
psi_analytical[mask_inside] = np.nan

# Plot comparison of stream functions
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.contour(X, Y, psi, levels=20, colors='blue')
circle = plt.Circle((0, 0), a, color='black', fill=True)
plt.gca().add_patch(circle)
plt.title("FD Stream Function")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.contour(X, Y, psi_analytical, levels=20, colors='red')
circle = plt.Circle((0, 0), a, color='black', fill=True)
plt.gca().add_patch(circle)
plt.title("Analytical Stream Function")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comparison.png'))
plt.close()

print("Finite difference solution completed. Results saved in 'output_fd' directory.")