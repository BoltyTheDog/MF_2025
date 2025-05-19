import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import griddata
from tqdm import tqdm 
import os

plt.style.use("seaborn-v0_8-muted")

def load_scalar_field(filepath):
    data = np.genfromtxt(filepath)
    return data[:, 0], data[:, 1], data[:, 2]

def load_vector_field(filepath):
    data = np.genfromtxt(filepath)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]

def create_grid(x, y, z, resolution=300):
    xi = np.linspace(x.min(), x.max(), resolution)
    yi = np.linspace(y.min(), y.max(), resolution)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    return xi, yi, zi

def load_mask(data_dir, resolution=300):
    x, y, m = load_scalar_field(os.path.join(data_dir, 'mask.data'))
    xi, yi, mi = create_grid(x, y, m, resolution)
    return xi, yi, mi

def overlay_mask(ax, mask_xi, mask_yi, mask_zi):
    ax.contourf(mask_xi, mask_yi, mask_zi < 0.5, levels=[0.5, 1],
                colors='gray', alpha=0.7, zorder=10)

def plot_scalar_field(xi, yi, zi, mask_xi, mask_yi, mask_zi, title, cmap, label, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    cf = ax.contourf(xi, yi, zi, levels=100, cmap=cmap, zorder=1)
    overlay_mask(ax, mask_xi, mask_yi, mask_zi)
    cbar = plt.colorbar(cf, ax=ax, label=label)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)

def plot_velocity_field(x, y, u, v, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.quiver(x, y, u, v, angles='xy', scale=40, width=0.002, headwidth=3, color='black', alpha=0.8)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)

def plot_streamlines(x, y, u, v, mask_xi, mask_yi, mask_zi, title, filename):
    xi = np.linspace(x.min(), x.max(), 300)
    yi = np.linspace(y.min(), y.max(), 300)
    xi, yi = np.meshgrid(xi, yi)
    ui = griddata((x, y), u, (xi, yi), method='cubic')
    vi = griddata((x, y), v, (xi, yi), method='cubic')

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw streamlines first
    ax.streamplot(xi, yi, ui, vi, density=1.2, linewidth=1, arrowsize=1.5, color='royalblue', zorder=1)
    
    # Then mask solid object on top
    ax.contourf(mask_xi, mask_yi, mask_zi < 0.5, levels=[0.5, 1],
                colors='gray', alpha=1.0, zorder=10)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def create_lagrangian_animation(x, y, u, v, mask_xi, mask_yi, mask_zi,
                                 filename, num_particles=150, steps=100):

    print("Initializing Lagrangian particle animation...")

    # Create velocity interpolation
    xi = np.linspace(x.min(), x.max(), 300)
    yi = np.linspace(y.min(), y.max(), 300)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    ui = griddata((x, y), u, (xi_grid, yi_grid), method='cubic')
    vi = griddata((x, y), v, (xi_grid, yi_grid), method='cubic')

    # All particles start at x = left inlet, evenly spaced along vertical axis
    inlet_x = x.min() + 0.01
    y_span = yi_grid[:, 0]
    y_valid = y_span[(mask_zi[:, 0] > 0.5)]
    chosen_y = np.linspace(y_valid.min(), y_valid.max(), num_particles)
    px = np.full_like(chosen_y, inlet_x)
    py = chosen_y

    # Save particle history for trails
    history = np.zeros((steps, len(px), 2))
    history[0, :, 0] = px
    history[0, :, 1] = py

    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_title("Lagrangian Particle Paths", fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis('equal')
    overlay_mask(ax, mask_xi, mask_yi, mask_zi)
    scat = ax.scatter(px, py, s=10, color='red', zorder=11)

    def animate(i):
        nonlocal px, py
        u_interp = griddata((xi_grid.flatten(), yi_grid.flatten()), ui.flatten(), (px, py), method='linear', fill_value=0)
        v_interp = griddata((xi_grid.flatten(), yi_grid.flatten()), vi.flatten(), (px, py), method='linear', fill_value=0)
        px += u_interp * 0.15
        py += v_interp * 0.15
        history[i, :, 0] = px
        history[i, :, 1] = py

        # Properly clear previous drawings
        for coll in ax.collections[:]:
            coll.remove()

        overlay_mask(ax, mask_xi, mask_yi, mask_zi)
        ax.scatter(px, py, s=10, color='red', zorder=11)

        # Optional trail effect
        for j in range(max(i - 15, 0), i):
            ax.scatter(history[j, :, 0], history[j, :, 1], s=2, color='black', alpha=0.05, zorder=10)

        return scat,


    print("Rendering frames...")

    ani = animation.FuncAnimation(fig, animate, frames=tqdm(range(steps)), interval=30, blit=False)
    ani.save(filename, writer='pillow', fps=30)
    plt.close(fig)

    print(f"Lagrangian animation saved to: {filename}")

def main():
    data_dir = 'data'
    visuals_dir = 'visuals'
    os.makedirs(visuals_dir, exist_ok=True)

    # Load mask
    mask_xi, mask_yi, mask_zi = load_mask(data_dir)

    # Scalar fields
    x, y, vel_mag = load_scalar_field(os.path.join(data_dir, 'velocity_magnitude.data'))
    xi, yi, zi = create_grid(x, y, vel_mag)
    plot_scalar_field(xi, yi, zi, mask_xi, mask_yi, mask_zi,
                      'Velocity Magnitude', cmap='viridis', label='|V|',
                      filename=os.path.join(visuals_dir, 'velocity_magnitude.png'))

    x, y, pressure = load_scalar_field(os.path.join(data_dir, 'pressure.data'))
    xi, yi, zi = create_grid(x, y, pressure)
    plot_scalar_field(xi, yi, zi, mask_xi, mask_yi, mask_zi,
                      'Pressure Field', cmap='coolwarm', label='P',
                      filename=os.path.join(visuals_dir, 'pressure_field.png'))

    x, y, psi = load_scalar_field(os.path.join(data_dir, 'stream_function.data'))
    xi, yi, zi = create_grid(x, y, psi)
    plot_scalar_field(xi, yi, zi, mask_xi, mask_yi, mask_zi,
                      'Stream Function (ψ)', cmap='plasma', label='ψ',
                      filename=os.path.join(visuals_dir, 'stream_function.png'))

    # Vector fields
    x, y, u, v = load_vector_field(os.path.join(data_dir, 'velocity_field.data'))
    plot_velocity_field(x, y, u, v, title='Velocity Field',
                        filename=os.path.join(visuals_dir, 'velocity_quiver.png'))

    plot_streamlines(x, y, u, v, mask_xi, mask_yi, mask_zi,
                     title='Streamlines', filename=os.path.join(visuals_dir, 'streamlines.png'))

    # Lagrangian Particle Animation
    create_lagrangian_animation(x, y, u, v, mask_xi, mask_yi, mask_zi,
                                 filename=os.path.join(visuals_dir, 'lagrangian_particles.gif'))

    print("All visualizations and animation are saved in the 'visuals/' folder.")

if __name__ == '__main__':
    main()
