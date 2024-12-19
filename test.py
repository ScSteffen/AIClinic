
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from spherical_harmonics_pn import SphericalHarmonicsPN


def main():
    """
    Test case demonstrating the PN solver for a simple radiation transport problem.
    """
    # Parameters
    N = 3  # Order of PN approximation
    nx = ny = nz = 20  # Grid points in each direction
    Lx = Ly = Lz = 1.0  # Domain size
    dt = 0.001  # Time step
    num_steps = 100  # Number of time steps
    sigma_t = 1.0  # Total cross section

    # Create grid
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    z = np.linspace(0, Lz, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Initialize density field (example: Gaussian bump)
    rho = np.ones((nx, ny, nz))  # Background density
    # Add a denser region in the center
    rho += 2.0 * np.exp(-((X - Lx / 2) ** 2 + (Y - Ly / 2) ** 2 + (Z - Lz / 2) ** 2) / 0.1 ** 2)

    # Initialize solver
    print("Initializing PN solver...")
    solver = SphericalHarmonicsPN(N)

    # Solve transport equation
    print("Solving transport equation...")
    scalar_flux = solver.solve_transport(rho, sigma_t, dt, num_steps)

    # Visualize results
    plot_results(X, Y, Z, rho, scalar_flux)


def plot_results(X, Y, Z, rho, scalar_flux):
    """
    Create visualizations of the density field and solution.
    """
    # Create slices at the middle of the domain
    mid_x = X.shape[0] // 2
    mid_y = Y.shape[1] // 2
    mid_z = Z.shape[2] // 2

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # Plot density field
    im1 = ax1.pcolormesh(Y[mid_x, :, :], Z[mid_x, :, :],
                         rho[mid_x, :, :], cmap='viridis')
    ax1.set_title('Density Field (x-slice)')
    ax1.set_xlabel('Y')
    ax1.set_ylabel('Z')
    plt.colorbar(im1, ax=ax1)

    # Plot scalar flux
    im2 = ax2.pcolormesh(Y[mid_x, :, :], Z[mid_x, :, :],
                         scalar_flux[mid_x, :, :], cmap='plasma')
    ax2.set_title('Scalar Flux (x-slice)')
    ax2.set_xlabel('Y')
    ax2.set_ylabel('Z')
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.show()

    # Print some statistics
    print("\nSolution Statistics:")
    print(f"Maximum scalar flux: {np.max(scalar_flux):.6f}")
    print(f"Minimum scalar flux: {np.min(scalar_flux):.6f}")
    print(f"Mean scalar flux: {np.mean(scalar_flux):.6f}")
    print(f"Standard deviation: {np.std(scalar_flux):.6f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise