import numpy as np
from scipy.special import sph_harm
from scipy.sparse import csr_matrix
from typing import Tuple, List


class SphericalHarmonicsPN:
    def __init__(self, order: int):
        """Initialize PN solver with given order N."""
        self.N = order
        self.n_moments = (order + 1) ** 2
        self._init_basis_indices()
        self._build_operators()

    def _init_basis_indices(self):
        """Initialize indices for spherical harmonics basis."""
        self.l_indices = []
        self.m_indices = []
        for l in range(self.N + 1):
            for m in range(-l, l + 1):
                self.l_indices.append(l)
                self.m_indices.append(m)

    def _build_operators(self):
        """Build streaming and collision operators."""
        # Generate quadrature points on unit sphere
        n_theta, n_phi = 2 * self.N + 1, 2 * self.N + 1
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi)
        theta_mesh, phi_mesh = np.meshgrid(theta, phi)
        # Compute spherical harmonics at quadrature points
        Y = np.zeros((self.n_moments, theta_mesh.size), dtype=complex)
        for idx, (l, m) in enumerate(zip(self.l_indices, self.m_indices)):
            Y[idx] = sph_harm(m, l, phi_mesh.ravel(), theta_mesh.ravel())
        # Build streaming matrices
        self.Ax = self._build_streaming_matrix(Y, theta_mesh, phi_mesh, 'x')
        self.Ay = self._build_streaming_matrix(Y, theta_mesh, phi_mesh, 'y')
        self.Az = self._build_streaming_matrix(Y, theta_mesh, phi_mesh, 'z')

    def _build_streaming_matrix(self, Y: np.ndarray, theta_mesh: np.ndarray,
                                phi_mesh: np.ndarray, direction: str) -> csr_matrix:
        """Build streaming matrix for given direction."""
        # Get direction weights
        if direction == 'x':
            weights = np.sin(theta_mesh.ravel()) * np.cos(phi_mesh.ravel())
        elif direction == 'y':
            weights = np.sin(theta_mesh.ravel()) * np.sin(phi_mesh.ravel())
        else:  # z direction
            weights = np.cos(theta_mesh.ravel())
        # Compute quadrature weights
        dtheta = np.pi / (theta_mesh.shape[0] - 1)
        dphi = 2 * np.pi / (phi_mesh.shape[1] - 1)
        quad_weights = np.sin(theta_mesh.ravel()) * dtheta * dphi
        # Compute matrix elements
        A = np.zeros((self.n_moments, self.n_moments))
        for i in range(self.n_moments):
            for j in range(self.n_moments):
                integrand = Y[i].conj() * weights * Y[j] * quad_weights
                A[i, j] = np.sum(integrand).real
        return csr_matrix(A)

    def solve_transport(self, rho: np.ndarray, sigma_t: float,
                        dt: float, num_steps: int) -> np.ndarray:
        """
        Solve the PN transport equations.
        Args:
            rho: Tissue density field (flattened)
            sigma_t: Total cross section
            dt: Time step size
            num_steps: Number of time steps
        Returns:
            Solution moments at final time
        """
        # Initialize solution vector
        u = np.zeros(self.n_moments)
        u[0] = 1.0  # Isotropic initial condition
        # Ensure rho is flattened
        rho_flat = rho.ravel()
        # Time stepping
        for step in range(num_steps):
            # Compute spatial derivatives
            dx_u = self.Ax.dot(u)
            dy_u = self.Ay.dot(u)
            dz_u = self.Az.dot(u)
            # Update solution
            flux_term = (dx_u + dy_u + dz_u) / rho_flat
            collision_term = sigma_t * u
            u = u - dt * (flux_term + collision_term)
        return u.reshape(rho.shape)

