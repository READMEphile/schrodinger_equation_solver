import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os
import time

class SchrodingerSolver:
    def __init__(self, x_min, x_max, n_points, potential_func, mass=1.0, hbar=1.0):
        self.x_min = x_min
        self.x_max = x_max
        self.n_points = n_points
        self.mass = mass
        self.hbar = hbar
        self.potential_func = potential_func
        
        # Set up the grid
        self.x = np.linspace(x_min, x_max, n_points)
        self.dx = (x_max - x_min) / (n_points - 1)
        
        # Compute the potential energy at each grid point
        self.potential = self.potential_func(self.x)
        
        # Initialize eigenstates
        self.eigenvalues = None
        self.eigenvectors = None
        
    def construct_hamiltonian(self):
        # Diagonal elements for potential energy
        potential_diag = sparse.diags(self.potential)
        
        # Construct kinetic energy operator using central difference
        diag_elements = np.ones(self.n_points)
        diag_elements[0] = 0  # Enforce boundary condition
        diag_elements[-1] = 0  # Enforce boundary condition
        
        # Coefficient for the second derivative operator
        coeff = -self.hbar**2 / (2 * self.mass * self.dx**2)
        
        # Create sparse matrix for Laplacian (central difference)
        main_diag = -2 * np.ones(self.n_points)
        off_diag = np.ones(self.n_points-1)
        
        # Construct the Laplacian operator
        laplacian = sparse.diags(
            [off_diag, main_diag, off_diag],
            [-1, 0, 1],
            shape=(self.n_points, self.n_points)
        )
        
        # Kinetic energy term
        kinetic = coeff * laplacian
        
        # Full Hamiltonian
        hamiltonian = kinetic + potential_diag
        
        return hamiltonian.tocsr()  # Convert to CSR format for efficiency
    
    def solve(self, k=10):
        # Construct the Hamiltonian
        hamiltonian = self.construct_hamiltonian()
        
        # Find the k lowest eigenvalues and eigenvectors
        # sigma=0 targets eigenvalues near 0 (lower end of spectrum)
        eigenvalues, eigenvectors = eigsh(hamiltonian, k=k, which='SA')
        
        # Sort by eigenvalue (energies)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Normalize the eigenvectors
        for i in range(k):
            norm = np.sqrt(np.sum(eigenvectors[:, i]**2) * self.dx)
            eigenvectors[:, i] = eigenvectors[:, i] / norm
        
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        
        return eigenvalues, eigenvectors
    
    def probability_density(self, state_idx=0):
        if self.eigenvectors is None:
            raise ValueError("Need to solve the system first using solve()")
        
        wavefunction = self.eigenvectors[:, state_idx]
        return np.abs(wavefunction)**2
    
    def plot_eigenstate(self, state_idx=0, title=None):
        if self.eigenvectors is None:
            raise ValueError("Need to solve the system first using solve()")
        
        wavefunction = self.eigenvectors[:, state_idx]
        energy = self.eigenvalues[state_idx]
        
        plt.figure(figsize=(12, 8))
        
        # Plot potential
        plt.plot(self.x, self.potential, 'k-', label='Potential V(x)')
        
        # Plot energy level
        plt.axhline(y=energy, color='r', linestyle='--', 
                   label=f'Energy = {energy:.4f}')
        
        # Plot wavefunction (scaled for visibility)
        scale_factor = 0.5 * (max(self.potential) - min(self.potential)) / max(abs(wavefunction))
        plt.plot(self.x, energy + scale_factor * wavefunction, 'b-', 
                label='Wavefunction (scaled)')
        
        # Plot probability density (scaled)
        prob_density = self.probability_density(state_idx)
        prob_scale = 0.3 * (max(self.potential) - min(self.potential)) / max(prob_density)
        plt.plot(self.x, energy + prob_scale * prob_density, 'g-', 
                label='Probability density (scaled)')
        
        if title:
            plt.title(title)
        else:
            plt.title(f'Eigenstate {state_idx}, Energy = {energy:.4f}')
            
        plt.xlabel('Position (x)')
        plt.ylabel('Energy')
        plt.legend()
        plt.grid(True)
        
        return plt.gcf()
    
    def plot_all_eigenstates(self, max_states=6):
        if self.eigenvectors is None:
            raise ValueError("Need to solve the system first using solve()")
        
        n_states = min(max_states, self.eigenvectors.shape[1])
        
        plt.figure(figsize=(14, 10))
        
        # Plot potential
        plt.plot(self.x, self.potential, 'k-', linewidth=2, label='Potential V(x)')
        
        # Plot all eigenstates
        for i in range(n_states):
            wavefunction = self.eigenvectors[:, i]
            energy = self.eigenvalues[i]
            
            # Plot energy level
            plt.axhline(y=energy, color=f'C{i}', linestyle='--', alpha=0.7)
            
            # Scale the wavefunction for better visibility
            scale_factor = 0.4 * (max(self.potential) - min(self.potential)) / max(abs(wavefunction))
            
            # Plot wavefunction centered at its energy level
            plt.plot(self.x, energy + scale_factor * wavefunction, color=f'C{i}', 
                    label=f'E{i} = {energy:.4f}')
        
        plt.title('Energy Eigenstates')
        plt.xlabel('Position (x)')
        plt.ylabel('Energy')
        plt.legend()
        plt.grid(True)
        
        return plt.gcf()
    
    def save_results(self, folder='results'):
        if self.eigenvectors is None:
            raise ValueError("Need to solve the system first using solve()")
            
        # Create folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)
        
        # Save eigenvalues
        np.savetxt(f"{folder}/eigenvalues.dat", self.eigenvalues)
        
        # Save eigenvectors (each column is an eigenvector)
        np.savetxt(f"{folder}/eigenvectors.dat", self.eigenvectors)
        
        # Save positions
        np.savetxt(f"{folder}/positions.dat", self.x)
        
        # Save potential
        np.savetxt(f"{folder}/potential.dat", self.potential)
        
        print(f"Results saved to {folder}/")


class SchrodingerSolver2D:
    
    def __init__(self, x_min, x_max, y_min, y_max, nx, ny, potential_func, mass=1.0, hbar=1.0):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.nx, self.ny = nx, ny
        self.mass = mass
        self.hbar = hbar
        self.potential_func = potential_func
        
        # Total number of grid points
        self.n_points = nx * ny
        
        # Set up the grids
        self.x = np.linspace(x_min, x_max, nx)
        self.y = np.linspace(y_min, y_max, ny)
        self.dx = (x_max - x_min) / (nx - 1)
        self.dy = (y_max - y_min) / (ny - 1)
        
        # Create 2D meshgrid
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # Compute the potential energy at each grid point
        self.potential = self.potential_func(self.X, self.Y)
        
        # Initialize eigenstates
        self.eigenvalues = None
        self.eigenvectors = None
    
    def construct_hamiltonian(self):
        nx, ny = self.nx, self.ny
        n = nx * ny
        
        # Constants for the kinetic energy term
        hbar_sq_over_2m = self.hbar**2 / (2 * self.mass)
        kx = hbar_sq_over_2m / (self.dx**2)
        ky = hbar_sq_over_2m / (self.dy**2)
        
        # Diagonal elements (potential energy + kinetic energy diagonal)
        diag = np.zeros(n)
        
        # Off-diagonal elements for x-direction connections
        diag_x_plus = np.zeros(n)
        diag_x_minus = np.zeros(n)
        
        # Off-diagonal elements for y-direction connections
        diag_y_plus = np.zeros(n)
        diag_y_minus = np.zeros(n)
        
        # Fill the diagonals
        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                
                # Potential energy
                diag[idx] = self.potential[i, j]
                
                # Kinetic energy diagonal term
                diag[idx] += 2 * kx + 2 * ky
                
                # Connections in x-direction
                if i > 0:
                    diag_x_minus[idx] = -kx
                if i < nx - 1:
                    diag_x_plus[idx] = -kx
                    
                # Connections in y-direction
                if j > 0:
                    diag_y_minus[idx] = -ky
                if j < ny - 1:
                    diag_y_plus[idx] = -ky
        
        # Offsets for the diagonals
        offsets = [0, -ny, ny, -1, 1]
        diagonals = [diag, diag_x_minus, diag_x_plus, diag_y_minus, diag_y_plus]
        
        # Create the sparse Hamiltonian matrix
        H = sparse.diags(diagonals, offsets, shape=(n, n), format='csr')
        
        return H
    
    def solve(self, k=10):
        # Construct the Hamiltonian
        hamiltonian = self.construct_hamiltonian()
        
        # Find the k lowest eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigsh(hamiltonian, k=k, which='SA')
        
        # Sort by eigenvalue (energies)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store the results
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        
        return eigenvalues, eigenvectors
    
    def get_eigenstate_2d(self, state_idx=0):
        if self.eigenvectors is None:
            raise ValueError("Need to solve the system first using solve()")
        
        # Reshape the eigenvector to 2D grid
        psi = self.eigenvectors[:, state_idx].reshape(self.nx, self.ny)
        
        # Normalize the wavefunction
        norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx * self.dy)
        psi = psi / norm
        
        return psi
    
    def plot_eigenstate_2d(self, state_idx=0, plot_type='contour'):
        """
        Plot a 2D eigenstate.
        
        Args:
            state_idx (int): Index of the eigenstate
            plot_type (str): Type of plot ('contour', 'surface', 'density')
        """
        if self.eigenvectors is None:
            raise ValueError("Need to solve the system first using solve()")
        
        # Get the 2D eigenstate
        psi = self.get_eigenstate_2d(state_idx)
        energy = self.eigenvalues[state_idx]
        
        # Calculate probability density
        prob_density = np.abs(psi)**2
        
        fig = plt.figure(figsize=(16, 5))
        
        if plot_type == 'contour' or plot_type == 'all':
            if plot_type == 'all':
                ax1 = fig.add_subplot(131)
            else:
                ax1 = fig.add_subplot(111)
                
            contour = ax1.contourf(self.X, self.Y, psi.real, cmap='RdBu', levels=50)
            ax1.set_title(f'Wavefunction (Real Part), E = {energy:.4f}')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            plt.colorbar(contour, ax=ax1)
        
        if plot_type == 'surface' or plot_type == 'all':
            if plot_type == 'all':
                ax2 = fig.add_subplot(132, projection='3d')
            else:
                ax2 = fig.add_subplot(111, projection='3d')
                
            surf = ax2.plot_surface(self.X, self.Y, psi.real, cmap='viridis', linewidth=0)
            ax2.set_title(f'Wavefunction 3D View, E = {energy:.4f}')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_zlabel('Wavefunction')
        
        if plot_type == 'density' or plot_type == 'all':
            if plot_type == 'all':
                ax3 = fig.add_subplot(133)
            else:
                ax3 = fig.add_subplot(111)
                
            density = ax3.imshow(prob_density.T, origin='lower', 
                                extent=[self.x_min, self.x_max, self.y_min, self.y_max],
                                cmap='hot', interpolation='bilinear')
            ax3.set_title(f'Probability Density, E = {energy:.4f}')
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')
            plt.colorbar(density, ax=ax3)
        
        plt.tight_layout()
        return fig
    
    def plot_energy_levels(self, n_levels=10):
        """
        Plot the energy levels as a bar chart.
        
        Args:
            n_levels (int): Number of energy levels to plot
        """
        if self.eigenvalues is None:
            raise ValueError("Need to solve the system first using solve()")
        
        num_levels = min(n_levels, len(self.eigenvalues))
        
        plt.figure(figsize=(10, 8))
        
        # Plot energy levels as horizontal lines
        for i in range(num_levels):
            plt.axhline(y=self.eigenvalues[i], color=f'C{i%10}', linestyle='-', linewidth=2)
            plt.text(0.1, self.eigenvalues[i] + 0.05, f'E{i} = {self.eigenvalues[i]:.4f}', 
                    fontsize=12)
        
        plt.title('Energy Levels')
        plt.ylabel('Energy')
        plt.xlim(-1, 1)
        plt.grid(axis='y')
        plt.tight_layout()
        
        return plt.gcf()


# Define potential functions
def infinite_well(x):
    """Infinite square well potential"""
    return np.zeros_like(x)

def harmonic_oscillator(x, k=1.0):
    """Harmonic oscillator potential V(x) = 0.5 * k * x^2"""
    return 0.5 * k * x**2

def double_well(x, a=1.0, b=0.5):
    """Double well potential V(x) = a * (x^2 - b)^2"""
    return a * (x**2 - b)**2

def finite_well(x, v0=10.0, width=1.0):
    """Finite square well potential"""
    potential = np.ones_like(x) * v0
    mask = (x >= -width/2) & (x <= width/2)
    potential[mask] = 0.0
    return potential

def triangular_well(x, slope=1.0):
    """Triangular well potential"""
    return np.maximum(slope * np.abs(x), 0)

def morse_potential(x, D=10.0, a=1.0, x0=0.0):
    """Morse potential for molecular vibration"""
    return D * (1 - np.exp(-a * (x - x0)))**2

def kronig_penney(x, v0=10.0, a=0.5, b=0.5, period=2.0):
    """Kronig-Penney potential (periodic square barriers)"""
    potential = np.zeros_like(x)
    
    # Map all points to [0, period)
    x_mod = np.mod(x, period)
    
    # Add barriers of width 'b' and height 'v0' starting at position 'a'
    mask = (x_mod >= a) & (x_mod < a + b)
    potential[mask] = v0
    
    return potential

# 2D potential functions
def harmonic_oscillator_2d(x, y, kx=1.0, ky=1.0):
    """2D harmonic oscillator potential V(x,y) = 0.5 * (kx * x^2 + ky * y^2)"""
    return 0.5 * (kx * x**2 + ky * y**2)

def infinite_well_2d(x, y):
    """2D infinite square well potential"""
    return np.zeros_like(x)

def coupled_oscillator_2d(x, y, kx=1.0, ky=1.0, kxy=0.1):
    """Coupled harmonic oscillators with coupling term kxy*x*y"""
    return 0.5 * (kx * x**2 + ky * y**2) + kxy * x * y

def crater_potential_2d(x, y, a=1.0, b=0.5, r0=1.0):
    """Radial crater-like potential: V(r) = a * (r - r0)^2 - b"""
    r = np.sqrt(x**2 + y**2)
    return a * (r - r0)**2 - b

def gaussian_well_2d(x, y, depth=10.0, width=1.0):
    """2D Gaussian potential well"""
    r_squared = x**2 + y**2
    return -depth * np.exp(-r_squared / (2 * width**2))


# Example usage
def main():
    print("Quantum Schrödinger Solver")
    print("--------------------------")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # 1D Solvers demonstration
    potentials = {
        "Harmonic Oscillator": lambda x: harmonic_oscillator(x, k=1.0),
        "Double Well": lambda x: double_well(x, a=1.0, b=0.5),
        "Finite Well": lambda x: finite_well(x, v0=20.0, width=2.0),
        "Morse Potential": lambda x: morse_potential(x, D=10.0, a=1.0)
    }
    
    for name, potential_func in potentials.items():
        print(f"\nSolving for {name} potential...")
        solver = SchrodingerSolver(
            x_min=-5.0, 
            x_max=5.0, 
            n_points=500, 
            potential_func=potential_func
        )
        
        start_time = time.time()
        eigenvalues, eigenvectors = solver.solve(k=8)
        end_time = time.time()
        
        print(f"Solution found in {end_time - start_time:.3f} seconds")
        print(f"First 5 energy levels: {eigenvalues[:5]}")
        
        # Plot and save all eigenstates
        fig = solver.plot_all_eigenstates(max_states=6)
        plt.tight_layout()
        plt.savefig(f"results/{name.lower().replace(' ', '_')}_eigenstates.png", dpi=300)
        plt.close(fig)
        
        # Save individual eigenstate plots
        for i in range(min(4, len(eigenvalues))):
            fig = solver.plot_eigenstate(state_idx=i, 
                                        title=f"{name}: State {i}, Energy = {eigenvalues[i]:.4f}")
            plt.tight_layout()
            plt.savefig(f"results/{name.lower().replace(' ', '_')}_state_{i}.png", dpi=300)
            plt.close(fig)
    
    # 2D example
    print("\nSolving 2D harmonic oscillator...")
    solver_2d = SchrodingerSolver2D(
        x_min=-3.0, x_max=3.0, y_min=-3.0, y_max=3.0,
        nx=50, ny=50,
        potential_func=lambda x, y: harmonic_oscillator_2d(x, y, kx=1.0, ky=1.5)
    )
    
    start_time = time.time()
    eigenvalues_2d, _ = solver_2d.solve(k=12)
    end_time = time.time()
    
    print(f"2D solution found in {end_time - start_time:.3f} seconds")
    print(f"First 5 energy levels: {eigenvalues_2d[:5]}")
    
    # Plot energy levels
    fig = solver_2d.plot_energy_levels(n_levels=10)
    plt.savefig("results/2d_harmonic_oscillator_levels.png", dpi=300)
    plt.close(fig)
    
    # Plot the first few eigenstates with different visualization methods
    for i in range(4):
        # Contour plot
        fig = solver_2d.plot_eigenstate_2d(state_idx=i, plot_type='contour')
        plt.tight_layout()
        plt.savefig(f"results/2d_harmonic_eigenstate_{i}_contour.png", dpi=300)
        plt.close(fig)
        
        # Density plot
        fig = solver_2d.plot_eigenstate_2d(state_idx=i, plot_type='density')
        plt.tight_layout()
        plt.savefig(f"results/2d_harmonic_eigenstate_{i}_density.png", dpi=300)
        plt.close(fig)
    
    print("\nAll results saved to 'results/' directory")
    
    plt.show()  # Show all plots (comment this out for non-interactive mode)


if __name__ == "__main__":
    main()