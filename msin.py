import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScalarizedChargedBlackHole:
    def __init__(self, rh: float, phi1: float, psi0: float, c: float, m: float, q: float):
        """
        Initialize the Scalarized Charged Black Hole.

        :param rh: Horizon radius
        :param phi1: Initial value for phi
        :param psi0: Initial value for psi
        :param c: Scalar field parameter
        :param m: Mass parameter
        :param q: Charge parameter
        """
        self.rh = rh
        self.phi1 = phi1
        self.psi0 = psi0
        self.c = c
        self.m = m
        self.q = q

    def equations(self, r: float, y: List[float]) -> List[float]:
        """Define the system of differential equations."""
        f, chi, phi, psi = y
        f = max(f, 1e-10)
        chi = min(chi, 700)
        dfdx = 1 - f + r * f * psi**2 + r * np.exp(chi) * phi**2 + r * f * self.W(psi**2)
        dchidx = r * (psi**2 + np.exp(chi) * phi**2 / f)
        dphidx = -2 * phi / r + self.q**2 * psi**2 * phi / f
        dpsidx = -(f + r * dfdx) * psi / f - (self.q**2 * np.exp(chi) * phi**2 / f - self.w(psi**2)) * psi / f
        return [dfdx, dchidx, dphidx, dpsidx]

    def W(self, x: float) -> float:
        """Potential function W."""
        return self.m**2 * self.c**2 * np.log(1 + x / (2 * self.c**2))

    def w(self, x: float) -> float:
        """Derivative of potential function W."""
        return self.m**2 / (1 + x / (2 * self.c**2))

    def solve(self, r_max: float) -> Tuple[np.ndarray, np.ndarray]:
        """Solve the differential equations."""
        def event(r: float, y: np.ndarray) -> float:
            return np.any(np.isnan(y)) or np.any(np.isinf(y))
        event.terminal = True

        r = np.geomspace(self.rh * 1.001, r_max, 10000)
        y0 = [1e-3, 1e-3, self.phi1, self.psi0]
        
        try:
            solution = solve_ivp(
                self.equations, 
                (self.rh * 1.001, r_max), 
                y0, 
                t_eval=r, 
                method='Radau',
                rtol=1e-10, 
                atol=1e-10,
                max_step=0.01,
                events=event
            )
        except Exception as e:
            logger.error(f"Integration failed: {e}")
            raise

        if not solution.success:
            raise ValueError(f"Integration failed: {solution.message}")
        if len(solution.t) < 1000:
            raise ValueError(f"Integration stopped after {len(solution.t)} steps")
        
        return solution.t, solution.y

    def mass_and_charge(self, solution: np.ndarray, r_max: float) -> Tuple[float, float]:
        """Calculate mass and charge of the black hole."""
        f = solution[0, -1]
        phi = solution[2, -1]
        M = -r_max**2 * (f - 1) / 2
        Q = r_max**2 * phi
        return M, Q

    def temperature(self, solution: np.ndarray) -> float:
        """Calculate the temperature of the black hole."""
        f_prime = (solution[0, 1] - solution[0, 0]) / (self.rh * 1e-3)
        return f_prime * np.exp(-solution[1, 0] / 2) / (4 * np.pi)

    def chemical_potential(self, solution: np.ndarray, r_max: float) -> float:
        """Calculate the chemical potential of the black hole."""
        phi = solution[2, -1]
        phi_prime = (solution[2, -1] - solution[2, -2]) / (r_max * 1e-3)
        return phi + r_max * phi_prime

    def entropy(self) -> float:
        """Calculate the entropy of the black hole."""
        return 4 * np.pi * self.rh**2

def find_scalarized_solution(bh: ScalarizedChargedBlackHole, r_max: float, target_mass: float, target_charge: float) -> Tuple[float, float]:
    """Find the scalarized solution using differential evolution."""
    def objective(params: np.ndarray) -> float:
        psi0, phi1 = params
        bh.psi0 = psi0
        bh.phi1 = phi1
        try:
            r, solution = bh.solve(r_max)
            M, Q = bh.mass_and_charge(solution, r_max)
            return (M - target_mass)**2 + (Q - target_charge)**2
        except ValueError:
            return np.inf

    bounds = [(0, 1), (0, 1)]
    result = differential_evolution(objective, bounds, popsize=20, mutation=(0.5, 1.5), recombination=0.7, seed=42)
    
    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")
    
    return tuple(result.x)

def analyze_stability(bh: ScalarizedChargedBlackHole, r_max: float) -> Tuple[float, float, float, float, float, float]:
    """Analyze the stability of the black hole solution."""
    r, solution = bh.solve(r_max)
    M, Q = bh.mass_and_charge(solution, r_max)
    T = bh.temperature(solution)
    mu = bh.chemical_potential(solution, r_max)
    S = bh.entropy()
    
    F = M - T * S
    G = F - mu * Q
    
    return M, Q, T, mu, F, G

def compute_qnms(bh: ScalarizedChargedBlackHole, r_max: float, l: int, n_max: int) -> List[complex]:
    """Compute quasi-normal modes using the WKB method."""
    r, solution = bh.solve(r_max)
    
    def V_eff(r: float) -> float:
        f = np.interp(r, r, solution[0])
        chi = np.interp(r, r, solution[1])
        df_dr = np.gradient(f, r)
        return l * (l + 1) / (r**2) * f * np.exp(-chi) + f / (2 * r) * (df_dr * np.exp(-chi / 2))
    
    def wkb_frequencies(n: int) -> complex:
        r0 = differential_evolution(lambda r: abs(np.gradient(V_eff(np.array([r])))[0]), 
                                    bounds=[(bh.rh, r_max)], popsize=20).x[0]
        V0 = V_eff(r0)
        V2 = np.gradient(np.gradient(V_eff(np.linspace(r0-1e-5, r0+1e-5, 1000))))[500]
        V3 = np.gradient(np.gradient(np.gradient(V_eff(np.linspace(r0-1e-5, r0+1e-5, 1000)))))[500]
        
        L = n + 0.5
        omega = np.sqrt(V0) - 1j * (n + 0.5) * np.sqrt(-V2 / (2 * V0))
        omega += 1 / (8 * V0) * ((1 / 4 + L**2) * V2**2 / V0**2 - V3 / (2 * V0))
        return omega
    
    return [wkb_frequencies(n) for n in range(n_max)]

def plot_results(bh: ScalarizedChargedBlackHole, r_max: float, M: float, Q: float, F: float, G: float, qnms: List[complex]):
    """Plot the results of the analysis."""
    r, solution = bh.solve(r_max)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    axs[0, 0].semilogx(r, solution[0], label='f(r)')
    axs[0, 0].semilogx(r, np.exp(-solution[1]), label='e^(-χ(r))')
    axs[0, 0].set_xlabel('r')
    axs[0, 0].set_ylabel('Metric functions')
    axs[0, 0].legend()
    axs[0, 0].set_title('Metric functions')

    axs[0, 1].semilogx(r, solution[2], label='φ(r)')
    axs[0, 1].semilogx(r, solution[3], label='ψ(r)')
    axs[0, 1].set_xlabel('r')
    axs[0, 1].set_ylabel('Matter fields')
    axs[0, 1].legend()
    axs[0, 1].set_title('Matter fields')

    axs[1, 0].scatter(np.real(qnms), np.imag(qnms))
    axs[1, 0].set_xlabel('Re(ω)')
    axs[1, 0].set_ylabel('Im(ω)')
    axs[1, 0].set_title('Quasi-normal modes')

    axs[1, 1].scatter(Q/M, F/M, label='Helmholtz free energy')
    axs[1, 1].scatter(Q/M, G/M, label='Gibbs free energy')
    axs[1, 1].set_xlabel('Q/M')
    axs[1, 1].set_ylabel('F/M or G/M')
    axs[1, 1].legend()
    axs[1, 1].set_title('Thermodynamic stability')

    plt.tight_layout()
    plt.show()

def main():
    bh = ScalarizedChargedBlackHole(rh=1, phi1=0.8, psi0=0.1988, c=0.1, m=0.01, q=1.1*0.01)
    r_max = 1000

    target_mass = 0.9676
    target_charge = 0.9675
    
    try:
        psi0_scalarized, phi1_scalarized = find_scalarized_solution(bh, r_max, target_mass, target_charge)
        logger.info(f"Found scalarized solution with psi0 = {psi0_scalarized:.6f}, phi1 = {phi1_scalarized:.6f}")
    except ValueError as e:
        logger.error(f"Error finding scalarized solution: {e}")
        return

    try:
        M, Q, T, mu, F, G = analyze_stability(bh, r_max)
        qnms = compute_qnms(bh, r_max, l=2, n_max=5)
        plot_results(bh, r_max, M, Q, F, G, qnms)

        logger.info(f"Mass: {M:.6f}")
        logger.info(f"Charge: {Q:.6f}")
        logger.info(f"Temperature: {T:.6f}")
        logger.info(f"Chemical potential: {mu:.6f}")
        logger.info(f"Helmholtz free energy: {F:.6f}")
        logger.info(f"Gibbs free energy: {G:.6f}")
    except Exception as e:
        logger.error(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()