import numpy as np
from scipy.integrate import odeint
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

class ScalarizedChargedBlackHole:
    def __init__(self, rh, phi1, psi0, c, m, q):
        self.rh = rh
        self.phi1 = phi1
        self.psi0 = psi0
        self.c = c
        self.m = m
        self.q = q
        
    def equations(self, y, r):
        f, chi, phi, psi = y
        dfdx = (1/r - r*psi**2/2 - r*np.exp(chi)*phi**2/2 - r*self.W(psi**2)/2 - f/r)
        dchidx = -r*np.exp(chi)*psi**2*phi**2/f - r*psi**2
        dphidx = -2*phi/r + self.q**2*psi**2*phi/(2*f)
        dpsidx = -(f + r*f.diff(r))/f*psi - (self.q**2*np.exp(chi)*phi**2/f**2 - self.w(psi**2)/f)*psi
        return [dfdx, dchidx, dphidx, dpsidx]
    
    def W(self, x):
        return self.m**2 * self.c**2 * np.log(1 + x/(2*self.c**2))
    
    def w(self, x):
        return self.m**2 / (1 + x/(2*self.c**2))
    
    def solve(self, r_max):
        r = np.linspace(self.rh, r_max, 1000)
        y0 = [0, 0, 0, self.psi0]
        solution = odeint(self.equations, y0, r)
        return r, solution
    
    def mass_and_charge(self, solution, r_max):
        f = solution[-1, 0]
        phi = solution[-1, 2]
        M = -r_max**2 * (f - 1) / 2
        Q = r_max**2 * phi
        return M, Q
    
    def temperature(self, solution):
        f_prime = (solution[1, 0] - solution[0, 0]) / (self.rh * 1e-6)
        return f_prime * np.exp(-solution[0, 1]/2) / (4*np.pi)
    
    def chemical_potential(self, solution, r_max):
        phi = solution[-1, 2]
        phi_prime = (solution[-1, 2] - solution[-2, 2]) / (r_max * 1e-6)
        return phi + r_max * phi_prime

def find_scalarized_solution(bh, r_max, target_mass, target_charge):
    def objective(psi0):
        bh.psi0 = psi0
        r, solution = bh.solve(r_max)
        M, Q = bh.mass_and_charge(solution, r_max)
        return (M - target_mass)**2 + (Q - target_charge)**2
    
    result = root_scalar(objective, bracket=[0, 1], method='brentq')
    return result.root

def analyze_stability(bh, r_max):
    r, solution = bh.solve(r_max)
    M, Q = bh.mass_and_charge(solution, r_max)
    T = bh.temperature(solution)
    mu = bh.chemical_potential(solution, r_max)
    
    # Compute free energies
    F = M - T * (4*np.pi*bh.rh**2)  # Helmholtz free energy
    G = M - T * (4*np.pi*bh.rh**2) - mu * Q  # Gibbs free energy
    
    return M, Q, T, mu, F, G

def compute_qnms(bh, r_max, l, n_max):
    def V_eff(r):
        f = np.interp(r, bh.r, bh.solution[:, 0])
        chi = np.interp(r, bh.r, bh.solution[:, 1])
        return l*(l+1)/(r**2) * f * np.exp(-chi) + f/(2*r) * (f.diff(r) * np.exp(-chi/2)).diff(r)
    
    def wkb_frequencies(n):
        r0 = root_scalar(lambda r: V_eff(r).diff(r), bracket=[bh.rh, r_max]).root
        V0 = V_eff(r0)
        V2 = V_eff(r0).diff(r, 2)
        V3 = V_eff(r0).diff(r, 3)
        
        L = n + 0.5
        omega = np.sqrt(V0) - 1j*(n+0.5)*np.sqrt(-V2/(2*V0))
        omega += 1/(8*V0) * ((1/4 + alpha**2)*V2**2/V0**2 - V3/(2*V0))
        return omega
    
    qnms = [wkb_frequencies(n) for n in range(n_max)]
    return qnms

# Main analysis
bh = ScalarizedChargedBlackHole(rh=1, phi1=0.8, psi0=0.1988, c=0.1, m=0.01, q=1.1*0.01)
r_max = 1000

# Find scalarized solution
target_mass = 0.9676
target_charge = 0.9675
psi0_scalarized = find_scalarized_solution(bh, r_max, target_mass, target_charge)
bh.psi0 = psi0_scalarized

# Analyze stability
M, Q, T, mu, F, G = analyze_stability(bh, r_max)

# Compute QNMs
qnms = compute_qnms(bh, r_max, l=2, n_max=5)

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.plot(bh.r, bh.solution[:, 0], label='f(r)')
plt.plot(bh.r, np.exp(-bh.solution[:, 1]), label='e^(-χ(r))')
plt.legend()
plt.title('Metric functions')

plt.subplot(222)
plt.plot(bh.r, bh.solution[:, 2], label='φ(r)')
plt.plot(bh.r, bh.solution[:, 3], label='ψ(r)')
plt.legend()
plt.title('Matter fields')

plt.subplot(223)
plt.scatter(np.real(qnms), np.imag(qnms))
plt.xlabel('Re(ω)')
plt.ylabel('Im(ω)')
plt.title('Quasi-normal modes')

plt.subplot(224)
plt.plot(Q/M, F/M, label='Helmholtz free energy')
plt.plot(Q/M, G/M, label='Gibbs free energy')
plt.xlabel('Q/M')
plt.ylabel('F/M or G/M')
plt.legend()
plt.title('Thermodynamic stability')

plt.tight_layout()
plt.show()

print(f"Mass: {M:.4f}")
print(f"Charge: {Q:.4f}")
print(f"Temperature: {T:.4f}")
print(f"Chemical potential: {mu:.4f}")
print(f"Helmholtz free energy: {F:.4f}")
print(f"Gibbs free energy: {G:.4f}")