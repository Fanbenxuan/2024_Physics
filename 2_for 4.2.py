# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 12:13:55 2024

@author: Mu Keming & Fan Benxuan
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
L = 50  # Length of the space station in meters
D = 4   # Diameter of the space station in meters
P0 = 101.3e3  # Initial pressure in Pa
Pf = 0.3 * 101.3e3  # Final pressure in Pa
Pout = 0 # Outside pressure in Pa
T = 20 + 273.15  # Temperature in Kelvin
R = 287.05  # Specific gas constant for air in J/(kg·K)
mu = 1.8e-5  # Dynamic viscosity of air in Pa·s
k_B = 1.38e-23  # Boltzmann constant in J/K
d_m = 3.7e-10  # Diameter of air molecule in meters
M = 28.97e-3  # Molar mass of air in kg/mol
N_A = 6.022e23  # Avogadro's number
m = M / N_A  # Mass of one air molecule in kg
d = 0.01  # Diameter of the hole in meters
A = np.pi * (d / 2)**2  # Area of the hole in m^2
V = np.pi * (D / 2)**2 * L  # Volume of the space station in m^3

# Functions for mean free path λ, Q_S, and Q_E
def mean_free_path(P):
    return k_B * T / (np.sqrt(2) * np.pi * d_m**2 * P)

def Q_S(delta_P, d, mu):
    return delta_P * d**3 / (24 * mu)

def Q_E(P_avg, d, T):
    return (P_avg * d**2) / (6 * np.pi * mu) * np.sqrt(8 * k_B * T / (np.pi * m))

# Differential equation for pressure change
def dPdt(t, P):
    if P > Pf:
        delta_P = P - Pout
        P_avg = (P + Pout) / 2
        λ = mean_free_path(P)
        Kn = λ / d  # Calculate Knudsen number
        
        # Calculate Q_S and Q_E
        QS = Q_S(delta_P, d, mu)
        QE = Q_E(P_avg, d, T)
        
        # Determine the total flow rate based on Knudsen number
        if Kn < 0.01:  # Viscous flow dominant
            Q_total = QS
        elif Kn > 10:  # Molecular flow dominant
            Q_total = QE
        else:  # Transitional flow, use combined model
            Q_total = QS / (1 + Kn) + QE / (1 + 1 / Kn)
        
        return -Q_total / V
    else:
        return 0

# Time span for the simulation
t_span = (0, 3.6e5)  # 1 hour in seconds
t_eval = np.linspace(*t_span, 1000000)  # Evaluation points

# Solve the ODE
solution = solve_ivp(dPdt, t_span, [P0], t_eval=t_eval)

# Extract results
P_values = solution.y[0]
time_values = solution.t

# Plotting the pressure vs. time
plt.figure(figsize=(10, 6))
plt.plot(time_values / 60, P_values / 1000, label='Pressure', color='blue')  # Convert to kPa
plt.axhline(y=Pf / 1000, color='red', linestyle='--', label='Final Pressure (0.3 atm)')
plt.title('Pressure vs Time in Space Station (Considering Mean Free Path and Knudsen Number)')
plt.xlabel('Time (minutes)')
plt.ylabel('Pressure (kPa)')
plt.legend()
plt.grid()
plt.ylim(bottom=0)  # Ensure y-axis starts at 0
plt.show()

# Time to reach final pressure
if np.any(P_values <= Pf):
    final_time = time_values[np.where(P_values <= Pf)[0][0]] / 60  # Convert to minutes
    print(f"Time to reduce pressure to 0.3 atm: {final_time:.2f} minutes")
else:
    print("Pressure did not reach 0.3 atm within the simulation time.")
