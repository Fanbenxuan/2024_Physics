# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 13:10:23 2024

@author: Mu Keming & Fan Benxuan
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
L = 50  # Length of the space station in meters
D = 4   # Diameter of the space station in meters
P0 = 101.3e3  # Initial pressure at the bottom in Pa
Pf = 0.3 * 101.3e3  # Final pressure at the bottom in Pa
T = 20 + 273.15  # Temperature in Kelvin
mu = 1.8e-5  # Dynamic viscosity of air in Pa·s
d = 0.01  # Diameter of the hole in meters
g = 9.81  # Gravity acceleration in m/s^2
M = 28.97e-3  # Molar mass of air in kg/mol
R = 8.314  # Gas constant in J/(mol·K)
k_B = 1.38e-23  # Boltzmann constant in J/K
N_A = 6.022e23  # Avogadro's number
d_m = 3.7e-10  # Diameter of air molecule in meters
m = M / N_A  # Mass of one air molecule in kg

A = np.pi * (d / 2)**2  # Area of the hole in m^2
V = np.pi * (D / 2)**2 * L  # Volume of the space station in m^3

# Calculate average molecular speed based on Maxwell-Boltzmann distribution
def average_speed(T):
    return np.sqrt(8 * k_B * T / (np.pi * m))

# Calculate root mean square speed
def rms_speed(T):
    return np.sqrt(3 * k_B * T / m)

# Mean free path as a function of pressure
def mean_free_path(P):
    return k_B * T / (np.sqrt(2) * np.pi * d_m**2 * P)

# Knudsen number
def knudsen_number(P):
    return mean_free_path(P) / d

# Modified flow model considering gravity, Knudsen number, and Maxwell distribution
def Q_total(P, h):
    delta_P = P - Pf
    P_avg = (P + Pf) / 2
    Kn = knudsen_number(P)
    # Calculate flow rates based on Knudsen number
    if Kn < 0.01:  # Viscous flow dominant
        # Poiseuille flow (viscous flow)
        Q_S = delta_P * d**3 / (24 * mu)
        return Q_S
    elif Kn > 10:  # Molecular flow dominant
        # Molecular flow considering Maxwell-Boltzmann distribution
        Q_E = (P_avg * d**2 / (6 * np.pi * mu)) * np.sqrt(8 * k_B * T / (np.pi * m))
        return Q_E
    else:  # Transitional flow (combined model)
        Q_S = delta_P * d**3 / (24 * mu)
        Q_E = (P_avg * d**2 / (6 * np.pi * mu)) * np.sqrt(8 * k_B * T / (np.pi * m))
        return Q_S / (1 + Kn) + Q_E / (1 + 1 / Kn)

# Calculate pressure at height h considering gravity (hydrostatic equilibrium)
def pressure_at_height(P0, h, T):
    return P0 * np.exp(-m * g * h / (k_B * T))

# Differential equation for pressure change considering gravity and Maxwell distribution
def dPdt(t, P):
    if P > Pf:
        h = 0  # Assume hole is at the bottom
        P_h = pressure_at_height(P, h, T)
        Q = Q_total(P_h, h)
        return -Q * P / V
    else:
        return 0

# Time span for the simulation
t_span = (0, 180)  # 1 hour in seconds
t_eval = np.linspace(*t_span, 10000)  # Evaluation points

# Solve the ODE
solution = solve_ivp(dPdt, t_span, [P0], t_eval=t_eval)

# Extract results
P_values = solution.y[0]
time_values = solution.t

# Plotting the pressure vs. time
plt.figure(figsize=(10, 6))
plt.plot(time_values / 60, P_values / 1000, label='Pressure', color='blue')  # Convert to kPa
plt.axhline(y=Pf / 1000, color='red', linestyle='--', label='Final Pressure (0.3 atm)')
plt.title('Pressure vs Time in Space Station (Gravity & Maxwell Distribution)')
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
