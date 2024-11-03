# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 10:43:15 2024

@author: Mu Keming & Fan Benxuan
"""

import numpy as np
import matplotlib.pyplot as plt

# Constant definitions
L = 50  # Length of the space station in meters
D = 4   # Diameter of the space station in meters
d = 0.001  # Diameter of the hole in meters
mw = 0.029       # Molecular weight of the gas (kg/mole) - Air = 0.029
temp = 293       # Temperature (K)
Rg = 8.314       # Universal gas constant (J / mole-K)
press = 101300   # Initial pressure (Pa)
vol = np.pi * (D / 2)**2 * L         # Volume of the space station in m^3
Ah = np.pi * (d / 2)**2      # Area of the hole in m^2

# Initialization
press0 = press   # Save initial pressure
tm = 0           # Start time (seconds)

# Calculate initial gas mass and density
mass = press * vol * mw / (Rg * temp)  # Initial gas mass
rho = mass / vol                          # Initial density

# Print title
print(f"Spacecraft depressurisation - Hole size = {Ah * 10000:.2f} sq cm")
print("Time(sec/min)  Mass(kg)  Density(kg/m^3)  Pressure(kPa)")
f1 = "{:<12.2f}{:<12.1f}{:<12.1f}{:<12.2f}{:<12.2f}"  # Output format, keep two decimal places

# Store time, mass, density, and pressure data
time_data = []
mass_data = []
rho_data = []
press_data = []

# Time interval
delta = 0.1

# Start time loop
while press >= 0.3 * press0:  # When the pressure is greater than 10% of the initial pressure
    # Store data
    time_data.append(tm)
    mass_data.append(mass)
    rho_data.append(rho)
    press_data.append(press / 1000)  # Convert to kPa
    print(f1.format(tm, tm / 60, mass, rho, press / 1000))
    
    # Advance time by one second
    tm += delta

    # Calculate mass loss per time interval
    # Use the gas leakage rate formula and introduce a time interval correction
    massloss = delta * Ah * np.sqrt(2 * press * rho)  
    mass -= massloss  # Calculate new mass
    rho = mass / vol  # Calculate new density
    press = rho * Rg * temp / mw  # Calculate new pressure

# Print final state
print(f"Final state after depressurization:")
print(f"Time: {tm:.1f} seconds")
print(f"Time: {tm/60:.1f} mins")
print(f"Remaining Mass: {mass:.2f} kg")
print(f"Remaining Pressure: {press / 1000:.2f} kPa")

# Plotting
plt.figure(figsize=(12, 8))

# Pressure change
plt.subplot(3, 1, 1)
plt.plot(time_data, press_data, label='Pressure (kPa)', color='blue')
plt.title('Depressurization of Spacecraft')
plt.ylabel('Pressure (kPa)')
plt.grid(True)

# Mass change
plt.subplot(3, 1, 2)
plt.plot(time_data, mass_data, label='Mass (kg)', color='orange')
plt.ylabel('Mass (kg)')
plt.grid(True)

# Density change
plt.subplot(3, 1, 3)
plt.plot(time_data, rho_data, label='Density (kg/m^3)', color='green')
plt.ylabel('Density (kg/m^3)')
plt.xlabel('Time (seconds)')
plt.grid(True)

# Show legend
plt.tight_layout()
plt.show()
