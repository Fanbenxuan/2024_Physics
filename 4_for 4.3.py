# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 23:27:42 2024

@author: Mu Keming & Fan Benxuan
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameter settings
n_particles = 100000  # Initial number of particles
initial_particles = n_particles
radius = 2  # Radius of the cylinder (meters)
length_cylinder = 50  # Length of the cylinder (meters)
hole_radius = 1  # Radius of the hole at the bottom (meters)
mass = 1e-26  # Assume particle mass is 1 (kg)
temperature = 293.15  # Initial temperature (Kelvin)
kB = 1.380649e-23  # Boltzmann constant
R = 8.314  # Ideal gas constant (J/(mol·K))
gamma = 1.4  # Adiabatic index of air

# Initial pressure (Pa)
initial_pressure = 101300  # 1 atmosphere
volume = np.pi * radius**2 * length_cylinder  # Volume of the cylinder (cubic meters)

dt = 0.005  # Time step
total_time = 10000  # Total simulation time
steps = int(total_time / dt)  # Total number of steps
time_data = []
pressure_data = []

# Initialize particle positions and velocities
angles = 2 * np.pi * np.random.rand(n_particles)  # Random angles
radii = radius * np.sqrt(np.random.rand(n_particles))  # Random radii, uniformly distributed within the circle
x_positions = radii * np.cos(angles)
y_positions = radii * np.sin(angles)
z_positions = length_cylinder * np.random.rand(n_particles)  # Uniform distribution in the z direction
positions = np.column_stack((x_positions, y_positions, z_positions))

speeds_x = np.sqrt(kB * temperature / mass) * np.random.randn(n_particles)  # x-direction speeds follow Maxwell distribution
speeds_y = np.sqrt(kB * temperature / mass) * np.random.randn(n_particles)  # y-direction speeds follow Maxwell distribution
speeds_z = np.sqrt(kB * temperature / mass) * np.random.randn(n_particles)  # z-direction speeds follow Maxwell distribution
speeds = np.column_stack((speeds_x, speeds_y, speeds_z))

# Monte Carlo molecular dynamics experiment
for t in range(steps):
    # Save positions before update
    previous_positions = positions.copy()

    # Update particle positions
    positions += speeds * dt

    # Handle collisions with the cylinder wall
    r = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
    outside_indices = r > radius
    if np.any(outside_indices):
        normals = positions[outside_indices, :2] / r[outside_indices, np.newaxis]  # Normal vectors
        speeds[outside_indices, :2] -= 2 * np.sum(speeds[outside_indices, :2] * normals, axis=1)[:, np.newaxis] * normals  # Reflect speeds
        speeds[outside_indices] -= 0.3 * np.random.randn(np.sum(outside_indices), 3)  # Reduce the effect of diffuse reflection randomness

    # Check if particles escape through the hole at the bottom, using parametric path check
    z_crossed = (previous_positions[:, 2] >= 0) & (positions[:, 2] < 0)  # Cross from positive to negative through the bottom
    if np.any(z_crossed):
        # For particles crossing the bottom, check if their path goes through the hole
        crossing_fraction = previous_positions[z_crossed, 2] / (previous_positions[z_crossed, 2] - positions[z_crossed, 2])
        interpolated_x = previous_positions[z_crossed, 0] + crossing_fraction * (positions[z_crossed, 0] - previous_positions[z_crossed, 0])
        interpolated_y = previous_positions[z_crossed, 1] + crossing_fraction * (positions[z_crossed, 1] - previous_positions[z_crossed, 1])
        interpolated_r = np.sqrt(interpolated_x**2 + interpolated_y**2)
        escaping_particles = interpolated_r < hole_radius  # Check if within hole radius

        if np.any(escaping_particles):
            escape_indices = np.where(z_crossed)[0][escaping_particles]
            speeds[escape_indices] = np.nan  # Set to NaN to indicate particles have escaped through the hole
            positions[escape_indices] = np.nan

    # Check for other collisions with the bottom
    bottom_indices = positions[:, 2] < 0
    if np.any(bottom_indices & ~np.isnan(positions[:, 0])):
        speeds[bottom_indices & ~np.isnan(positions[:, 0]), 2] *= -1  # Reflect speeds upon collision with the bottom

    # Check if particles exceed the top of the cylinder
    top_indices = positions[:, 2] > length_cylinder
    if np.any(top_indices):
        speeds[top_indices, 2] *= -1  # Reflect speeds upon collision with the top of the cylinder

    # Remove particles that have escaped through the hole
    valid_indices = ~np.isnan(speeds[:, 0])
    positions = positions[valid_indices]
    speeds = speeds[valid_indices]
    n_particles = len(positions)

    # Update temperature and pressure
    if n_particles > 0:
        # temperature = 293.15 * (n_particles / initial_particles) ** (gamma - 1)
        temperature = 293.15 
        current_pressure = (n_particles / initial_particles) * initial_pressure * (temperature / 293.15)
    else:
        current_pressure = 0

    # Output current particle count, pressure, and temperature every 1000 time steps
    if t % 1000 == 0:
        kinetic_energy = 0.5 * mass * np.sum(speeds**2)
        average_kinetic_energy = kinetic_energy / n_particles if n_particles > 0 else 0
        print(f'Time: {t * dt:.2f} s, Particle Count: {n_particles}, Pressure: {current_pressure:.2f} Pa, Temperature: {temperature:.2f} K')
        time_data.append(t * dt)
        pressure_data.append(current_pressure)

        
    # Check if particle count is below 30% of the initial count
    if current_pressure < 0.3 * initial_pressure:
        print(f'Particle count below 30% of initial count. Terminating at Time: {t * dt:.2f} s')
        break
        
# Visualize the leakage process
plt.plot(time_data, pressure_data)
plt.xlabel('Time (s)')
plt.ylabel('Pressure (Pa)')
plt.title('Pressure vs. Time during Leak')
plt.grid(False)
plt.show()