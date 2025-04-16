import numpy as np

# Constants
G = 6.67434e-20
Me = 5.974e24 # [kg]
AU = 1.496e8 # [km]
r_earth = 6368 #[km]
mu = G*Me # [km^3/s^2]

# Calculate Hohmann Transfer Parameters
r1 = 400 + r_earth # [km] LEO
r2 = 42241 + r_earth # [km] GEO
a_h = (r1 + r2) / 2 # [km] semi-major axis of the transfer orbit
e_h = (r2 - r1) / (r2 + r1) # eccentricity of the transfer orbit
T_h = 2*np.pi * np.sqrt(a_h**3 / mu) # [s] period of the transfer orbit
t_h = T_h / 2 # [s] time to transfer from LEO to GEO
v1plus = np.sqrt(mu * (2/r1 - 1/a_h))
v2minus = np.sqrt(mu/r1)
delta_v1 = v1plus - v2minus # [km/s] delta-v at LEO
v2plus = np.sqrt(mu/r2)
v2minus = np.sqrt(mu * (2/r2 - 1/a_h))
delta_v2 = v2plus - v2minus # [km/s] delta-v at GEO
delta_v = delta_v1 + delta_v2 # [km/s] total delta-v for the transfer

# print results
print(f"Transfer orbit semi-major axis a: {a_h:.2f} km")
print(f"delta_v1: {delta_v1:.2f} km/s")
print(f"delta_v2: {delta_v2:.2f} km/s")
print(f"Total delta_v: {delta_v:.2f} km/s")
print(f"Time of flight: {t_h/3600:.2f} hours")


# Phase Change
