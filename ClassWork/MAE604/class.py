import numpy as np

G = 6.67434e-20
Me = 5.974e24 # [kg]
AU = 1.496e8 # [km]
r_earth = 6368 #[km]
mu = G*Me # [km^3/s^2]
r_circ = r_earth + 15000  # [km]
T = 2*np.pi*np.sqrt(r_circ**3/mu) # [s]
t_flight = 0.75*T # [s]
print(f'Time to do phase transfer is {t_flight/3600:.2f} [hours]')
a = pow(mu*(t_flight/(2*np.pi))**2, 1/3) # [km]
print(f'Semi-major axis of transfer orbit is {a:.2f} [km]')
v1 = np.sqrt(mu*(2/r_circ - 1/a)) # [km/s]
print(f'Velocity of spacecraft in elliptical orbit is {v1:.2f} [km/s]')
v0 = np.sqrt(mu/r_circ) # [km/s]
print(f'Velocity of spacecraft in circular orbit is {v0:.2f} [km/s]')
delta_v = v1 - v0 # [km/s]
print(f'Delta V to do phase transfer is {delta_v:.2f} [km/s]')

