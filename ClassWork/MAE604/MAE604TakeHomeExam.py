import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import chardet
import matplotlib.animation as animation

# Constants
G = 6.6743e-20 # [km^3/kg/s^2]
Me = 5.974e24 # [kg]
AU = 1.496e8 # [km]
M_mars = 0.1074*Me # [kg]
mars_d = 1.52*AU # [km]
mars_rkm = 3389.5 # [km]

# Question 1
v1 = 6 # [km/s]
mu_mars = G*M_mars # [km^3/s^2]
r1 = mu_mars/(v1**2) # [km]
h1 = r1 - mars_rkm # [km]
vmax = np.sqrt(mu_mars/mars_rkm) # [km/s]
print('\nQuestion 1:\n')
if h1 <= 0:
    print(f'Orbit is not possible. The altitude is less than 0: h1 = {h1:.2f} [km]')
    print(f'Maximum velocity for circular orbit around Mars is {vmax:.2f} [km/s] for an orbit radius of {mars_rkm:.2f}')
else:
    print(f'Altitude of spacecraft orbiting Mars: {h1:.2f} [km]')

v_esc = np.sqrt(2) * v1 # [km/s]
print(f'Escape velocity: {v_esc:.2f} [km/s]\n')

# Question 2
# Checling the encoding of the csv file
with open('Lagrange Sol.csv', 'rb') as f:
    result = chardet.detect(f.read())  

# Reading the data from the csv file and printing the columns
data = pd.read_csv('Lagrange Sol.csv', encoding=result['encoding'])

# Extracting the data from the csv file using the column names
planet_smas = data['semimajor_axis(xe9m)']*1e9
L1_values = data['L1(xe9m)']*1e9
L2_values = data['L2(xe9m)']*1e9
L3_values = data['L3(xe9m)']*-1e9

# Defining the planet names in simpler list than csv file
planet_names = [
    'Mercury',
    'Venus',
    'Earth',
    'Mars',
    'Jupiter',
    'Saturn',
    'Uranus',
    'Neptune'
]

# Creating a plot
plt.figure(figsize=(10, 10))

# Plotting the orbits
for sma, name in zip(planet_smas, planet_names):
    theta = np.linspace(0, 2*np.pi, 100)
    x = sma * np.cos(theta)
    y = sma * np.sin(theta)
    plt.plot(x, y, label=name)

# Plot the sun and the lagrange points for each planet
plt.plot(0, 0, 'yo', label='Sun')
for L1, L2, L3, name in zip(L1_values, L2_values, L3_values, planet_names):
    plt.plot(L1, 0, 'x', label='L1 for ' + name)
    plt.plot(L2, 0, 'o', label='L2 for ' + name)
    plt.plot(L3, 0, '.', label='L3 for ' + name)

# Adding labels and legend
plt.title('Solar System Circular Orbits')
plt.axis('equal')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.grid()
plt.legend()
plt.show()

print('Question 2:\n')
print('Based on the plot, it is not possible for any of the planets in the solar system to pass through one (or more) of each others Lagrange points. Even though L4 and L5 were not plotted for the planets, since L4 and L5 lay on the planets orbit, a different planet in the solar system would not be able to get caught in it.\n')

# Question 3
print('Question 3:\n')
print('In order to travel from one co-orbital planet to the other, the spacecraft must burst with a delta v to get onto an elliptical orbit around the focus point (sun) that intersects the path of the target planet. Then, once it reaches the target planet, it needs to execute delta v to get into circular orbit around that planet.\n')

# Problem 1
# Parameters
M_moon = 7.34767309e22 # [kg]
r_moon = 1737.4 # [km]
h_moon = 100 # [km]
mu_moon = G*M_moon # [km^3/s^2]
v_sp = np.sqrt(mu_moon/(r_moon + h_moon)) # [km/s]
T_sp = 2*np.pi*np.sqrt((r_moon + h_moon)**3/mu_moon) # [s]
print('Problem 1:\n')
print(f'Velocity of spacecraft in circular orbit around Moon: {v_sp} [km/s]')
print(f'Period of spacecraft in circular orbit around Moon: {T_sp} [s]\n')

# Problem 2
# Parameters
r_earth = 6378 # [km]
mu = G*Me  # [km^3/s^2]
ra = 10000 + r_earth  # [km]
rp = 5000 + r_earth  # [km]
e = (ra - rp)/(ra + rp)  # eccentricity
a = ra/(1 + e)  # semi-major axis [km]
T_sp_earth = 2*np.pi*np.sqrt(a**3/mu)  # [s]
E = np.linspace(0, 2*np.pi, 5)
M_ab = E - e*np.sin(E)
n = np.sqrt(mu/a**3)
t = M_ab/n

print('Problem 2:\n')
print(f'(a) Time it takes to get from point P to point A is {(t[2]-t[0])/3600:.2f} [hours]')
print(f'(b) Time it takes to get from point B to point D is {(t[3]-t[1])/3600:.2f} [hours]')

# Calculating mean anomaly for t = 4.5 hours
M_i = n * 4.5 * 3600 # initial guess for mean anomaly [rad]
def g(M, E, e):
    return M - (E - e*np.sin(E))

def dg_dE(E, e):
    return e*np.cos(E) - 1

t_vals = np.linspace(0, 4.5*3600, 1000)
M_vals = n*t_vals
E_vals = np.zeros_like(M_vals)

# Newton-Raphson method
for i, M in enumerate(M_vals):
    Ei = M
    for _ in range(100):
        E_new = Ei - g(M, Ei, e)/dg_dE(Ei, e)
        if np.max(np.abs(E_new - Ei)) < 1e-6:
            break
        Ei = E_new
    E_vals[i] = Ei

f_vals = 2 * np.arctan(np.sqrt((1 + e)/(1 - e)) * np.tan(E_vals/2))  # [rad]
P = a*(1 - e**2)  # [km]
r_vals = P/(1 + e*np.cos(f_vals))  # [km]

x_vals = r_vals*np.cos(f_vals)  # [km]
y_vals = r_vals*np.sin(f_vals)  # [km]
print(f'(c) Position of spacecraft after 4.5 hours is ({x_vals[-1]:.2f}, {y_vals[-1]:.2f}) [km]\n    Angle from x-axis is {f_vals[-1]:.2f} [rad]\n')

theta = np.linspace(0, 2*np.pi, 500)
r_orbit = P/(1 + e*np.cos(theta))
x_orbit = r_orbit*np.cos(theta)
y_orbit = r_orbit*np.sin(theta)

b = np.sqrt(a**2*(1 - e**2))  # [km]
print(f'Semi-minor axis of the orbit is {b:.2f} [km]')
print(f'Semi-major axis of the orbit is {a:.2f} [km]\n')
# Set up figure
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x_orbit, y_orbit, 'b', label='Orbit')  # Full orbit
ax.plot(a * e, 0, 'go', markersize=8, label='Earth')  # Earth
spacecraft, = ax.plot([], [], 'ro', markersize=8, label='Spacecraft')  # Spacecraft marker
trail, = ax.plot([], [], 'r-', linewidth=2, label='Trajectory')
line_to_spacecraft, = ax.plot([], [], 'k--', label='Line to spacecraft')
x_y_position = ax.text([], [], '')

ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_title('Spacecraft Orbiting Earth')
ax.legend()
ax.axis('equal')
ax.grid()

# Animation function
def update(frame):
    spacecraft.set_data([x_vals[frame]], [y_vals[frame]])
    trail.set_data(x_vals[:frame+1], y_vals[:frame+1])
    line_to_spacecraft.set_data([a * e, x_vals[frame]], [0, y_vals[frame]])
    x_y_position.set_text(f'x = {x_vals[frame]:.2f} km\ny = {y_vals[frame]:.2f} km')
    x_y_position.set_position((x_vals[frame], y_vals[frame]))
    return spacecraft, trail, line_to_spacecraft, x_y_position

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(x_vals), interval=20, blit=True, repeat=False)
# ani.save('problem2.mp4', writer='ffmpeg', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()

# Problem 3
mL = 1000  # [kg]
mp1 = mp2 = mp3 = 10000  # [kg]
ms1 = 2000 # [kg]
ms2 = 1500 # [kg]
ms3 = 1000 # [kg]
c = 2500 # [km/s^2]

# Stage 1
mo1 = mp1 + ms1 + mp2 + ms2 + mp3 + ms3 + mL  # [kg]

# Stage 2
mo2 = ms2 + mp2 + ms3 + mp3 + mL  # [kg]

# Stage 3
mo3 = ms3 + mp3 + mL  # [kg]

mL1 = mo2
mL2 = mo3
mL3 = mL

z1 = mo1/(ms1 + mL1)
z2 = mo2/(ms2 + mL2)
z3 = mo3/(ms3 + mL3)

delta_v1 = c*np.log(z1)
delta_v2 = c*np.log(z2)
delta_v3 = c*np.log(z3)
delta_v = delta_v1 + delta_v2 + delta_v3
print('Problem 3:\n')
print(f'Velocity change: {delta_v/1000:.2f} [km/s]\n')