import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import chardet

# Question 2
# Checling the encoding of the csv file
with open('Lagrange Sol.csv', 'rb') as f:
    result = chardet.detect(f.read())  
    print(result['encoding'])

# Reading the data from the csv file and printing the columns
data = pd.read_csv('Lagrange Sol.csv', encoding='windows-1252')
print(f'{data}\n')
print(f'{data.columns}\n')

# Extracting the data from the csv file using the column names
planet_smas = data['semimajor_axis(xe9m)']*1e9
L1_values = data['L1(xe9m)']*1e9
L2_values = data['L2(xe9m)']*1e9
L3_values = data['L3(xe9m)']*1e9

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

# Problem 2 part (c)

# Parameters
neu = 3.987e5  # [km^3/s^2]
ra = 10000  # [km]
rp = 5000  # [km]
e = (ra - rp)/(ra + rp)  # eccentricity
a = ra/(1 + e)  # semi-major axis [km]

# Calculating mean anomaly for t = 4.5 hours
M = np.sqrt(neu/a**3)*(4.5*3600)  # [rad]
print(f'Mean anomaly for t = 4.5 hours: {M} [rad]\n')
def g(M, E, e):
    return M - (E - e*np.sin(E))

def dg_dE(E, e):
    return e*np.cos(E) - 1

# Newton-Raphson method
def nr_method(M, tol=1e-6, max_iter=100):
    Ei = M
    iter = 0
    for i in range(max_iter):
        E_new = Ei - g(M, Ei, e)/dg_dE(Ei, e)
        if abs(E_new - Ei) < tol:
            return E_new, iter
        Ei = E_new
        iter += 1
    return Ei, iter

E, iter = nr_method(M)
print(f'Eccentric anomaly for t = 4.5 hours: {E} [rad]\nNumber of iterations: {iter}\n')

f = 2 * np.arctan(np.sqrt((1 + e)/(1 - e)) * np.tan(E/2))  # [rad]
print(f'True anomaloy for t = 4.5 hours: {f} [rad]\n')

# Problem 3
mL = 1000  # [kg]
mp1 = mp2 = mp3 = 10000  # [kg]
ms1 = 2000 # [kg]
ms2 = 1500 # [kg]
ms3 = 1000 # [kg]

# Stage 1
m0 = mp1 + ms1 + mp2 + ms2 + mp3 + ms3 + mL  # [kg]
m1 = m0 - mp1  # [kg]

# Stage 2
m2 = ms2 + mp2 + ms3 + mp3 + mL  # [kg]
m3 = m2 - mp2  # [kg]

# Stage 3
m4 = ms3 + mp3 + mL  # [kg]
m5 = m4 - mp3  # [kg]

z1 = m0/m1
z2 = m1/m2
z3 = m2/m3

c = 9.81  # [m/s^2]
delta_v1 = 2*c*np.log(z1)
delta_v2 = 2*c*np.log(z2)
delta_v3 = 2*c*np.log(z3)
delta_v = delta_v1 + delta_v2 + delta_v3
print(f'Velocity change: {delta_v} [km/s]\n')