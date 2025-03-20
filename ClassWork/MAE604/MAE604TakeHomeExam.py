import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import chardet

# Checling the encoding of the csv file
with open('Lagrange Sol.csv', 'rb') as f:
    result = chardet.detect(f.read())  
    print(result['encoding'])

# Reading the data from the csv file and printing the columns
data = pd.read_csv('Lagrange Sol.csv', encoding='windows-1252')
print(data)
print(data.columns)

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