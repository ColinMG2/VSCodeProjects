import matplotlib.pyplot as plt
import numpy as np

b = np.linspace(0, 10, 100)
a = 1
c = 1
P1_real = np.zeros(len(b))
P1_imag = np.zeros(len(b))
P2_real = np.zeros(len(b))
P2_imag = np.zeros(len(b))

for i in range(len(b)):
    P1 = (-b[i] + np.sqrt(b[i]**2 - 4*a*c))/(2*a)
    P2 = (-b[i] - np.sqrt(b[i]**2 - 4*a*c))/(2*a)
    P1_real[i] = P1.real
    P1_imag[i] = P1.imag
    P2_real[i] = P2.real
    P2_imag[i] = P2.imag

plt.figure()
plt.plot(P1_real, P1_imag, label='P1')
plt.plot(P2_real, P2_imag, label='P2')
plt.xlabel('Sigma')
plt.ylabel('j Omega')
plt.title('Pole Locations')
plt.legend()
plt.grid()
plt.show()
