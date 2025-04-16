import matplotlib.pyplot as plt
import numpy as np
from control import tf

G = 30 * tf([1], [1, 10, 30])
print(G)