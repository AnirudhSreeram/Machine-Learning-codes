import matplotlib.pypot as plt
import numpy as np

h = []
p = -10
till = 20
while p <= till :
    function = -(1/3)*p**2 + p + (1/5)
    h.append(function)
    p += 0.1
plt.plot(h,label='linear')
plt.show()