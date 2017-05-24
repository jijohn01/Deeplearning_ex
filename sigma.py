import numpy as math
import matplotlib.pyplot as plt

x = math.arange(0,6,0.1)
y = math.log(x)

plt.plot(x,y, linestyle="--", label="sin")
plt.xlabel("x")
plt.ylabel("y")
plt.title('sin&cos')
plt.show()

