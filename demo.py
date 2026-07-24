import numpy as np 
import matplotlib.pyplot as plt 

x = np.linspace(-np.pi, np.pi, 100)
y1= np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(8,6))
plt.plot(x, y1, color ='red', linewidth=2, label='sine')
plt.plot(x,y2, color='blue',linewidth=2, label='cosine')

plt.xlabel('x')
plt.ylabel('y=f(x)')
plt.title('Sine and Cosine functions')
plt.legend()
plt.show()

print('Executed')