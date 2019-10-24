import numpy as np
import matplotlib.pyplot as plt


def model(w0, w1):
    t = np.arange(0.0, 30.0, 0.01)
    s = t*w0 + w1
    plt.plot(t, s)


file = np.loadtxt(".\\Data\\ex1data1.txt", delimiter=',')
x = file[:, 0]
y = file[:, 1]

plt.scatter(x, y)
plt.title("Scatter Plot for City population and profit")
plt.xlabel("City Population")
plt.ylabel("Profit")
ymin, ymax = min(y), max(y)
plt.ylim(ymin, 1.05 * ymax)

plt.show()

ws = np.loadtxt(".\\Data\\ws.txt", delimiter=' ')
for w in ws:
    plt.scatter(x, y)
    plt.title("Scatter Plot for City population and profit")
    plt.xlabel("City Population")
    plt.ylabel("Profit")
    ymin, ymax = min(y), max(y)
    plt.ylim(ymin, 1.05 * ymax)

    model(w[0], w[1])
    plt.show()


plt.show()