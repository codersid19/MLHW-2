# Load libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# given data
d1 = np.array([-1, -1])
d2 = np.array([-1, +1])
d3 = np.array([+1, -1])
d4 = np.array([+1, +1])

X = np.array([d1, d2, d3, d4])
Y = np.array([-1, 1, 1, -1])  # outputs, 1 = Positive, -1 = Negative

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.grid()

# plot each point in the input data
for i, x in enumerate(X):
    if Y[i] > 0:
        # positive points are colored green
        plt.plot(x[0], x[1], "go", label='Positive')
    else:
        # negative points are colored red
        plt.plot(x[0], x[1], "ro", label='Negative')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()


for i, x in enumerate(X):
    print(f'{x} -> {[x[0], x[0] * x[1]]}')
    X[i] = [x[0], x[0] * x[1]]
X


# plot each point in the input data
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.grid()

for i, x in enumerate(X):
    if Y[i] > 0:
        # positive points are colored green
        plt.plot(x[0], x[1], "go", label='Positive')
    else:
        # negative points are colored red
        plt.plot(x[0], x[1], "ro", label='Negative')

# plt.plot((-1, 1), (0, 0), c='green')
plt.axhline(y=0, color='b', linestyle='-', label='maximal margin separator')
handles, labels = plt.gca().get_legend_handles_labels() 
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()

# find the distance of the point from the line (a point on the separator line)
mid = (x+d2)/2 # find a point on the line on the separator
d = abs(np.cross(d2-d1,mid-d1)/np.linalg.norm(d2-d1))
print(f'The margin is {d}')