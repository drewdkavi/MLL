import random
import numpy as np

from Binary_SVM import SVMLinearOVR
import matplotlib.pyplot as plt

NUM_DATA_POINTS = 500
NUM_FEATURES = 2
EPSILON = 0.2

svm = SVMLinearOVR(input_dim=NUM_FEATURES, num_classes=4)

'''
Artificial Data set:
NUM_DATA_POINTS data points to train on, each of form (x1, x2, y):


x2
| xx / xxxx
|xx /xxxxxxx
|x /xx xxxxxx
| /     xxxx
|/______________: x1

'''
N = NUM_DATA_POINTS
xs = []  # : npt.NDArray[npt.NDArray[float]]
ys = []  # : npt.NDArray[int]
x1s = []
x2s = []
colours = []

to_colour = {0: 'red', 1: 'green', 2: 'blue', 3: 'purple'}
to_light_colour = {0: 'pink', 1: 'teal', 2: 'lightblue', 3: 'violet'}

for _ in range(N):

    y = random.choice([0, 1, 2, 3])
    if y == 0:
        ys.append(0)
        colours.append('red')
        x1_init = 0.25
        x2_init = 0.75

    elif y == 1:
        ys.append(1)
        colours.append('green')
        x1_init = 0.45
        x2_init = 0.25

    elif y == 2:
        ys.append(2)
        colours.append('blue')
        x1_init = 0.75
        x2_init = 0.45

    elif y == 3:
        ys.append(3)
        colours.append('purple')
        x1_init = 0.75
        x2_init = 0.8

    x1 = x1_init + random.uniform(-EPSILON, EPSILON)
    x2 = x2_init + random.uniform(-EPSILON, EPSILON)

    x1s.append(x1)
    x2s.append(x2)
    xs.append([x1, x2, 1])

# ys.append(1)
# colours.append('green')
# x1s.append(0.75)
# x2s.append(0.75)
# xs.append([0.75, 0.75, 1])

w = svm.train(xs, ys)

print(f"w[0]\n: {w[0]}")
print(f"w[1]\n: {w[1]}")
print(f"w[2]\n: {w[2]}")
print(f"w[3]\n: {w[3]}")

fig, ax = plt.subplots()


mesh_x1s = []
mesh_x2s = []
mesh_colours = []
for i in range(1000):
    for j in range(1000):

        x1ij = i/1000
        x2ij = j/1000
        point = [x1ij, x2ij, 1]
        mesh_x1s.append(x1ij)
        mesh_x2s.append(x2ij)
        mesh_colours.append(to_light_colour[svm.predict(np.array(point))])

ax.scatter(mesh_x1s, mesh_x2s, c=mesh_colours, s=1)

ax.scatter(x1s, x2s, c=colours, edgecolor='black')

# x1*w[0] + x2*w[1] + w[2] = 0
# => x2 = -1/w[1] * (x1*w[0] + w[2])


def separator(x1_dp, i):
    # print(f"Creating a separator for w[{i}] = {w[i]}")
    return (-1 / w[i][1]) * (x1_dp * w[i][0] + w[i][2])


def sepbin(x1_dp):
    return (-1 / w[1]) * (x1_dp * w[0] + w[2])


# separator_line_1 = [separator(x1, 0) for x1 in x1s]
# separator_line_2 = [separator(x1, 1) for x1 in x1s]
# separator_line_3 = [separator(x1, 2) for x1 in x1s]
# separator_line_4 = [separator(x1, 3) for x1 in x1s]
# ax.plot(x1s, separator_line_1, 'red')
# ax.plot(x1s, separator_line_2, 'green')
# ax.plot(x1s, separator_line_3, 'blue')
# ax.plot(x1s, separator_line_4, 'purple')

# plt.plot(x1s, [sepbin(x1) for x1 in x1s], 'purple')

plt.xlim(0, 1)
plt.ylim(0, 1)

x_dp = np.array([0.25, 0.75, 1])
print(svm.predict(x_dp))

def onclick(event):
    x, y = event.xdata, event.ydata
    if x is not None and y is not None:  # Check if the click was inside the axes
        print(f"{x}, {y} -> {to_colour[svm.predict(np.array([x, y, 1]))]}")


# Create a figure and an axis
# Connect the onclick event to the function
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()


x_test = []
y_test = []
for _ in range(100):

    y = random.choice([0, 1, 2, 3])
    if y == 0:
        y_test.append(0)
        x1_init = 0.25
        x2_init = 0.75

    elif y == 1:
        y_test.append(1)
        x1_init = 0.45
        x2_init = 0.25

    elif y == 2:
        y_test.append(2)
        x1_init = 0.75
        x2_init = 0.45

    elif y == 3:
        y_test.append(3)
        x1_init = 0.75
        x2_init = 0.8

    x1 = x1_init + random.uniform(-EPSILON, EPSILON)
    x2 = x2_init + random.uniform(-EPSILON, EPSILON)

    x_test.append([x1, x2, 1])

x_test = np.array(x_test)
y_test = np.array(y_test)

print(f"accuracy: {svm.test(x_test, y_test)}")





