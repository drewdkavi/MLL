import numpy as np
import matplotlib.pyplot as plt


def get_extra_2d_features(x):
    x1, x2 = x[0], x[1]
    x1_sqr, x2_sqr = x1 ** 2, x2 ** 2
    x1_sin, x2_sin = np.sin(x1), np.sin(x2)
    return np.array([x1, x2, x1_sqr, x2_sqr, x1_sin, x2_sin])


def map_extra_2d_features(xs):
    mapping = get_extra_2d_features
    return np.apply_along_axis(mapping, 1, xs)


def plot_2d_predictions(model, x_train, y_train, x1_min, x1_max, x2_min, x2_max, extras: bool = False, add_extra_dim: bool = False, mesh_points: int = 200) -> None:

    # TODO: Add the rest of these
    to_colour = {0: 'red', 1: 'green', 2: 'blue', 3: 'purple'}
    to_light_colour = {0: 'pink', 1: 'teal', 2: 'lightblue', 3: 'violet'}

    train_x1s = x_train[:, 0:1]
    train_x2s = x_train[:, 1:2]
    train_colours = [to_colour[ydp] for ydp in y_train]

    MESH_SIZE = mesh_points

    xs = np.zeros(MESH_SIZE * MESH_SIZE, dtype=object)
    k = 0
    delta_x = (x1_max - x1_min) / MESH_SIZE
    delta_y = (x2_max - x2_min) / MESH_SIZE

    mesh_x1s, mesh_x2s = [], []
    mesh_colours = []

    if extras:
        for i in range(MESH_SIZE):
            for j in range(MESH_SIZE):
                x = x1_min + i * delta_x
                y = x2_min + j * delta_y
                x_sqr = x ** 2
                y_sqr = y ** 2
                x_sin = np.sin(x)
                y_sin = np.sin(y)
                mesh_x1s.append(x)
                mesh_x2s.append(y)
                mesh_colours.append(to_light_colour[model.predict(np.array([x, y, x_sqr, y_sqr, x_sin, y_sin]))])
                xs[k] = np.array([x, y, x_sqr, y_sqr, x_sin, y_sin])

                def onclick(event):
                    x, y = event.xdata, event.ydata
                    if x is not None and y is not None:  # Check if the click was inside the axes
                        print(f"{x}, {y} -> {to_colour[model.predict(np.array([x, y, x**2, y**2, np.sin(x), np.sin(y)]))]}")
    elif add_extra_dim:
        for i in range(MESH_SIZE):
            for j in range(MESH_SIZE):
                x = x1_min + i * delta_x
                y = x2_min + j * delta_y
                mesh_x1s.append(x)
                mesh_x2s.append(y)
                mesh_colours.append(to_light_colour[model.predict(np.array([x, y, 1]))])
                xs[k] = np.array([x, y, 1])

                def onclick(event):
                    x, y = event.xdata, event.ydata
                    if x is not None and y is not None:  # Check if the click was inside the axes
                        print(f"{x}, {y} -> {to_colour[model.predict(np.array([x, y, 1]))]}")

    else:
        for i in range(MESH_SIZE):
            for j in range(MESH_SIZE):
                x = x1_min + i * delta_x
                y = x2_min + j * delta_y
                point = np.array([x, y])
                xs[k] = point
                mesh_x1s.append(x)
                mesh_x2s.append(y)
                mesh_colours.append(to_light_colour[model.predict(np.array([x, y]))])

                def onclick(event):
                    x, y = event.xdata, event.ydata
                    if x is not None and y is not None:  # Check if the click was inside the axes
                        print(f"{x}, {y} -> {to_colour[model.predict(np.array([x, y]))]}")

    fig, ax = plt.subplots()
    ax.scatter(mesh_x1s, mesh_x2s, c=mesh_colours, s=0.25)

    ax.scatter(train_x1s, train_x2s, c=train_colours)

    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)

    # Create a figure and an axis
    # Connect the onclick event to the function
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()
