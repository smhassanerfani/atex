import matplotlib.pyplot as plt


def plot_2d(features, labels, classes):

    import matplotlib.patches as mpatches

    cmap = plt.get_cmap('tab20c', lut=len(classes))

    fig, ax = plt.subplots()

    fig.set_figheight(10)
    fig.set_figwidth(10)
    patches = [mpatches.Patch(color=cmap(idx), label=name)
               for idx, name in enumerate(classes)]

    ax.scatter(features[:, 0], features[:, 1], 50, labels, cmap=cmap)
    ax.legend(handles=patches, loc='upper right')
    plt.show()
