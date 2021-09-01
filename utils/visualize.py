# Adapted from score written by KellerJordan
# https://github.com/KellerJordan/tSNE-Animation

from __future__ import print_function

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'tab20'
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches


def init_plot(classes):
    mpl.use('Agg')
    cmap = plt.get_cmap(lut=len(classes))
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)

    # fig.patch.set_facecolor((234 / 255, 234 / 255, 242 / 255))
    # fig.patch.set_alpha(0.7)

    ax.axis("off")
    # ax.patch.set_facecolor('orange')
    # ax.patch.set_alpha(0.1)

    patches = [mpatches.Patch(color=cmap(idx), label=name)
               for idx, name in enumerate(classes)]
    return fig, ax, patches


def savegif(Y_seq, labels, fig_name, path, classes, limits=None):
    fig, ax, patches = init_plot(classes)

    def init():
        return scatter,

    def update(i):

        if (i + 1) % 50 == 0:
            print('[%d / %d] Animating frames' % (i + 1, len(Y_seq)))
        ax.clear()
        ax.axis("off")
        if limits is not None:
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
        # plt.legend(handles=patches, prop={"size": 12}, loc='upper right')
        plt.legend(handles=patches, prop={"size": 14}, bbox_to_anchor=(
            0., -.12, 1., 0.), loc='lower left', ncol=5, mode="expand", borderaxespad=0.)
        ax.scatter(Y_seq[i][:, 0], Y_seq[i][:, 1], s=2, c=labels)
        ax.set_title('%s (epoch %d)' %
                     (fig_name, i), fontdict={'fontsize': 18})
        return ax, scatter

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(Y_seq), interval=50)
    print('[*] Saving animation as %s' % path)
    anim.save(path, writer='imagemagick', fps=30)


def savepng(Y, labels, fig_name, path):
    fig, ax, patches = init_plot()
    ax.scatter(Y[:, 0], Y[:, 1], 1, labels)
    ax.set_title(fig_name)
    print('[*] Saving figure as %s' % path)
    plt.savefig(path)


def scatter(Y, labels):
    fig, ax, patches = init_plot()
    ax.scatter(Y[:, 0], Y[:, 1], 1, labels)
    plt.show()


def plot_2d(features, labels, classes, legend=True):

    import matplotlib.patches as mpatches

    font = {'font.family': 'Times New Roman', 'font.size': 14}
    plt.rcParams.update(**font)

    cmap = plt.get_cmap('tab20', lut=len(classes))

    fig, ax = plt.subplots()

    fig.set_figheight(10)
    fig.set_figwidth(10)
    patches = [mpatches.Patch(color=cmap(idx), label=name)
               for idx, name in enumerate(classes)]

    ax.scatter(features[:, 0], features[:, 1], 50, labels, cmap=cmap)
    if legend:
        # ax.legend(handles=patches, loc='upper right')
        ax.legend(handles=patches, prop={"size": 14}, bbox_to_anchor=(
            0., -.12, 1., 0.), loc='lower left', ncol=5, mode="expand", borderaxespad=0.)
    plt.show()
