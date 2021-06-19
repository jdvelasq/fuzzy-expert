import matplotlib.pyplot as plt


def format_plot(title=None, view_xaxis=True, view_yaxis=False):

    plt.gca().set_ylim(-0.05, 1.05)

    plt.gca().spines["bottom"].set_visible(True)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    plt.gca().spines["bottom"].set_color("gray")

    plt.gca().get_yaxis().set_visible(False)
    if view_yaxis == "left":
        plt.gca().get_yaxis().set_visible(True)
    if view_yaxis == "right":
        plt.gca().get_yaxis().set_visible(True)
        plt.gca().yaxis.tick_right()

    plt.gca().get_xaxis().set_visible(view_xaxis)

    if title is not None:
        plt.gca().set_title(title)


def plot_fuzzyvariable(
    universe, memberships, labels, title, fmt, linewidth, view_xaxis, view_yaxis
):
    #
    for label, membership in zip(labels, memberships):
        plt.gca().plot(universe, membership, fmt, label=label, linewidth=linewidth)
    plt.gca().legend()
    #
    format_plot(
        title=title,
        view_xaxis=view_xaxis,
        view_yaxis=view_yaxis,
    )

    # plt.gca().spines["left"].set_color("lightgray")
