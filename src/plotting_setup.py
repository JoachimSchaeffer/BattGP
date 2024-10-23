from typing import List

import matplotlib.pyplot as plt
import seaborn as sns


# Define colors from the sunlight foundation color palette
# https://sunlightfoundation.com/2013/05/01/visualizing-the-2013-federal-budget/
# Path: src/plotting_setup.py
def get_colors() -> List[str]:
    colors = [
        "#E3BA22",
        "#E6842A",
        "#137B80",
        "#8E6C8A",
        "#978F80",
    ]
    return colors


def get_special_colors() -> List[str]:
    colors = [
        "#9A3E25",
        "#E6842A",
        "#708259",
        "#5C8100",
        "#BD2D28",
        "#0F8C79",
    ]
    return colors


def get_many_colors() -> List[str]:
    # Combine two lists
    colors = get_colors() + get_special_colors()
    return colors


def setup_plots():
    sns.set(rc={"figure.figsize": (10, 10)})
    sns.set_style("white")
    # matplotlib.rcParams["figure.figsize"] = (7, 7)
    # matlibplot settings
    # plt.rcParams["xtick.labelsize"] = 15
    # plt.rcParams["ytick.labelsize"] = 15
    # plt.rcParams["axes.labelsize"] = 15
    # plt.rcParams["axes.titlesize"] = 15
    # rcParams legend fontsize
    # plt.rcParams["legend.fontsize"] = 14
    # plt.rcParams["axes.titleweight"] = "bold"
    # plt.rcParams["font.weight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=get_many_colors())
