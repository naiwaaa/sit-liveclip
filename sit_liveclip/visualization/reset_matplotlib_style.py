import matplotlib.pyplot as plt


def reset_matplotlib_style() -> None:
    # Style sheets:
    # https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
    plt.style.use("seaborn-paper")

    plt.rcParams["font.family"] = "serif"
