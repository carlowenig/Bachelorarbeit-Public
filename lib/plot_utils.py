import locale
import math
import matplotlib.pyplot as plt
import numpy as np

from .discretization import x_vals

# Set to German locale to get comma decimal separater
locale.setlocale(locale.LC_NUMERIC, "de_DE")

# Pretty matplotlib plots
plt.rcParams.update(
    {
        "figure.dpi": 200,
        "figure.figsize": (7, 4),
        "font.family": "serif",
        "font.serif": "cm",
        "mathtext.fontset": "cm",
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{icomma}",
        "axes.grid": True,
        "axes.labelsize": 12,
        "axes.formatter.use_locale": True,
        "grid.linewidth": 0.2,
        "grid.alpha": 0.5,
        "lines.linewidth": 1,
        "lines.dashed_pattern": (6, 4),
    }
)


def scatter_complex(x, y, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.scatter(x, np.real(y), s=2, c="seagreen", label="Realteil")
    ax.scatter(x, np.imag(y), s=2, c="dodgerblue", label="Imaginärteil")
    ax.legend()


def colorize(iterable, cmap="tab10", *, continuous=False):
    if isinstance(cmap, str):
        cmap_fn = plt.get_cmap(cmap)
    elif isinstance(cmap, list):
        cmap_fn = lambda x: cmap[x]
    else:
        raise TypeError(f"cmap must be str or list, not {type(cmap)}.")

    max_i = len(iterable) - 1
    for i, element in enumerate(iterable):
        color = cmap_fn(i / max_i) if continuous else cmap_fn(i)
        yield (color, element)


def colorize_plots(iterable, cmap_name="tab10", *, continuous=False):
    with plt.rc_context():
        for color, element in colorize(iterable, cmap_name, continuous=continuous):
            plt.rcParams["lines.color"] = color
            yield element


def round_up(value, decimals=0):
    return math.ceil(value * 10 ** decimals) / 10 ** decimals


def fmt(value, format="%.3g"):
    if isinstance(format, int):
        decimals = format
        format = f"%.{decimals}f"
    elif isinstance(format, str) and format.endswith("f"):
        decimals = int(format[2:-1])
    else:
        decimals = None

    if isinstance(value, complex):
        return fmt(value.real, format) + " + " + fmt(value.imag, format) + "i"
    elif isinstance(value, tuple) and len(value) == 2:
        num, err = value
        if decimals is not None:
            err = round_up(err, decimals)
        return fmt(num, format) + " +/- " + fmt(err, format)
    return (format % value).replace(".", ",")


def fmt_tex(value, format="%.3g"):
    if isinstance(value, complex):
        return fmt_tex(value.real, fmt) + " + " + fmt_tex(value.imag) + "i"
    elif isinstance(value, tuple) and len(value) == 2:
        num, err = value
        return fmt_tex(num, format) + " \\pm " + fmt_tex(err, format)

    s = (format % value).replace(".", "{,}")
    if "e" in s:
        s = s.replace("e", r"\cdot 10^{") + "}"
    return s


def plot_phi(phi):
    x = x_vals(len(phi)).real
    scatter_complex(x, phi)
    plt.xlabel("$x_k$")
    plt.ylabel("$\\phi_k$")
