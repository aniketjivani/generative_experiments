# toy reproductions of unusual / interesting / sometimes informative plots seen in the wild.

# https://matplotlib.org/stable/gallery/color/named_colors.html

import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
def plot_colortable(colors, *, ncols=4, sort_colors=True):

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        names = sorted(
            colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
    else:
        names = list(colors)

    n = len(names)
    nrows = math.ceil(n / ncols)

    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-margin)/height)
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )

    return fig
xkcd_fig = plot_colortable(mcolors.XKCD_COLORS)
# xkcd_fig.savefig("XKCD_Colors.png")

# %%
# cool rainbow line plot based on the talk: https://youtu.be/Ums_VKKf_s4?si=G8av_yBeN7XWgqNW&t=1438

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Generate sample data
np.random.seed(42)
x = np.linspace(0, 2000, 20)
# y = 0.4 * (1 - np.exp(-x/400)) + 0.95 * (1 - np.exp(-x/1000))
y = np.sin(x/100) * np.exp(-2 * x / 1000)
confidence = 0.1 * np.ones_like(x)  # Constant confidence interval width

# Create figure
plt.figure(figsize=(10, 6))

# Create custom colormap that transitions through multiple colors
# colors = ['blue', 'cyan', 'yellow', 'red']

# change colors to rainbow colors
colors=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']

n_bins = 1000
custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

n_points = 200
x_smooth = np.linspace(x.min(), x.max(), n_points)
y_smooth = np.interp(x_smooth, x, y)

conf_smooth = np.interp(x_smooth, x, confidence)

# Plot the confidence interval with color gradient
for i in range(n_points-1):
    # Calculate color for this segment
    # color = custom_cmap(i / (len(x)-1))
    color = custom_cmap((x_smooth[i] - x[0]) / (x[-1] - x[0]))
    
    plt.fill_between(
        x_smooth[i:i+2],
        y_smooth[i:i+2] - conf_smooth[i:i+2],
        y_smooth[i:i+2] + conf_smooth[i:i+2],
        color=color,
        alpha=0.4,
        edgecolor='none'
    )
    # Plot the line segment with the same color
    plt.plot(
        x_smooth[i:i+2],
        y_smooth[i:i+2],
        color=color,
        linewidth=2.5
    )

for i in range(len(x)):
    color = custom_cmap((x[i] - x[0]) / (x[-1] - x[0]))
    plt.plot(x[i], y[i], marker='o', markersize=15, color=color)

# Customize the plot
plt.title('Random Rainbow Plot', fontsize=14)
plt.xlabel('x')
plt.ylabel('y')
# plt.ylim(0, 1.0)
plt.grid(True, alpha=0.3)

plt.legend()

plt.tight_layout()
plt.show()
# %%
