
"""
Module containing function for creating plots related to churn modelling.
"""

# %% Import packages
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%

def line_with_dist_plot(x: pd.Series(), y: pd.Series(), unc: np.ndarray, x_vals: np.ndarray):
    """
    Plots a line plot, with uncertainty band. Furthermore plots a histogram of x-values on top of the line plot.
    """
    # Create joint grid:
    g = sns.JointGrid()
    # Remove marginal y plot
    g.ax_marg_y.remove()

    # line plot and confidence interval:
    g.ax_joint.plot(x, y, label='Prediction')
    g.ax_joint.fill_between(x=x, y1=unc[: ,0], y2=unc[: ,1], alpha=0.2, label="Prediction interval")

    # distribution plot:
    sns.histplot(x=x_vals ,ax=g.ax_marg_x)

    # Remove axes, change fonts, add grid and text.
    g.ax_joint.spines['left'].set_visible(False)
    g.ax_joint.spines['bottom'].set_visible(False)
    plt.yticks(fontfamily='Open Sans', fontsize=14)
    plt.xticks(fontfamily='Open Sans', fontsize=14)
    g.ax_joint.grid(alpha=0.3)
    plt.suptitle(x.name, fontproperties={'family': 'Ubuntu', "size": 20})
    g.ax_joint.set_ylabel(y.name)

    plt.tight_layout()
    plt.show()