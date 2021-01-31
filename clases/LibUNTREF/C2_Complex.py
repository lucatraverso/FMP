"""
Module: LibUNTREF.C2_Complex
"""
import numpy as np
from matplotlib import pyplot as plt

def generate_figure(figsize=(2,2),xlim=[0,1],ylim=[0,1]):
    """Generate figure for plotting complex numbers"""
    plt.figure(figsize=figsize)
    plt.grid()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('$\mathrm{Re}$')
    plt.ylabel('$\mathrm{Im}$');

def plot_vector(c,color='k',start=0,linestyle='-'):
    """Plot arrow corresponding to difference between two complex numbers
    Args:
        c: Complex number
        color: Color of arrow
        start: Complex number encoding the start position
        linestyle: Linestyle of arrow
    Returns:
        plt.arrow: matplotlib.patches.FancyArrow
    """
    return plt.arrow(np.real(start),np.imag(start),np.real(c),np.imag(c),
              linestyle=linestyle,head_width=0.05,fc=color,ec=color,overhang=0.3,
              length_includes_head=True) 
