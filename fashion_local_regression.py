#!/usr/bin/env python3
"""
Local Regression Analysis on Fashion Trend Data

This script generates synthetic fashion trend data and applies a local regression
(smoothing) using LOWESS (Locally Weighted Scatterplot Smoothing) to reveal the underlying trend.
It then plots the original data alongside the smoothed curve.

"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

def generate_data():
    # Generate synthetic data representing fashion trends over time.
    # x represents time (in months) and y represents a fashion trend indicator.
    np.random.seed(42)
    x = np.linspace(0, 24, 100)  # Data over 24 months
    # Create a cyclic trend (e.g., seasonal effects) with added noise
    y = 10 + 2 * np.sin(2 * np.pi * x / 12) + np.random.normal(scale=1, size=x.shape)
    return x, y

def perform_lowess(x, y, frac=0.2):
    """
    Apply LOWESS (Locally Weighted Scatterplot Smoothing) to the data.

    Parameters:
    x (np.array): Independent variable (e.g., time, month).
    y (np.array): Dependent variable (e.g., fashion trend indicator).
    frac (float): The fraction of the data used when estimating each y-value.

    Returns:
    np.array: Smoothed data as a Nx2 array where the first column is x and the second is the smoothed y.
    """
    smoothed = lowess(y, x, frac=frac)
    return smoothed

def plot_results(x, y, smoothed):
    # Plot original data and LOWESS smoothed curve
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Original Data', color='blue', alpha=0.5)
    plt.plot(smoothed[:, 0], smoothed[:, 1], label='Lowess Smoothed', color='red', linewidth=2)
    plt.xlabel('Month')
    plt.ylabel('Fashion Trend Indicator')
    plt.title('Local Regression (LOWESS) on Fashion Trend Data')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    print("\nExplanation:")
    print("The blue points represent the original synthetic fashion trend data (with noise and seasonal effects).")
    print("The red line is the LOWESS smoothed curve, which reveals the underlying trend by smoothing the data,")
    print("making it easier to interpret the overall trend and patterns in fashion over time.")
    x, y = generate_data()
    smoothed = perform_lowess(x, y, frac=0.2)
    plot_results(x, y, smoothed)
if __name__ == "__main__":
    main()
