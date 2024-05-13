import pandas as pd
from sklearn.calibration import CalibrationDisplay
import numpy as np
import matplotlib.pyplot as plt


def plot_calibration_curve(df: pd.DataFrame, mc_dropout: bool = True):
    name = "Median Calibration" if mc_dropout else "Calibration Curve"

    df = df.transpose()
    df.label = df.label.astype(int)

    df = get_cis(df)

    # Generate the basic calibration curve
    fig, ax = plt.subplots()
    CalibrationDisplay.from_predictions(df["label"], df["median_preds"], ax=ax, strategy="uniform", name=name)

    if mc_dropout:
        CalibrationDisplay.from_predictions(df["label"], df["upper_ci"], ax=ax, strategy="uniform", name="Upper CI")
        CalibrationDisplay.from_predictions(df["label"], df["lower_ci"], ax=ax, strategy="uniform", name="Lower CI")

        lines = ax.get_lines()
        upper_ci_line, lower_ci_line = lines[2], lines[3]
        upper_ci_y, lower_ci_y, ci_x = upper_ci_line.get_ydata(), lower_ci_line.get_ydata(), lower_ci_line.get_xdata()

        for line in ax.get_lines():
            if line.get_label() == "Upper CI" or line.get_label() == "Lower CI":
                line.remove()

        ax.fill_between(ci_x, upper_ci_y, lower_ci_y, color="gray", alpha=0.2)

    plt.savefig("results/calibration_curve.png")


def get_cis(mcd_df: pd.DataFrame, confidence_level: float = 0.95):
    mcd_df["median_preds"] = mcd_df["preds"].apply(lambda x: np.median(x))
    mcd_df["lower_ci"] = mcd_df["preds"].apply(lambda x: np.percentile(x, (1 - confidence_level) / 2 * 100))
    mcd_df["upper_ci"] = mcd_df["preds"].apply(lambda x: np.percentile(x, (1 + confidence_level) / 2 * 100))

    return mcd_df
