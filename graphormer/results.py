import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.calibration import CalibrationDisplay
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_conover_friedman
from sklearn.metrics import balanced_accuracy_score, f1_score
import os
import warnings


# Results dict: Dict[mc_sample, Dict[label, List[preds]]]
def save_results(results_dict: Dict[int, Dict[str, List[float] | int]], model_name: str, mc_samples) -> None:
    result_type = "mc_dropout" if mc_samples else "regular"
    global_results_path = "results"
    model_results_path = f"results/{model_name}_model"

    os.makedirs(model_results_path, exist_ok=True)

    results_df = pd.DataFrame(results_dict)
    # assert False, results_df

    results_df.to_pickle(f"{model_results_path}/{result_type}_preds.pkl")
    plot_calibration_curve(results_df, model_name, model_results_path, mc_samples)
    save_mc_bacs(results_df, model_name, global_results_path)

    ece = calculate_ece(results_df.T["label"], results_df.T["preds"].apply(lambda x: float(x[0])))
    print(f"Predictions, calibration curve, BACs, F1s, saved to {model_results_path}\n ECE: {ece:.3f}")


def calculate_ece(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges, right=True)

    ece = 0.0
    for i in range(1, n_bins + 1):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            bin_accuracy = np.mean(y_true[bin_mask])
            bin_confidence = np.mean(y_prob[bin_mask])
            bin_size = np.sum(bin_mask)
            ece += (bin_size / len(y_true)) * np.abs(bin_accuracy - bin_confidence)

    return ece


def plot_calibration_curve(df: pd.DataFrame, model_name: str, save_path: str, mc_dropout: bool = True) -> None:
    """
    Plots the calibration curve for a given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the calibration data.
        name (str): The name of the model.
        mc_dropout (bool, optional): Whether to include the median calibration curve and confidence intervals.
            Defaults to True.

    Returns:
        None
    """

    chart_type = "CI Calibration" if mc_dropout else "Calibration Curve"

    df = df.T
    df.label = df.label.astype(int)

    df = get_cis(df)

    # Generate the basic calibration curve
    fig, ax = plt.subplots()
    CalibrationDisplay.from_predictions(df["label"], df["median_preds"], ax=ax, strategy="uniform", name=model_name)

    if mc_dropout:
        CalibrationDisplay.from_predictions(df["label"], df["upper_ci"], ax=ax, strategy="uniform", name="Upper CI")
        CalibrationDisplay.from_predictions(df["label"], df["lower_ci"], ax=ax, strategy="uniform", name="Lower CI")

        lines = ax.get_lines()
        upper_ci_line, lower_ci_line = lines[2], lines[3]
        upper_ci_y, lower_ci_y, ci_x = upper_ci_line.get_ydata(), lower_ci_line.get_ydata(), lower_ci_line.get_xdata()

        for line in ax.get_lines():
            if line.get_label() == "Upper CI" or line.get_label() == "Lower CI":
                line.remove()

        assert (
            upper_ci_y.shape == lower_ci_y.shape == ci_x.shape
        ), f"upper_ci_y: {upper_ci_y} and lower_ci_y: {lower_ci_y} shapes do not match."
        ax.fill_between(ci_x, upper_ci_y, lower_ci_y, color="gray", alpha=0.2)

    plt.savefig(f"{save_path}/{chart_type}.png")
    print(f"Calibration curve saved to results/{save_path}/{chart_type}.png")


def save_mc_bacs(df: pd.DataFrame, model_name: str, global_results_path: str) -> None:
    """
    Save the balanced accuracy scores for each Monte Carlo (MC) sample to a CSV file.

    Args:
        df (pd.DataFrame): The input DataFrame containing the predictions and labels.
        model_name (str): The name of the model.
        global_results_path (str): The path to the directory containing the csv to which results will be appended.

    Returns:
        None
    """
    bac_list = []
    f1_list = []

    df = df.T
    bac_csv = os.path.join(global_results_path, "MC_BACs.csv")
    f1_csv = os.path.join(global_results_path, "MC_F1s.csv")

    preds_df = pd.DataFrame(df["preds"].tolist(), index=df.index)  # Preds_df has a col for each mc sample
    preds_df.columns = [f"mc_sample_{i+1}" for i in preds_df.columns]
    df = pd.concat([df, preds_df], axis=1)
    df.drop("preds", axis=1, inplace=True)

    mc_samples = len(df.columns) - 1

    for sample in range(mc_samples):
        mc_sample = df[f"mc_sample_{sample+1}"]
        prediction = [1 if x >= 0.5 else 0 for x in mc_sample]

        bac = balanced_accuracy_score(df["label"].astype(int), prediction)
        f1 = f1_score(df["label"].astype(int), prediction)

        bac_list.append(bac)
        f1_list.append(f1)

    bac_scores_df = pd.DataFrame(
        [bac_list], columns=[f"mc_sample_{i+1}" for i in range(mc_samples)], index=[model_name]
    )

    f1_scores_df = pd.DataFrame([f1_list], columns=[f"mc_sample_{i+1}" for i in range(mc_samples)], index=[model_name])

    update_or_create_csv(bac_csv, bac_scores_df, model_name)
    update_or_create_csv(f1_csv, f1_scores_df, model_name)


def friedman_from_bac_csv(
    bac_csv_path: str, models_to_friedman: list, alpha: float = 0.05
) -> Tuple[float, float, Optional[pd.DataFrame]]:
    bac_df = pd.read_csv(bac_csv_path, index_col=0)
    model_rows = {}

    for index, row in bac_df.iterrows():
        if index in models_to_friedman:
            model_rows[index] = row.tolist()

    stat, p = friedmanchisquare(*model_rows.values())

    print("Statistics=%.3f, p=%.3f" % (stat, p))

    if p < alpha:
        print("Different distributions (reject H0) - Performing post hoc test")
        posthoc_results_df = posthoc_conover_friedman(bac_df.T)
        return stat, p, posthoc_results_df
    else:
        print("Same distributions (fail to reject H0) - No need for post hoc")
        return stat, p


def get_cis(mcd_df: pd.DataFrame, confidence_level: float = 0.95):
    mcd_df["median_preds"] = mcd_df["preds"].apply(lambda x: np.median(x))
    mcd_df["lower_ci"] = mcd_df["preds"].apply(lambda x: np.percentile(x, (1 - confidence_level) / 2 * 100))
    mcd_df["upper_ci"] = mcd_df["preds"].apply(lambda x: np.percentile(x, (1 + confidence_level) / 2 * 100))

    return mcd_df


def update_or_create_csv(file_path: str, new_df: pd.DataFrame, model_name: str) -> None:
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path, index_col=0)
        if len(existing_df.columns) != len(new_df.columns):
            warnings.warn(
                f"Existing scores in {file_path} have {len(existing_df.columns)} MC_sample columns, while new scores from {model_name} have {len(new_df.columns)} MC_sample columns.",
                UserWarning,
            )
        print(f"Scores of {model_name} appended to {file_path}")
        updated_df = pd.concat([existing_df, new_df])
    else:
        print(f"{file_path} does not exist - Now creating")
        updated_df = new_df

    updated_df.to_csv(file_path)
