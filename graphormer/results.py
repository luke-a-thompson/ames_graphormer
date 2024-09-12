import pandas as pd
from typing import List, Dict, Tuple, Optional, TypedDict, NotRequired, Any
from sklearn.calibration import CalibrationDisplay
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_conover_friedman
from sklearn.metrics import balanced_accuracy_score, f1_score
import os
import warnings
import json


class ResultsDict(TypedDict):
    labels: List[int]
    logits: List[float]
    preds: List[int]
    bac: float
    f1: float
    ece: float


class MCResultsDict(TypedDict):
    labels: List[int]
    median_logits: List[np.floating[Any]]

    upper_ci_bac: np.floating[Any]
    median_bac: np.floating[Any]
    lower_ci_bac: np.floating[Any]

    upper_ci_f1: np.floating[Any]
    median_f1: np.floating[Any]
    lower_ci_f1: np.floating[Any]

    upper_ci_ece: np.floating[Any]
    median_ece: np.floating[Any]
    lower_ci_ece: np.floating[Any]


def generate_results_dict(results: Dict[str, List[float]]) -> ResultsDict:
    preds = [1 if x >= 0.5 else 0 for x in results["logits"]]

    results_dict: ResultsDict = {
        "labels": [int(label) for label in results["labels"]],
        "preds": preds,
        "logits": results["logits"],
        "bac": balanced_accuracy_score(results["labels"], preds),
        "f1": f1_score(results["labels"], preds),  # type: ignore
        "ece": calculate_ece(results["labels"], preds),
    }

    return results_dict


def save_results(
    results: List[Dict[str, List[float]]] | Dict[str, List[float]], model_name: str, mc_samples: Optional[int] = None
) -> None:
    if mc_samples is not None and isinstance(results, list):
        assert len(results) == mc_samples, "Results length does not match MC samples length"
        model_results_path = f"results/{model_name}_model_{mc_samples}MC_samples"
        os.makedirs(model_results_path, exist_ok=True)

        mc_sample_results: list[ResultsDict] = []
        mc_sample_results_dict: dict[str, dict[str, float]] = {}
        for mc_sample in range(len(results)):
            sample_results: ResultsDict = generate_results_dict(results[mc_sample])

            mc_sample_results.append(sample_results)

            filtered_results = {"bac": sample_results["bac"], "f1": sample_results["f1"], "ece": sample_results["ece"]}
            mc_sample_results_dict[f"mc_sample_{mc_sample}"] = filtered_results

        assert len(mc_sample_results) == len(results), "Computed results length does not match input results length"

        num_datapoints = len(mc_sample_results[0]["logits"])
        median_logits = []
        for i in range(num_datapoints):
            logits_across_samples = [sample["logits"][i] for sample in mc_sample_results]

            median_logits.append(np.median(logits_across_samples))

        # Construct the final results dictionary
        mc_results: MCResultsDict = {
            "labels": mc_sample_results[0]["labels"],  # Labels are the same across all MC samples
            "median_logits": median_logits,
            "upper_ci_bac": np.percentile([sample["bac"] for sample in mc_sample_results], 97.5),
            "median_bac": np.median([sample["bac"] for sample in mc_sample_results]),
            "lower_ci_bac": np.percentile([sample["bac"] for sample in mc_sample_results], 2.5),
            "upper_ci_f1": np.percentile([sample["f1"] for sample in mc_sample_results], 97.5),
            "median_f1": np.median([sample["f1"] for sample in mc_sample_results]),
            "lower_ci_f1": np.percentile([sample["f1"] for sample in mc_sample_results], 2.5),
            "upper_ci_ece": np.percentile([sample["ece"] for sample in mc_sample_results], 97.5),
            "median_ece": np.median([sample["ece"] for sample in mc_sample_results]),
            "lower_ci_ece": np.percentile([sample["ece"] for sample in mc_sample_results], 2.5),
        }

        assert abs(mc_results["median_bac"] - mc_sample_results[0]["bac"]) < 1, "BACs are not close"

        save_mc_sample_subset = {
            k: v for k, v in mc_sample_results_dict.items() if k not in ["labels", "logits", "preds"]
        }
        save_mc_results_subset = {k: v for k, v in mc_results.items() if k not in ["labels", "median_logits"]}

        fig, ax = plt.subplots()
        CalibrationDisplay.from_predictions(
            mc_results["labels"], mc_results["median_logits"], ax=ax, strategy="uniform", name=model_name
        )

        plt.savefig(f"{model_results_path}/Calibration_Curve.png")

        with open(f"{model_results_path}/MC_Results.json", "w") as json_file:
            json.dump(save_mc_results_subset, json_file, indent=4)
        with open(f"{model_results_path}/MC_Results_All_Samples.json", "w") as json_file:
            json.dump(save_mc_sample_subset, json_file, indent=4)

        print(
            f"Median BAC: {mc_results['median_bac']:.3f}, Median F1: {mc_results['median_f1']:.3f}, Median ECE: {mc_results['median_ece']:.3f}.\nFurther results saved to {model_results_path}"
        )

    elif mc_samples is None and isinstance(results, dict):
        model_results_path = f"results/{model_name}_model"
        os.makedirs(model_results_path, exist_ok=True)

        results_dict: ResultsDict = generate_results_dict(results)

        save_mc_resultsdict = {
            "BAC": results_dict["bac"],
            "F1": results_dict["f1"],
            "ECE": results_dict["ece"],
        }

        fig, ax = plt.subplots()
        CalibrationDisplay.from_predictions(
            results_dict["labels"], results_dict["logits"], ax=ax, strategy="uniform", name=model_name
        )

        plt.savefig(f"{model_results_path}/Calibration_Curve.png")
        with open(f"{model_results_path}/Results.json", "w") as json_file:
            json.dump(save_mc_resultsdict, json_file, indent=4)

        print(
            f"BAC: {results_dict['bac']:.3f}, F1: {results_dict['f1']:.3f}, ECE: {results_dict['ece']:.3f}.\nFurther results saved to {model_results_path}"
        )


def calculate_ece(y_true, y_prob, n_bins=10) -> float:
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    assert len(y_true) == len(y_prob), "y_true and y_prob must have the same length"
    assert np.all((y_prob >= 0) & (y_prob <= 1)), "y_prob values must be between 0 and 1"

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
