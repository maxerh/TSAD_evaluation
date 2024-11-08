import pandas
import argparse
import numpy as np
import pandas as pd
from tabulate import tabulate
from combine_csvs import get_combined_data


def get_p_r_f1_auc(elements_count: list):
    """
    Calculate precision, recall, f1, and auc scores
    :param elements_count: list with scores
    :return:
    """
    tp, tn, fp, fn = elements_count
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    tnr = tn / (tn + fp)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    auc = 0.5 - 0.5 * fpr + 0.5 * tpr
    return precision, recall, f1, auc


def get_scores(df: pd.DataFrame):
    """
    Calculate true positives, true negatives, false positives, false negatives
    :param df: dataframe with scores
    :return: list with scores
    """
    tp = df["TP"].sum()
    tn = df["TN"].sum()
    fp = df["FP"].sum()
    fn = df["FN"].sum()
    return [tp, tn, fp, fn]


def get_total_f1(df: pd.DataFrame, datasets: list, algorithms: list, window_sizes: list):
    """
    Get the combined F1 scores for datasets and algorithms
    :param df: DataFrame containing all results
    :param datasets: list of dataset names
    :param algorithms: list of algorithm names
    :param window_sizes: list of window sizes
    :return: list with f1 scores
    """
    f1_total = []
    for algo in algorithms:
        df_algo = df.loc[df["algorithm"] == algo]  # filter algorithm
        df_dataset = df_algo.loc[df_algo["dataset"].isin(datasets)]  # filter dataset
        for s_w in window_sizes:
            df_s_w = df_dataset.loc[df_dataset["seq_len"] == s_w]  # filter window_sizes
            p, r, f1, auc = get_p_r_f1_auc(get_scores(df_s_w))
            f1_total.append(f1)
    return [round(elem, 4) for elem in f1_total]


def evaluate(filename: str, algorithms: list, datasets: list, printformat: str):
    """
    Evaluate the scores for given datasets and algorithms
    :param filename: filename of the csv file with the results
    :param algorithms: list of algorithm names
    :param datasets: list of dataset names
    :param printformat: table format for tabulate table
    """
    df = pandas.read_csv(filename)  # combined results
    f1_dict_summary = {f"f1_{dataset}": [] for dataset in datasets}
    window_sizes = sorted(df["seq_len"].unique().tolist())  # get all available window sizes

    for algo in algorithms:
        df_algo = df.loc[df["algorithm"] == algo]  # filter algorithm

        # init dicts
        names_dict = {"": [f"{algo}_{s}" for s in window_sizes]}
        P_dict = {f"P-{ds}": [] for ds in datasets}
        R_dict = {f"R-{ds}": [] for ds in datasets}
        F1_dict = {f"F1-{ds}": [] for ds in datasets}
        AUC_dict = {f"ROC/AUC-{ds}": [] for ds in datasets}

        for i, s_w in enumerate(window_sizes):
            df_s_w = df_algo.loc[df_algo["seq_len"] == s_w]  # filter window_sizes
            for dataset in datasets:
                df_dataset = df_s_w.loc[df_s_w["dataset"] == dataset]  # filter dataset
                p, r, f1, auc = get_p_r_f1_auc(get_scores(df_dataset))
                P_dict[f"P-{dataset}"].append(p)
                R_dict[f"R-{dataset}"].append(r)
                F1_dict[f"F1-{dataset}"].append(f1)
                AUC_dict[f"ROC/AUC-{dataset}"].append(auc)
                f1_dict_summary[f"f1_{dataset}"].append(f1)

        # combine dictionaries
        for p, r, f, a, f1ds in zip(P_dict, R_dict, F1_dict, AUC_dict, f1_dict_summary):
            names_dict[p] = [round(elem, 4) for elem in P_dict[p]]
            names_dict[r] = [round(elem, 4) for elem in R_dict[r]]
            names_dict[f] = [round(elem, 4) for elem in F1_dict[f]]
            names_dict[a] = [round(elem, 4) for elem in AUC_dict[a]]
            f1_dict_summary[f1ds] = [round(elem, 4) for elem in f1_dict_summary[f1ds]]

        if printformat == "latex":
            for k in names_dict.keys():
                if "F1" in k or "ROC" in k:
                    best_idx = np.argmax(names_dict[k])
                    names_dict[k][best_idx] = r"textbf{{{}}}".format(names_dict[k][best_idx])

        print(tabulate(names_dict, headers="keys", tablefmt=printformat))

    names_dict = {"SUMMARY": [f"{a}_{s}" for a in algorithms for s in window_sizes]}
    f1_dict_summary["f1_total"] = get_total_f1(df, datasets, algorithms, window_sizes)

    for f in f1_dict_summary:
        best_idx = np.argmax(f1_dict_summary[f])
        if printformat == "latex":
            f1_dict_summary[f][best_idx] = r"textbf{{{}}}".format(f1_dict_summary[f][best_idx])
        names_dict[f] = f1_dict_summary[f]

    print(tabulate(names_dict, headers="keys", tablefmt=printformat))

def main(args):
    printformat = args.format
    filename = args.output_filename
    algorithms = args.algorithms.split("_")
    datasets = args.datasets.split("_")
    input_path = args.input_path
    get_combined_data(input_path, filename)
    evaluate(filename, algorithms, datasets, printformat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-out", "--output_filename",
                        help="output filename with all results",
                        default="results.csv",
                        type=str)
    parser.add_argument("-a", "--algorithms",
                        help="algorithms to evaluate. Multiple algorithms are separated by '_' in a single string (algo1_algo2_algo3)",
                        default="TranAD_AnomalyTransformer_OmniAnomaly_TimesNet_Swin1D",
                        type=str)
    parser.add_argument("-d", "--datasets",
                        help="datasets to evaluate. Multiple datasets are separated by '_' in a single string (data1_data2_data3)",
                        default="SMD_PSM_SWaT_WADI_MSL_SMAP",
                        type=str)
    parser.add_argument("-f", "--format",
                        help="format of table in output",
                        default="grid",
                        type=str)
    parser.add_argument("-in", "--input_path",
                        help="format of table in output",
                        default="all_csvs",
                        type=str)
    args = parser.parse_args()
    main(args)
