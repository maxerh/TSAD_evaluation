import os
import glob
import pandas as pd


def get_combined_data(input_folder, out_filename):
    """
    Combine data of different csv files
    :param input_folder: folder name with different csv files
    :param out_filename: the filename to save the combined data
    :return:
    """
    csvs = glob.glob(f"{input_folder}/*.csv")

    data = {
        "algorithm": [],
        "dataset": [],
        "entity": [],
        "seq_len": [],
        "TP": [],
        "TN": [],
        "FP": [],
        "FN": [],
    }

    for file in csvs:
        df = pd.read_csv(file)
        for k in data.keys():
            if k in df:
                data[k].extend(df[k].tolist())
            else:
                if k == "algorithm":
                    # algorithm name should appear in the csv-file, e.g. "results_MyAlgo.csv"
                    data[k].extend([file.split("_")[-1].split(".")[0]]*len(df))
                elif k == "entity":
                    # if entity is not available, use dataset name as entity
                    data[k].extend(df["dataset"].tolist())

    df = pd.DataFrame(data)
    if os.path.exists(out_filename):
        os.remove(out_filename)
    df.to_csv(out_filename)

if __name__ == "__main__":
    get_combined_data("all_csvs", "results.csv")