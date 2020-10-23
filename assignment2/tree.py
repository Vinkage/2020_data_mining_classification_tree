import sklearn
import re
from pathlib import Path
import pandas as pd


def read_into_pandas_dataframe(data=None):
    if data is None:
        data_path = Path.cwd() / "data"
    else:
        data_path = Path(data)

    data_dict = {}
    hidden_file = "\."
    for folds in data_path.iterdir():
        match = re.match(hidden_file, folds.name)
        if not match:
            data_dict[folds.name] = {}
            # print(data_dict)
            print(folds)
            for fold in folds.iterdir():
                match = re.match(hidden_file, fold.name)
                if not match:
                    # print(fold.name)
                    data_dict[folds.name][fold.name] = []
                    # print(data_dict)
                    for review in fold.glob("*"):
                        # print(review)
                        with open(review, "r") as rev:
                            rev_string = rev.read()
                            # print(string)
                            data_dict[folds.name][fold.name].append(rev_string)
                    print(fold.name)
                    print(len(data_dict[folds.name][fold.name]))
    df = pd.DataFrame.from_dict(data_dict)
    return df



    # for folds in data_path.iterdir():
    #     print(folds)

    # for folds in ["deceptive", "truthful"]:
    #     folds_path = data_path / folds
    #     print(folds_path.is_dir())

# read_into_pandas_dataframe(data="/Users/mikevink/Documents/python/2020_data_mining_assignments/assignment2/data")
df = read_into_pandas_dataframe()
