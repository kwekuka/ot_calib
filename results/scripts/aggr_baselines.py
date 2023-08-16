import os
import argparse

import numpy as np
import pandas as pd


"""
Sample run .. @yotam should work as long as the file tree structure is the same as ours on collab 
python3 aggr_baselines.py --dataset cifar10 --method ResnetWide32
"""



parser = argparse.ArgumentParser(
                    prog='AggrBaselines',
                    description='Aggregates the baseline runs into a single file for easy analysis or something',
                    epilog='got nothing for ya sorry')

parser.add_argument('--dataset', type=str, required=True)
parser.add_argument("--method", type=str, required=True)
parser.add_argument("--results", "--results", default="..")


#supported datasets
datasets = ["cifar10"]
methods = ["ResnetWide32"]

args = parser.parse_args()
if args.dataset not in datasets:
    raise ValueError(f"the dataset you input is not supported: \"{args.dataset}\" ")

if args.method not in methods:
    raise ValueError(f"the method you input is not supported: \"{args.method}\" ")


path_containing_csv = os.path.join(args.results, args.dataset, args.method, "baseline")
if not os.path.exists(path_containing_csv):
    raise ValueError(f"The csv's you wanted to consolidate need to be located at the following path based "
                     f"on the arguments you're passing to this script: {path_containing_csv}")

#Collect all csv file names that are seeded
collect_csv_files = [file for file in os.listdir(path_containing_csv) if ".csv" in file and "seed" in file]

#Read the cvs
csv_from_file = [pd.read_csv(os.path.join(path_containing_csv, csv)) for csv in collect_csv_files]

#Convert to numpy
values_from_cvs = [df.iloc[:,3:].values for df in csv_from_file]

#Convert to single numpy thing
combined_values = np.stack(values_from_cvs, axis=0)

#Compute avg and std
avg, std = combined_values.mean(axis=0), combined_values.std(axis=0)

csv_rows = []
for i, _ in enumerate(avg):
    row = []
    for j, _ in enumerate(avg[i]):
        cell = ("%.4f" % avg[i,j]) + "+/-" + ("%.4f" % std[i,j])
        row.append(cell)
    csv_rows.append(row)
#Needed to grab stuff -- don't want to index every time, so this is a lazy resource for the following lines
sample_df = csv_from_file[0]

#Make dataframe, the 3: indexing is because the first 3 columns r junky, we don't need them
df = pd.DataFrame(csv_rows, columns=sample_df.columns[3:])

#Add the method names as a column at the beginning (this is where the lazy sample_df thing come in hand)
df.insert(0, "Method", sample_df["Method"])

#Save as csv
df.to_csv(os.path.join(path_containing_csv, f"{args.method}_{args.dataset}_combined.csv"), index=False)