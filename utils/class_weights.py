"""
==============================================================
PrismShieldAI

Automatic Class Weight Computation

Computes class weights ONLY from the training split.

Supports future datasets automatically.

==============================================================
"""

import torch
import pandas as pd

from configs.settings import CSV_FILE


class ClassWeights:

    def __init__(self):

        self.csv_file = CSV_FILE

    def compute(self):

        df = pd.read_csv(self.csv_file)

        train_df = df[df["split"] == "train"]

        counts = train_df["label"].value_counts().sort_index()

        print()

        print("=" * 60)
        print("CLASS DISTRIBUTION")
        print("=" * 60)

        for cls, count in counts.items():

            print(f"Class {cls}: {count}")

        total = counts.sum()

        weights = total / (len(counts) * counts)

        weights = weights / weights.sum()

        weights = torch.tensor(

            weights.values,

            dtype=torch.float32,

        )

        print()

        print("=" * 60)
        print("CLASS WEIGHTS")
        print("=" * 60)

        print(weights)

        return weights


if __name__ == "__main__":

    cw = ClassWeights()

    weights = cw.compute()

    print()

    print("Ready!")