"""
==============================================================
PrismShieldAI

Seed Utility

Ensures reproducible experiments.

==============================================================
"""

import random

import numpy as np

import torch


def set_seed(

    seed: int = 42,

):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

    print()

    print("=" * 60)

    print("SEED")

    print("=" * 60)

    print(f"Random Seed : {seed}")


if __name__ == "__main__":

    set_seed(42)