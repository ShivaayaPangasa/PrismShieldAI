"""
==============================================================
PrismShieldAI

Checkpoint Utility

Save and load model checkpoints.

==============================================================
"""

from pathlib import Path

import torch


class CheckpointManager:

    def __init__(

        self,

        checkpoint_dir="weights",

    ):

        self.checkpoint_dir = Path(checkpoint_dir)

        self.checkpoint_dir.mkdir(

            exist_ok=True

        )

    def save(

        self,

        model,

        optimizer,

        epoch,

        best_score,

        filename,

    ):
        
        checkpoint_path = self.checkpoint_dir / filename

        tmp_path = checkpoint_path.with_suffix(".tmp")

        checkpoint = {

            "epoch": epoch,

            "best_score": best_score,

            "model_state_dict": model.state_dict(),

            "optimizer_state_dict": optimizer.state_dict(),

        }

        torch.save(

            checkpoint,

            tmp_path,

        )

        tmp_path.replace(

            checkpoint_path,

        )

        print()

        print("=" * 60)

        print("CHECKPOINT SAVED")

        print("=" * 60)

        print(checkpoint_path)

    def load(

        self,

        model,

        optimizer,

        filename,

    ):

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = torch.load(

            checkpoint_path,

            map_location="cpu",

        )

        model.load_state_dict(

            checkpoint["model_state_dict"]

        )

        optimizer.load_state_dict(

            checkpoint["optimizer_state_dict"]

        )

        print()

        print("=" * 60)

        print("CHECKPOINT LOADED")

        print("=" * 60)

        print(checkpoint_path)

        return (

            checkpoint["epoch"],

            checkpoint["best_score"],

        )


if __name__ == "__main__":

    import torch.nn as nn

    model = nn.Linear(

        10,

        2,

    )

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=1e-3,

    )

    manager = CheckpointManager()

    manager.save(

        model=model,

        optimizer=optimizer,

        epoch=1,

        best_score=0.87,

        filename="checkpoint_test.pth",

    )

    epoch, score = manager.load(

        model=model,

        optimizer=optimizer,

        filename="checkpoint_test.pth",

    )

    print()

    print("Epoch      :", epoch)

    print("Best Score :", score)