"""
==============================================================
PrismShieldAI

Early Stopping Utility

Stops training when validation metric
does not improve.

==============================================================
"""

class EarlyStopping:

    def __init__(

        self,

        patience=8,

        min_delta=0.0,

    ):

        self.patience = patience

        self.min_delta = min_delta

        self.best_score = None

        self.counter = 0

        self.should_stop = False

    def step(

        self,

        score,

    ):

        if self.best_score is None:

            self.best_score = score

            return False

        if score > self.best_score + self.min_delta:

            self.best_score = score

            self.counter = 0

            return False

        self.counter += 1

        print(

            f"EarlyStopping: {self.counter}/{self.patience}"

        )

        if self.counter >= self.patience:

            self.should_stop = True

            return True

        return False


if __name__ == "__main__":

    stopper = EarlyStopping(

        patience=3

    )

    validation_scores = [

        0.60,

        0.65,

        0.66,

        0.66,

        0.65,

        0.65,

        0.64,

    ]

    for epoch, score in enumerate(

        validation_scores,

        start=1,

    ):

        print()

        print(

            f"Epoch {epoch} | Validation F1 = {score:.2f}"

        )

        stop = stopper.step(score)

        if stop:

            print()

            print("=" * 60)

            print("EARLY STOPPING TRIGGERED")

            print("=" * 60)

            break