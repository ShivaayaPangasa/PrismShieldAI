"""
==============================================================
PrismShieldAI

Metrics Utility

Reusable evaluation metrics for all models.

==============================================================
"""

import torch

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    confusion_matrix,

    roc_auc_score,

    average_precision_score,

    balanced_accuracy_score,

    matthews_corrcoef,

)

class MetricsCalculator:

    def __init__(

        self,

        average="binary",

    ):

        self.average = average
        
    def calculate(
        
        self,
        
        predictions,
        
        labels,
        
        probabilities=None,

    ):

        if isinstance(predictions, torch.Tensor):

            predictions = predictions.cpu().numpy()

        if isinstance(labels, torch.Tensor):

            labels = labels.cpu().numpy()
        
        if probabilities is not None:
            
            if isinstance(probabilities, torch.Tensor):
                
                probabilities = probabilities.cpu().numpy()
        
        metrics = {
            
            "accuracy": accuracy_score(
                
                labels,
                
                predictions,
            ),

            "balanced_accuracy": balanced_accuracy_score(
                
                labels,

                predictions,

            ),

            "precision": precision_score(
                
                labels,

                predictions,

                average=self.average,

                zero_division=0,

            ),

            "recall": recall_score(
                
                labels,

                predictions,

                average=self.average,

                zero_division=0,

            ),

            "f1": f1_score(
                
                labels,

                predictions,

                average=self.average,

                zero_division=0,

            ),

            "mcc": matthews_corrcoef(
                
                labels,

                predictions,

            ),
            
            "roc_auc": (
                
                roc_auc_score(
                    
                    labels,
                    
                    probabilities,

               )

               if probabilities is not None
               
               else 0.0

            ),

            "pr_auc": (

                average_precision_score(
                    
                    labels,

                    probabilities,

                )

                if probabilities is not None

                else 0.0

            ),

            "confusion_matrix": confusion_matrix(

                labels,

                predictions,

            ),

        }

        return metrics

if __name__ == "__main__":

    predictions = torch.tensor([

        0,

        1,

        1,

        0,

        1,

        0,

    ])

    labels = torch.tensor([

        0,

        1,

        0,

        0,

        1,

        1,

    ])

    calculator = MetricsCalculator()

    results = calculator.calculate(

        predictions,

        labels,

    )

    print()

    print("=" * 60)

    print("METRICS")

    print("=" * 60)

    print(f"Accuracy : {results['accuracy']:.4f}")

    print(f"Precision: {results['precision']:.4f}")

    print(f"Recall   : {results['recall']:.4f}")

    print(f"F1 Score : {results['f1']:.4f}")

    print()

    print("Confusion Matrix")

    print(results["confusion_matrix"])