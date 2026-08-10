import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.lstm_predictor import train

if __name__ == "__main__":
    try:
        result = train()
    except RuntimeError as e:
        print(f"\nRefusing to train:\n{e}")
        sys.exit(1)
    if result:
        model, metrics = result
        print(f"\nTraining complete. Held-out (newest) validation:")
        print(f"  accuracy={metrics['accuracy']:.1%} balanced={metrics['balanced_accuracy']:.1%} "
              f"auc={metrics['auc'] if metrics['auc'] is not None else 'N/A'} "
              f"majority_baseline={metrics['majority_baseline']:.1%} "
              f"persistence={metrics['persistence_baseline']:.1%}")
        if metrics["balanced_accuracy"] <= metrics["majority_baseline"]:
            print("  WARNING: no evidence this model beats a no-signal baseline. Do not deploy.")
        else:
            print("Model ready for predictions.")
