import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.lstm_predictor import train

if __name__ == "__main__":
    result = train()
    if result:
        model, metrics = result
        print(f"\nTraining complete. Validation (held-out newest):")
        print(f"  accuracy={metrics['accuracy']:.1%} balanced={metrics['balanced_accuracy']:.1%} "
              f"auc={metrics['auc'] if metrics['auc'] is not None else 'N/A'} "
              f"baseline={metrics['majority_baseline']:.1%}")
        print("Model ready for predictions.")