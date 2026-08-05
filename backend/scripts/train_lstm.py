import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.lstm_predictor import train

if __name__ == "__main__":
    result = train()
    if result:
        model, acc = result
        print(f"\nTraining complete. Accuracy: {acc:.1%}")
        print("Model ready for predictions.")