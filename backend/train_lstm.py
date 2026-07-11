from services.lstm_predictor import train

if __name__ == "__main__":
    result = train()
    if result:
        model, acc = result
        print(f"\nTraining complete. Accuracy: {acc:.1%}")
        print("Model ready for predictions.")