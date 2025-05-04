import argparse
import subprocess
import re
import os

def run_predict_script(directory, script, symptoms):
    """
    Runs a predict script in a given directory with the command:
      python.exe script --predict "symptoms"
    Returns a tuple (prediction, confidence, full_output).
    """
    # Build the command
    cmd = ["python.exe", script, "--predict", symptoms]
    try:
        # Run the command in the given directory, capturing stdout and stderr.
        result = subprocess.run(cmd, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout
        # Optionally, you can print result.stderr if debugging errors.
        prediction = None
        confidence = 0.0
        
        # Look for a line in the output that starts with a dash,
        # e.g. "- ConditionName: 0.XX ..."
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("-"):
                # Use regex to capture the condition and the confidence value.
                m = re.match(r"^- (.+):\s*([0-9]*\.?[0-9]+)", line)
                if m:
                    prediction = m.group(1).strip()
                    confidence = float(m.group(2))
                    break
        return prediction, confidence, output
    except Exception as e:
        print(f"Error running script in {directory}: {e}")
        return None, 0.0, ""

def main():
    parser = argparse.ArgumentParser(
        description="Run predictions across multiple models and select the one with the highest confidence."
    )
    parser.add_argument("symptoms", type=str, help="Symptoms for disease prediction (in quotes)")
    args = parser.parse_args()
    symptoms = args.symptoms

    # Define the model directories and corresponding prediction scripts.
    model_configs = [
        {"name": "BERT", "directory": "bert", "script": "bert.py"},
        {"name": "LSTM_BioBERT", "directory": "lstm_biobert", "script": "bio2.py"},
        {"name": "LSTM_GloVe", "directory": "lstm_glove", "script": "lstm2.py"},
        {"name": "LSTM_GloVe_BioBERT", "directory": "lstm_glove_biobert", "script": "bio.py"}
    ]

    results = []

    for config in model_configs:
        print(f"\nRunning prediction for {config['name']}...")
        pred, conf, out = run_predict_script(config["directory"], config["script"], symptoms)
        print(f"Output from {config['name']}:\n{out}\n")
        results.append({"model": config["name"], "prediction": pred, "confidence": conf})

    # Print all results
    print("Results from all models:")
    for res in results:
        print(f"{res['model']}: Prediction: {res['prediction']}, Confidence: {res['confidence']:.4f}")

    # Select the result with the highest confidence.
    best = max(results, key=lambda r: r["confidence"])
    print("\nModel with highest confidence:")
    print(f"{best['model']}: Prediction: {best['prediction']}, Confidence: {best['confidence']:.4f}")

if __name__ == "__main__":
    main()
