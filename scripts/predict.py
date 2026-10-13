"""Load a trained ensemble and rebuild a format-validated Kaggle submission."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from .preprocess import read_competition_data, build_submission
from .train import ordered_probabilities
from .evaluate import decide


def predict(data, model_path, output="submissions/submission.csv", label_style="numeric"):
    """Predict with saved fitted Pipelines; preserve sample_submission row order."""
    _, test, sample = read_competition_data(data)
    bundle = joblib.load(model_path)
    p = np.zeros((len(test), 5), dtype=float)
    for name, weight in bundle["weights"].items():
        p += weight * ordered_probabilities(bundle["models"][name], test)
    predictions = decide(p, bundle["offsets"])
    return build_submission(sample, test, predictions, Path(output), label_style)


def main():
    """Provide a command-line interface for model inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--model", default="submissions/model_bundle.joblib")
    parser.add_argument("--output", default="submissions/submission.csv")
    parser.add_argument("--label-style", choices=["auto", "numeric", "grade"], default="numeric")
    args = parser.parse_args()
    result = predict(args.data, args.model, args.output, args.label_style)
    print(result.head(), f"\nSaved {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
