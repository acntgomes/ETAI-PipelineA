
import argparse
import joblib
import pandas as pd
import yaml

from src.preprocessing import clean_dataset, split_features_target


def predict(df: pd.DataFrame, pipeline, config: dict) -> pd.DataFrame:
    """
    Predict new observations without retraining.
    Preserve every original row and ID.
    """

    clean = clean_dataset(df, config["diagnostics"])

    X, _, _ = split_features_target(
        clean,
        config["data"],
        config["preprocessing"]["mnar_indicator_sources"]
    )

    probability = pipeline.predict_proba(X)[:, 1]

    return pd.DataFrame({
        "id": df[config["diagnostics"]["id_column"]].to_numpy(),
        "probability": probability,
        "prediction": (probability >= 0.5).astype(int)
    })


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Predict new data with a saved model."
    )

    parser.add_argument("input")
    parser.add_argument("output", nargs="?", default="predictions.csv")

    args = parser.parse_args()

    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pipeline = joblib.load(config["output"]["model_path"])

    new_data = pd.read_csv(args.input)
    predictions = predict(new_data, pipeline, config)

    predictions.to_csv(args.output, index=False)

    print(f"{len(predictions)} rows predicted -> {args.output}")
