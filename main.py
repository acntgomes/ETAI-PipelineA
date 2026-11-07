"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full pipeline:
    load config -> load data -> diagnose/clean (week 3) -> split features/target
    -> leak-safe train/test split -> preprocess + train (week 3's encoder/scaler pair)
    -> evaluate (accuracy, fairness) -> save results
"""

from datetime import datetime

import shutil
import tempfile
import optuna
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import (
    clean_dataset,
    drop_duplicate_rows,
    add_engineered_features,
    split_features_target,
    split_dev_test,
)

from src.model import build_pipeline

from src.evaluate import (
    build_scorers,
    cross_validate_pipeline,
    cv_report,
    oof_classification_report,
    fairness_report,
)

from src.tuning import (
    tune_pipeline,
    nested_cross_validate,
    tuning_report,
)

from src.results import save_run, save_scores



def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    run = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = config["model"]["type"]

    # 1. Load and clean data
    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])
    df_clean = drop_duplicate_rows(
        df_clean, config["diagnostics"].get("id_column")
    )

    # Optional feature engineering (Week 6)
    if config.get("feature_engineering", {}).get("enabled", False):
        df_clean = add_engineered_features(df_clean)
        new_features = [
            "priors_per_adult_year",
            "juvenile_offenses_total",
            "combined_offenses",
        ]
        numeric = config["preprocessing"]["numeric_features"]
        for feature in new_features:
            if feature not in numeric:
                numeric.append(feature)

    mnar_sources = config["preprocessing"].get(
        "mnar_indicator_sources", []
    )

    X, y, extras = split_features_target(
        df_clean, config["data"], mnar_sources
    )

    # 2. Locked test set
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = (
        split_dev_test(
            X, y, extras,
            test_size=config["test_set"]["size"],
            random_state=config["test_set"]["random_state"],
        )
    )

    # 3. Cross-validation and metrics
    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)

    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=shuffle,
        random_state=(
            cv_config.get("random_state") if shuffle else None
        ),
    )

    metrics = cv_config["metrics"]
    selection_metric = cv_config["selection_metric"]
    scorers = build_scorers(metrics)
    n_jobs = cv_config.get("n_jobs", 1)

    if selection_metric not in scorers:
        raise ValueError(
            "selection_metric must be included in cv.metrics"
        )

    print(
        f"Model: {model_type} | "
        f"Selection metric: {selection_metric}"
    )
    print(
        f"Development rows: {len(X_dev)} | "
        f"Locked test rows: {len(X_test)}"
    )

    # 4. Build model and dummy baseline
    
    selection_config = config.get("feature_selection", {})
    cache_dir = tempfile.mkdtemp(prefix="pipeline_cache_")

    pipeline = build_pipeline(
        config["preprocessing"],
        config["model"],
        selection_config,
        memory=cache_dir
    )


    floor_pipeline = build_pipeline(
        config["preprocessing"], {"type": "dummy"}
    )

    floor_scores, _ = cross_validate_pipeline(
        floor_pipeline, X_dev, y_dev, cv,
        {selection_metric: scorers[selection_metric]},
        selection_metric, n_jobs
    )

    fold_scores, oof_proba = cross_validate_pipeline(
        pipeline, X_dev, y_dev, cv,
        scorers, selection_metric, n_jobs
    )

    floor = floor_scores["validation"].mean()
    untuned = fold_scores["validation"].mean()
    final_scores = fold_scores

    report = (
        f"Hyperparameters from config.yaml: "
        f"{config['model'].get('params', {})}\n\n"
        + cv_report(
            fold_scores, metrics, selection_metric,
            "Cross-validation"
        )
        + f"\n\nNo-model floor: {floor:.3f}"
    )

    print(
        f"\nUntuned {selection_metric}: "
        f"{untuned:.3f} | Floor: {floor:.3f}"
    )

    # 5. Hyperparameter tuning
    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)

    if tuning_enabled:
        search_spaces = tuning_config.get(
            "search_spaces", {}
        )

        if model_type not in search_spaces:
            raise ValueError(
                f"No tuning search space for {model_type}"
            )

        search_space = search_spaces[model_type].copy()

        if selection_config.get("enabled", False):
            search_space.update(
                selection_config.get("search_space", {})
            )

        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]

        optuna.logging.set_verbosity(
            optuna.logging.WARNING
        )

        inner_cv = StratifiedKFold(
            n_splits=tuning_config["n_splits"],
            shuffle=True,
            random_state=tuning_seed,
        )

        print(
            f"\nNested CV: {n_trials} Optuna trials",
            flush=True
        )

        final_scores, oof_proba = nested_cross_validate(
            pipeline, X_dev, y_dev,
            cv, inner_cv,
            scorers, selection_metric,
            search_space, n_trials,
            tuning_seed, n_jobs=n_jobs,
        )

        report += "\n\n" + cv_report(
            final_scores, metrics, selection_metric,
            "Nested cross-validation"
        )

        pipeline, study = tune_pipeline(
            pipeline, X_dev, y_dev,
            inner_cv, scorers[selection_metric],
            search_space, n_trials,
            tuning_seed, n_jobs=n_jobs,
        )

        report += "\n\n" + tuning_report(
            study, final_scores, selection_metric
        )

    # 6. Report all metrics
    print("\nModel metrics:")
    for name in metrics:
        mean = final_scores[name].mean()
        std = final_scores[name].std(ddof=1)
        print(f"{name}: {mean:.3f} +/- {std:.3f}")

    report += "\n\n" + oof_classification_report(
        y_dev, oof_proba
    )

    report += "\n\n" + fairness_report(
        y_dev, oof_proba, extras_dev,
        sensitive_attr=config["data"]["sensitive_attr"],
    )

    # 7. Fit final model on development data
    pipeline.fit(X_dev, y_dev)
    
    if selection_config.get("enabled", False):
        feature_names = pipeline.named_steps[
            "prep"
        ].get_feature_names_out()

        selected = pipeline.named_steps[
            "select"
        ].get_support()

        kept_features = feature_names[selected]

        selection_text = (
            f"Features kept: {len(kept_features)} "
            f"out of {len(feature_names)}\n"
            + "\n".join(kept_features)
        )

        print("\n" + selection_text)
        report += "\n\n" + selection_text


    refit = (
        f"Final model: {model_type} fitted on "
        f"{len(X_dev)} development rows."
    )

    report += "\n\n" + refit
    report += (
        f"\nLocked test set: {len(X_test)} rows, "
        "not evaluated."
    )

    # 8. Save scoreboard
    row = {
        "run": run,
        "data": config["data"]["path"],
        "model": model_type,
        "tuned": tuning_enabled,
        "feature_selection": selection_config.get("enabled", False),
        "encoder": config["preprocessing"]["encoder"],
        "scaler": config["preprocessing"]["scaler"],
        "feature_engineering": config.get(
            "feature_engineering", {}
        ).get("enabled", False),
        "params": str(config["model"].get("params", {})),
        "tuned_params": (
            str(study.best_params) if tuning_enabled else ""
        ),
        "metric": selection_metric,
        "mean": final_scores["validation"].mean(),
        "std": final_scores["validation"].std(ddof=1),
        "untuned": untuned,
        "floor": floor,
        **{
            f"fold_{i}": score
            for i, score in enumerate(
                final_scores["validation"], start=1
            )
        },
    }

    scoreboard_path = config.get("output", {}).get(
        "scoreboard", "results/runs.csv"
    )

    runs = save_scores(scoreboard_path, row)

    same = runs[
        (runs["data"] == row["data"])
        & (runs["metric"] == selection_metric)
        & (
            runs["feature_engineering"].astype(str).str.lower()
            == str(row["feature_engineering"]).lower()
        )
    ]

    board = same.sort_values("mean", ascending=False)[
        ["run", "model", "tuned", "mean",
         "std", "untuned", "floor"]
    ]

    print("\nScoreboard:")
    print(board.to_string(index=False))

    report += "\n\nScoreboard:\n"
    report += board.to_string(index=False)

    # 9. Save full report
    results_dir = config.get("output", {}).get(
        "results_dir", "results"
    )

    path = save_run(
        results_dir, run, config, report
    )

    print(f"\nFull results saved to {path}")
    shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
