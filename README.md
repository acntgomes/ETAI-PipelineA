# Baseline Predictive Pipeline -- ETAI
20231680 António Gomes
This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.


## week 2 Resultados da primeira aula (mudança de logistic regression para decision trees e comparação dos erros)
After switching from logistic regression to a decision tree, the metrics changed noticeably.
Accuracy dropped from 0.68 to 0.63, and the f1‑score for class1 fell from 0.63 to 0.55, showing weaker performance on reoffenders.
Precision also decreased slightly (0.66 → 0.62), while recall for class 0 remained similar (0.75 → 0.74).
Overall, the decision tree produced less balanced results between classes.
These differences confirm that model choice affects predictive behavior and group performance.

## week 6
| 6 | Ensembles: Random Forest, Oblique Random Forest and AdaBoost | Added ensemble models to the pipeline, including custom builders for Oblique Random Forest and AdaBoost. Extended the hyperparameter search spaces in config.yaml and evaluated default and tuned models using nested cross-validation. |
## 6.1

The pipeline was extended to include Random Forest, Oblique Random Forest, and AdaBoost, alongside the previously implemented Logistic Regression and Decision Tree. All five models were evaluated using five-fold cross-validation and nested cross-validation with 30 hyperparameter optimization trials.

| Model | Default CV mean ± std (gap) | Tuned nested CV mean ± std (gap) | Best tuning score | Chosen hyperparameters |
|---|---|---|---|---|
| Logistic Regression | 0.672 ± 0.013 (+0.003) | 0.671 ± 0.015 (+0.005) | 0.677 | C = 0.0746 |
| Decision Tree | 0.610 ± 0.018 (+0.085) | 0.672 ± 0.017 (+0.013) | 0.679 | max_depth = 6, min_samples_leaf = 101, criterion = gini |
| Random Forest | 0.650 ± 0.019 (+0.082) | 0.681 ± 0.015 (+0.009) | 0.684 | max_depth = 22, min_samples_leaf = 42, max_features = 0.5 |
| Oblique Random Forest | 0.647 ± 0.011 (+0.114) | 0.679 ± 0.013 | — | max_depth = 6, min_samples_leaf = 3, n_components = 5 |
| AdaBoost | 0.676 ± 0.012 (+0.006) | 0.682 ± 0.017 | — | See tuning report |

### 6.2  Model comparison and interpretation

AdaBoost achieved the highest tuned nested cross-validation accuracy (0.682), closely followed by Random Forest (0.681) and Oblique Random Forest (0.679). However, these differences are considerably smaller than the standard deviations across folds, so there is no clear evidence that one ensemble consistently outperforms the others. Hyperparameter tuning substantially improved the Decision Tree and Random Forest models while reducing their train-validation gaps, suggesting better generalisation. I would report AdaBoost as the highest-scoring model, while acknowledging the comparable performance of Random Forest and the uncertainty associated with cross-validation variability.


## Week 7 — Evaluation Metrics and Model Comparison

### Selection Metric

ROC-AUC was selected as the primary evaluation metric because it measures the model's ability to distinguish between individuals who reoffend and those who do not, without depending on a fixed classification threshold. This is particularly relevant for the COMPAS dataset, where false positives and false negatives have important consequences. Accuracy, balanced accuracy, precision, recall, F1, F-beta (β=2) and average precision are also reported to provide a more complete assessment of model performance.

### Scoreboard

A persistent scoreboard was implemented in `results/runs.csv` to compare all models using the same development dataset, preprocessing configuration and cross-validation folds. The scoreboard records each model's selection metric, mean score, standard deviation, fold-level results, untuned performance and Dummy Classifier baseline.

**Initial results — ROC-AUC (5-fold stratified cross-validation):**

| Model | Untuned ROC-AUC | Tuned ROC-AUC (nested CV) | No-model floor |
|---|---:|---:|---:|
| Dummy Classifier | 0.500 | N/A | 0.500 |
| Logistic Regression | Pending | Pending | 0.500 |
| Gaussian Naive Bayes | Pending | N/A | 0.500 |
| Decision Tree | Pending | Pending | 0.500 |
| Random Forest | Pending | Pending | 0.500 |
| Oblique Random Forest | Pending | Pending | 0.500 |
| AdaBoost | Pending | Pending | 0.500 |
| Gradient Boosting | 0.727 | Pending | 0.500 |
| Histogram Gradient Boosting | Pending | Pending | 0.500 |

The initial Gradient Boosting evaluation achieved a mean ROC-AUC of **0.727 ± 0.015**, outperforming the Dummy Classifier baseline of 0.500. Its accuracy was 0.680, precision 0.679, recall 0.550 and F1-score 0.607. These results are preliminary, as the remaining models and tuned configurations have not yet been evaluated.

All model selection and hyperparameter tuning are performed using the development set. The locked test set remains untouched until the final evaluation.




Week 7 — Evaluation Metrics and Model Comparison
Implementation and evaluation protocol
The pipeline now supports Gradient Boosting, Histogram Gradient Boosting and Gaussian Naive Bayes, alongside the existing classifiers. The build_scorers() function creates scorers from the eight metrics configured in config.yaml: accuracy, balanced accuracy, precision, recall, F1, F-beta (β=2), ROC-AUC and average precision. Each model is evaluated using the same stratified five-fold splits on the development set. For models with a defined Optuna search space, tuning uses 30 trials with five inner folds and performance is assessed by five outer folds (nested cross-validation). The Dummy Classifier supplies the no-model ROC-AUC floor of 0.500.
Selection metric
ROC-AUC was chosen as the selection metric because it measures how well the model ranks individuals who reoffend above those who do not, across classification thresholds rather than at a single cutoff. In the COMPAS setting, both false positives and false negatives have important consequences, so relying on accuracy alone would obscure relevant trade-offs. Precision, recall, F1, F-beta (β=2), balanced accuracy and average precision are therefore reported alongside ROC-AUC to assess threshold-dependent performance and positive-class detection.
Scoreboard — ROC-AUC
The persistent scoreboard results/runs.csv records model configurations, tuning status, mean ROC-AUC, standard deviation across folds, fold scores and the Dummy baseline. The following table summarises the completed development-set evaluations (rounded to three decimals). “Untuned” refers to ordinary five-fold cross-validation; “Tuned” refers to the outer-fold performance of nested cross-validation, not the best internal Optuna score.


| Model | Untuned ROC-AUC | Tuned ROC-AUC (nested CV) | No-model floor |
|---|---:|---:|---:|
| Dummy Classifier | 0.500 | N/A | 0.500 |
| Logistic Regression | 0.718 | 0.718 | 0.500 |
| Gaussian Naive Bayes | 0.685 | N/A | 0.500 |
| Decision Tree | 0.610 | 0.717 | 0.500 |
| Random Forest | 0.684 | 0.725 | 0.500 |
| Oblique Random Forest | 0.685 | 0.726 | 0.500 |
| AdaBoost | 0.723 | 0.726 | 0.500 |
| Gradient Boosting | 0.727 | 0.727 | 0.500 |
| Histogram Gradient Boosting | 0.717 | **0.728** | 0.500 |


Histogram Gradient Boosting achieved the highest tuned nested-CV mean ROC-AUC (0.728 ± 0.016), closely followed by Gradient Boosting (0.727 ± 0.014), AdaBoost (0.726 ± 0.016) and Oblique Random Forest (0.726 ± 0.015). These differences are small relative to the fold-to-fold variability, so they do not establish that one of the leading models is statistically superior. The Decision Tree showed the largest difference between the untuned CV and tuned nested-CV estimates (0.610 versus 0.717); as these are different evaluation procedures, this difference should not be interpreted as a causal estimate of tuning's benefit.

Class-weight experiment — Histogram Gradient Boosting
To examine the effect of class weighting, Histogram Gradient Boosting was evaluated without tuning, with identical preprocessing and folds, once with default class weights and once with class_weight="balanced".

| Metric | Default weights | Balanced class weights |
|---|---:|---:|
| Accuracy | 0.668 | 0.667 |
| Balanced accuracy | 0.658 | 0.664 |
| Precision | 0.654 | 0.632 |
| Recall | 0.559 | **0.628** |
| F1 | 0.603 | **0.630** |
| F-beta (β=2) | 0.576 | **0.629** |
| ROC-AUC | 0.7167 | 0.7169 |
| Average precision | 0.680 | 0.682 |

Balanced class weights increased recall by 0.069 (6.9 percentage points), while precision decreased by 0.022 (2.2 percentage points). ROC-AUC remained effectively unchanged. This illustrates the trade-off between identifying more positive cases and increasing false-positive predictions at a fixed classification threshold; weighting alone is not evidence of improved fairness across groups.

All tuning, model comparisons and class-weight experiments were conducted on the 5,771-row development set. The 1,443-row locked test set has not been used for model selection or evaluation during this stage.





## Week 8 — Evaluation Metrics and Model Comparison

Feature Selection reduced the number of features from 10 to 2, while achieving a similar ROC-AUC (0.7250 vs. 0.7274). The corrected paired t-test showed no statistically significant difference between the models (p = 0.425), although this does not prove equivalence. The smaller model is simpler and requires fewer input features, which may be beneficial in practice. I would keep Feature Selection enabled when simplicity is a priority, since the observed loss in predictive performance is very small.

## Week 8 — Feature Selection and XGBoost

### Pipeline Improvements

XGBoost was added to the pipeline, together with a VoteSelector combining Random Forest, Gradient Boosting and L1 Logistic Regression. Feature selection is performed inside the pipeline to avoid data leakage, and the minimum number of votes (`min_votes`) is included in Optuna's hyperparameter search.

### XGBoost Results

Both configurations were evaluated using 5-fold nested cross-validation, 30 Optuna trials and ROC-AUC as the selection metric.

| Configuration | ROC-AUC (mean ± std) |
|---|---|
| XGBoost without Feature Selection | 0.7274 ± 0.0148 |
| XGBoost with Feature Selection | 0.7250 ± 0.0129 |

### Corrected Paired t-test

The corrected paired t-test was used to compare the ROC-AUC scores across the same five outer folds.

- Mean difference (with − without selection): −0.0024
- t-statistic: −0.888
- p-value: 0.425
- Folds won by Feature Selection: 2/5

The difference was not statistically significant at the 5% significance level.

### Selected Features

The final tuned XGBoost model retained **2 out of 10 features**:

- `numeric__age`
- `numeric__priors_count`

This represents an 80% reduction in the number of features.

### Conclusion

Feature Selection reduced the number of features from 10 to 2, while achieving a similar ROC-AUC (0.7250 vs. 0.7274). The corrected paired t-test showed no statistically significant difference between the models (p = 0.425), although this does not prove equivalence. The smaller model requires fewer input features, which may be beneficial in practice. I would keep Feature Selection enabled when simplicity is a priority, since the observed loss in predictive performance is very small.