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


##Resultados da primeira aula (mudança de logistic regression para decision trees e comparação dos erros)
After switching from logistic regression to a decision tree, the metrics changed noticeably.
Accuracy dropped from 0.68 to 0.63, and the f1‑score for class1 fell from 0.63 to 0.55, showing weaker performance on reoffenders.
Precision also decreased slightly (0.66 → 0.62), while recall for class 0 remained similar (0.75 → 0.74).
Overall, the decision tree produced less balanced results between classes.
These differences confirm that model choice affects predictive behavior and group performance.

## week 6
| 6 | Ensembles: Random Forest, Oblique Random Forest and AdaBoost | Added ensemble models to the pipeline, including custom builders for Oblique Random Forest and AdaBoost. Extended the hyperparameter search spaces in config.yaml and evaluated default and tuned models using nested cross-validation. |


## Model evaluation

### Ensembles — Week 6

The pipeline was extended to include Random Forest, Oblique Random Forest, and AdaBoost, alongside the previously implemented Logistic Regression and Decision Tree. All five models were evaluated using five-fold cross-validation and nested cross-validation with 30 hyperparameter optimization trials.

| Model | Default CV mean ± std (gap) | Tuned nested CV mean ± std (gap) | Best tuning score | Chosen hyperparameters |
|---|---|---|---|---|
| Logistic Regression | 0.672 ± 0.013 (+0.003) | 0.671 ± 0.015 (+0.005) | 0.677 | C = 0.0746 |
| Decision Tree | 0.610 ± 0.018 (+0.085) | 0.672 ± 0.017 (+0.013) | 0.679 | max_depth = 6, min_samples_leaf = 101, criterion = gini |
| Random Forest | 0.650 ± 0.019 (+0.082) | 0.681 ± 0.015 (+0.009) | 0.684 | max_depth = 22, min_samples_leaf = 42, max_features = 0.5 |
| Oblique Random Forest | 0.647 ± 0.011 (+0.114) | 0.679 ± 0.013 | — | max_depth = 6, min_samples_leaf = 3, n_components = 5 |
| AdaBoost | 0.676 ± 0.012 (+0.006) | 0.682 ± 0.017 | — | See tuning report |

### Model comparison and interpretation

AdaBoost achieved the highest tuned nested cross-validation accuracy (0.682), closely followed by Random Forest (0.681) and Oblique Random Forest (0.679). However, these differences are considerably smaller than the standard deviations across folds, so there is no clear evidence that one ensemble consistently outperforms the others. Hyperparameter tuning substantially improved the Decision Tree and Random Forest models while reducing their train-validation gaps, suggesting better generalisation. I would report AdaBoost as the highest-scoring model, while acknowledging the comparable performance of Random Forest and the uncertainty associated with cross-validation variability.
