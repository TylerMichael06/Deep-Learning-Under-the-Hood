# Deep Learning Under the Hood

Open **[engine_fault.ipynb](engine_fault.ipynb)**. All code is in this notebook:
data checks, plots, train/validation/test split, Dummy, Random Forest, MLP,
model comparison, and a prediction demo.

Each model has its own short cell with visible `.fit()` and `.predict()` calls.
Everything runs locally on CPU using scikit-learn.

## Run

Use Python 3.11 or newer. If `.venv` already exists, activate it:

```bash
source .venv/bin/activate
python -m jupyterlab engine_fault.ipynb
```

For a new setup:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab engine_fault.ipynb
```

Select the `.venv` kernel and choose **Restart Kernel and Run All**.
Run from the repository root. Each full run trains the models and replaces
its generated outputs. The original CSV stays unchanged.

## Files

- `engine_fault.ipynb`: all analysis and model code, with saved results.
- `EngineFaultDB_Final.csv`: original dataset, excluded from Git.
- `proposal.pdf`: original project proposal.
- `requirements.txt`: Python packages.
- `outputs/`: saved split, selected model, and results; excluded from Git.

If the CSV is missing, download it from the
[authors' repository](https://github.com/leoxthomas/EngineFaultDB) and place it
next to the notebook. The source declares GPL-3.0; retain its license and attribution.

The first run selected Random Forest: validation macro F1 **0.752**, test macro
F1 **0.762**. Classes 2 and 3 are the main difficulty. The same test was already
consulted, so repeating this notebook is not a new independent evaluation.
Results do not establish early warning or performance on another vehicle.
