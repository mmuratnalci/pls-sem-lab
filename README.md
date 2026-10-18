# PLS-SEM Lab

A small Python implementation of composite-based PLS path modeling. Define
constructs and their indicators, specify directed structural paths, then fit
the model to numeric indicator columns.

## Install

```bash
python -m pip install -e .
```

## Example

```python
import numpy as np
from pls_sem import PLSSEMModel

rng = np.random.default_rng(42)
quality = rng.normal(size=200)
satisfaction = 0.7 * quality + rng.normal(scale=0.5, size=200)

data = {
    "quality_1": quality + rng.normal(scale=0.2, size=200),
    "quality_2": quality + rng.normal(scale=0.2, size=200),
    "satisfaction_1": satisfaction + rng.normal(scale=0.2, size=200),
    "satisfaction_2": satisfaction + rng.normal(scale=0.2, size=200),
}
model = PLSSEMModel(
    {"Quality": ["quality_1", "quality_2"],
     "Satisfaction": ["satisfaction_1", "satisfaction_2"]},
    [("Quality", "Satisfaction")],
).fit(data)

print(model.path_coefficients_)
print(model.r_squared_)
print(model.loadings_)
```

The estimator standardizes indicators, iterates outer weights using the
centroid inner-weighting scheme (Mode A by default; Mode B can be selected per
construct with `modes={"Construct": "B"}`), then estimates structural paths by
least squares. Fitted results are available as `weights_`, `scores_`,
`loadings_`, `path_coefficients_`, `r_squared_`, and `n_iter_`.

Run the tests with:

```bash
python -m unittest discover -s tests
```