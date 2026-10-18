from collections.abc import Mapping, Sequence

import numpy as np


class PLSSEMModel:
    """Estimate a PLS path model from indicator columns."""

    def __init__(
        self,
        measurements: Mapping[str, Sequence[str]],
        paths: Sequence[tuple[str, str]],
        modes: Mapping[str, str] | None = None,
        *,
        max_iter: int = 300,
        tol: float = 1e-7,
    ) -> None:
        if not measurements or any(
            not isinstance(name, str)
            or not name
            or isinstance(items, (str, bytes))
            or not len(items)
            for name, items in measurements.items()
        ):
            raise ValueError("Each construct must have a name and at least one indicator.")
        if any(not isinstance(item, str) or not item for items in measurements.values() for item in items):
            raise ValueError("Indicator names must be non-empty strings.")

        self.measurements = {name: tuple(items) for name, items in measurements.items()}
        all_indicators = [item for items in self.measurements.values() for item in items]
        if len(all_indicators) != len(set(all_indicators)):
            raise ValueError("Each indicator can belong to only one construct.")

        self.paths = tuple(paths)
        constructs = set(self.measurements)
        if any(
            source not in constructs or target not in constructs or source == target
            for source, target in self.paths
        ):
            raise ValueError("Each path must connect two different defined constructs.")
        if len(set(self.paths)) != len(self.paths):
            raise ValueError("Structural paths must be unique.")
        if max_iter < 1 or tol <= 0:
            raise ValueError("max_iter and tol must be positive.")

        self.modes = {name: "A" for name in self.measurements}
        if modes:
            if set(modes) - constructs or any(mode not in {"A", "B"} for mode in modes.values()):
                raise ValueError("Measurement modes must map defined constructs to 'A' or 'B'.")
            self.modes.update(modes)
        self.max_iter = max_iter
        self.tol = tol
        self.path_coefficients_: dict[tuple[str, str], float] = {}
        self.r_squared_: dict[str, float] = {}
        self.loadings_: dict[str, float] = {}
        self.weights_: dict[str, dict[str, float]] = {}
        self.scores_: dict[str, np.ndarray] = {}
        self.n_iter_: int | None = None

    def fit(self, data: Mapping[str, Sequence[float]]) -> "PLSSEMModel":
        """Fit the model using a mapping of indicator names to numeric columns."""
        missing = {
            indicator
            for indicators in self.measurements.values()
            for indicator in indicators
            if indicator not in data
        }
        if missing:
            raise ValueError(f"Missing indicator columns: {', '.join(sorted(missing))}.")

        columns = {
            indicator: np.asarray(data[indicator], dtype=float)
            for indicators in self.measurements.values()
            for indicator in indicators
        }
        if any(column.ndim != 1 for column in columns.values()):
            raise ValueError("Each indicator must be a one-dimensional numeric column.")
        lengths = {len(column) for column in columns.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 3:
            raise ValueError("Indicator columns must have equal lengths and at least three rows.")
        if any(not np.all(np.isfinite(column)) for column in columns.values()):
            raise ValueError("Indicator data must contain only finite numeric values.")

        standardized: dict[str, np.ndarray] = {}
        for indicator, column in columns.items():
            deviation = column.std()
            if deviation == 0:
                raise ValueError(f"Indicator '{indicator}' has no variation.")
            standardized[indicator] = (column - column.mean()) / deviation

        indicator_matrices = {
            construct: np.column_stack([standardized[name] for name in indicators])
            for construct, indicators in self.measurements.items()
        }
        weights = {
            construct: np.full(len(indicators), 1 / np.sqrt(len(indicators)))
            for construct, indicators in self.measurements.items()
        }
        neighbors = {construct: [] for construct in self.measurements}
        for source, target in self.paths:
            neighbors[source].append(target)
            neighbors[target].append(source)

        for iteration in range(1, self.max_iter + 1):
            scores = {
                construct: self._standardize(matrix @ weights[construct])
                for construct, matrix in indicator_matrices.items()
            }
            updated = {}
            for construct, matrix in indicator_matrices.items():
                adjacent = neighbors[construct]
                if not adjacent:
                    updated[construct] = weights[construct]
                    continue
                proxy = sum(
                    np.sign(np.dot(scores[construct], scores[neighbor]))
                    * scores[neighbor]
                    for neighbor in adjacent
                )
                proxy = self._standardize(proxy)
                if self.modes[construct] == "A":
                    candidate = matrix.T @ proxy / len(proxy)
                else:
                    candidate = np.linalg.lstsq(matrix, proxy, rcond=None)[0]
                norm = np.linalg.norm(candidate)
                if norm <= np.finfo(float).eps:
                    candidate = weights[construct]
                else:
                    candidate = candidate / norm
                if np.dot(candidate, weights[construct]) < 0:
                    candidate = -candidate
                updated[construct] = candidate

            change = max(
                np.linalg.norm(updated[name] - weights[name]) for name in weights
            )
            weights = updated
            if change < self.tol:
                break

        scores = {
            construct: self._standardize(matrix @ weights[construct])
            for construct, matrix in indicator_matrices.items()
        }
        path_coefficients: dict[tuple[str, str], float] = {}
        r_squared: dict[str, float] = {}
        for target in self.measurements:
            predictors = [source for source, outcome in self.paths if outcome == target]
            if not predictors:
                continue
            design = np.column_stack([scores[source] for source in predictors])
            coefficients = np.linalg.lstsq(design, scores[target], rcond=None)[0]
            fitted = design @ coefficients
            r_squared[target] = float(1 - np.mean((scores[target] - fitted) ** 2))
            path_coefficients.update(
                ((source, target), float(coefficient))
                for source, coefficient in zip(predictors, coefficients)
            )

        self.weights_ = {
            construct: dict(zip(indicators, weights[construct].tolist()))
            for construct, indicators in self.measurements.items()
        }
        self.scores_ = scores
        self.loadings_ = {
            indicator: float(np.corrcoef(
                standardized[indicator], scores[construct]
            )[0, 1])
            for construct, indicators in self.measurements.items()
            for indicator in indicators
        }
        self.path_coefficients_ = path_coefficients
        self.r_squared_ = r_squared
        self.n_iter_ = iteration
        return self

    @staticmethod
    def _standardize(values: np.ndarray) -> np.ndarray:
        centered = values - values.mean()
        deviation = centered.std()
        if deviation <= np.finfo(float).eps:
            raise ValueError("A construct score has no variation; check the model data.")
        return centered / deviation
