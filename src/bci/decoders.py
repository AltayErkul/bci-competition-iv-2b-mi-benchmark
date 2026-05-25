"""Classical REST-vs-ACTIVE motor-imagery decoders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import mne
from mne.decoding import CSP
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import DEFAULT_FILTER_BANK, FilterBand, filter_bank_epochs, normalize_bands


REST_LABEL = 0
ACTIVE_LABEL = 1

mne.set_log_level("ERROR")


class ActiveRestDecoder(Protocol):
    """Shared protocol for REST-vs-ACTIVE decoders."""

    algorithm: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> ActiveRestDecoder:
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def save(self, path: str | Path) -> None:
        ...


@dataclass(frozen=True)
class DecoderConfig:
    """Minimal decoder configuration saved with trained models."""

    sfreq: float
    bands: tuple[tuple[float, float], ...] = DEFAULT_FILTER_BANK
    n_csp_components: int = 2
    filter_order: int = 4


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """Apply CSP in each frequency band and concatenate log-variance features."""

    def __init__(
        self,
        *,
        sfreq: float,
        bands: tuple[tuple[float, float], ...] = DEFAULT_FILTER_BANK,
        n_components: int = 2,
        filter_order: int = 4,
    ) -> None:
        self.sfreq = sfreq
        self.bands = bands
        self.n_components = n_components
        self.filter_order = filter_order

    def fit(self, X: np.ndarray, y: np.ndarray) -> FilterBankCSP:
        self._bands: tuple[FilterBand, ...] = normalize_bands(self.bands)
        filtered = filter_bank_epochs(
            X,
            sfreq=self.sfreq,
            bands=self._bands,
            order=self.filter_order,
        )
        self._csps = []
        for band_X in filtered:
            csp = CSP(
                n_components=self.n_components,
                reg="ledoit_wolf",
                log=True,
                norm_trace=False,
            )
            csp.fit(band_X, y)
            self._csps.append(csp)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        filtered = filter_bank_epochs(
            X,
            sfreq=self.sfreq,
            bands=self._bands,
            order=self.filter_order,
        )
        features = [
            csp.transform(band_X)
            for csp, band_X in zip(self._csps, filtered, strict=True)
        ]
        return np.concatenate(features, axis=1)


class CappedSelectKBest(SelectKBest):
    """SelectKBest that caps k at the available feature count."""

    def __init__(self, score_func=mutual_info_classif, *, k: int = 8) -> None:
        super().__init__(score_func=score_func, k=k)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> CappedSelectKBest:
        if isinstance(self.k, int) and self.k > X.shape[1]:
            self.k = X.shape[1]
        return super().fit(X, y)


class ParzenWindowClassifier(BaseEstimator, ClassifierMixin):
    """Class-conditional Parzen window density classifier.

    This approximates the Naive Bayes Parzen Window style used by the winning
    BCI Competition IV 2b FBCSP submission, while staying compact enough for
    local benchmarking.
    """

    def __init__(self, *, bandwidth: float = 1.0) -> None:
        self.bandwidth = bandwidth

    def fit(self, X: np.ndarray, y: np.ndarray) -> ParzenWindowClassifier:
        self.classes_, counts = np.unique(y, return_counts=True)
        self.log_priors_ = np.log(counts / counts.sum())
        self.models_ = []
        for class_label in self.classes_:
            kde = KernelDensity(kernel="gaussian", bandwidth=self.bandwidth)
            kde.fit(X[y == class_label])
            self.models_.append(kde)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_scores = np.column_stack(
            [
                model.score_samples(X) + log_prior
                for model, log_prior in zip(self.models_, self.log_priors_, strict=True)
            ]
        )
        log_scores -= log_scores.max(axis=1, keepdims=True)
        scores = np.exp(log_scores)
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class NaiveBayesParzenWindowClassifier(BaseEstimator, ClassifierMixin):
    """Naive Bayes classifier with one Parzen density per feature and class."""

    def __init__(self, *, bandwidth: float = 0.8) -> None:
        self.bandwidth = bandwidth

    def fit(self, X: np.ndarray, y: np.ndarray) -> NaiveBayesParzenWindowClassifier:
        self.classes_, counts = np.unique(y, return_counts=True)
        self.log_priors_ = np.log(counts / counts.sum())
        self.models_: list[list[KernelDensity]] = []
        for class_label in self.classes_:
            X_class = X[y == class_label]
            feature_models = []
            for feature_idx in range(X.shape[1]):
                kde = KernelDensity(kernel="gaussian", bandwidth=self.bandwidth)
                kde.fit(X_class[:, [feature_idx]])
                feature_models.append(kde)
            self.models_.append(feature_models)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        class_log_scores = []
        for feature_models, log_prior in zip(
            self.models_,
            self.log_priors_,
            strict=True,
        ):
            log_score = np.full(X.shape[0], log_prior)
            for feature_idx, kde in enumerate(feature_models):
                log_score += kde.score_samples(X[:, [feature_idx]])
            class_log_scores.append(log_score)
        log_scores = np.column_stack(class_log_scores)
        log_scores -= log_scores.max(axis=1, keepdims=True)
        scores = np.exp(log_scores)
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class FBCSPDecoder:
    """Filter-bank CSP followed by a classical classifier."""

    algorithm = "fbcsp_lda"

    def __init__(self, config: DecoderConfig, *, estimator: str = "lda") -> None:
        self.config = config
        estimator = estimator.lower()
        self.algorithm = f"fbcsp_{_estimator_name(estimator)}"
        self.pipeline = Pipeline(
            steps=[
                (
                    "fbcsp",
                    FilterBankCSP(
                        sfreq=config.sfreq,
                        bands=config.bands,
                        n_components=config.n_csp_components,
                        filter_order=config.filter_order,
                    ),
                ),
                *_classifier_steps(estimator),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> FBCSPDecoder:
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> FBCSPDecoder:
        return joblib.load(path)


class CSPDecoder:
    """Single-band CSP followed by a classical classifier baseline."""

    algorithm = "csp_lda"

    def __init__(self, *, n_components: int = 2, estimator: str = "lda") -> None:
        estimator = estimator.lower()
        self.algorithm = f"csp_{_estimator_name(estimator)}"
        self.pipeline = Pipeline(
            steps=[
                (
                    "csp",
                    CSP(
                        n_components=n_components,
                        reg="ledoit_wolf",
                        log=True,
                        norm_trace=False,
                    ),
                ),
                *_classifier_steps(estimator),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> CSPDecoder:
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> CSPDecoder:
        return joblib.load(path)


class RiemannianDecoder:
    """Covariance tangent-space decoder.

    pyRiemann is imported lazily so the rest of the project can still run when
    the optional dependency has not been installed yet.
    """

    algorithm = "riemannian_lda"

    def __init__(self, *, estimator: str = "lda") -> None:
        try:
            from pyriemann.estimation import Covariances
            from pyriemann.tangentspace import TangentSpace
        except (ImportError, ModuleNotFoundError) as exc:
            message = (
                "RiemannianDecoder needs pyriemann. Install it with "
                "`pip install pyriemann` or use method='fbcsp_lda'."
            )
            raise RuntimeError(message) from exc

        if estimator == "lda":
            classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            self.algorithm = "riemannian_lda"
        elif estimator in {"logreg", "logistic"}:
            classifier = LogisticRegression(max_iter=1000)
            self.algorithm = "riemannian_logreg"
        elif estimator == "tsclassifier":
            raise ValueError(
                "estimator='tsclassifier' is not supported across current "
                "pyriemann versions; use 'lda' or 'logreg'."
            )
        else:
            raise ValueError("estimator must be 'lda', 'logreg', or 'tsclassifier'")

        self.pipeline = Pipeline(
            steps=[
                ("cov", Covariances(estimator="oas")),
                ("ts", TangentSpace(metric="riemann")),
                ("clf", classifier),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> RiemannianDecoder:
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> RiemannianDecoder:
        return joblib.load(path)


def load_decoder(path: str | Path) -> ActiveRestDecoder:
    """Load any saved decoder."""

    payload = joblib.load(path)
    if isinstance(payload, dict) and "decoder" in payload:
        return payload["decoder"]
    return payload


def _estimator_name(estimator: str) -> str:
    if estimator in {"lda", "shrinkage_lda"}:
        return "lda"
    if estimator in {"svm", "linear_svm", "svc"}:
        return "svm"
    if estimator in {"logreg", "logistic", "logistic_regression"}:
        return "logreg"
    if estimator in {"parzen", "kde"}:
        return "parzen"
    if estimator in {
        "nbpw",
        "nb_parzen",
        "naive_parzen",
        "naive_bayes_parzen",
        "nbpw_independent",
    }:
        return "nbpw"
    raise ValueError(f"Unsupported classifier estimator: {estimator}")


def _classifier_steps(estimator: str) -> list[tuple[str, object]]:
    estimator_name = _estimator_name(estimator)
    if estimator_name == "lda":
        return [
            (
                "lda",
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            )
        ]
    if estimator_name == "logreg":
        return [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    if estimator_name == "svm":
        return [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    C=1.0,
                    class_weight="balanced",
                    kernel="linear",
                    probability=True,
                    random_state=13,
                ),
            ),
        ]
    if estimator_name == "parzen":
        return [
            ("scaler", StandardScaler()),
            ("parzen", ParzenWindowClassifier(bandwidth=1.0)),
        ]
    if estimator_name == "nbpw":
        return [
            ("scaler", StandardScaler()),
            ("nbpw", NaiveBayesParzenWindowClassifier(bandwidth=0.8)),
        ]
    raise ValueError(f"Unsupported classifier estimator: {estimator}")


def _with_mibif(steps: list[tuple[str, object]], *, k: int = 8) -> list[tuple[str, object]]:
    return [
        (
            "mibif",
            CappedSelectKBest(score_func=mutual_info_classif, k=k),
        ),
        *steps,
    ]


def make_decoder(
    *,
    method: str,
    sfreq: float,
    bands: tuple[tuple[float, float], ...] = DEFAULT_FILTER_BANK,
    n_csp_components: int = 2,
    filter_order: int = 4,
) -> ActiveRestDecoder:
    """Create a decoder by config method name."""

    method = method.lower()
    use_mibif = method.startswith("fbcsp_mibif_")
    if use_mibif:
        method = method.replace("fbcsp_mibif_", "fbcsp_", 1)

    if method == "csp_lda":
        return CSPDecoder(n_components=n_csp_components, estimator="lda")
    if method == "csp_svm":
        return CSPDecoder(n_components=n_csp_components, estimator="svm")
    if method in {"csp_logreg", "csp_logistic"}:
        return CSPDecoder(n_components=n_csp_components, estimator="logreg")
    if method in {"csp_parzen", "csp_kde"}:
        return CSPDecoder(n_components=n_csp_components, estimator="parzen")
    if method in {"csp_nbpw", "csp_nbpw_independent", "csp_naive_parzen"}:
        return CSPDecoder(n_components=n_csp_components, estimator="nbpw")

    if method == "fbcsp_lda":
        decoder = FBCSPDecoder(
            DecoderConfig(
                sfreq=sfreq,
                bands=bands,
                n_csp_components=n_csp_components,
                filter_order=filter_order,
            ),
            estimator="lda",
        )
        if use_mibif:
            decoder.pipeline.steps = [
                decoder.pipeline.steps[0],
                *_with_mibif(decoder.pipeline.steps[1:]),
            ]
            decoder.algorithm = "fbcsp_mibif_lda"
        return decoder
    if method == "fbcsp_svm":
        decoder = FBCSPDecoder(
            DecoderConfig(
                sfreq=sfreq,
                bands=bands,
                n_csp_components=n_csp_components,
                filter_order=filter_order,
            ),
            estimator="svm",
        )
        if use_mibif:
            decoder.pipeline.steps = [
                decoder.pipeline.steps[0],
                *_with_mibif(decoder.pipeline.steps[1:]),
            ]
            decoder.algorithm = "fbcsp_mibif_svm"
        return decoder
    if method in {"fbcsp_logreg", "fbcsp_logistic"}:
        decoder = FBCSPDecoder(
            DecoderConfig(
                sfreq=sfreq,
                bands=bands,
                n_csp_components=n_csp_components,
                filter_order=filter_order,
            ),
            estimator="logreg",
        )
        if use_mibif:
            decoder.pipeline.steps = [
                decoder.pipeline.steps[0],
                *_with_mibif(decoder.pipeline.steps[1:]),
            ]
            decoder.algorithm = "fbcsp_mibif_logreg"
        return decoder
    if method in {"fbcsp_parzen", "fbcsp_kde"}:
        decoder = FBCSPDecoder(
            DecoderConfig(
                sfreq=sfreq,
                bands=bands,
                n_csp_components=n_csp_components,
                filter_order=filter_order,
            ),
            estimator="parzen",
        )
        if use_mibif:
            decoder.pipeline.steps = [
                decoder.pipeline.steps[0],
                *_with_mibif(decoder.pipeline.steps[1:]),
            ]
            decoder.algorithm = "fbcsp_mibif_parzen"
        return decoder
    if method in {"fbcsp_nbpw", "fbcsp_nbpw_independent", "fbcsp_naive_parzen"}:
        decoder = FBCSPDecoder(
            DecoderConfig(
                sfreq=sfreq,
                bands=bands,
                n_csp_components=n_csp_components,
                filter_order=filter_order,
            ),
            estimator="nbpw",
        )
        if use_mibif:
            decoder.pipeline.steps = [
                decoder.pipeline.steps[0],
                *_with_mibif(decoder.pipeline.steps[1:]),
            ]
            decoder.algorithm = "fbcsp_mibif_nbpw"
        return decoder
    if method == "riemannian_lda":
        return RiemannianDecoder(estimator="lda")
    if method in {"riemannian_logreg", "riemannian_logistic"}:
        return RiemannianDecoder(estimator="logreg")
    raise ValueError(f"Unsupported decoder method: {method}")
