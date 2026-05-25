from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import mne
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

mne.set_log_level("ERROR")


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bci.decoders import ACTIVE_LABEL, make_decoder


EEG_CHANNELS = ["EEG:C3", "EEG:Cz", "EEG:C4"]
EVENT_TRIAL_START = 768
EVENT_LEFT = 769
EVENT_RIGHT = 770
EVENT_REJECTED = 1023
METHODS = (
    "csp_lda",
    "csp_svm",
    "csp_logreg",
    "csp_parzen",
    "csp_nbpw",
    "fbcsp_lda",
    "fbcsp_svm",
    "fbcsp_logreg",
    "fbcsp_parzen",
    "fbcsp_nbpw",
    "fbcsp_mibif_nbpw",
    "riemannian_lda",
    "riemannian_logreg",
)


def read_raw(path: Path) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
    raw.pick(EEG_CHANNELS)
    data = raw.get_data()
    if np.isnan(data).any():
        raw._data[:] = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    raw.filter(8.0, 32.0, method="iir", verbose="ERROR")
    return raw


def extract_events(raw: mne.io.BaseRaw) -> np.ndarray:
    wanted = ["768", "769", "770", "1023"]
    available = set(str(item) for item in raw.annotations.description)
    event_id = {code: int(code) for code in wanted if code in available}
    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose="ERROR")
    return events


def clean_cues(events: np.ndarray, sfreq: float, *, epoch_s: float) -> np.ndarray:
    trial_starts = events[events[:, 2] == EVENT_TRIAL_START][:, 0]
    rejected = events[events[:, 2] == EVENT_REJECTED][:, 0]
    cues = events[np.isin(events[:, 2], [EVENT_LEFT, EVENT_RIGHT])]
    clean = []
    for sample, _, code in cues:
        prev_starts = trial_starts[trial_starts <= sample]
        trial_start = prev_starts[-1] if len(prev_starts) else sample
        epoch_end = sample + int(epoch_s * sfreq)
        has_reject = np.any((rejected >= trial_start) & (rejected <= epoch_end))
        if not has_reject:
            clean.append([sample, 0, code])
    return np.asarray(clean, dtype=int)


def load_left_right(files: list[Path]) -> tuple[np.ndarray, np.ndarray, float]:
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    sfreq = 0.0
    for path in files:
        raw = read_raw(path)
        sfreq = float(raw.info["sfreq"])
        events = extract_events(raw)
        clean = clean_cues(events, sfreq, epoch_s=4.0)
        epochs = mne.Epochs(
            raw,
            clean,
            event_id={"left": EVENT_LEFT, "right": EVENT_RIGHT},
            tmin=0.5,
            tmax=4.0,
            baseline=None,
            preload=True,
            reject_by_annotation=False,
            verbose="ERROR",
        )
        X_parts.append(epochs.get_data())
        y_parts.append(np.where(epochs.events[:, -1] == EVENT_LEFT, 0, 1))
    return np.concatenate(X_parts), np.concatenate(y_parts), sfreq


def load_rest_active_pseudo(
    files: list[Path],
    *,
    window_s: float,
    active_offset_s: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    sfreq = 0.0
    for path in files:
        raw = read_raw(path)
        sfreq = float(raw.info["sfreq"])
        data = raw.get_data()
        events = extract_events(raw)
        cues = clean_cues(events, sfreq, epoch_s=active_offset_s + window_s)
        trial_starts = events[events[:, 2] == EVENT_TRIAL_START][:, 0]
        window_samples = int(round(window_s * sfreq))
        active_offset = int(round(active_offset_s * sfreq))
        rest_offset = int(round(0.5 * sfreq))

        for cue_sample, _, _ in cues:
            start = int(cue_sample) + active_offset
            end = start + window_samples
            if end <= data.shape[1]:
                X_parts.append(data[:, start:end][np.newaxis, :, :])
                y_parts.append(np.asarray([1]))

        for trial_start in trial_starts:
            future_cues = cues[cues[:, 0] > trial_start]
            if len(future_cues) == 0:
                continue
            cue_sample = int(future_cues[0, 0])
            start = int(trial_start) + rest_offset
            end = start + window_samples
            if end < cue_sample - int(0.25 * sfreq):
                X_parts.append(data[:, start:end][np.newaxis, :, :])
                y_parts.append(np.asarray([0]))

    if not X_parts:
        raise RuntimeError("No pseudo REST/ACTIVE windows could be extracted.")
    return np.concatenate(X_parts), np.concatenate(y_parts), sfreq


def evaluate(
    *,
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sfreq: float,
) -> dict[str, object]:
    try:
        decoder = make_decoder(method=method, sfreq=sfreq)
        decoder.fit(X_train, y_train)
        pred = decoder.predict(X_test)
        proba = decoder.predict_proba(X_test)
    except Exception as exc:
        print(f"{method}: failed ({exc})")
        return {
            "method": method,
            "status": "failed",
            "reason": str(exc),
        }

    positive_col = ACTIVE_LABEL if proba.shape[1] > ACTIVE_LABEL else 0
    auc = roc_auc_score(y_test, proba[:, positive_col])
    bal = balanced_accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    print(
        f"{method}: balanced_accuracy={bal:.3f}, "
        f"roc_auc={auc:.3f}"
    )
    print(cm)
    return {
        "method": method,
        "status": "ok",
        "balanced_accuracy": bal,
        "roc_auc": auc,
        "true_0_pred_0": int(cm[0, 0]),
        "true_0_pred_1": int(cm[0, 1]),
        "true_1_pred_0": int(cm[1, 0]),
        "true_1_pred_1": int(cm[1, 1]),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def subject_files(data_dir: Path, subject: str) -> tuple[list[Path], list[Path]]:
    train_files = [
        data_dir / f"B{subject}01T.gdf",
        data_dir / f"B{subject}02T.gdf",
    ]
    test_files = [data_dir / f"B{subject}03T.gdf"]
    missing = [path for path in [*train_files, *test_files] if not path.exists()]
    if missing:
        missing_text = ", ".join(path.name for path in missing)
        raise FileNotFoundError(f"Missing GDF files for subject {subject}: {missing_text}")
    return train_files, test_files


def run_subject(
    *,
    data_dir: Path,
    subject: str,
    mode: str,
    window_s: float,
) -> list[dict[str, object]]:
    train_files, test_files = subject_files(data_dir, subject)

    if mode == "left_right_debug":
        X_train, y_train, sfreq = load_left_right(train_files)
        X_test, y_test, _ = load_left_right(test_files)
    else:
        X_train, y_train, sfreq = load_rest_active_pseudo(
            train_files,
            window_s=window_s,
            active_offset_s=0.5,
        )
        X_test, y_test, _ = load_rest_active_pseudo(
            test_files,
            window_s=window_s,
            active_offset_s=0.5,
        )

    print(
        f"\nsubject={subject}, mode={mode}, "
        f"X_train={X_train.shape}, X_test={X_test.shape}"
    )
    rows = []
    for method in METHODS:
        row = evaluate(
            method=method,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            sfreq=sfreq,
        )
        row.update(
            {
                "subject": subject,
                "mode": mode,
                "sfreq": sfreq,
                "window_s": window_s,
            }
        )
        rows.append(row)
    return rows


def save_results(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject",
        "mode",
        "method",
        "status",
        "balanced_accuracy",
        "roc_auc",
        "true_0_pred_0",
        "true_0_pred_1",
        "true_1_pred_0",
        "true_1_pred_1",
        "n_train",
        "n_test",
        "sfreq",
        "window_s",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def save_summary_outputs(
    rows: list[dict[str, object]],
    *,
    csv_path: Path,
    plot_dir: Path | None = None,
) -> None:
    """Save readable summary tables and figures next to the raw benchmark CSV."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:
        print(f"Could not create plots because a plotting dependency is missing: {exc}")
        return

    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print("No successful benchmark rows to summarize.")
        return

    for col in [
        "balanced_accuracy",
        "roc_auc",
        "true_0_pred_0",
        "true_0_pred_1",
        "true_1_pred_0",
        "true_1_pred_1",
        "n_test",
    ]:
        ok[col] = pd.to_numeric(ok[col])

    ok["class0_to_class1_rate"] = ok["true_0_pred_1"] / (
        ok["true_0_pred_0"] + ok["true_0_pred_1"]
    )
    ok["class1_to_class0_rate"] = ok["true_1_pred_0"] / (
        ok["true_1_pred_0"] + ok["true_1_pred_1"]
    )

    plot_dir = plot_dir or csv_path.with_suffix("").parent / f"{csv_path.stem}_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    summary = (
        ok.groupby("method", as_index=False)
        .agg(
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_class0_to_class1_rate=("class0_to_class1_rate", "mean"),
            mean_class1_to_class0_rate=("class1_to_class0_rate", "mean"),
            n_subjects=("subject", "nunique"),
        )
        .sort_values("mean_balanced_accuracy", ascending=False)
    )
    best_by_subject = (
        ok.sort_values(["subject", "balanced_accuracy"], ascending=[True, False])
        .groupby("subject", as_index=False)
        .first()[
            [
                "subject",
                "method",
                "balanced_accuracy",
                "roc_auc",
                "class0_to_class1_rate",
                "class1_to_class0_rate",
            ]
        ]
    )

    summary_path = plot_dir / "method_summary.csv"
    best_path = plot_dir / "best_method_by_subject.csv"
    summary.to_csv(summary_path, index=False)
    best_by_subject.to_csv(best_path, index=False)

    print("\nMethod summary:")
    print(
        summary[
            [
                "method",
                "mean_balanced_accuracy",
                "std_balanced_accuracy",
                "mean_roc_auc",
                "mean_class0_to_class1_rate",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print("\nBest method by subject:")
    print(
        best_by_subject[
            ["subject", "method", "balanced_accuracy", "roc_auc"]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )

    _plot_subject_accuracy(ok, plot_dir / "balanced_accuracy_by_subject.png", plt)
    _plot_method_summary(summary, plot_dir / "method_mean_balanced_accuracy.png", plt)
    _plot_error_rates(ok, plot_dir / "class_error_rates_by_method.png", plt)
    _plot_best_method_counts(
        best_by_subject,
        plot_dir / "best_method_counts.png",
        plt,
    )

    print(f"\nSaved summary tables and plots in: {plot_dir}")


def _plot_subject_accuracy(df: object, path: Path, plt: object) -> None:
    pivot = df.pivot(index="subject", columns="method", values="balanced_accuracy")
    ax = pivot.plot(kind="bar", figsize=(12, 6), width=0.82)
    ax.set_title("Balanced Accuracy by Subject")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Method", ncols=2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_method_summary(summary: object, path: Path, plt: object) -> None:
    ax = summary.plot(
        x="method",
        y="mean_balanced_accuracy",
        yerr="std_balanced_accuracy",
        kind="bar",
        figsize=(8, 5),
        legend=False,
        capsize=4,
    )
    ax.set_title("Mean Balanced Accuracy Across Subjects")
    ax.set_xlabel("Method")
    ax.set_ylabel("Mean balanced accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_error_rates(df: object, path: Path, plt: object) -> None:
    summary = (
        df.groupby("method", as_index=False)
        .agg(
            class0_to_class1=("class0_to_class1_rate", "mean"),
            class1_to_class0=("class1_to_class0_rate", "mean"),
        )
        .set_index("method")
    )
    ax = summary.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Mean Class Error Rates")
    ax.set_xlabel("Method")
    ax.set_ylabel("Error rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(["class 0 -> class 1", "class 1 -> class 0"])
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_best_method_counts(best_by_subject: object, path: Path, plt: object) -> None:
    counts = best_by_subject["method"].value_counts().sort_index()
    ax = counts.plot(kind="bar", figsize=(7, 5))
    ax.set_title("Best Method Count Across Subjects")
    ax.set_xlabel("Method")
    ax.set_ylabel("Number of subjects")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark BCI Competition IV 2b decoders.")
    parser.add_argument(
        "--data-dir",
        default="BCICIV_2b_gdf",
        help="Directory containing B0x0yT.gdf files.",
    )
    parser.add_argument(
        "--subject",
        default="02",
        help="Two-digit subject id, e.g. 02, or 'all'.",
    )
    parser.add_argument(
        "--mode",
        choices=("left_right_debug", "rest_active_pseudo"),
        default="rest_active_pseudo",
    )
    parser.add_argument("--window-s", type=float, default=1.0)
    parser.add_argument(
        "--save-csv",
        default=None,
        help="Optional CSV path for benchmark results.",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Optional output directory for summary tables and PNG plots.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only save the raw CSV; do not generate summary plots.",
    )
    args = parser.parse_args(argv)

    data_dir = ROOT / args.data_dir
    subjects = (
        [f"{idx:02d}" for idx in range(1, 10)]
        if args.subject.lower() == "all"
        else [args.subject.zfill(2)]
    )

    rows: list[dict[str, object]] = []
    for subject in subjects:
        rows.extend(
            run_subject(
                data_dir=data_dir,
                subject=subject,
                mode=args.mode,
                window_s=args.window_s,
            )
        )

    if args.save_csv is not None:
        output = ROOT / args.save_csv
        save_results(rows, output)
        print(f"\nSaved benchmark CSV: {output}")
        if not args.no_plots:
            plot_dir = ROOT / args.plot_dir if args.plot_dir is not None else None
            save_summary_outputs(rows, csv_path=output, plot_dir=plot_dir)


if __name__ == "__main__":
    main()
