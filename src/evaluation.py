from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from .config import LABELLED_DATASET_CANDIDATES
from .preprocessing import normalize_sentiment_labels, prepare_dataframe
from .sentiment_models import (
    build_logistic_model,
    build_svm_model,
    save_model,
    vader_predict,
    transformer_predict,
)

LABELS = ["negative", "neutral", "positive"]


def locate_labelled_dataset():
    for path in LABELLED_DATASET_CANDIDATES:
        if path.exists():
            return path
    return None


def load_labelled_dataset():
    path = locate_labelled_dataset()
    if path is None:
        expected = "\n".join(str(p) for p in LABELLED_DATASET_CANDIDATES)
        raise FileNotFoundError(
            "No internal labelled dataset was found. Expected one of:\n" + expected
        )

    df = pd.read_csv(path)
    df = prepare_dataframe(df)
    df = normalize_sentiment_labels(df)

    if len(df) < 30:
        raise ValueError("The labelled dataset is too small for the evaluation pipeline.")

    return df, path


def metric_result(model_name, y_true, y_pred, cv_mean=None, cv_std=None):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cv_mean_f1": cv_mean,
        "cv_std_f1": cv_std,
        "predictions": list(y_pred),
        "report": classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=LABELS,
        ),
    }


def evaluate_classical_model(model, model_name, x_train, y_train, x_test, y_test, filename):
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    folds = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    cv_scores = cross_val_score(
        model,
        x_train,
        y_train,
        cv=folds,
        scoring="f1_weighted",
    )

    save_model(model, filename)

    return metric_result(
        model_name,
        y_test,
        predictions,
        cv_mean=float(cv_scores.mean()),
        cv_std=float(cv_scores.std()),
    )


def run_evaluation(roberta_classifier=None):
    df, dataset_path = load_labelled_dataset()

    x = df["processed_text"].astype(str)
    y = df["human-sentiment"].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    results = []
    detailed = {}

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------
    lr_result = evaluate_classical_model(
        build_logistic_model(),
        "TF-IDF + Logistic Regression",
        x_train,
        y_train,
        x_test,
        y_test,
        "tfidf_logistic_regression.joblib",
    )
    results.append(lr_result)
    detailed[lr_result["model"]] = lr_result

    # --------------------------------------------------------
    # Linear SVM
    # --------------------------------------------------------
    svm_result = evaluate_classical_model(
        build_svm_model(),
        "TF-IDF + Linear SVM",
        x_train,
        y_train,
        x_test,
        y_test,
        "tfidf_linear_svm.joblib",
    )
    results.append(svm_result)
    detailed[svm_result["model"]] = svm_result

    # --------------------------------------------------------
    # VADER on SAME holdout set
    # --------------------------------------------------------
    vader_labels, vader_scores = vader_predict(
        df.loc[x_test.index, "sentiment_text"]
    )
    vader_result = metric_result(
        "VADER",
        y_test,
        vader_labels,
    )
    vader_result["scores"] = vader_scores
    results.append(vader_result)
    detailed["VADER"] = vader_result

    # --------------------------------------------------------
    # RoBERTa on SAME holdout set
    # --------------------------------------------------------
    if roberta_classifier is not None:
        roberta_labels, roberta_scores = transformer_predict(
            df.loc[x_test.index, "sentiment_text"],
            classifier=roberta_classifier,
        )
        roberta_result = metric_result(
            "RoBERTa",
            y_test,
            roberta_labels,
        )
        roberta_result["scores"] = roberta_scores
        results.append(roberta_result)
        detailed["RoBERTa"] = roberta_result

    results_df = pd.DataFrame([
        {
            key: value
            for key, value in result.items()
            if key in {
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "cv_mean_f1",
                "cv_std_f1",
            }
        }
        for result in results
    ])

    return {
        "dataset": df,
        "dataset_path": str(dataset_path),
        "train_size": len(x_train),
        "test_size": len(x_test),
        "y_test": y_test,
        "results": results_df,
        "details": detailed,
    }
