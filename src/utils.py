import pandas as pd
import matplotlib.pyplot as plt
from typing import Any
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV, train_test_split, TunedThresholdClassifierCV
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay


def plot_label_distribution(target_labels: pd.Series, title: str):
    """
    Plot a bar plot of class labels distribution

    Args:
        traget_labels: Target feature class labels
        title: Title of the plot
    """
    plt.bar(target_labels.value_counts().index, target_labels.value_counts().values, color=["blue", "orange"])
    plt.xticks([0, 1], ["Legit", "Fraud"])
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(title)
    plt.show()


def plot_confusion_matrix_from_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series):
    """
    Display the confusion matrix of a pipeline

    Args:
        pipeline: The pipeline
        X: Input features matrix
        y: Target labels series
    """
    ConfusionMatrixDisplay.from_estimator(pipeline, X, y)
    plt.show()


def plot_roc_curve_from_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, plot_chance_level: bool = False):
    """
    Display ROC curve form pipeline

    Args:
        pipeline: The pipeline to plot a ROC curve of
        X: Input features matrix
        y: Target labels series
        plot_chance_level: Whether to plot the chance level
    """
    RocCurveDisplay.from_estimator(pipeline, X, y, plot_chance_level=plot_chance_level)
    plt.show()


def cross_validate_pipeline(
        pipeline: Pipeline, 
        X: pd.DataFrame, 
        y: pd.Series, 
        cv: StratifiedKFold, 
        scoring_metrics: dict[str, str]
    ):
    """
    Cross validate a pipeline

    Args:
        pipeline: The pipeline to cross-validate
        X: Input features matrix
        y: Target labels series
        cv: Cross-validation iterator
        scoring_metrics: Metrics to evaluate the pipeline on each split
    """
    results = cross_validate(
        pipeline,
        X, y,
        cv=cv,
        scoring=scoring_metrics,
        return_train_score=False,
    )

    for metric in scoring_metrics:
        print(f"{metric}: {results['test_' + metric].mean():.4f}")


def tune_pipeline_hyper_parameters(
        pipeline: Pipeline, 
        param_dist: dict[str, Any], 
        n_iter: int, 
        cv: StratifiedKFold, 
        scoring_metric: str, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> tuple[BaseEstimator, dict, float]:
    """
    Tune the hyper-parameters of a pipeline using random searching

    Args:
        pipeline: The pipeline to tune
        param_dist: Hyper-parameter distributions to choose values from
        n_iter: The number of iterations of random searching
        cv: Cross-validation iterator
        scoring_metric: Metric to evaluate the model on each split
        X: input feature matrix
        y: Target labels series

    Returns:
        The best performing pipeline, the optimal hyper-parameter values and the best score
    """
    rand_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,                      
        cv=cv,
        scoring=scoring_metric,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    rand_search.fit(X, y)

    return rand_search.best_estimator_, rand_search.best_params_, rand_search.best_score_


def split_data(X: pd.DataFrame, y: pd.Series, test_size: int | float) -> tuple:
    """
    Split data into training set and test set

    Args:
        X: Input features matrix
        y: Target labels series
        test_size: Size of the test set. Number of samples if int, fraction from the training set otherwise
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test


def evaluate_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series):
    """
    Evaluate a trained pipeline

    Args:
        pipeline: The pipeline to evaluate
        X: Input features matrix
        y: Target labels series
    """
    y_pred = pipeline.predict(X) # predicted class labels
    y_pred_proba = pipeline.predict_proba(X)[:, 1] # confidence scores

    print(f"Test set f1 score: {f1_score(y, y_pred):.4f}")
    print(f"Test set average precision: {average_precision_score(y, y_pred):.4f}")
    print(f"Test set roc auc score: {roc_auc_score(y, y_pred_proba):.4f}")


def tune_classification_threshold(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, scoring_metric: str, cv: StratifiedKFold):
    """
    Tune classification threshold of a pipeline

    Args:
        pipeline: The pipeline to tune
        X: Input features matrix
        y: Target labels series

    Returns:
        The pipeline with the optimal threshold
    """
    thresholded_clf = TunedThresholdClassifierCV(
        pipeline, 
        scoring=scoring_metric, 
        cv=cv,
        n_jobs=-1,
        random_state=42,
    )

    thresholded_clf.fit(X, y)
    print(f"Cut-off point found at {thresholded_clf.best_threshold_:.3f}")

    return thresholded_clf