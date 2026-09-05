import argparse
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# DAGSHUB + MLFLOW INITIALIZATION
# ============================================================

dagshub.init(
    repo_owner="adityakmr152000",
    repo_name="mlops-project-customer-churn",
    mlflow=True,
)


# ============================================================
# READ PARAMETERS
# ============================================================

def read_params(config_path):
    """
    Read parameters from params.yaml.

    Args:
        config_path: Path to params.yaml

    Returns:
        Configuration dictionary
    """

    with open(config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    return config


# ============================================================
# SPLIT FEATURES AND TARGET
# ============================================================

def get_feat_and_target(df, target):
    """
    Separate features and target.

    Args:
        df: Input dataframe
        target: Target column name

    Returns:
        X: Features
        y: Target
    """

    X = df.drop(columns=[target])
    y = df[target]

    return X, y


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(y_test, predictions):
    """
    Calculate classification metrics.
    """

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print("\n==============================")
    print("CLASSIFICATION REPORT")
    print("==============================")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print("\n==============================")
    print("CONFUSION MATRIX")
    print("==============================")

    print(confusion_matrix(y_test, predictions))

    print("\n==============================")
    print("MODEL METRICS")
    print("==============================")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    return accuracy, precision, recall, f1


# ============================================================
# TRAIN + EVALUATE + MLFLOW
# ============================================================

def train_and_evaluate(config_path):

    # --------------------------------------------------------
    # 1. Read configuration
    # --------------------------------------------------------

    config = read_params(config_path)

    # --------------------------------------------------------
    # 2. Read paths and parameters
    # --------------------------------------------------------

    train_data_path = config["processed_data_config"]["train_data_csv"]
    test_data_path = config["processed_data_config"]["test_data_csv"]

    target = config["raw_data_config"]["target"]

    max_depth = config["random_forest"]["max_depth"]
    n_estimators = config["random_forest"]["n_estimators"]

    mlflow_config = config["mlflow_config"]

    experiment_name = mlflow_config["experiment_name"]
    run_name = mlflow_config["run_name"]
    registered_model_name = mlflow_config["registered_model_name"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    # --------------------------------------------------------
    # 3. Configure MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(remote_server_uri)

    mlflow.set_experiment(experiment_name)

    # --------------------------------------------------------
    # 4. Load processed data
    # --------------------------------------------------------

    print("\nLoading training data...")
    train = pd.read_csv(train_data_path)

    print("Loading testing data...")
    test = pd.read_csv(test_data_path)

    print(f"Training data shape : {train.shape}")
    print(f"Testing data shape  : {test.shape}")

    # --------------------------------------------------------
    # 5. Separate features and target
    # --------------------------------------------------------

    train_x, train_y = get_feat_and_target(
        train,
        target,
    )

    test_x, test_y = get_feat_and_target(
        test,
        target,
    )

    print(f"\nNumber of features: {train_x.shape[1]}")

    # --------------------------------------------------------
    # 6. Start MLflow experiment
    # --------------------------------------------------------

    with mlflow.start_run(run_name=run_name):

        print("\nStarting model training...")

        # ----------------------------------------------------
        # 7. Create Random Forest model
        # ----------------------------------------------------

        model = RandomForestClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=42,
        )

        # ----------------------------------------------------
        # 8. Train model
        # ----------------------------------------------------

        model.fit(
            train_x,
            train_y,
        )

        print("Model training completed.")

        # ----------------------------------------------------
        # 9. Make predictions
        # ----------------------------------------------------

        predictions = model.predict(test_x)

        # ----------------------------------------------------
        # 10. Evaluate model
        # ----------------------------------------------------

        accuracy, precision, recall, f1 = evaluate_model(
            test_y,
            predictions,
        )

        # ----------------------------------------------------
        # 11. Log parameters to MLflow
        # ----------------------------------------------------

        mlflow.log_param(
            "model_type",
            "RandomForestClassifier",
        )

        mlflow.log_param(
            "max_depth",
            max_depth,
        )

        mlflow.log_param(
            "n_estimators",
            n_estimators,
        )

        mlflow.log_param(
            "random_state",
            42,
        )

        # ----------------------------------------------------
        # 12. Log metrics to MLflow
        # ----------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy,
        )

        mlflow.log_metric(
            "precision",
            precision,
        )

        mlflow.log_metric(
            "recall",
            recall,
        )

        mlflow.log_metric(
            "f1_score",
            f1,
        )

        # ----------------------------------------------------
        # 13. Log trained model to MLflow
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=registered_model_name,
        )

        # ----------------------------------------------------
        # 14. Print MLflow information
        # ----------------------------------------------------

        run_id = mlflow.active_run().info.run_id

        print("\n==============================")
        print("MLFLOW RUN COMPLETED")
        print("==============================")

        print(f"Experiment : {experiment_name}")
        print(f"Run name   : {run_name}")
        print(f"Run ID     : {run_id}")
        print(f"Model      : {registered_model_name}")

        print("\nMLflow tracking URI:")
        print(remote_server_uri)

        print("\nModel successfully logged to MLflow.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="params.yaml",
        help="Path to params.yaml",
    )

    args = parser.parse_args()

    train_and_evaluate(
        config_path=args.config
    )