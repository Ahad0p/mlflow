# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis.

import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import logging
import dagshub
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Setup DagsHub with credentials
    try:
        dagshub.init(
            repo_owner="Ahad0p",
            repo_name="mlflow",
            mlflow=True
        )
        logger.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
    except Exception as e:
        logger.error(f"Failed to initialize DAGShub: {e}")
        logger.warning("Falling back to local tracking URI")
        mlflow.set_tracking_uri("file:./mlruns")

    # Validate credentials from .env
    if not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        logger.warning("MLFLOW_TRACKING_PASSWORD not set. DAGShub may not log properly.")

    # Download dataset
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception("Failed to load dataset")
        sys.exit(1)

    train, test = train_test_split(data, test_size=0.25, random_state=42)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    try:
        with mlflow.start_run():
            lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            lr.fit(train_x, train_y)

            predicted_qualities = lr.predict(test_x)
            rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

            print(f"Elasticnet model (alpha={alpha:.6f}, l1_ratio={l1_ratio:.6f}):")
            print(f"  RMSE: {rmse}")
            print(f"  MAE: {mae}")
            print(f"  R2: {r2}")

            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            signature = infer_signature(train_x, lr.predict(train_x))

            try:
                # Log model to DagsHub or fallback store
                mlflow.sklearn.log_model(
                    sk_model=lr,
                    artifact_path="model",
                    signature=signature
                )
            except mlflow.exceptions.RestException as e:
                if "unsupported endpoint" in str(e):
                    logger.warning("DagsHub does not support full model registry. Saving as artifact instead.")
                    import joblib
                    model_path = "model.pkl"
                    joblib.dump(lr, model_path)
                    mlflow.log_artifact(model_path, artifact_path="model")
                    os.remove(model_path)
                    mlflow.log_param("model_note", "Logged as artifact due to DagsHub limitation")
                else:
                    raise e

    except Exception as e:
        logger.exception("Training or logging failed.")
        sys.exit(1)
