import dagshub
dagshub.init(repo_owner='Ahad0p', repo_name='mlflow', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

  https://dagshub.com/Ahad0p/mlflow.mlflow
  69e08b7864798c3e1c586a5fb76798bd743cc426