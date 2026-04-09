from __future__ import annotations

try:
    from prefect import flow, task
except ImportError:
    flow = None
    task = None

try:
    from .cluster import main as cluster_main
    from .evaluate import main as evaluate_main
    from .ingest import main as ingest_main
    from .preprocess import main as preprocess_main
    from .train import main as train_main
except ImportError:
    from cluster import main as cluster_main
    from evaluate import main as evaluate_main
    from ingest import main as ingest_main
    from preprocess import main as preprocess_main
    from train import main as train_main


if flow is not None and task is not None:
    @task
    def ingest_task():
        return ingest_main()


    @task
    def preprocess_task():
        return preprocess_main()


    @task
    def cluster_task():
        return cluster_main()


    @task
    def train_task():
        return train_main()


    @task
    def evaluate_task():
        return evaluate_main()


    @flow(name="marketing-pipeline")
    def marketing_pipeline():
        ingest_task()
        preprocess_task()
        cluster_task()
        train_task()
        return evaluate_task()


else:
    def marketing_pipeline():
        """Fallback sequential pipeline when Prefect is not installed."""
        print("Prefect is not installed. Running the pipeline sequentially.")
        ingest_main()
        preprocess_main()
        cluster_main()
        train_main()
        return evaluate_main()


if __name__ == "__main__":
    marketing_pipeline()
