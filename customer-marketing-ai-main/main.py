from src.cluster import main as cluster_main
from src.evaluate import main as evaluate_main
from src.ingest import main as ingest_main
from src.preprocess import main as preprocess_main
from src.train import main as train_main


def run_pipeline() -> dict:
    """Run the whole project pipeline from ingestion to evaluation."""
    ingest_main()
    preprocess_main()
    cluster_main()
    train_main()
    return evaluate_main()


if __name__ == "__main__":
    run_pipeline()
