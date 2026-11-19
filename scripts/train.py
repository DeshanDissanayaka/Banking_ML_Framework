from banking_ml.utils.config import load_yaml
from banking_ml.pipelines.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    config = load_yaml("config/config.yaml")
    params = load_yaml("config/params.yaml")
    results = TrainingPipeline(config, params).run()
    print("Training completed. Test metrics:")
    print(results)
