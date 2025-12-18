from documind.config.configuration import ConfigurationManager
from documind.components.model_evaluation import ModelEvaluation
from documind import logger

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            model_evaluation.evaluation()
        except Exception as e:
            raise e