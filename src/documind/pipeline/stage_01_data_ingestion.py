from documind.config.configuration import ConfigurationManager
from documind.components.data_ingestion import DataIngestion
from documind import logger

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logger.info(">>> Stage 01: Data Ingestion started <<<")
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_data()
            logger.info(">>> Stage 01: Data Ingestion completed <<< \n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e