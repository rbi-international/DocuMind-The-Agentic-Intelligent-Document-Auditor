import os
import pandas as pd
from documind import logger
from documind.entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = True
            
            # Read the ingested data
            data = pd.read_csv(os.path.join("artifacts", "data_ingestion", "train.csv"))
            all_cols = list(data.columns)

            # Fix: all_schema already contains the column names as keys
            schema_keys = self.config.all_schema.keys()

            logger.info(f"Validating columns. Expected: {list(schema_keys)}, Found: {all_cols}")

            for col in all_cols:
                if col not in schema_keys:
                    validation_status = False
                    logger.error(f"Validation Error: Column '{col}' is not defined in schema.yaml")
                else:
                    # Optional: Check data type consistency
                    logger.info(f"Column '{col}' validated.")

            # Write status to file
            with open(self.config.report_file, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            if validation_status:
                logger.info("Data Validation Stage Successful.")
            else:
                logger.error("Data Validation Stage Failed.")

            return validation_status

        except Exception as e:
            logger.exception(e)
            raise e