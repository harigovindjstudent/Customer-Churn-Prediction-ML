import pandas as pd
import logging
import yaml

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineering:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.pipeline = None

    def build_pipeline(self, df):
        try:
            # Automatically detect columns
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

            logger.info(f"Categorical columns: {categorical_columns}")
            logger.info(f"Numerical columns: {numerical_columns}")

            # Preprocessing pipelines
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')
            numerical_transformer = StandardScaler()

            # Column Transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', categorical_transformer, categorical_columns),
                    ('num', numerical_transformer, numerical_columns)
                ]
            )

            # Final pipeline (only preprocessing for now)
            self.pipeline = Pipeline(steps=[
                ('preprocessing', preprocessor)
            ])

            logger.info("Pipeline built successfully")

        except Exception as e:
            logger.error(f"Error building pipeline: {e}")
            raise

    def process_features(self, df, is_training=True):
        try:
            df_processed = df.copy()

            # Build pipeline only during training
            if is_training:
                self.build_pipeline(df_processed)
                processed_array = self.pipeline.fit_transform(df_processed)
            else:
                processed_array = self.pipeline.transform(df_processed)

            # Convert to DataFrame (optional but useful)
            feature_names = self.pipeline.named_steps['preprocessing'].get_feature_names_out()

            df_processed = pd.DataFrame(
                processed_array,
                columns=feature_names,
                index=df.index
            )

            logger.info("Feature preprocessing completed successfully")

            return df_processed

        except Exception as e:
            logger.error(f"Error processing features: {e}")
            raise

    def create_features(self, df):
        try:
            df_new = df.copy()
            
            # Example feature: Balance per product
            df_new['BalancePerProduct'] = df_new['Balance'] / (df_new['NumOfProducts'] + 1)
            
            # Example feature: Customer engagement score
            df_new['EngagementScore'] = (
                df_new['IsActiveMember'] * 0.5 +
                df_new['HasCrCard'] * 0.3 +
                (df_new['NumOfProducts'] / 4) * 0.2
            )
            
            logger.info("Feature creation completed successfully")
            return df_new
            
        except Exception as e:
            logger.error(f"Error in feature creation: {str(e)}")
            raise