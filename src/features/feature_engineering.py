import pandas as pd
import logging
import yaml

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineering:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.pipeline = None

    def build_pipeline(self, df):
        try:
            #detect columns
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            logger.info(f"Categorical columns: {categorical_columns}")
            logger.info(f"Numerical columns: {numerical_columns}")
            logger.info("="*50)

            categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            numerical_transformer = StandardScaler()

            #column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', categorical_transformer, categorical_columns),
                    ('num', numerical_transformer, numerical_columns)
                ]
            )

            self.pipeline = Pipeline(steps=[
                ('preprocessing', preprocessor)
            ])

            logger.info("Pipeline built successfully")
            logger.info("="*50)

        except Exception as e:
            logger.error(f"Error building pipeline: {e}")
            logger.error("="*50)
            raise

    def process_features(self, df, is_training=True):
        try:
            
            df_processed = df.copy()

            if is_training:
                self.build_pipeline(df_processed)
                processed_array = self.pipeline.fit_transform(df_processed)
            else:
                if self.pipeline is None:
                    raise ValueError("Pipeline has not been built. Call process_features with is_training=True first.")
                processed_array = self.pipeline.transform(df_processed)

            #convert array back to df
            column_names = self.pipeline.named_steps['preprocessing'].get_feature_names_out()
            df_processed = pd.DataFrame(
                processed_array, 
                columns=column_names,
                index=df.index)
            
            logger.info("Feature preprocessing completed successfully")
            logger.info("="*50)

            return df_processed

        except Exception as e:
            logger.error(f"Error processing features: {e}")
            logger.error("="*50)
            raise

    def create_features(self, df):
        try:
            df_new = df.copy()
            
            # Example feature: Balance per product
            df_new['BalancePerProduct'] = df_new['Balance'] / (df_new['NumOfProducts'] + 1)
            
            # Example feature: Customer engagement score
            df_new['EngagementScore'] = df_new['IsActiveMember'] * 0.5 + df_new['HasCrCard'] * 0.3 + \
                                      (df_new['NumOfProducts'] / 4) * 0.2
            
            logger.info("Feature creation completed successfully")
            logger.info("="*50)
            return df_new
            
        except Exception as e:
            logger.error(f"Error in feature creation: {str(e)}")
            logger.error("="*50)
            raise

    def smote(self, X_train, y_train):
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=self.config['data']['random_state'])
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            logger.info("SMOTE oversampling completed successfully")
            logger.info("="*50)

            return X_train_resampled, y_train_resampled
        except Exception as e:
            logger.error(f"Error in SMOTE oversampling: {str(e)}")
            logger.error("="*50)
            raise

    def select_k_features(self, X_train, y_train, X_val, X_test, k=10):
        try:
            from sklearn.feature_selection import SelectKBest, f_classif
            selector = SelectKBest(score_func=f_classif, k=k)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_val_selected = selector.transform(X_val)
            X_test_selected = selector.transform(X_test)

            selected_feature_names = selector.get_feature_names_out(input_features=X_train.columns).tolist()
            logger.info(f"Selected {k} features: {selected_feature_names}")
            logger.info("="*50)

            X_train_selected = pd.DataFrame(X_train_selected, columns=selected_feature_names)
            X_val_selected = pd.DataFrame(X_val_selected, columns=selected_feature_names, index=X_val.index)
            X_test_selected = pd.DataFrame(X_test_selected, columns=selected_feature_names, index=X_test.index)

            logger.info(f"Feature selection completed successfully. Selected top {k} features.")
            logger.info("="*50)

            return X_train_selected, X_val_selected, X_test_selected    
        
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            logger.error("="*50)
            raise
        