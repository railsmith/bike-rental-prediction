from feature_engineering import feature_engineering
from config.config import Config
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def build_pipeline():

    weekday_imputer = feature_engineering.WeekdayImputer()
    weathersit_imputer = feature_engineering.WeathersitImputer()
    mapper = feature_engineering.Mapper()
    outlier_handler = feature_engineering.OutlierHandler(method='IQR', factor=1.5)
    weekday_encoder = feature_engineering.WeekdayOneHotEncoder()
    column_dropper = feature_engineering.ColumnDropper(columns_to_drop=['dteday'])

    # Define the regressor (RandomForestRegressor as an example)
    regressor = RandomForestRegressor(n_estimators=Config.NESTIMATORS, random_state=Config.RANDOM_STATE)

    pipeline = Pipeline([
      ('weekday_imputer', weekday_imputer),
      ('weathersit_imputer', weathersit_imputer),
      ('drop_columns', column_dropper),
      ('mapper', mapper),
      ('outlier_handler', outlier_handler),
      ('weekday_encoder', weekday_encoder),
      ('scaler', StandardScaler()),
      ('regressor', regressor) 
    ])

    return pipeline

