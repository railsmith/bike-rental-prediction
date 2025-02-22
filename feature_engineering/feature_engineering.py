import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Ensure 'dteday' exists and convert to datetime if needed
        if 'dteday' in X_copy.columns:
            X_copy['dteday'] = pd.to_datetime(X_copy['dteday'], errors='coerce')
            # Extract day names (Weekdays) from the dteday column
            X_copy['weekday'] = X_copy['dteday'].dt.strftime('%a')  # Extract short weekday name like 'Mon', 'Tue'
        else:
            raise ValueError("'dteday' column is missing from the dataset")

        return X_copy


class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self):
        # Initialize with no parameters needed for this transformation
        self.most_frequent_value = None

    def fit(self, X, y=None):
        """ Find the most frequent category in the 'weathersit' column """
        # Calculate the most frequent value in the 'weathersit' column
        self.most_frequent_value = X['weathersit'].mode()[0]  # Mode returns a series, take the first element
        return self

    def transform(self, X):
        """ Impute missing values in 'weathersit' with the most frequent category value """
        # Make a copy to avoid modifying the original data
        X_copy = X.copy()

        # Replace missing values with the most frequent value
        X_copy['weathersit'] = X_copy['weathersit'].fillna(self.most_frequent_value)

        return X_copy


class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """
    
    def __init__(self):
        # Define the mappings for each column as per their ordinal relationships
        self.mappings = {
            'year': {2011: 0, 2012: 1},
            'month': {
                'January': 1, 'February': 2, 'March': 3, 'April': 4,
                'May': 5, 'June': 6, 'July': 7, 'August': 8,
                'September': 9, 'October': 10, 'November': 11, 'December': 12
            },
            'season': {
                'spring': 1, 'summer': 2, 'fall': 3, 'winter': 4
            },
            'weathersit': {
                'Clear': 1, 'Mist': 2, 'Light Rain': 3, 'Heavy Rain': 4
            },
            'holiday': {
                'No': 0, 'Yes': 1
            },
            'workingday': {
                'No': 0, 'Yes': 1
            },
            'hr': {
                '12am': 0, '1am': 1, '2am': 2, '3am': 3, '4am': 4, '5am': 5, '6am': 6,
                '7am': 7, '8am': 8, '9am': 9, '10am': 10, '11am': 11, '12pm': 12,
                '1pm': 13, '2pm': 14, '3pm': 15, '4pm': 16, '5pm': 17, '6pm': 18,
                '7pm': 19, '8pm': 20, '9pm': 21, '10pm': 22, '11pm': 23
            }
        }
    
    def fit(self, X, y=None):
        """ This method doesn't need to do anything as we are using predefined mappings """
        return self

    def transform(self, X):
        """ Apply the mappings to the appropriate columns """
        # Make a copy of the DataFrame to avoid modifying the original data
        X_copy = X.copy()
        
        # Apply the mappings to each column in the mappings dictionary
        for col, mapping in self.mappings.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].map(mapping).fillna(X_copy[col])
        
        return X_copy

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - To upper-bound if the value is higher than the upper-bound
        - To lower-bound if the value is lower than the lower-bound
    """
    
    def __init__(self, method='IQR', factor=1.5):
        """
        Initialize the OutlierHandler with the method to calculate bounds.
        
        :param method: Method to calculate bounds ('IQR' or 'z-score')
        :param factor: The factor to use for calculating the bounds (default is 1.5 for IQR)
        """
        self.method = method
        self.factor = factor
        self.lower_bounds = None
        self.upper_bounds = None
    
    def fit(self, X, y=None):
        """ 
        Compute the lower and upper bounds for each numerical column.
        """
        # Select only numerical columns
        X_num = X.select_dtypes(include=['number'])
        
        if self.method == 'IQR':
            # Using IQR method to compute bounds
            Q1 = X_num.quantile(0.25)
            Q3 = X_num.quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds = Q1 - self.factor * IQR
            self.upper_bounds = Q3 + self.factor * IQR
        elif self.method == 'z-score':
            # Using Z-score method to compute bounds
            mean = X_num.mean()
            std = X_num.std()
            self.lower_bounds = mean - self.factor * std
            self.upper_bounds = mean + self.factor * std
        else:
            raise ValueError("Unsupported method. Choose 'IQR' or 'z-score'.")
        
        return self
    
    def transform(self, X):
        """
        Replace the outliers with upper and lower bounds in numerical columns.
        """
        # Select only numerical columns
        X_copy = X.copy()
        X_num = X_copy.select_dtypes(include=['number'])
        
        # Iterate over all numerical columns
        for col in X_num.columns:
            # Apply upper and lower bounds to each numerical column
            X_copy[col] = X_copy[col].clip(lower=self.lower_bounds[col], upper=self.upper_bounds[col])
        
        return X_copy


class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode the 'weekday' column """
    
    def __init__(self):
        """
        Initialize the transformer. No specific parameters are needed for this transformation.
        """
        self.columns_ = None  # To store the column names after encoding (optional)
    
    def fit(self, X, y=None):
        """
        Fit method (learn column names for encoding).
        """
        # Check if 'weekday' column exists in the DataFrame
        if 'weekday' not in X.columns:
            raise ValueError("'weekday' column not found in the DataFrame")
        
        # Perform one-hot encoding of the 'weekday' column and store column names
        weekday_dummies = pd.get_dummies(X['weekday'], prefix='weekday')
        self.columns_ = weekday_dummies.columns.tolist()  # Store column names for future reference
        
        return self
    
    def transform(self, X):
        """
        One-hot encode the 'weekday' column in the DataFrame.
        """
        X_copy = X.copy()  # Copy the DataFrame to avoid modifying the original data
        
        # Check if 'weekday' column exists in the DataFrame
        if 'weekday' not in X_copy.columns:
            raise ValueError("'weekday' column not found in the DataFrame")
        
        # Perform one-hot encoding of the 'weekday' column
        weekday_dummies = pd.get_dummies(X_copy['weekday'], prefix='weekday')
        
        # Ensure that the columns match those learned during the fit
        # Reindex to the training columns, filling with zeros for any new categories
        weekday_dummies = weekday_dummies.reindex(columns=self.columns_, fill_value=0)
        
        # Drop the original 'weekday' column
        X_copy = X_copy.drop(columns=['weekday'])
        
        # Concatenate the original DataFrame with the one-hot encoded weekday columns
        X_copy = pd.concat([X_copy, weekday_dummies], axis=1)
        
        return X_copy


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop if columns_to_drop else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=self.columns_to_drop, errors='ignore')