from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib as jb

def build_preproccessor(x_train):
    categorical_features = x_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = x_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"Categorical features: {categorical_features}")
    print(f"Numerical features: {numerical_features}")

    categorical_pipeline = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    numerical_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preproccess = ColumnTransformer(
        transformers=[
            ('cat', categorical_pipeline, categorical_features),
            ('num', numerical_pipeline, numerical_features),
        ]
    )
    preproccess.fit(x_train)
    return preproccess

def transform_data(preproccess, x):
    return preproccess.transform(x)

    



