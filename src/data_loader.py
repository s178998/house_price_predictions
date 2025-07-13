import pandas as  pd
from sklearn.model_selection import train_test_split

def load_data(path, target_column, drop_duplicates=False, test_size=.2, random_state=42):
    df = pd.read_csv(path)
    if drop_duplicates:
        df = df.drop_duplicates()
    x = df.drop(target_column, axis=1)
    y = df[target_column]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test