from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def split_data(df, target_col='Churn', test_size=0.2, random_state=42):
    """Split a cleaned DataFrame into train/test features and target, arranged by target."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Fit a StandardScaler on X_train only, then transform both train and test sets."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_logistic_regression(X_train_scaled, y_train):
    """Train a Logistic Regression model on scaled training data."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model