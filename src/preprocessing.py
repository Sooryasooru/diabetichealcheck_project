"""Preprocessing module."""

from sklearn.preprocessing import StandardScaler


def preprocess_data(df):
    """Scale features."""

    features = [
        "age_num",
        "hospital_usage",
        "glucose_level",
        "bmi",
    ]

    x = df[features]

    scaler = StandardScaler()

    x_scaled = scaler.fit_transform(x)

    return x_scaled
