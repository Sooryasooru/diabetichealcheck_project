import pandas as pd

from src.preprocessing import preprocess_data


def test_preprocess_data():
    data = {
        "age_num": [20, 30],
        "hospital_usage": [1, 2],
        "glucose_level": [100, 120],
        "bmi": [22.5, 25.1],
    }

    df = pd.DataFrame(data)

    result = preprocess_data(df)

    assert result.shape[0] == 2
