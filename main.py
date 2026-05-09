from src.data_cleaning import load_data
from src.preprocessing import preprocess_data
from src.clustering import train_kmeans
from src.evaluation import evaluate_model


def main():
    df = load_data("data/diabetes_dataset.csv")

    x_scaled = preprocess_data(df)

    model, labels = train_kmeans(x_scaled)

    results = evaluate_model(x_scaled, labels)

    print(results)


if __name__ == "__main__":
    main()
