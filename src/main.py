from data_vizualization import DataVizualization
from label_data import LabelData
from trend_recommender import TrendRecommender
from sentiment_analysis import SentimentAnalysis
import os
import pandas as pd


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_menu():
    print("\n=== Trend Analysis Menu ===")
    print("1. Run Data Visualization")
    print("2. Run Trend Recommender")
    print("3. Run Sentiment Analysis")
    print("4. Run Data Labeling")
    print("0. Exit")
    print("=========================")


def main():
    data_path = "data/trndset-dump-Top_Creators_(DK).json"

    while True:
        clear_screen()
        print_menu()
        choice = input("\nEnter your choice (0-5): ")

        if choice == "0":
            print("Exiting...")
            break
        elif choice == "1":
            print("\nRunning Data Visualization...")
            data_viz = DataVizualization(data_path)
            data_viz.run()
        elif choice == "2":
            print("\nRunning Trend Recommender...")
            recommender = TrendRecommender(data_path)
            # recommender.plot_trend_clusters()
            trend = "Home Renovation and DIY Project Inspirations"
            print("Finding Similar Trends for: ", trend)
            print(recommender.find_similar_trends(trend))
        elif choice == "3":
            print("\nRunning Sentiment Analysis...")
            data_df = pd.read_json(data_path, encoding="utf8")
            model_path = "checkpoints/20250407_1450_roberta_sentiment"
            sentiment_analysis = SentimentAnalysis(model_path)
            print(sentiment_analysis.analyze_trend_sentiment(data_df.iloc[1]))

        elif choice == "4":
            print("\nRunning Data Labeling...")
            labeler = LabelData(data_path)
            labeler.run()
        else:
            print("Invalid choice. Please try again.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
