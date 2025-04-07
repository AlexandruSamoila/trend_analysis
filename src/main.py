from data_vizualization import DataVizualization
from label_data import LabelData
from trend_recommender import TrendRecommender


def main():

    data_path = "data/trndset-dump-Top_Creators_(DK).json"
    data_vizualization = DataVizualization(data_path)
    data_vizualization.run()
    # trend_recommender = TrendRecommender(data_path)


if __name__ == "__main__":
    main()
