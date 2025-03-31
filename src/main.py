from data_vizualization import DataVizualization
from label_data import LabelData


def main():

    data_path = "data/trndset-dump-Top_Creators_(DK).json"
    # data_vizualization = DataVizualization(data_path)
    # data_vizualization.run()
    label_data = LabelData(data_path)
    label_data.run()


if __name__ == "__main__":
    main()
