import json
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime


class DataVizualization:
    """Distribution of trend types ("type" field).

    Most common topics across trends.
    """

    def __init__(self, data_path):
        with open(data_path, "r", encoding="utf8") as file:
            data = json.load(file)
        self.data = data

    def get_features(self):
        engagement = []
        impressions = []
        trend_durations = []
        platform_counter = Counter()
        for trend in self.data:
            start_date = datetime.strptime(trend["start_date"], "%Y-%m-%d %H:%M:%S")
            end_date = datetime.strptime(trend["end_date"], "%Y-%m-%d %H:%M:%S")
            duration = end_date - start_date
            trend_durations.append(duration.days)
            engagement.append(trend["engagement"])
            impressions.append(trend["impressions"])
            for post in trend["posts"]:
                platform_counter[post["platform"]] += 1
        return platform_counter, engagement, impressions, trend_durations

    def run(self):
        platform_counter, engagement, impressions, trend_durations = self.get_features()
        plt.figure(figsize=(10, 5))
        plt.bar(platform_counter.keys(), platform_counter.values())
        plt.xlabel("Platform")
        plt.ylabel("Number of trends")
        plt.show()
        plt.boxplot(engagement)
        plt.show()
        plt.boxplot(impressions)
        plt.show()
        plt.boxplot(trend_durations)
        plt.show()
