from openai import OpenAI
import os
import json
import time
from typing import Dict, List, Any
from ftfy import fix_text
import pandas as pd
import numpy as np

# Use a pipeline as a high-level helper
import tweetnlp

model = tweetnlp.Classifier(
    "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual", max_length=128
)

# from transformers import pipeline

# pipe = pipeline("text-classification", model="mirfan899/da-sentiment")

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


class LabelData:
    """Label data using the OpenAI API.

    This class is responsible for labeling the data using the OpenAI API.
    """

    def __init__(self, data_path: str):
        with open(data_path, "r", encoding="utf8") as file:
            data = json.load(file)
        self.data = data

    def get_sentiment(self, text: str) -> str:
        """Get sentiment analysis for a given text using OpenAI API."""
        prompt = f"""Analyze the sentiment of the following social media content and return just the word describing if it has a negative/neutral/positive angle.
        Consider the overall tone, emotions, and context.
        Return exactly one of these words: 'Positive', 'Neutral', or 'Negative'
        
        Content: {text}"""

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        sentiment = completion.choices[0].message.content.strip()

        return sentiment

    def analyze_trend_sentiment(self) -> Dict[str, Any]:
        """Analyze sentiment for a single trend including its examples and posts."""
        sentiment_scores = {"positive": 0, "negative": 0, "neutral": 0}

        # Analyze examples
        # for example in trend.get("examples", []):
        #     if "text" in example:
        #         sentiment = self.get_sentiment(example["text"])
        #         print(sentiment)
        #         sentiment_scores["examples"][sentiment.lower()] += 1
        #         time.sleep(0.5)  # Rate limiting

        # for bullet in trend.get("about_bullets", []):
        #     sentiment = self.get_sentiment(bullet)
        #     print(bullet, sentiment)
        #     sentiment_scores["examples"][sentiment.lower()] += 1
        #     time.sleep(0.5)  # Rate limiting
        # sentiment = self.get_sentiment(trend.get("description", []))
        # print(sentiment)
        # sentiment_scores["examples"][sentiment.lower()] += 1
        # time.sleep(0.5)  # Rate limiting

        # Analyze topics
        # topic_text = ""
        # for topic in trend.get("topics", []):
        #     topic_text += topic + " "
        # sentiment = self.get_sentiment(topic_text)
        # print(topic_text, sentiment)
        # sentiment_scores[sentiment.lower()] += 1
        # time.sleep(0.5)  # Rate limiting

        result = []
        # Analyze posts
        for trend in self.data:
            for post in trend.get("posts", []):
                if post.get("text") == "":
                    continue
                sentiment = self.get_sentiment(post["text"])
                sentiment = sentiment.lower()
                if sentiment not in ["positive", "neutral", "negative"]:
                    continue
                if sentiment == "positive":
                    label = 2
                elif sentiment == "neutral":
                    label = 1
                else:
                    label = 0
                result.append(
                    {"post": post["text"], "sentiment": sentiment, "label": label}
                )
                result_roberta = model.predict(post["text"])
                print("Chatgpt:", sentiment, "RoBERTa:", result_roberta)
                # sentiment_scores[sentiment.lower()] += 1
                time.sleep(0.5)  # Rate limiting
            # print(result)

        # Determine dominant sentiment
        # max_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])

        # trend["sentiment_analysis"] = {
        #     "scores": sentiment_scores,
        #     "dominant_sentiment": max_sentiment[0],
        #     "confidence": max_sentiment[1] / sum(sentiment_scores.values()),
        # }
        # print(trend)
        return result

    def run(self) -> None:
        """Run sentiment analysis on all trends and save results."""
        # analyzed_data = []

        # for trend in self.data:
        #     print(f"Analyzing trend: {trend.get('name', 'Unknown')}")
        #     analyzed_trend = self.analyze_trend_sentiment(trend)
        #     analyzed_data.append(analyzed_trend)
        #     print(
        #         f"Dominant sentiment: {analyzed_trend['sentiment_analysis']['dominant_sentiment']}"
        #     )
        #     print(
        #         f"Confidence: {analyzed_trend['sentiment_analysis']['confidence']:.2f}"
        #     )
        #     print("-" * 50)

        analyzed_data = self.analyze_trend_sentiment()

        # Save results
        output_path = "data/analyzed_trends.json"
        with open(output_path, "w", encoding="utf8") as file:
            json.dump(analyzed_data, file, ensure_ascii=False, indent=4)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    data_path = "data/trndset-dump-Top_Creators_(DK).json"
    # label_data = LabelData(data_path)
    # label_data.run()

    train = pd.read_json("data/analyzed_trends.json")
    print(train.shape)
    print(train.head())
    print("Unique labels: ", train["label"].unique())
    print("Total samples", train["label"].value_counts())
    print(train.describe())
    train, validate, test = np.split(
        train.sample(frac=1, random_state=42),
        [int(0.8 * len(train)), int(0.9 * len(train))],
    )
    print(train.shape, validate.shape, test.shape)
