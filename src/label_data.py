from openai import OpenAI
import os
import json
import time
from typing import Dict, List, Any


client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


class LabelData:
    """
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
        """Analyze sentiment for all posts."""

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

                time.sleep(0.5)  # Rate limiting

        return result

    def run(self) -> None:
        """Run sentiment analysis on all trends and save results."""

        analyzed_data = self.analyze_trend_sentiment()

        # Save results
        output_path = "data/analyzed_trends.json"
        with open(output_path, "w", encoding="utf8") as file:
            json.dump(analyzed_data, file, ensure_ascii=False, indent=4)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    data_path = "data/trndset-dump-Top_Creators_(DK).json"
    label_data = LabelData(data_path)
    label_data.run()
