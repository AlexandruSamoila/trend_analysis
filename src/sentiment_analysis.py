from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from collections import Counter
import pandas as pd


class SentimentAnalysis:
    def __init__(self, model_path):
        """
        Initialize the sentiment analysis class with model and tokenizer.

        Args:
            model_path (str): Path to the fine-tuned sentiment classification model.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        self.max_len = 512  # Adding max_len attribute for tokenization

    def predict_sentiment(self, text):
        """
        Predict the sentiment of a single text.

        Args:
            text (str): The input text.

        Returns:
            tuple: (predicted_label, score)
        """
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(**inputs).logits
        _, predictions = torch.max(outputs.data, dim=1)
        label = self.model.config.id2label[predictions.item()]
        return label

    def analyze_trend_sentiment(self, trend):
        """
        Analyze sentiment for each post in a trend.

        Args:
            trend (dict or pd.Series): A dictionary or pandas Series with at least a "name" and a list of "posts".

        Returns:
            dict: Trend with appended sentiment analysis results.
        """
        sentiments = []
        posts = (
            trend.get("posts", [])
            if isinstance(trend, dict)
            else trend.get("posts", [])
        )

        for post in posts:
            if isinstance(post, str):
                label = self.predict_sentiment(post)
                sentiments.append(label)
            else:
                # If post is not a string, try to get text content
                text = str(post)  # Convert to string if it's not already
                label = self.predict_sentiment(text)
                sentiments.append(label)

        sentiment_counts = Counter(sentiments)
        dominant_sentiment, count = sentiment_counts.most_common(1)[0]
        confidence = count / len(sentiments) if sentiments else 0.0

        result = {
            "scores": sentiment_counts,
            "dominant_sentiment": dominant_sentiment,
            "confidence": confidence,
        }

        if isinstance(trend, dict):
            trend["sentiment_analysis"] = result
            return trend
        else:
            return result


if __name__ == "__main__":
    model_path = "checkpoints/20250407_1450_roberta_sentiment"
    data_path = "data/trndset-dump-Top_Creators_(DK).json"
    data_df = pd.read_json(data_path, encoding="utf8")
    sentiment_analysis = SentimentAnalysis(model_path)
    print(sentiment_analysis.analyze_trend_sentiment(data_df.iloc[1]))
