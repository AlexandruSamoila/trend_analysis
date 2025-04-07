import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.cluster import KMeans
from keybert import KeyBERT


class TrendRecommender:
    def __init__(self, data_path):
        """
        A content-based trend recommender system using text embeddings and topic encoding.

        Args:
            data_path (String): Path of the JSON file containing trend data.
        """

        self.df_trends = pd.read_json(data_path, encoding="utf8")

        # MultiLabelBinarizer to encode topic lists into binary arrays
        self.mlb = MultiLabelBinarizer()

        # Load the SentenceTransformer model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer("all-MiniLM-L12-v2", device=device)

    def encode_trends(self):
        """
        Encodes the trend description using SentenceTransformer and one-hot encodes topics.
        """
        print("Encoding trends...")

        # Text embeddings
        text_embeddings = self.model.encode(
            self.df_trends["about"].tolist(), convert_to_tensor=True
        )

        # Topic encoding (one-hot encoding)
        topic_encoded = self.mlb.fit_transform(self.df_trends["topics"])
        topic_embeddings = torch.tensor(topic_encoded, dtype=torch.float32)

        # Combine text embeddings and topic vectors
        combined_embeddings = torch.cat((text_embeddings, topic_embeddings), dim=1)

        return combined_embeddings

    def find_similar_trends(self, trend_name, top_k=5):
        """
        Finds trends similar to the given trend name based on embeddings.

        Args:
            trend_name (str): The name of the trend to find similarities for.
            top_k (int): Number of similar trends to return.

        Returns:
            List[str]: Names of similar trends.
        """
        try:
            # Encode trends
            combined_embeddings = self.encode_trends()

            # Find index of the given trend
            trend_index = self.df_trends[
                self.df_trends["name"].str.lower() == trend_name.lower()
            ].index[0]

            # Compute similarity matrix
            similarities = cosine_similarity(combined_embeddings.cpu().numpy())

            # Retrieve similarities for the target trend
            sim_scores = list(enumerate(similarities[trend_index]))

            # Sort by similarity score (descending)
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Remove the trend itself from results
            sim_scores = [score for score in sim_scores if score[0] != trend_index]

            # Get top-k similar trends
            top_indices = [i[0] for i in sim_scores[:top_k]]
            similar_trends = self.df_trends.iloc[top_indices]["name"].tolist()

            return similar_trends
        except:
            print(f"Trend '{trend_name}' not found!")
            return []

    def plot_trend_clusters(self, n_clusters=15, perplexity=10):
        """
        Visualizes trend embeddings in 2D using PCA + t-SNE, clusters them, and summarizes clusters with KeyBERT.
        """
        print("Visualizing trends...")

        combined_embeddings = self.encode_trends().cpu().numpy()

        features_df = self._reduce_dimensions(combined_embeddings, perplexity)

        features_df = self._apply_clustering(features_df, n_clusters)
        cluster_labels = self._generate_cluster_labels(features_df)
        features_df["cluster_label"] = features_df["cluster"].map(cluster_labels)

        self._plot_clusters(features_df)
        # print("Cluster labels:", cluster_labels)

    def _reduce_dimensions(self, embeddings, perplexity=10):
        """
        Reduces dimensionality using PCA + TSNE.
        """
        reduced_embeddings_pca = PCA(n_components=50, random_state=42).fit_transform(
            embeddings
        )
        reduced_embeddings_tsne = TSNE(
            n_components=2, perplexity=perplexity, random_state=42
        ).fit_transform(reduced_embeddings_pca)
        features_df = pd.concat(
            [
                pd.DataFrame(reduced_embeddings_tsne, columns=["x", "y"]),
                self.df_trends,
            ],
            axis=1,
        )
        return features_df

    def _apply_clustering(self, features_df, n_clusters):
        """
        Applies KMeans clustering to reduced embeddings.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        features_df["cluster"] = kmeans.fit_predict(features_df[["x", "y"]])
        return features_df

    def _generate_cluster_labels(self, features_df):
        """
        Combines text from each cluster and uses KeyBERT to generate a label.
        """
        cluster_texts = {}

        for _, row in features_df.iterrows():
            cluster_id = row["cluster"]
            about_bullets = row.get("about_bullets", [])
            if cluster_id not in cluster_texts:
                cluster_texts[cluster_id] = ""

            for bullet in about_bullets:
                if isinstance(bullet, str):
                    cluster_texts[cluster_id] += bullet.replace(".", ",") + " "

        kw_model = KeyBERT()
        cluster_labels = {}

        for cluster_id, text in cluster_texts.items():
            keywords = kw_model.extract_keywords(
                text, keyphrase_ngram_range=(1, 6), stop_words="english", top_n=1
            )
            cluster_labels[cluster_id] = (
                keywords[0][0] if keywords else f"Cluster {cluster_id}"
            )

        return cluster_labels

    def _plot_clusters(self, features_df):
        """
        Uses Plotly to visualize clustered trends.
        """
        fig = px.scatter(
            features_df,
            x="x",
            y="y",
            color="cluster_label",
            hover_data={
                "name": True,
                "topics": True,
                "contributors_count": True,
                "cluster_label": True,
            },
            title="Trend Similarity Visualization",
        )
        fig.show()

    def tune_kmeans(self, k_min=2, k_max=20):
        """
        Visualizes inertia over different K values to find the optimal number of clusters.

        Args:
            k_min (int): Minimum number of clusters.
            k_max (int): Maximum number of clusters.
        """
        inertias = []
        k_range = range(k_min, k_max + 1)
        combined_embeddings = self.encode_trends().cpu().numpy()

        features_df = self._reduce_dimensions(combined_embeddings, perplexity=10)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            kmeans.fit(features_df[["x", "y"]])
            inertias.append(kmeans.inertia_)

        plt.plot(k_range, inertias, marker="o")
        plt.title("Elbow Method for Optimal K")
        plt.xlabel("Number of clusters")
        plt.ylabel("Inertia")
        plt.show()

    def tune_tsne_perplexity(self, perplexities=[5, 10, 30, 50]):
        """
        Visualizes t-SNE reduction for various perplexity values to help tune this hyperparameter.

        Args:
            embeddings (np.ndarray): High-dimensional trend embeddings (e.g., shape [298, 1182]).
            perplexities (list): List of perplexity values to test.
        """
        n_plots = len(perplexities)
        n_cols = 2
        n_rows = (n_plots + 1) // 2

        combined_embeddings = self.encode_trends().cpu().numpy()

        reduced_embeddings_pca = PCA(n_components=50, random_state=42).fit_transform(
            combined_embeddings
        )

        plt.figure(figsize=(n_cols * 6, n_rows * 5))

        for i, perplexity in enumerate(perplexities, 1):
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            reduced = tsne.fit_transform(reduced_embeddings_pca)

            plt.subplot(n_rows, n_cols, i)
            plt.scatter(reduced[:, 0], reduced[:, 1], s=10, alpha=0.7)
            plt.title(f"Perplexity: {perplexity}")
            plt.xlabel("x")
            plt.ylabel("y")

        plt.tight_layout()
        plt.suptitle(
            "t-SNE Visualizations for Different Perplexities", fontsize=16, y=1.02
        )
        plt.show()


if __name__ == "__main__":
    data_path = "data/trndset-dump-Top_Creators_(DK).json"
    recommender = TrendRecommender(data_path)
    # print(
    #     recommender.find_similar_trends("Home Renovation and DIY Project Inspirations")
    # )
    recommender.plot_trend_clusters(n_clusters=14, perplexity=10)
    # recommender.tune_kmeans(k_min=5, k_max=20)
    # recommender.tune_tsne_perplexity(perplexities=[5, 10, 30, 50])
