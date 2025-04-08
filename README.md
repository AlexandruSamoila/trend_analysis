# Trend Sentiment Analysis

This project analyzes trends based on sentiment and extracts meaningful
insights from social media posts.

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### Main Menu Interface

First, edit the data_path to the desired location.
Next, run:

```bash
python src/main.py
```

This will launch an interactive menu where you can:

1. Run Data Visualization
2. Run Trend Recommendation
3. Run Sentiment Analysis
4. Run Data Labeling
5. Exit

### Individual Components

1. **Data Visualization**

   ```bash
   python src/data_vizualization.py
   ```

   Generates all plots and charts in the results folder.

2. **Trend Recommendation**

   ```bash
   python src/trend_recommender.py
   ```

   Runs the trend recommendation system.

3. **Sentiment Analysis**

   ```bash
   python src/sentiment_analysis.py
   ```

   Performs sentiment analysis on the data.

4. **Data Labeling**

   ```bash
   python src/label_data.py
   ```

   Processes and labels the raw data.

5. **Model Training**
   - Open `src/finetune_roberta.ipynb` in Google Colab
   - Upload the notebook to your Google Drive
   - Run all cells to train the model

## Results

### Sentiment Analysis Performance

- Base XLM RoBERTa accuracy: 0.56
- Fine-tuned XLM RoBERTa accuracy: 0.84 (after 3 epochs)

The analysis generates several visualizations in the `results` folder:

![Trend Clusters](results/clusters.png)
_Visualization of trend clusters_

![Perplexity Tuning](results/perplexity_tuning.png)
_Visualization of perplexity tuning_

### Sentiment Analysis

The project uses XLM RoBERTa for sentiment analysis, fine-tuned on our specific dataset to achieve high accuracy in trend sentiment classification.
