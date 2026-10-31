import os
import sys
import warnings
import logging
import torch
import pandas as pd

# -------------------------------------------------------------
# 0. Environment Setup & Logging Suppression
# -------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, trim, count
from pyspark.sql.types import StringType, StructType, StructField, IntegerType, FloatType

# -------------------------------------------------------------
# 1. Global Lazy Load for RoBERTa Model (Optimized)
# -------------------------------------------------------------
_SENTIMENT_PIPELINE = None

def get_sentiment_pipeline():
    global _SENTIMENT_PIPELINE
    if _SENTIMENT_PIPELINE is None:
        # Automatically route to GPU (device=0) if available, otherwise CPU (device=-1)
        device = 0 if torch.cuda.is_available() else -1
        
        _SENTIMENT_PIPELINE = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=device,
            truncation=True,
            max_length=128  # Reduced sequence length to significantly speed up transformer attention
        )
    return _SENTIMENT_PIPELINE

# -------------------------------------------------------------
# 2. Spark Session Builder
# -------------------------------------------------------------
def get_spark_session():
    spark = SparkSession.builder \
        .appName("YouTubeCommentSentimentBDA") \
        .master("local[*]") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    return spark

# -------------------------------------------------------------
# 3. Fast PyTorch Batched Inference (With Confidence Thresholding)
# -------------------------------------------------------------
def analyze_comment_batch(comments_list):
    nlp = get_sentiment_pipeline()
    
    # Pre-clean all strings
    cleaned_texts = [
        str(text).strip() if text and str(text).strip() else "Neutral" 
        for text in comments_list
    ]
    
    scores = []
    labels = []
    
    # Pass entire array to Transformers with larger batch size
    try:
        # Increased batch size to 256 for optimal hardware throughput
        raw_results = nlp(cleaned_texts, batch_size=256, truncation=True, top_k=None)
        
        for res_list in raw_results:
            # Map probabilities: {'positive': 0.12, 'neutral': 0.43, 'negative': 0.45}
            prob_map = {r['label'].lower(): float(r['score']) for r in res_list}
            
            pos_prob = prob_map.get('positive', 0.0)
            neg_prob = prob_map.get('negative', 0.0)
            
            # Confidence Thresholding Logic:
            # Require at least 0.55 confidence to classify as Positive/Negative
            if pos_prob > 0.55 and pos_prob > neg_prob:
                label = 'Positive'
                score = pos_prob
            elif neg_prob > 0.55 and neg_prob > pos_prob:
                label = 'Negative'
                score = -neg_prob
            else:
                label = 'Neutral'
                score = 0.0
                
            scores.append(score)
            labels.append(label)
            
    except Exception:
        # Fallback in case of batch failure
        for _ in comments_list:
            scores.append(0.0)
            labels.append("Neutral")
            
    return scores, labels

# -------------------------------------------------------------
# 4. Universal Data Normalizer
# -------------------------------------------------------------
def normalize_to_pandas(raw_data) -> pd.DataFrame:
    if isinstance(raw_data, pd.DataFrame):
        df = raw_data.copy()
    else:
        parsed_rows = []
        for item in raw_data:
            if isinstance(item, dict):
                author = item.get("author") or item.get("Author") or "Anonymous"
                comment = item.get("comment") or item.get("Comment") or item.get("text") or ""
                likes = item.get("likes") or item.get("Likes") or item.get("like_count") or 0
                timestamp = item.get("timestamp") or item.get("Timestamp") or "2026-01-01"
                parsed_rows.append({
                    "Author": str(author), 
                    "Comment": str(comment), 
                    "Likes": int(likes),
                    "Timestamp": str(timestamp)
                })
            elif isinstance(item, (list, tuple)):
                parsed_rows.append({
                    "Author": str(item[0]) if len(item) > 0 else "Anonymous",
                    "Comment": str(item[1]) if len(item) > 1 else "",
                    "Likes": int(item[2]) if len(item) > 2 else 0,
                    "Timestamp": str(item[3]) if len(item) > 3 else "2026-01-01"
                })
        
        df = pd.DataFrame(parsed_rows)

    if 'Author' not in df.columns:
        df['Author'] = 'Anonymous'
    if 'Comment' not in df.columns:
        df['Comment'] = ''
    if 'Likes' not in df.columns:
        df['Likes'] = 0
    if 'Timestamp' not in df.columns:
        df['Timestamp'] = '2026-01-01'

    df['Author'] = df['Author'].fillna('Anonymous').astype(str)
    df['Comment'] = df['Comment'].fillna('').astype(str)
    df['Likes'] = pd.to_numeric(df['Likes'], errors='coerce').fillna(0).astype(int)
    df['Timestamp'] = df['Timestamp'].fillna('2026-01-01').astype(str)

    return df[['Author', 'Comment', 'Likes', 'Timestamp']]

# -------------------------------------------------------------
# 5. AI Creator Action Plan Generator
# -------------------------------------------------------------
def generate_creator_action_plan(df: pd.DataFrame):
    if df.empty:
        return {
            "praise": ["No data available."],
            "criticism": ["No data available."],
            "actions": ["Gather more audience feedback."]
        }

    top_pos = df[df['Sentiment'] == 'Positive'].sort_values(by='Likes', ascending=False).head(3)
    top_neg = df[df['Sentiment'] == 'Negative'].sort_values(by='Likes', ascending=False).head(3)

    praise_list = top_pos['Comment'].tolist() if not top_pos.empty else ["Overall audience reception is balanced."]
    criticism_list = top_neg['Comment'].tolist() if not top_neg.empty else ["No major audience backlash detected."]

    actions = []
    if len(top_neg) > 0:
        actions.append("Address key concerns highlighted in top-liked negative comments in your next video intro or pinned comment.")
        actions.append("Review video pacing or specific sections mentioned in viewer critiques.")
    else:
        actions.append("Maintain current content format and pacing, as viewer satisfaction remains high.")
    
    actions.append("Double down on topics mentioned in top positive feedback to drive higher subscriber retention.")

    return {
        "praise": praise_list,
        "criticism": criticism_list,
        "actions": actions
    }

# -------------------------------------------------------------
# 6. Main PySpark Processing Pipeline
# -------------------------------------------------------------
def analyze_sentiments_spark(raw_data):
    if raw_data is None or len(raw_data) == 0:
        return pd.DataFrame(), {}, {}

    input_pd_df = normalize_to_pandas(raw_data)

    if input_pd_df.empty or input_pd_df['Comment'].str.strip().eq('').all():
        return pd.DataFrame(), {}, {}

    # Run PyTorch Batched RoBERTa scoring
    scores, labels = analyze_comment_batch(input_pd_df['Comment'].tolist())
    input_pd_df['Score'] = scores
    input_pd_df['Sentiment'] = labels

    # Ingest into PySpark for distributed cleaning and aggregation
    spark = get_spark_session()
    
    schema = StructType([
        StructField("Author", StringType(), True),
        StructField("Comment", StringType(), True),
        StructField("Likes", IntegerType(), True),
        StructField("Timestamp", StringType(), True),
        StructField("Score", FloatType(), True),
        StructField("Sentiment", StringType(), True)
    ])

    spark_df = spark.createDataFrame(input_pd_df, schema=schema)

    cleaned_spark_df = spark_df.withColumn(
        "Comment",
        regexp_replace(col("Comment"), r"http\S+|www\S+|https\S+", "")
    ).withColumn(
        "Comment",
        trim(regexp_replace(col("Comment"), r"\s+", " "))
    )

    total_count = cleaned_spark_df.count()

    metrics_df = cleaned_spark_df.groupBy("Sentiment").agg(count("Sentiment").alias("Count"))
    metrics = {row['Sentiment']: row['Count'] for row in metrics_df.collect()}

    pos_count = metrics.get('Positive', 0)
    neg_count = metrics.get('Negative', 0)
    neu_count = metrics.get('Neutral', 0)

    stats = {
        'total': total_count,
        'positive_count': pos_count,
        'negative_count': neg_count,
        'neutral_count': neu_count,
        'positive_pct': round((pos_count / total_count) * 100, 2) if total_count > 0 else 0,
        'negative_pct': round((neg_count / total_count) * 100, 2) if total_count > 0 else 0,
        'neutral_pct': round((neu_count / total_count) * 100, 2) if total_count > 0 else 0,
    }

    final_pd_df = cleaned_spark_df.select('Author', 'Comment', 'Likes', 'Timestamp', 'Sentiment', 'Score').toPandas()

    action_plan = generate_creator_action_plan(final_pd_df)

    return final_pd_df, stats, action_plan