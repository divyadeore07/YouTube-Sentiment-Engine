# YouTube-Sentiment-Engine
AI-powered YouTube comment sentiment analysis using PySpark, RoBERTa, and Streamlit, featuring interactive dashboards, sentiment trends, word clouds, and actionable audience insights.
# YouTube Sentiment Engine 📊

An AI-powered YouTube comment sentiment analysis system built using **PySpark, RoBERTa, and Streamlit**. The project analyzes audience reactions to YouTube videos and transforms comments into meaningful insights through interactive dashboards and automated content recommendations.

## 🚀 Features

- **YouTube Comment Extraction:** Fetch comments, author details, likes, and timestamps.
- **AI Sentiment Analysis:** Classify comments as Positive, Negative, or Neutral.
- **Sentiment Analytics:** Visualize audience sentiment distribution and engagement.
- **Timeline Trends:** Explore how comment sentiment changes over time.
- **AI Creator Action Plan:** Generate audience praise, criticism, and strategic content recommendations.
- **Word Cloud:** Identify frequently occurring words in audience comments.
- **Engagement Analysis:** Identify highly liked comments.
- **Dataset Export:** Download analyzed comment data as a CSV file.
- **Analysis History:** Revisit previously analyzed videos during the current session.

## 🛠️ Technologies Used

- Python
- PySpark
- RoBERTa Transformer Model
- Streamlit
- Pandas and NumPy
- Plotly
- Matplotlib
- WordCloud
- yt-dlp

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YouTube-Sentiment-Engine.git
cd YouTube-Sentiment-Engine
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**macOS/Linux**
```bash
source venv/bin/activate
```

### 3. Install dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise, create one containing the dependencies required by the project before installing them.

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to access the dashboard.

## 📈 How It Works

1. Enter a YouTube video URL.
2. Select the maximum number of comments to analyze.
3. Extract comments using the comment-fetching module.
4. Process the comments through the sentiment analysis pipeline.
5. Explore sentiment distributions, trends, engagement metrics, and word clouds.
6. Review the generated creator action plan.
7. Export the analyzed dataset as a CSV file.

## 🎓 Project Information

**Subject:** Big Data Analytics (BDA)

**Project Name:** YouTube Sentiment Engine

**Objective:** To apply big data processing, natural language processing, and data visualization techniques to understand audience sentiment and derive actionable insights from YouTube comments.

## 🔮 Future Enhancements

- Support sentiment analysis across multiple videos.
- Add creator and channel-level analytics.
- Improve sarcasm and context detection.
- Deploy the application online for public access.
- Add historical analytics storage and comparison.


**Note:** This project is intended for educational and analytical purposes. Comment extraction depends on YouTube availability, video settings, and the capabilities of the extraction library.
