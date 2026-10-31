import importlib
import os
import math
import numpy as np
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

from scraper import fetch_youtube_comments
import sentiment_engine

# Force module reload to prevent Streamlit caching old logic
importlib.reload(sentiment_engine)
from sentiment_engine import analyze_sentiments_spark as analyze_sentiments

# -------------------------------------------------------------
# 1. Page Config & CSS (Datapulse Modern Animated 3D Dark Theme)
# -------------------------------------------------------------
st.set_page_config(
    page_title="DataPulse | Enterprise Sentiment Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        /* Hide Default Streamlit Menu & Footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Main App Canvas */
        .stApp {
            background: #080c14;
            color: #f1f5f9;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }

        /* Sidebar Custom Dark Styling */
        [data-testid="stSidebar"] {
            background-color: #0d1322 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        }
        
        [data-testid="stSidebarUserContent"] {
            padding: 1.5rem 1rem !important;
        }

        /* Keyframe Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(18px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes floating3D {
            0% { transform: translateY(0px) perspective(600px) rotateX(0deg); }
            50% { transform: translateY(-8px) perspective(600px) rotateX(2deg); }
            100% { transform: translateY(0px) perspective(600px) rotateX(0deg); }
        }

        .animated-entry {
            animation: fadeIn 0.7s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }

        /* Hero Badge (Floating 3D) */
        .badge-pill {
            display: inline-flex;
            align-items: center;
            padding: 6px 14px;
            border-radius: 9999px;
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.3);
            color: #60a5fa;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.8px;
            text-transform: uppercase;
            margin-bottom: 12px;
            animation: floating3D 4s ease-in-out infinite;
            box-shadow: 0px 5px 15px rgba(59, 130, 246, 0.2), inset 0px 1px 1px rgba(255,255,255,0.2);
        }

        /* Animated Title Header */
        .hero-title {
            font-size: 3rem;
            font-weight: 900;
            letter-spacing: -1.5px;
            color: #ffffff;
            margin-bottom: 0.4rem;
            line-height: 1.1;
        }

        .hero-highlight {
            background: linear-gradient(270deg, #3b82f6, #8b5cf6, #ec4899, #3b82f6);
            background-size: 300% 300%;
            animation: gradientShift 6s ease infinite;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-sub {
            font-size: 1.1rem;
            color: #94a3b8;
            margin-bottom: 1.8rem;
            font-weight: 400;
        }

        /* Custom KPI Metric Cards */
        .metric-card {
            background: linear-gradient(145deg, #131b2c, #0f1523);
            border: 1px solid rgba(255, 255, 255, 0.03);
            border-radius: 16px;
            padding: 20px;
            text-align: left;
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(12px);
            transition: transform 0.4s ease, box-shadow 0.4s ease;
            transform-style: preserve-3d;
            box-shadow: 8px 8px 16px rgba(0,0,0,0.6), -4px -4px 12px rgba(255,255,255,0.02);
        }

        .metric-card:hover {
            border-color: rgba(139, 92, 246, 0.4);
            transform: perspective(1000px) rotateX(5deg) rotateY(-5deg) translateZ(10px) translateY(-5px);
            box-shadow: 15px 15px 25px rgba(0,0,0,0.7), -5px -5px 15px rgba(139, 92, 246, 0.15);
        }

        /* 3D Inputs & Controls */
        .stTextInput > div > div > input {
            background-color: #0b111a !important;
            color: #f8fafc !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 12px !important;
            padding: 14px 18px !important;
            font-size: 1rem !important;
            box-shadow: inset 4px 4px 10px rgba(0,0,0,0.6), inset -4px -4px 10px rgba(255,255,255,0.03) !important;
        }

        .stTextInput > div > div > input:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.4), inset 2px 2px 5px rgba(0,0,0,0.4) !important;
        }

        /* Smaller Square-like 3D Action Button */
        .stButton > button {
            border-radius: 16px !important;
            width: 120px !important;
            height: 54px !important;
            background: linear-gradient(145deg, #2a6df0, #6e33d4) !important;
            border: none !important;
            font-weight: 800 !important;
            font-size: 1.05rem !important;
            color: #ffffff !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            margin: 0 auto !important;
            box-shadow: 0px 6px 0px #1e4bb8, 0px 10px 15px rgba(0,0,0,0.4), inset 0px 2px 2px rgba(255,255,255,0.2) !important;
            text-shadow: 0px 2px 4px rgba(0,0,0,0.3);
            transition: all 0.2s ease !important;
        }

        .stButton > button:hover {
            transform: translateY(2px) !important;
            box-shadow: 0px 4px 0px #1e4bb8, 0px 8px 10px rgba(0,0,0,0.4), inset 0px 2px 2px rgba(255,255,255,0.2) !important;
        }
        
        .stButton > button:active {
            transform: translateY(6px) !important;
            box-shadow: 0px 0px 0px #1e4bb8, 0px 2px 5px rgba(0,0,0,0.4), inset 0px 4px 6px rgba(0,0,0,0.4) !important;
        }

        /* Improved Table Borders */
        .table-container {
            border: 2px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 6px;
            background: rgba(15, 23, 42, 0.4);
            box-shadow: inset 0 0 15px rgba(0,0,0,0.5), 0 5px 15px rgba(0,0,0,0.3);
            margin-bottom: 20px;
        }

        /* AI Action Plan Enhanced UI */
        .info-card {
            background: linear-gradient(145deg, #1e293b, #0f172a);
            border-left: 5px solid #3b82f6;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 12px;
            color: #e2e8f0;
            box-shadow: 4px 4px 10px rgba(0,0,0,0.4);
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .error-card {
            background: linear-gradient(145deg, #451a1a, #2a0f0f);
            border-left: 5px solid #ef4444;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 12px;
            color: #fca5a5;
            box-shadow: 4px 4px 10px rgba(0,0,0,0.4);
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .success-card {
            background: linear-gradient(145deg, #064e3b, #022c22);
            border-left: 5px solid #10b981;
            border-radius: 12px;
            padding: 18px 22px;
            margin-bottom: 14px;
            color: #6ee7b7;
            box-shadow: 4px 4px 10px rgba(0,0,0,0.4);
            font-size: 1rem;
            line-height: 1.5;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Session State Initialization (History Store)
# -------------------------------------------------------------
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "current_url" not in st.session_state:
    st.session_state.current_url = ""

# -------------------------------------------------------------
# 3. Sidebar: Analysis History
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🕒 Analysis History")
    st.caption("Previously analyzed YouTube videos in this session")

    if not st.session_state.analysis_history:
        st.info("No videos analyzed yet. Enter a link to start.")
    else:
        for idx, item in enumerate(reversed(st.session_state.analysis_history)):
            st.markdown(f"""
                <div class="history-card">
                    <div class="history-title">🎬 {item['title']}</div>
                    <div class="history-meta">
                        <span>💬 {item['total']} comments</span>
                        <span style="color: {item['grade_color']}; font-weight: bold;">⭐ {item['score']}/10</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if st.button(f"Load Video #{len(st.session_state.analysis_history) - idx}", key=f"hist_btn_{idx}"):
                st.session_state.current_url = item['url']
                st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear History"):
            st.session_state.analysis_history = []
            st.rerun()

# -------------------------------------------------------------
# 4. Hero Branding Section
# -------------------------------------------------------------
st.markdown("""
    <div class="animated-entry">
        <div class="badge-pill">⚡ PySpark + RoBERTa Transformer Core</div>
        <div class="hero-title">DataPulse <span class="hero-highlight">Sentiment Intelligence</span></div>
        <div class="hero-sub">Real-time YouTube comment mining engine built for distributed big data processing.</div>
    </div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 5. Search & Control Panel
# -------------------------------------------------------------
with st.container():
    url_input = st.text_input(
        "YouTube Video URL", 
        value=st.session_state.current_url,
        placeholder="https://www.youtube.com/watch?v=..."
    )
    st.markdown("<br>", unsafe_allow_html=True)
    
    max_comments = st.slider(
        "Sample Extraction Limit", 
        min_value=100, max_value=20000, value=1000, step=100
    )
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Use columns to strictly center the smaller button
    c1, c2, c3 = st.columns([4, 2, 4])
    with c2:
        analyze_btn = st.button("🚀 Analyze")

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# 6. Pipeline Execution & Dashboards
# -------------------------------------------------------------
if analyze_btn:
    if not url_input.strip():
        st.warning("⚠️ Please enter a valid YouTube Video URL.")
    else:
        st.session_state.current_url = url_input.strip()
        with st.spinner("⚡ Running Distributed PySpark Pipeline & RoBERTa AI Models..."):
            try:
                # Step 1: Scrape
                video_title, raw_comments = fetch_youtube_comments(url_input.strip(), max_comments=max_comments)
                
                if not raw_comments:
                    st.error("❌ No comments found or the comment section is disabled for this video.")
                else:
                    # Step 2: PySpark Sentiment Analysis & Action Plan
                    df, stats, action_plan = analyze_sentiments(raw_comments)
                    
                    pos_cnt = stats.get('positive_count', 0)
                    neu_cnt = stats.get('neutral_count', 0)
                    tot_cnt = stats.get('total', 1)

                    sentiment_index = ((pos_cnt * 1.0) + (neu_cnt * 0.5)) / tot_cnt
                    content_score = round(sentiment_index * 10, 1)

                    if content_score >= 8.0:
                        grade, grade_color, gradient_bg = "A+ (Highly Positive)", "#10b981", "rgba(16, 185, 129, 0.15)"
                    elif content_score >= 6.5:
                        grade, grade_color, gradient_bg = "B (Good Engagement)", "#3b82f6", "rgba(59, 130, 246, 0.15)"
                    elif content_score >= 5.0:
                        grade, grade_color, gradient_bg = "C (Mixed / Sarcastic)", "#f59e0b", "rgba(245, 158, 11, 0.15)"
                    else:
                        grade, grade_color, gradient_bg = "D (High Backlash)", "#ef4444", "rgba(239, 68, 68, 0.15)"

                    history_entry = {
                        "title": video_title, "url": url_input.strip(),
                        "total": stats['total'], "score": content_score,
                        "grade": grade, "grade_color": grade_color
                    }
                    if not any(h['url'] == url_input.strip() for h in st.session_state.analysis_history):
                        st.session_state.analysis_history.append(history_entry)

                    st.success(f"Successfully Analyzed: **{video_title}**")
                    
                    # Margin of Error
                    sample_size = stats['total']
                    if sample_size > 0:
                        z_score = 2.576 
                        p = 0.5         
                        moe = z_score * math.sqrt((p * (1 - p)) / sample_size)
                        moe_pct = round(moe * 100, 2)
                    else:
                        moe_pct = 0.0

                    st.caption(f"📊 **Sampling Validity:** Evaluated **{sample_size}** comments (99% Confidence Level, **±{moe_pct}%** Error Margin).")

                    # Hero Score Banner
                    st.markdown(f"""
                        <div class="animated-entry" style="background: {gradient_bg}; border: 1px solid {grade_color}; border-radius: 16px; padding: 24px; margin-bottom: 25px; text-align: center; backdrop-filter: blur(12px);">
                            <span style="color: #94a3b8; font-size: 0.95rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;">Overall Content Rating Score</span>
                            <h1 style="color: {grade_color}; font-size: 3.8rem; font-weight: 900; margin: 8px 0;">{content_score} <span style="font-size: 1.4rem; color: #94a3b8;">/ 10</span></h1>
                            <span style="background-color: rgba(0, 0, 0, 0.3); color: {grade_color}; padding: 8px 20px; border-radius: 20px; font-weight: 800; font-size: 1rem; border: 1px solid {grade_color};">
                                Audience Grade: {grade}
                            </span>
                        </div>
                    """, unsafe_allow_html=True)

                    # Custom Animated KPI Cards
                    k1, k2, k3, k4 = st.columns(4)
                    with k1:
                        st.markdown(f"""
                            <div class="metric-card animated-entry">
                                <span class="metric-icon">⚡</span>
                                <div class="metric-val">{stats['total']}</div>
                                <div class="metric-lbl">Comments Processed</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with k2:
                        st.markdown(f"""
                            <div class="metric-card animated-entry">
                                <span class="metric-icon">🟢</span>
                                <div class="metric-val" style="color: #10b981;">{stats['positive_pct']}%</div>
                                <div class="metric-lbl">Positive ({stats['positive_count']})</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with k3:
                        st.markdown(f"""
                            <div class="metric-card animated-entry">
                                <span class="metric-icon">🔴</span>
                                <div class="metric-val" style="color: #ef4444;">{stats['negative_pct']}%</div>
                                <div class="metric-lbl">Negative ({stats['negative_count']})</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with k4:
                        st.markdown(f"""
                            <div class="metric-card animated-entry">
                                <span class="metric-icon">⚪</span>
                                <div class="metric-val" style="color: #94a3b8;">{stats['neutral_pct']}%</div>
                                <div class="metric-lbl">Neutral ({stats['neutral_count']})</div>
                            </div>
                        """, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                    def style_sentiment(val):
                        if val == 'Positive': 
                            return 'color: #10b981; background-color: rgba(16, 185, 129, 0.15); font-weight: bold;'
                        if val == 'Negative': 
                            return 'color: #ef4444; background-color: rgba(239, 68, 68, 0.15); font-weight: bold;'
                        return 'color: #94a3b8; background-color: rgba(148, 163, 184, 0.15); font-weight: bold;'

                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "📊 Sentiment Analytics", "📈 Timeline Trends", "🤖 AI Action Plan", "☁️ Word Cloud", "📋 Spark Dataset"
                    ])

                    # TAB 1: Visualizations & Top Liked
                    with tab1:
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.subheader("Sentiment Distribution")
                            pie_fig = px.pie(
                                values=[stats['positive_count'], stats['negative_count'], stats['neutral_count']],
                                names=['Positive', 'Negative', 'Neutral'],
                                color=['Positive', 'Negative', 'Neutral'],
                                color_discrete_map={'Positive': '#10b981', 'Negative': '#ef4444', 'Neutral': '#64748b'},
                                hole=0.6
                            )
                            pie_fig.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#f8fafc', family="Inter"),
                                margin=dict(t=20, b=20, l=10, r=10), showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                            )
                            st.plotly_chart(pie_fig, use_container_width=True)

                        with c2:
                            st.subheader("🔥 Top Engaged Audience Comments")
                            top_comments = df.sort_values(by='Likes', ascending=False).head(8)
                            max_likes = int(df['Likes'].max()) if not df.empty and df['Likes'].max() > 0 else 10
                            styled_top = top_comments[['Author', 'Comment', 'Likes', 'Sentiment']].style.map(
                                style_sentiment, subset=['Sentiment']
                            )
                            
                            st.markdown('<div class="table-container">', unsafe_allow_html=True)
                            st.dataframe(
                                styled_top, 
                                use_container_width=True,
                                height=400,
                                hide_index=True,
                                column_config={
                                    "Likes": st.column_config.ProgressColumn(
                                        "Engagement (Likes)", help="Total likes on this comment",
                                        format="%d", min_value=0, max_value=max_likes,
                                    ),
                                    "Comment": st.column_config.TextColumn("Comment", width="large")
                                }
                            )
                            st.markdown('</div>', unsafe_allow_html=True)

                    # TAB 2: Time-Series Sentiment Trend Line Chart & Insights
                    with tab2:
                        st.subheader("📈 Sentiment Evolution Across Comment Timeline")
                        
                        trend_df = df.groupby(['Timestamp', 'Sentiment']).size().reset_index(name='Count')
                        trend_df['Timestamp'] = pd.to_datetime(trend_df['Timestamp'])
                        trend_df = trend_df.sort_values(by='Timestamp')

                        line_fig = px.line(
                            trend_df, x='Timestamp', y='Count', color='Sentiment',
                            color_discrete_map={'Positive': '#10b981', 'Negative': '#ef4444', 'Neutral': '#64748b'},
                            markers=True
                        )
                        line_fig.update_traces(line_shape='spline', line=dict(width=3))
                        line_fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#f8fafc', family="Inter"),
                            xaxis=dict(showgrid=False, title="Publication Date"),
                            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)', title="Comment Volume"),
                            hovermode="x unified"
                        )
                        st.plotly_chart(line_fig, use_container_width=True)

                        # Dynamic Graph Insight Engine
                        if not trend_df.empty:
                            peak_row = trend_df.loc[trend_df['Count'].idxmax()]
                            peak_date = peak_row['Timestamp'].strftime('%B %d, %Y')
                            peak_sentiment = peak_row['Sentiment']
                            
                            st.markdown(f"""
                            <div class="info-card" style="margin-top: 10px; border-left-color: #8b5cf6;">
                                💡 <b>Trend Insight:</b> Audience interaction reached its maximum peak on <b>{peak_date}</b>, largely driven by a surge in <b>{peak_sentiment.lower()}</b> engagement.
                            </div>
                            """, unsafe_allow_html=True)

                    # TAB 3: AI Creator Action Plan
                    with tab3:
                        st.subheader("🤖 AI Creator Action Plan & Intelligence Summary")
                        st.markdown("Automated insights synthesized from audience reaction signals:")
                        
                        col_praise, col_critique = st.columns(2)
                        
                        with col_praise:
                            st.markdown("#### 🌟 Key Viewer Praise")
                            for p in action_plan['praise']:
                                st.markdown(f'<div class="info-card">💬 "{p}"</div>', unsafe_allow_html=True)
                                
                        with col_critique:
                            st.markdown("#### ⚠️ Key Viewer Critiques")
                            for c in action_plan['criticism']:
                                st.markdown(f'<div class="error-card">💬 "{c}"</div>', unsafe_allow_html=True)

                        st.markdown("---")
                        st.markdown("#### 🎯 Strategic Content Recommendations")
                        for idx, act in enumerate(action_plan['actions'], 1):
                            st.markdown(f'<div class="success-card">✅ <b>Step {idx}:</b> {act}</div>', unsafe_allow_html=True)

                    # TAB 4: WordCloud
                    with tab4:
                        st.subheader("☁️ High-Frequency Keyword Cloud")
                        all_text = " ".join(df['Comment'].astype(str).tolist())
                        if all_text.strip():
                            mask_img = None
                            if os.path.exists("youtube_mask.png"):
                                try:
                                    mask_img = np.array(Image.open("youtube_mask.png"))
                                except Exception:
                                    pass
                            
                            wordcloud = WordCloud(
                                width=1200 if mask_img is None else None, 
                                height=500 if mask_img is None else None,
                                background_color='#080c14', colormap='plasma', 
                                max_words=150, mask=mask_img,
                                contour_width=2 if mask_img is not None else 0,
                                contour_color='#3b82f6'
                            ).generate(all_text)

                            fig, ax = plt.subplots(figsize=(12, 5), facecolor='#080c14')
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis("off")
                            plt.tight_layout(pad=0)
                            st.pyplot(fig)
                        else:
                            st.info("Not enough text data available.")

                    # TAB 5: Dataset Export (Bigger Table)
                    with tab5:
                        st.subheader("📋 PySpark Distributed Execution Output")
                        max_likes = int(df['Likes'].max()) if not df.empty and df['Likes'].max() > 0 else 10
                        styled_df = df.style.map(style_sentiment, subset=['Sentiment'])
                        
                        st.markdown('<div class="table-container">', unsafe_allow_html=True)
                        st.dataframe(
                            styled_df, 
                            use_container_width=True, 
                            height=700,
                            hide_index=True,
                            column_config={
                                "Likes": st.column_config.ProgressColumn("Likes", format="%d", min_value=0, max_value=max_likes),
                                "Score": st.column_config.NumberColumn("AI Score", help="RoBERTa Raw Output Score", format="%.4f"),
                                "Comment": st.column_config.TextColumn("Comment", width="large")
                            }
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                        clean_title = "".join(c if c.isalnum() else "_" for c in video_title)
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Export Processed Dataset (CSV)",
                            data=csv, file_name=f"datapulse_{clean_title[:30]}.csv", mime="text/csv"
                        )

            except Exception as e:
                st.error(f"An error occurred during execution: {e}")