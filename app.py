import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# PATH
DATA_URL = st.sidebar.text_input("Enter file path for Tweets data:", "/Users/adityapandey/Desktop/GitHub Repo/DashBoard-Streamlit/Tweets.csv")

# Title and Description
st.title("Sentiment Analysis of Tweets about US Airlines")
st.sidebar.title("Sentiment Analysis of Tweets")
st.markdown("This application is a Streamlit dashboard used to analyze sentiments of tweets 🐦")
st.sidebar.markdown("This application is a Streamlit dashboard used to analyze sentiments of tweets 🐦")

# load and cache data
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    return data

try:
    data = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Random tweet based on sentiment
st.sidebar.subheader("Show random tweet")
random_tweet = st.sidebar.radio('Sentiment', ('positive', 'neutral', 'negative'), key='random_tweet_radio')
st.sidebar.markdown(data.query("airline_sentiment == @random_tweet")[["text"]].sample(n=1).iat[0, 0])

# Number of tweets by sentiment
st.sidebar.markdown("### Number of tweets by sentiment")
select = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='sentiment_plot_type')
sentiment_count = data['airline_sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment': sentiment_count.index, 'Tweets': sentiment_count.values})

if not st.sidebar.checkbox("Hide", True, key='sentiment_checkbox'):
    st.markdown("### Number of tweets by sentiment")
    if select == 'Bar plot':
        fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
        st.plotly_chart(fig)

# Tweets by hour
st.sidebar.subheader("When and where are users tweeting from?")
hour = st.sidebar.slider("Hour to look at", 0, 23, key='hour_slider')
modified_data = data[data['tweet_created'].dt.hour == hour]

if not st.sidebar.checkbox("Hide", True, key='hour_checkbox'):
    st.markdown("### Tweet locations based on time of day")
    st.markdown(f"{len(modified_data)} tweets between {hour}:00 and {(hour + 1) % 24}:00")
    st.map(modified_data)
    if st.sidebar.checkbox("Show raw data", False, key='raw_data_checkbox'):
        st.write(modified_data)

# Total tweets for each airline
st.sidebar.subheader("Total number of tweets for each airline")
each_airline = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='airline_plot_type')
airline_sentiment_count = data.groupby('airline')['airline_sentiment'].count().sort_values(ascending=False)
airline_sentiment_count = pd.DataFrame({'Airline': airline_sentiment_count.index, 'Tweets': airline_sentiment_count.values})

if not st.sidebar.checkbox("Hide", True, key='airline_checkbox'):
    if each_airline == 'Bar plot':
        st.subheader("Total number of tweets for each airline")
        fig_1 = px.bar(airline_sentiment_count, x='Airline', y='Tweets', color='Tweets', height=500)
        st.plotly_chart(fig_1)
    elif each_airline == 'Pie chart':
        st.subheader("Total number of tweets for each airline")
        fig_2 = px.pie(airline_sentiment_count, values='Tweets', names='Airline')
        st.plotly_chart(fig_2)

# Plot sentiment breakdown
@st.cache_data
def plot_sentiment(airline):
    df = data[data['airline'] == airline]
    count = df['airline_sentiment'].value_counts()
    return pd.DataFrame({'Sentiment': count.index, 'Tweets': count.values})

# Breakdown by sentiment for selected airlines
st.sidebar.subheader("Breakdown airline by sentiment")
choice = st.sidebar.multiselect('Pick airlines', data['airline'].unique(), key='airline_choice_multiselect')

if len(choice) > 0:
    st.subheader("Breakdown airline by sentiment")
    breakdown_type = st.sidebar.selectbox('Visualization type', ['Pie chart', 'Bar plot'], key='breakdown_type')
    
    if breakdown_type == 'Bar plot':
        # Create bar plot subplots
        fig_3 = make_subplots(rows=1, cols=len(choice), subplot_titles=choice)
        for i, airline in enumerate(choice):
            sentiment_data = plot_sentiment(airline)
            fig_3.add_trace(
                go.Bar(x=sentiment_data['Sentiment'], y=sentiment_data['Tweets'], name=airline),
                row=1, col=i + 1
            )
    elif breakdown_type == 'Pie chart':
        # Create pie chart subplots
        fig_3 = make_subplots(
            rows=1,
            cols=len(choice),
            specs=[[{'type': 'domain'} for _ in range(len(choice))]],
            subplot_titles=choice
        )
        for i, airline in enumerate(choice):
            sentiment_data = plot_sentiment(airline)
            fig_3.add_trace(
                go.Pie(labels=sentiment_data['Sentiment'], values=sentiment_data['Tweets'], name=airline),
                row=1, col=i + 1
            )

    fig_3.update_layout(height=600, width=800 * len(choice))
    st.plotly_chart(fig_3)

# Word Cloud
st.sidebar.header("Word Cloud")
word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?', ('positive', 'neutral', 'negative'), key='wordcloud_radio')

if not st.sidebar.checkbox("Hide", True, key='wordcloud_checkbox'):
    st.subheader(f'Word cloud for {word_sentiment} sentiment')
    df = data[data['airline_sentiment'] == word_sentiment]
    words = ' '.join(df['text'])
    processed_words = ' '.join(word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT')
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(processed_words)
    
    # Create figure and axes for the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")  # Hide the axes

    # Display the word cloud using the figure
    st.pyplot(fig)