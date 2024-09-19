import streamlit as st
import contractions
import nltk
from nltk.corpus import stopwords
from transformers import pipeline
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import pandas as pd
import plotly.express as px
@st.cache_resource
def load_model():
    sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    return sentiment_model
def preprocess_text(text):
    # Expand contractions
    expanded_text = contractions.fix(text)
    
    # Convert text to lowercase
    expanded_text = expanded_text.lower()
    
    # Sentence splitting (if you want to process sentence by sentence)
    sentences = sent_tokenize(expanded_text)
    
    # Process each sentence
    cleaned_sentences = []
    for sentence in sentences:
        # Tokenize sentence into words
        words = word_tokenize(sentence)
        
        # Join words back into a cleaned sentence (no stopword removal)
        cleaned_sentence = ' '.join(words)
        cleaned_sentences.append(cleaned_sentence)
    
    # Return the cleaned sentences as a single string
    return ' '.join(cleaned_sentences)
# Function to determine overall sentiment based on sentiment distribution
def get_overall_sentiment(positive_count, negative_count):
    if positive_count > negative_count:
        return "POSITIVE"
    elif negative_count > positive_count:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

# Function to analyze review sentiment
def analyze_review_sentiment(review1):
    sentiment_model=load_model()
    sentences = sent_tokenize(review1)  # Split the review into sentences
    positive_count = 0
    negative_count = 0

    for sentence in sentences:
        sentiment = sentiment_model(sentence)[0]['label']
        if sentiment in ["4 stars", "5 stars"]:
            positive_count += 1
        elif sentiment in ["1 star", "2 stars"]:
            negative_count += 1


    return get_overall_sentiment(positive_count, negative_count)

# Create a function to perform sentiment analysis on a CSV file
def analyze_sentiment(csv_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Check if the CSV has a 'review' column
    if 'reviews.text' not in df.columns:
        st.error("CSV file must contain a 'review' column.")
        return None
    
    # Preprocess reviews and perform sentiment analysis
    #sentiment_model = load_model()
    df['cleaned_review'] = df['reviews.text'].apply(preprocess_text)
    df['sentiment'] = df['cleaned_review'].apply(lambda x: analyze_review_sentiment(x))

    
    return df



# Streamlit UI
st.title("Sentiment Analysis")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file with a 'reviews.text' column", type=["csv"])

if uploaded_file is not None:
    # Perform sentiment analysis
    result_df = analyze_sentiment(uploaded_file)
    
    if result_df is not None:
        # Display the result
        st.write(result_df)
        
        # Option to download the result as a CSV file
        csv = result_df.to_csv(index=False)
        st.download_button("Download Result", csv, "sentiment_analysis_result.csv", "text/csv")
         # Select the sentiment column (assuming you know the column name, e.g., "Sentiment")
    if 'sentiment' in result_df.columns:
        sentiment_counts = result_df['sentiment'].value_counts()

        # Plot a bar chart using Plotly
        
        fig = px.bar(
         sentiment_counts, 
         x=sentiment_counts.index, 
         y=sentiment_counts.values, 
         color=sentiment_counts.index,  # Add color to bars based on sentiment
         labels={'x': 'Sentiment', 'y': 'Count'},
         title='Sentiment Distribution',
          text=sentiment_counts.values,  # Display the count on each bar
           color_discrete_sequence=px.colors.qualitative.Pastel  # Use a vibrant color palette
)
        fig.update_traces(
    textposition='outside',  # Position text outside the bars for clarity
    hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>',  # Customize hover information
    marker_line_width=1,  # Add a border to bars for a sharp look
)

        fig.update_layout(
    title_font_size=24,  # Make the title larger
    xaxis_title_font_size=18,
    yaxis_title_font_size=18,
    plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    paper_bgcolor='white',  # Light grey background for the chart
    xaxis=dict(showgrid=False),  # Remove x-axis gridlines
    yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='white'),  # Light y-axis grid
    font=dict(family="Arial", size=14),  # Change font for a modern look
)

        # Display the plot
    st.plotly_chart(fig)
