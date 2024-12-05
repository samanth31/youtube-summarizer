!pip install youtube-transcript-api # Install the required module

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
from textblob import TextBlob
import re
import nltk

# ... (rest of your code) ...

# Lazy loading for NLTK resources
def ensure_nltk_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# Initialize summarization pipeline
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to summarize text
def summarize_text(text, max_length=50000):
    chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]  # Break into chunks for summarization
    summaries = [
        summarization_pipeline(chunk, max_length=130, min_length=50, do_sample=False)[0]['summary_text']
        for chunk in chunks
    ]
    return ' '.join(summaries)

# Function to extract keywords
def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum()]
    keywords = [word for word in words if word not in stop_words and len(word) > 1]
    vectorizer = CountVectorizer()
    vectorizer.fit([' '.join(keywords)])
    keywords_freq = vectorizer.vocabulary_
    sorted_keywords = sorted(keywords_freq.items(), key=lambda item: item[1], reverse=True)[:5]
    return [word for word, freq in sorted_keywords]

# Function to perform topic modeling (LDA)
def topic_modeling(text):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tf = vectorizer.fit_transform([text])
    lda_model = LatentDirichletAllocation(n_components=5, max_iter=5, learning_method='online', random_state=42)
    lda_model.fit(tf)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        topics.append([feature_names[i] for i in topic.argsort()[:-6:-1]])
    return topics

# Function to extract YouTube video ID from URL
def extract_video_id(url):
    patterns = [
        r'v=([^&]+)',  # Pattern for URLs with 'v=' parameter
        r'youtu.be/([^?]+)',  # Pattern for shortened URLs
        r'youtube.com/embed/([^?]+)'  # Pattern for embed URLs
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Main Streamlit app
def main():
    ensure_nltk_resources()  # Ensure NLTK resources are available
    st.title("YouTube Video Summarizer")

    # User input for YouTube video URL
    video_url = st.text_input("Enter YouTube Video URL:", "")
    max_summary_length = st.slider("Max Summary Length:", 1000, 20000, 50000)

    if st.button("Summarize"):
        try:
            # Extract video ID
            video_id = extract_video_id(video_url)
            if not video_id:
                st.error("Invalid YouTube URL. Please enter a valid URL.")
                return

            # Get video transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            if not transcript:
                st.error("Transcript not available for this video.")
                return

            # Combine transcript text
            video_text = ' '.join([line['text'] for line in transcript])

            # Perform summarization
            summary = summarize_text(video_text, max_length=max_summary_length)

            # Extract keywords
            keywords = extract_keywords(video_text)

            # Perform topic modeling
            topics = topic_modeling(video_text)

            # Perform sentiment analysis
            sentiment = TextBlob(video_text).sentiment

            # Display results
            st.subheader("Video Summary:")
            st.write(summary)

            st.subheader("Keywords:")
            st.write(keywords)

            st.subheader("Topics:")
            for idx, topic in enumerate(topics):
                st.write(f"Topic {idx + 1}: {', '.join(topic)}")

            st.subheader("Sentiment Analysis:")
            st.write(f"Polarity: {sentiment.polarity}")
            st.write(f"Subjectivity: {sentiment.subjectivity}")

        except TranscriptsDisabled:
            st.error("Transcripts are disabled for this video.")
        except NoTranscriptFound:
            st.error("No transcript found for this video.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
