import streamlit as st
from sentence_transformers import SentenceTransformer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import chromadb
import numpy as np
import speech_recognition as sr
import tempfile
import subprocess
import torch
import time 

st.set_page_config(page_title="Subtitle Search", layout="wide", page_icon="🎥")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Connect to ChromaDB
client = chromadb.PersistentClient(path=r"C:\Users\Akhira kodam\streamlit_26\GenAI\Shazam\chroma_db")
try:
    collection = client.get_collection("subtitle_chunks")
except ValueError:
    collection = client.create_collection("subtitle_chunks")

# Function to extract audio from video
def extract_audio_from_video(video_path, audio_path):
    command = ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

# Text Cleaning
def clean_text(text):
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    tokens = word_tokenize(clean_text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words]
    return ' '.join(clean_tokens).strip()

# Audio to Text Extraction
def extract_audio_text(audio_file):
    recognizer = sr.Recognizer()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        if audio_file.type.startswith("video"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(audio_file.read())
                temp_video_path = temp_video.name
            extract_audio_from_video(temp_video_path, temp_audio.name)
        else:
            temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name
    
    with sr.AudioFile(temp_audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Could not understand the audio."
        except sr.RequestError:
            return "Error with the speech recognition service."

# Retrieve Relevant Subtitles
def get_most_relevant_subtitles(query):
    cleaned_query = clean_text(query)
    query_embedding = model.encode([cleaned_query])
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5,
        include=['documents', 'metadatas'])

    documents = results.get('documents', [])
    metadatas = results.get('metadatas', [])

    relevant_subtitles = []
    if documents:
        for doc, meta in zip(documents[0], metadatas[0]):
            subtitle_info = {'subtitle': doc}
            if 'subtitle_id' in meta:
                subtitle_info['link'] = f"https://www.opensubtitles.org/en/subtitles/{meta['subtitle_id']}"
            relevant_subtitles.append(subtitle_info)

    return relevant_subtitles


# Sidebar
st.sidebar.title("📂 Upload File")
uploaded_file = st.sidebar.file_uploader("Upload Audio/Video", type=["mp3", "wav", "mp4"])

# Sidebar - Settings
with st.sidebar.expander("Settings", expanded=False):
    num_results = st.slider("Number of Results", 1, 10, 5)

# Title Section
st.markdown(
    """
    <h1 style="text-align: center; color: #2b95f3; font-family: 'Poppins', sans-serif;">Subtitle Search : Enhancing Video Content Accessibility (Shazam for Subtitles)🎬</h1>
    <p style="text-align: center; font-family: 'Poppins', sans-serif;">Find relevant subtitles by searching with text or audio queries.</p>
    """,
    unsafe_allow_html=True,
)

if uploaded_file:
    with st.spinner("🚀 Extracting audio & processing... Please wait."):

        # Audio to text extraction with progress bar
        query = extract_audio_text(uploaded_file)
        st.progress(100)

        st.success("Audio Processed !")

    st.markdown(f"### **Extracted Text :** `{query}`")

    if query:
        results = get_most_relevant_subtitles(query)
        
        if results:
            st.subheader("**Matching Subtitles Found :**")

            for res in results:
                with st.container():
                    st.markdown(
                        f"""
                        <div style="padding: 15px; border-radius: 10px; background-color: #282828; color: white; margin-bottom: 10px;">
                            <p style="font-size:18px; font-family: 'Poppins', sans-serif; text-align: left;"><b>🎞 Subtitle:</b> {res['subtitle']}</p>
                            {'<a href="' + res['link'] + '" target="_blank" style="color:#FFD700; font-size:16px;">🔗 View Full Subtitle</a>' if 'link' in res else ''}
                        </div>""",unsafe_allow_html=True,)
                    st.write("")
        else:
            st.error("No relevant subtitles found.")
