import streamlit as st
import os
import tempfile
import speech_recognition as sr
from huggingface_hub import InferenceClient
import docx2txt

st.set_page_config(page_title="Kolachi", layout="wide")
st.title("🏫 Kolachi😋 - AI Assistant🤖")

HF_TOKEN = st.secrets["HF_TOKEN"]

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
hf_client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

@st.cache_resource
def load_hotel_knowledge():
    try:
        path = "HotelData.docx"
        if os.path.exists(path):
            return docx2txt.process(path)
        else:
            return "Error: HotelData.docx not found"
    except Exception as e:
        return f"Error reading document: {e}"

hotel_context = load_hotel_knowledge()

def speech_to_text(audio_bytes):
    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        audio_path = f.name
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

def generate_hotel_response(user_query):
    system_prompt = f"""
    You are the official AI assistant for Kolachi (managed by Raja).
    Use the following hotel information to answer questions.
    If the answer isn't in the info, politely say "Please contact our manager Raja directly for that!"

    HOTEL INFORMATION:
    {hotel_context}
    """

    messages = [{"role": "system", "content": system_prompt}]
    for msg in st.session_state.messages[-5:]:
        messages.append(msg)
    messages.append({"role": "user", "content": user_query})

    try:
        response = ""
        for chunk in hf_client.chat_completion(messages, max_tokens=500, stream=True):
            if chunk.choices and chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content
        return response if response else "I'm sorry, I couldn't generate a response. Please try again."
    except Exception as e:
        return f"⚠️ Service is busy: {str(e)}"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to Kolachi! How can I help you today?"}
    ]

if "last_processed_audio" not in st.session_state:
    st.session_state.last_processed_audio = None

with st.sidebar:
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome back! How can I help you?"}
        ]
        st.session_state.last_processed_audio = None
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

col1, col2 = st.columns([0.85, 0.15])
with col1:
    text_input = st.chat_input("Ask about the menu, check-in, or hotel rules...")
with col2:
    voice_data = st.audio_input("🎤", label_visibility="collapsed")

user_query = None

if text_input:
    user_query = text_input
elif voice_data:
    current_audio_id = hash(voice_data.getvalue())
    if st.session_state.last_processed_audio != current_audio_id:
        with st.spinner("🎧 Transcribing your request..."):
            try:
                user_query = speech_to_text(voice_data.getvalue())
                st.session_state.last_processed_audio = current_audio_id
            except Exception:
                st.error("Speech not recognized. Try speaking closer to the mic.")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Checking records..."):
            answer = generate_hotel_response(user_query)
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    st.rerun()
