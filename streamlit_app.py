# streamlit_app.py (Streamlit Application with local transcription)
import streamlit as st
import whisper
import os
from tempfile import NamedTemporaryFile

st.title("Audio to Text Transcription App")

# Load Whisper model once for the Streamlit app
model = whisper.load_model("small", device="cpu")

# Allow the user to upload an audio file (MP3, WAV)
uploaded_file = st.file_uploader("Choose an audio file...", type=["mp3", "wav"])

if uploaded_file is not None:
    # Display the uploaded file name and play it
    st.audio(uploaded_file, format='audio/mp3')
    st.write(f"Uploaded file: {uploaded_file.name}")

    # Transcribe the audio locally when the button is clicked
    if st.button('Transcribe'):
        # Save the uploaded file to a temporary location
        with NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            temp_audio_file.write(uploaded_file.read())
            temp_audio_file_path = temp_audio_file.name

        try:
            # Transcribe the audio using Whisper
            result = model.transcribe(temp_audio_file_path)
            transcription = result['text']

            # Display the transcription
            st.write(f"Transcription: {transcription}")

            # Delete the temporary file after transcription
            os.remove(temp_audio_file_path)

        except Exception as e:
            st.write(f"Error: {str(e)}")
