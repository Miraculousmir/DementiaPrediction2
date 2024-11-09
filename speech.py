import pickle
import streamlit as st
import speech_recognition as sr
import io
import pandas as pd
import liwc
import re
from collections import Counter
import random


# Tokenize function (same as before)
def tokenize(speech_text):
    for match in re.finditer(r'\w+', speech_text, re.UNICODE):
        yield match.group(0)


# Function to compute category frequencies normalized by total word count
def compute_liwc_categories(speech_text, category_names, parse):
    tokens = list(tokenize(speech_text.lower()))  # Tokenize and convert to lowercase
    total_tokens = len(tokens)  # Total number of tokens

    # Initialize all categories with zero count
    category_frequencies = {category: 0 for category in category_names}

    # Count categories in the text
    category_counts = Counter(category for token in tokens for category in parse(token))

    # Update category frequencies based on counts
    for category, count in category_counts.items():
        category_frequencies[category] = count / total_tokens if total_tokens > 0 else 0

    return category_frequencies


def show_page():
    st.header("Predict with Speech Data", divider="blue")
    st.subheader("Image Description")

    # Check if the selected image is already in session state
    if "selected_image" not in st.session_state:
        # Randomly select a new image only once per session
        pic_list = ["picture0.jpg", "picture1.jpg", "picture2.jpg", "picture3.jpg"]
        st.session_state.selected_image = random.choice(pic_list)

    # Display the selected image
    st.image(st.session_state.selected_image)

    # Add dropdown for description method
    description_method = st.selectbox("Choose a description method:", ("Describe with text", "Describe with audio"))

    text = ""
    if description_method == "Describe with text":
        # Text input for manual description
        text = st.text_input("Please describe the picture here:")

    elif description_method == "Describe with audio":
        # Initialize the recognizer
        recognizer = sr.Recognizer()

        # Record a voice message
        audio_value = st.experimental_audio_input("Describe the picture",
                                                  help="Press the record button to record your description")

        if audio_value:
            # Display the audio player
            st.audio(audio_value)

            # Read the audio data from the UploadedFile object
            audio_bytes = audio_value.read()  # Read the content as bytes

            # Use BytesIO to create a file-like object from the bytes data
            audio_file_like = io.BytesIO(audio_bytes)

            # Load the audio data from the BytesIO object
            with sr.AudioFile(audio_file_like) as source:
                # Adjust for ambient noise (optional)
                recognizer.adjust_for_ambient_noise(source)

                # Record the audio data
                audio = recognizer.record(source)

            # Recognize the speech in the audio data
            try:
                # Using Google's Web Speech API for recognition
                text = recognizer.recognize_google(audio)
                st.write("Transcription: ", text)
            except sr.UnknownValueError:
                st.write("Could not understand audio")
            except sr.RequestError as e:
                st.write(f"Could not request results from the service; {e}")

    # Load LIWC dictionary
    parse, category_names = liwc.load_token_parser('LIWC2007_English.dic')

    # Apply the LIWC category frequency computation to each transcript
    liwc_results = compute_liwc_categories(text, category_names, parse)

    # Convert the results into a DataFrame with one column for each LIWC category
    liwc_df = pd.DataFrame([liwc_results])

    # Load the model from the pickle file
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    if st.button("Predict"):
        # Displaying a confirmation message and the entered information
        st.success("Submitted Successfully!")

        input_row_format = pd.read_csv('input_row_format.csv')

        # Ensure that liwc_df has the same columns as X_train
        liwc_df = liwc_df.reindex(columns=input_row_format.columns, fill_value=0)

        y_prob = model.predict_proba(liwc_df)

        # The output of predict_proba() is an array with two columns:
        # Column 0: Probability of the class '0' (no dementia)
        # Column 1: Probability of the class '1' (dementia)

        # To get the probability of dementia (class 1):
        dementia_prob = y_prob[:, 1]  # This gives the probability of class 1 (dementia)

        # Multiply by 10 and round
        dementia_prob_rounded = (dementia_prob * 10).round().astype(int)

        # Display the prediction result
        st.write(f"You have a {dementia_prob_rounded[0]} out of 10 chance of having dementia.")
