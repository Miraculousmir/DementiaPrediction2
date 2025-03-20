import wave
import librosa
import numpy as np
import pickle
import streamlit as st
import speech_recognition as sr
import pandas as pd
import liwc
import re
from collections import Counter
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

# spell = SpellChecker()

# Load the trained model
MODEL_PATH = "mfcc.pkl"
with open(MODEL_PATH, "rb") as model_file:
    modelmfcc = pickle.load(model_file)


# Tokenize function (same as before)
def tokenize(speech_text):
    for match in re.finditer(r'\w+', speech_text, re.UNICODE):
        yield match.group(0)


'''def cosine_to_probability_piecewise(cosine_similarity):
    if cosine_similarity <= 0.2:
        # Linear transformation for cosine similarity between 0 and 0.2
        probability = 90 + 25 * cosine_similarity
    elif 0.2 < cosine_similarity <= 0.6:
        # Linear transformation for cosine similarity between 0.2 and 0.6
        probability = 95 - 112.5 * (cosine_similarity - 0.2)
    else:
        # Linear transformation for cosine similarity between 0.6 and 1
        probability = 6 + 5 * (1 - cosine_similarity)

    return round(probability, 2)'''


def classify_dementia_scale(cosine_similarity, dementia_prob):
    # First, apply the cosine_to_probability_piecewise transformation to map cosine similarity to a range
    def cosine_to_probability_piecewise(cosine_similarity):
        # Ensure cosine_similarity is a scalar
        if isinstance(cosine_similarity, (np.ndarray, list)):
            cosine_similarity = float(cosine_similarity[0])  # Convert to scalar if array-like

        if cosine_similarity <= 0.2:
            # Linear transformation for cosine similarity between 0 and 0.2
            probability = 90 + 25 * cosine_similarity
        elif 0.2 < cosine_similarity <= 0.6:
            # Linear transformation for cosine similarity between 0.2 and 0.6
            probability = 95 - 112.5 * (cosine_similarity - 0.2)
        else:
            # Linear transformation for cosine similarity between 0.6 and 1
            probability = 6 + 5 * (1 - cosine_similarity)

        return round(probability)

    # Ensure inputs are scalars
    if isinstance(cosine_similarity, (np.ndarray, list)):
        cosine_similarity = float(cosine_similarity[0])
    if isinstance(dementia_prob, (np.ndarray, list)):
        dementia_prob = float(dementia_prob[0])

    # Get the transformation of cosine similarity to probability
    similarity_prob = cosine_to_probability_piecewise(cosine_similarity)

    # Scale the dementia probability (between 0 and 1) to the range [32, 96]
    scaled_dementia_prob = 32 + (dementia_prob * (96 - 32))

    # Now, blend the similarity-based probability and the dementia probability
    # Higher cosine similarity should push the result higher in the range (more towards 96%)
    if cosine_similarity >= 0.4:
        # If cosine similarity is high, give more weight to the dementia probability
        final_probability = (0.6 * scaled_dementia_prob) + (0.4 * similarity_prob)
    else:
        # If cosine similarity is low, give more weight to the similarity-based probability
        final_probability = (0.4 * scaled_dementia_prob) + (0.6 * similarity_prob)

    return round(final_probability, 2)


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


def extract_468_features(audio_file, sr=22050, n_mfcc=13):
    y, sr = librosa.load(audio_file, sr=sr)
    window_length = int(0.023 * sr)  # 23ms window
    hop_length = int(0.010 * sr)  # 10ms step size

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=window_length, window='hamming')
    delta_mfcc = librosa.feature.delta(mfcc, order=1)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    features_per_frame = np.vstack([mfcc, delta_mfcc, delta2_mfcc])

    min_features = np.min(features_per_frame, axis=1)
    max_features = np.max(features_per_frame, axis=1)
    mean_features = np.mean(features_per_frame, axis=1)
    std_features = np.std(features_per_frame, axis=1)

    summary_features_per_turn = np.hstack([min_features, max_features, mean_features, std_features])
    summary_features_per_turn = summary_features_per_turn.reshape(1, -1)  # Shape (1, 156)

    max_across_turns = np.max(summary_features_per_turn, axis=0)
    mean_across_turns = np.mean(summary_features_per_turn, axis=0)
    std_across_turns = np.std(summary_features_per_turn, axis=0)

    final_468_features = np.concatenate([max_across_turns, mean_across_turns, std_across_turns])  # Shape (468,)
    return final_468_features.reshape(1, -1)  # Reshape for model input


# Function to predict dementia probability
def predict_dementia(features):
    probabilities = modelmfcc.predict_proba(features)[0]  # Get probabilities
    return probabilities


def show_page():
    st.header("Predict with text or audio", divider="blue")
    st.subheader("Image Description")

    # Ensure ran_idx is stored in session state
    if "ran_idx" not in st.session_state:
        st.session_state.ran_idx = random.randint(0, 3)

    # Use the persistent ran_idx value from session state
    ran_idx = st.session_state.ran_idx

    pic_desc = [
        "This is a picture featuring a chaotic kitchen scene. The man is busy cutting veggies while the girls are cooking something. The dustbin is smelly and overfilled with waste. The mop and bucket are lying on the floor with water spilled over. The cat is sitting in the middle. There are many items on the table. The water in the pots in the oven is boiling. The kitchen is in complete disarray.",
        "This is a picture of a typical organized kitchen. The pans are neatly hanging on the wall. There is a fridge, oven, and chimney. The sink is kept clean with no dishes to wash. There's a small vase that adds to the aesthetics of the kitchen. The floor is made of vitrified checkered tiles, which are shiny and spick-free. Such an organized and neat place makes people happy.",
        "This picture features a mom with her two kids, a girl and a boy. The mom is busy doing the dishes with the sink overflowing with water, while the children are up to some naughty behavior. It seems both are busy stealing cookies from the shelf behind their mom's back. The boy is about to fall as the stool on which he is standing seems to topple while his sister is giggling or laughing and demands more cookies from her brother.",
        "This is a lively playground scene. All people seem so happy and cheerful, especially the children. Some are enjoying the slide while others are on the swings. A girl seems to be busy sharing something with her friend sitting on the bench, while her friend seems uninterested and more focused on eating. Two children are skipping ropes. An elder seems to have come with his baby in a stroller. One person seems to walk his dog. The two children seem thirsty, as they are quenching their thirst by drinking from the tap. Some children are playing tag. The person sitting on the bench seems to be speaking on the phone. Overall, the atmosphere seems merry."]

    pic_desc_key = [
        "chaotic kitchen  man  cutting veggies  girls  cooking  smelly dustbin  overfilled waste  mop  bucket  floor  spilled water   cat middle  items  table  boiling water pots oven disarray ",
        "typical organized kitchen pans neatly hanging fridge oven chimney clean sink no dishes small vase aesthetics vitrified checkered tiles shiny spick-free neat place happy",
        " mom kids girl boy dishes sink overflowing water naughty behavior stealing cookies shelf behind mom boy falling stool toppling sister giggling laughing demanding cookies mother kitchen cupboard tool curtain basin ",
        " playground  lively scene  happy people  cheerful children  slide  swings  girl  sharing  friend  bench  uninterested  eating  skipping ropes  elder  baby  stroller  person  dog  children  thirsty  drinking tap water  playing tag  bench  phone  merry atmosphere park "
    ]

    # ran_idx = random.randint(0, 3)
    # Check if the selected image is already in session state
    if "selected_image" not in st.session_state:
        # Randomly select a new image only once per session
        pic_list = ["picture0.jpg", "picture1.jpg", "picture2.jpg", "picture3.jpg"]

        # ran_idx = random.randint(0,4)
        # st.image(pic_list[ran_idx])
        st.session_state.selected_image = pic_list[ran_idx]
    # Display the selected image
    st.image(st.session_state.selected_image)

    # Add dropdown for description method
    description_method = st.selectbox("Choose a description method:", ("Describe with text", "Describe with audio"))

    text = ""
    flag = False
    if description_method == "Describe with text":
        # Text input for manual description
        text = st.text_input("Please describe the picture here:")
        flag = True
    elif description_method == "Describe with audio":
        # Initialize the recognizer
        recognizer = sr.Recognizer()

        # Record a voice message
        audio_value = st.audio_input("Describe the picture",
                                     help="Press the record button to record your description")

        if audio_value:
            # Display the audio player
            st.audio(audio_value, format="audio/wav")

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

            # Convert raw audio bytes to numpy array (assuming 16-bit PCM)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

            # Create an in-memory WAV file using the wave module
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit PCM
                wav_file.setframerate(22050)  # Match sample rate from original function
                wav_file.writeframes(audio_np.tobytes())

            buffer.seek(0)  # Move to the beginning of the file

            features_468 = extract_468_features(buffer)

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

        '''# Multiply by 10 and round
        dementia_prob_rounded = (dementia_prob * 10).round().astype(int)
        dementia_prob_rounded_final = dementia_prob_rounded[0] * 0.2
        '''
        desc_text = pic_desc[ran_idx]

        # Initialize the TF-IDF Vectorizer
        vectorizer = TfidfVectorizer()

        # Compute TF-IDF vectors for the texts
        tfidf_matrix = vectorizer.fit_transform([text, desc_text])

        # Compute cosine similarity between the two vectors
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

        # rounded_similarity= round(similarity[0][0],5)
        '''taken_prob= cosine_to_probability_piecewise(similarity[0][0])

        final_taken_prob = taken_prob


        words = text.split()
        misspelled = spell.unknown(words)
        if similarity == 0.0 and len(misspelled) == 0:
            st.write("You are talking out of context. Please try again. ")
        else:
            st.write(f"Probability of having Dementia out of 10: {dementia_prob_rounded_final}")
        # Display the prediction result
        ###st.write(f"{rounded_similarity}")'''
        # final_prediction = (2 + (8 * dementia_prob) - (2 * similarity)) * 10
        # final_prediction= classify_dementia_scale(cosine_similarity, dementia_prob)
        if not flag:
            dementia_prob2 = predict_dementia(features_468)
        col1, col2 = st.columns(2)
        with col1:
            st.write("Linguistic Score(LIWC)")
            st.write(f"### {dementia_prob}%")

        with col2:
            if not flag:
                st.write("Acoustic Score(MFCC)")
                st.write(f"### {dementia_prob2[1] * 100:.2f}%")
