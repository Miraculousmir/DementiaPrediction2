# from spellchecker import SpellChecker
import pickle
import streamlit as st
import speech_recognition as sr
import io
import pandas as pd
import liwc
import re
from collections import Counter
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# spell = SpellChecker()
# Tokenize function (same as before)
def tokenize(speech_text):
    for match in re.finditer(r'\w+', speech_text, re.UNICODE):
        yield match.group(0)

def cosine_to_probability_piecewise(cosine_similarity):
    if cosine_similarity <= 0.2:
        # Linear transformation for cosine similarity between 0 and 0.2
        probability = 90 + 25 * cosine_similarity
    elif 0.2 < cosine_similarity <= 0.6:
        # Linear transformation for cosine similarity between 0.2 and 0.6
        probability = 95 - 112.5 * (cosine_similarity - 0.2)
    else:
        # Linear transformation for cosine similarity between 0.6 and 1
        probability = 6 + 5 * (1 - cosine_similarity)

    return round(probability, 2)

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

    pic_desc = ["This is a picture featuring a chaotic kitchen scene. The man is busy cutting veggies while the girls are cooking something. The dustbin is smelly and overfilled with waste. The mop and bucket are lying on the floor with water spilled over. The cat is sitting in the middle. There are many items on the table. The water in the pots in the oven is boiling. The kitchen is in complete disarray.",
                "This is a picture of a typical organized kitchen. The pans are neatly hanging on the wall. There is a fridge, oven, and chimney. The sink is kept clean with no dishes to wash. There's a small vase that adds to the aesthetics of the kitchen. The floor is made of vitrified checkered tiles, which are shiny and spick-free. Such an organized and neat place makes people happy.",
                "This picture features a mom with her two kids, a girl and a boy. The mom is busy doing the dishes with the sink overflowing with water, while the children are up to some naughty behavior. It seems both are busy stealing cookies from the shelf behind their mom's back. The boy is about to fall as the stool on which he is standing seems to topple while his sister is giggling or laughing and demands more cookies from her brother.",
                "This is a lively playground scene. All people seem so happy and cheerful, especially the children. Some are enjoying the slide while others are on the swings. A girl seems to be busy sharing something with her friend sitting on the bench, while her friend seems uninterested and more focused on eating. Two children are skipping ropes. An elder seems to have come with his baby in a stroller. One person seems to walk his dog. The two children seem thirsty, as they are quenching their thirst by drinking from the tap. Some children are playing tag. The person sitting on the bench seems to be speaking on the phone. Overall, the atmosphere seems merry."]
   
    pic_desc_key = ["chaotic kitchen  man  cutting veggies  girls  cooking  smelly dustbin  overfilled waste  mop  bucket  floor  spilled water   cat middle  items  table  boiling water pots oven disarray ",
    "typical organized kitchen pans neatly hanging fridge oven chimney clean sink no dishes small vase aesthetics vitrified checkered tiles shiny spick-free neat place happy",
    " mom kids girl boy dishes sink overflowing water naughty behavior stealing cookies shelf behind mom boy falling stool toppling sister giggling laughing demanding cookies mother kitchen cupboard tool curtain basin ",   
" playground  lively scene  happy people  cheerful children  slide  swings  girl  sharing  friend  bench  uninterested  eating  skipping ropes  elder  baby  stroller  person  dog  children  thirsty  drinking tap water  playing tag  bench  phone  merry atmosphere park "
    ]
    
    ran_idx = random.randint(0, 3)
    # Check if the selected image is already in session state
    if "selected_image" not in st.session_state:
        # Randomly select a new image only once per session
        pic_list = ["picture0.jpg", "picture1.jpg", "picture2.jpg", "picture3.jpg"]

        #ran_idx = random.randint(0,4)
        #st.image(pic_list[ran_idx])
        st.session_state.selected_image = pic_list[ran_idx]
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
        dementia_prob_rounded_final = dementia_prob_rounded[0] * 0.2
        desc_text = pic_desc_key[ran_idx]

        # Initialize the TF-IDF Vectorizer
        vectorizer = TfidfVectorizer()

        # Compute TF-IDF vectors for the texts
        tfidf_matrix = vectorizer.fit_transform([text, desc_text])

        # Compute cosine similarity between the two vectors
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        #rounded_similarity= round(similarity[0][0],5)
        taken_prob= similarity[0][0]

        final_taken_prob = taken_prob

    
        # words = text.split()
        # misspelled = spell.unknown(words)
        # if similarity == 0.0 and len(misspelled) == 0:
        # st.write("You are talking out of context. Please try again. ")
        # else:
    st.write(f"Probability of having Dementia out of 10: {final_taken_prob}")
        # Display the prediction result
        ###st.write(f"{rounded_similarity}")
