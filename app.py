import streamlit as st

# Set the title for the main app
st.title("Dementia Prediction System")

# Create a sidebar for navigation
st.sidebar.title("Menu")
page = st.sidebar.radio("Select an option:", ("Predict with demographic data", "Predict with speech data"))

# Render the selected page
if page == "Predict with demographic data":
    from demographics import show_page
    show_page()
elif page == "Predict with speech data":
    from speech import show_page
    show_page()
