
import streamlit as st
import requests
import io
import base64
from PIL import Image

# Hugging Face API Key
hf_api_key = "put_your_key"
headers = {"Authorization": f"Bearer {hf_api_key}"}

# API URLs
ANALYSIS_API_URL = "https://api-inference.huggingface.co/models/dandelin/vilt-b32-finetuned-vqa"
GENERATION_API_URL = "https://api-inference.huggingface.co/models/thejagstudio/3d-animation-style-sdxl"

# Function to query analysis (VQA) model
def query_analysis(image_bytes, question):
    payload = {
        "inputs": {"question": question, "image": base64.b64encode(image_bytes).decode('utf-8')}
    }
    try:
        response = requests.post(ANALYSIS_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()[0].get('answer', 'unspecified')
    except Exception as e:
        st.error(f"Error: {e}")
    return 'unspecified'

# Function to query image generation model
def query_generation(prompt, image_bytes):
    payload = {
        "inputs": prompt,
        "image": base64.b64encode(image_bytes).decode('utf-8')
    }
    try:
        response = requests.post(GENERATION_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.content
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Function to save feedback to a file
def save_feedback(name, feedback, rating):
    with open("feedback.txt", "a") as f:
        f.write(f"Name: {name}\nFeedback: {feedback}\nRating: {rating}/5\n\n")

# Streamlit app title
st.title("Image Insight & Generation Studio 👻")

# Upload image section
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image bytes
    image_bytes = uploaded_file.read()

    # Display the uploaded image separately before generating the new image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Text input for additional description for image generation
    user_prompt = st.text_input("Enter additional description for image generation (optional):")

    if st.button("Generate Image"):
        # Analyze image (VQA) and extract features
        with st.spinner("Analyzing the image..."):
            gender = query_analysis(image_bytes, "What is the gender of the person in the image?")
            clothing = query_analysis(image_bytes, "What is the person wearing and which color?")
            hair_color = query_analysis(image_bytes, "What is the hair color of the person?")
            facial_expression = query_analysis(image_bytes, "What is the facial expression of the person?")
            age = query_analysis(image_bytes, "What is the estimated age of the person?")
            background = query_analysis(image_bytes, "What is in the background of the image?")  # New query for background

        # Build generation prompt based on VQA responses and user input
        if gender.lower() == "female":
            prompt = (f"Create a {age}-year-old girl with {hair_color} hair, "
                      f"wearing {clothing}, showing a {facial_expression}. "
                      f"Include a {background} background. {user_prompt}")
        else:
            prompt = (f"Create a {age}-year-old person with {hair_color} hair, "
                      f"wearing {clothing}, showing a {facial_expression}. "
                      f"Include a {background} background. {user_prompt}")

        # Call image generation API
        with st.spinner("Generating the image..."):
            generated_image_data = query_generation(prompt, image_bytes)
            if generated_image_data:
                # Store the generated image in session state
                st.session_state.generated_image_data = generated_image_data
                st.success("Image generated successfully!")

    # Display the generated image if available
    if 'generated_image_data' in st.session_state:
        st.markdown("### Generated Image")
        generated_image = Image.open(io.BytesIO(st.session_state.generated_image_data))
        st.image(generated_image, caption="Generated Image", use_column_width=True)

        # Provide download option for the generated image
        buffered = io.BytesIO()
        generated_image.save(buffered, format="PNG")
        st.download_button(
            label="Download Generated Image",
            data=buffered.getvalue(),
            file_name="generated_image.png",
            mime="image/png"
        )

        # Ask for feedback after the image is generated
        with st.form(key='feedback_form'):
            name = st.text_input("Your Name")
            feedback = st.text_area("Please leave your feedback")
            rating = st.slider("Rate the image quality", 1, 5)
            submit_button = st.form_submit_button(label='Submit Feedback')

            if submit_button:
                save_feedback(name, feedback, rating)
                st.success("Thank you for your feedback!")

# Ensure that the generated image does not disappear after feedback or download
if 'generated_image_data' in st.session_state:
    pass

# Footer for a better UI experience
st.markdown("---")
st.markdown("❤️‍🔥 *Made by Sujal Tamrakar*")
st.markdown("💡 *Powered by Hugging Face and Streamlit*")
