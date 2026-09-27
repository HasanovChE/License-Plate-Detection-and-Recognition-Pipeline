import streamlit as st
import requests
from PIL import Image

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="LPR System", page_icon="🚗")

st.title("🚗 License Plate Recognition (LPR)")
st.write("Upload a car image and automatically read the license plate.")

uploaded_file = st.file_uploader(
    "Select an image (JPG, PNG)...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Read Number (Detect)"):
        with st.spinner("AI processes the image..."):
            files = {"file": uploaded_file.getvalue()}
            
            try:
                response = requests.post(API_URL, files=files)
                result = response.json()
                
                if result.get("success") and len(result["plates"]) > 0:
                    st.success(f"{len(result['plates'])} license plate found!")
                    
                    for idx, plate in enumerate(result["plates"], 1):
                        st.markdown(f"### Result #{idx}")
                        st.write(f"**Number:** `{plate['plate']}`")
                        st.write(f"**Country format:** {plate['format']['country']}")
                        st.write(f"**OCR Raw Text:** {plate['raw_ocr']}")
                        st.write(f"**Belief in disclosure (Confidence):** {plate['detection_confidence']:.2f}")
                        st.markdown("---")
                else:
                    st.warning("No license mark was found or could be read in the frame.")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API server. Ensure that the FastAPI server is running (on port 8000).")