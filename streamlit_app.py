import requests
import streamlit as st 

st.title('COVID X-Ray Classification')

uploaded_file = st.file_uploader("Upload a Chest X-Ray Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    st.image(uploaded_file, caption='Uploaded X-Ray Image', use_container_width=True)

    with st.spinner('Classifying...'):
        files = {"file": uploaded_file.getvalue()}

        try:
            response = requests.post("http://localhost:8000/predict", files=files)

            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: **{result['predicted_class']}**")
                st.info(f"Confidence: {float(result['confidence'])*100:.2f}%")
            else:
                st.error(f"Server Error: {response.text}")

        except Exception as e:
            st.error(f"Request Failed: {e}")





