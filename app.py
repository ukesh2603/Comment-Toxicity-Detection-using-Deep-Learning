import streamlit as st
import pickle
import torch
import pandas as pd

from Model.model import TextLSTM
from utils import predict_comments,word2idx


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


model=TextLSTM(vocab_size=len(word2idx),embedding_dim=128,hidden_size=64,num_classes=6).to(device)


model.load_state_dict(torch.load("tox_model.pth",map_location=device))

model.eval()


st.set_page_config(
    page_title="Comments Toxicity Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Toxic Comment Detection")


st.title("💬 Toxic Comment Detection")

tab1, tab2, tab3 = st.tabs([
    "🏠 Home",
    "⚡ Real Time Prediction",
    "📂 Bulk Prediction"
])

with tab1:
    st.header("Welcome")

    st.write("""
    This application detects toxic comments using a Deep Learning (LSTM) model.

    Features:
    - Real time prediction
    - Bulk CSV prediction
    - Download prediction results
    """)
    
    
with tab2:
    comment = st.text_area("Enter Comment")

    if st.button("Predict"):

        result = predict_comments(comment, model)

        st.subheader("Prediction")

        for label, value in result.items():

            if value["prediction"] == "Yes":
                st.success(f"{label} : {value['prediction']} ({value['probability']}%)")
            else:
                st.info(f"{label} : {value['prediction']} ({value['probability']}%)")
                
with tab3:
    st.header("Bulk Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)
        df=df.head(100)
        st.write("Preview")
        st.dataframe(df.head())


        column = st.selectbox(
            "Select Comment Column",
            df.columns
        )

        if st.button("Predict CSV"):

            results = []

            for comment in df[column]:

                prediction = predict_comments(str(comment), model)

                row = {}

                for label, value in prediction.items():
                    row[label] = value["prediction"]

                results.append(row)

            prediction_df = pd.DataFrame(results)

            final_df = pd.concat([df, prediction_df], axis=1)

            st.success("Prediction Completed!")

            st.dataframe(final_df)

            csv = final_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Predictions",
                csv,
                "toxicity_predictions.csv",
                "text/csv"
            )