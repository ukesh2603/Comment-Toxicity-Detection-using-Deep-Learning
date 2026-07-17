import streamlit as st
import pickle
import torch
import pandas as pd
import matplotlib.pyplot as plt

from Model.model import TextLSTM
from utils import predict_comments,word2idx


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


model=TextLSTM(vocab_size=len(word2idx),embedding_dim=128,hidden_size=64,num_classes=6).to(device)


model.load_state_dict(torch.load("tox_model.pth",map_location=device))

model.eval()

df=pd.read_csv(r"/Users/suriya/Ukesh_AIML_Projects/Deep_Learning_Comment_Toxicity/Data/train.csv")

st.set_page_config(
    page_title="Comments Toxicity Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.title("💬 Toxic Comment Detection")

tab1, tab2, tab3, tab4 , tab5= st.tabs([
    "🏠 Home",
    "📊 Data Insights",
    "📈 Model Performance",
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
    st.header("Dataset Overview")

    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    st.dataframe(df.head(10))
    
    

with tab3:
    st.header("Model Performance")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Accuracy","89%")
    col2.metric("Precision","70%")
    col3.metric("Recall","68%")
    col4.metric("F1 Score","69%")

    examples = {
    "Positive":
        "Thank you for your help.",
    "Toxic":
        "You are an idiot.",
    "Threat":
        "I will kill you.",
    "Obscene":
        "*****"
    }

    choice = st.selectbox(
        "Choose Sample",
        examples.keys()
    )

    st.text_area(
        "Comment",
        examples[choice]
    )

    if st.button("Try Sample"):
        result = predict_comments(
            examples[choice],
            model
        )
        st.write(result)
    
with tab4:
    comment = st.text_area("Enter Comment")

    if st.button("Predict"):

        result = predict_comments(comment, model)

        st.subheader("Prediction")

        for label, value in result.items():

            if value["prediction"] == "Yes":
                st.success(f"{label} : {value['prediction']} ({value['probability']}%)")
            else:
                st.info(f"{label} : {value['prediction']} ({value['probability']}%)")
                
with tab5:
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