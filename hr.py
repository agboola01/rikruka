import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os

# Load the SBERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load CV database
@st.cache
def load_cv_data(file_path):
    return pd.read_csv(file_path)

# Embedding function for CV data
def embed_cvs(cv_data):
    cv_texts = cv_data.apply(lambda x: f"{x['education']} {x['skills']} {x['projects']} {x['certifications']} {x['experience']}", axis=1)
    cv_embeddings = model.encode(cv_texts, convert_to_tensor=True)
    return cv_embeddings, cv_data

# Function to save job details
def save_job_details(company_name, position_title, job_description):
    job_details = pd.DataFrame([[job_description, position_title, company_name]], 
                               columns=['Job Description', 'Position Title', 'Company Name'])
    if not os.path.exists("job_details.csv"):
        job_details.to_csv("job_details.csv", index=False)
    else:
        job_details.to_csv("job_details.csv", mode='a', header=False, index=False)

# Main function for JD matching
def find_best_matches(cv_embeddings, job_description, top_n=7):
    job_description_embedding = model.encode(job_description, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(job_description_embedding, cv_embeddings)[0]
    top_results = np.argpartition(-similarities, range(top_n))[:top_n]
    return top_results, similarities

# Streamlit UI
st.title("JD-CV Matching System")
st.header("Input Job Description")

# User inputs for job details
company_name = st.text_input("Company Name")
job_description = st.text_area("Job Description")
position_title = st.text_input("Position Title")

# Number of CV matches to display
top_n = st.number_input("Number of CVs to display", min_value=1, max_value=20, value=7, step=1)

if st.button("Find Top Matches"):
    # Process job description
    full_job_description = f"{position_title} at {company_name}: {job_description}"
    
    # Save job details
    save_job_details(company_name, position_title, job_description)

    # Load and embed CVs
    cv_data = load_cv_data('data/cv_data.csv')
    cv_embeddings, cv_data = embed_cvs(cv_data)

    # Find top matches
    top_results, similarities = find_best_matches(cv_embeddings, full_job_description, top_n=top_n)

    # Display top CVs
    st.subheader("Top Matching CVs")
    for idx in top_results:
        match_percentage = similarities[idx].item() * 100  # Convert similarity to percentage
        st.write(f"Match Score: {match_percentage:.2f}%")
        st.write(cv_data.iloc[idx].to_dict())
