import streamlit as st
import pandas as pd
from joblib import load
import re
import requests
from bs4 import BeautifulSoup
import time
from job_automation import run_automation


# Load models once
kmeans = load("kmeans_model.joblib")
vectorizer = load("tfidf_vectorizer.joblib")

def clean_skills(skills):
    if pd.isna(skills):
        return ""
    skills = skills.lower()
    skills = re.sub(r"[^a-zA-Z0-9, ]", "", skills)
    skills = [skill.strip() for skill in skills.split(",") if skill.strip()]
    return " ".join(skills)

# Use your existing scraper logic (simplified here)
def scrape_karkidi_jobs(keyword="data science", pages=1):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    jobs_list = []

    for page in range(1, pages + 1):
        url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        job_blocks = soup.find_all("div", class_="ads-details")

        for job in job_blocks:
            try:
                title = (job.find("h4") or job.find("h2")).get_text(strip=True) if job.find("h4") or job.find("h2") else ""
                company_tag = job.find("a", href=lambda x: x and "Employer-Profile" in x)
                company = company_tag.get_text(strip=True) if company_tag else ""
                skills = ""
                key_skills_tag = job.find("span", string="Key Skills")
                if key_skills_tag:
                    skills = key_skills_tag.find_next("p").get_text(strip=True)
                if not skills:
                    skills_block = job.find("div", class_="job-skills")
                    skills = skills_block.get_text(strip=True) if skills_block else ""

                jobs_list.append({
                    "Title": title,
                    "Company": company,
                    "Skills": skills
                })
            except:
                continue
        time.sleep(1)

    return pd.DataFrame(jobs_list)

def classify_new_jobs(df_new_jobs):
    df_new_jobs["Cleaned_Skills"] = df_new_jobs["Skills"].apply(clean_skills)
    X_new = vectorizer.transform(df_new_jobs["Cleaned_Skills"])
    df_new_jobs["Predicted_Cluster"] = kmeans.predict(X_new)
    return df_new_jobs

def notify_user(df_new_jobs, preferred_cluster):
    matched_jobs = df_new_jobs[df_new_jobs["Predicted_Cluster"] == preferred_cluster]
    if not matched_jobs.empty:
        return matched_jobs[["Title", "Company"]]
    else:
        return pd.DataFrame()

# --- Streamlit UI ---
st.title("Job Posting Classifier and Notifier")

keyword = st.text_input("Enter skill keyword(s) to search jobs:", "data science")
pages = st.slider("Number of pages to scrape:", 1, 5, 1)

if st.button("Scrape and Classify Jobs"):
    with st.spinner("Scraping jobs..."):
        df_jobs = scrape_karkidi_jobs(keyword, pages)

    with st.spinner("Classifying jobs..."):
        df_classified = classify_new_jobs(df_jobs)

    st.success(f"Found {len(df_classified)} jobs and classified into clusters.")

    cluster_options = df_classified["Predicted_Cluster"].unique().tolist()
    preferred_cluster = st.selectbox("Select your preferred cluster:", cluster_options)

    matched_jobs = notify_user(df_classified, preferred_cluster)

    if not matched_jobs.empty:
        st.markdown(f"### 🔔 Jobs in Cluster {preferred_cluster} matching your interest:")
        for idx, row in matched_jobs.iterrows():
            st.write(f"**{row['Title']}** at *{row['Company']}*")
    else:
        st.write(f"No new jobs found in Cluster {preferred_cluster}.")
