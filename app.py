import streamlit as st
from pypdf import PdfReader
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from io import BytesIO


# -----------------------------
# Load LLM
# -----------------------------
@st.cache_resource
def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        tokenizer="google/flan-t5-large"
    )

llm = load_llm()


# -----------------------------
# Load embedding model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()


# -----------------------------
# Skill extraction
# -----------------------------
def extract_skills(text, skills_list):

    text = text.lower()

    found = []

    for skill in skills_list:
        if skill.lower() in text:
            found.append(skill)

    return list(set(found))


# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("🤖 AI Resume Analyzer")
st.write("Upload your resume to analyze skills and job matches.")


uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])


# -----------------------------
# Load datasets
# -----------------------------
with open("skills.txt") as f:
    skills_list = f.read().splitlines()

jobs = pd.read_csv("jobs.csv")


# -----------------------------
# Create FAISS index
# -----------------------------
@st.cache_resource
def create_index():

    job_embeddings = model.encode(jobs["description"].tolist())

    dimension = job_embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(job_embeddings))

    return index, job_embeddings

index, job_embeddings = create_index()


# -----------------------------
# Resume Processing
# -----------------------------
if uploaded_file:

    st.success("Resume uploaded successfully!")

    reader = PdfReader(uploaded_file)

    resume_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            resume_text += text


    # -----------------------------
    # Detect skills
    # -----------------------------
    detected_skills = extract_skills(resume_text, skills_list)


    # -----------------------------
    # Resume embedding
    # -----------------------------
    resume_embedding = model.encode([resume_text])


    # -----------------------------
    # Find best job matches
    # -----------------------------
    D, I = index.search(np.array(resume_embedding), k=10)


    col1, col2 = st.columns(2)


    # -----------------------------
    # Skills Section
    # -----------------------------
    with col1:

        st.subheader("Detected Skills")

        if detected_skills:

            for skill in detected_skills:

                st.markdown(
                    f"""
                    <div style="
                    padding:10px;
                    margin:5px;
                    border-radius:10px;
                    background:#1E293B;">
                    {skill}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.write("No skills detected")


    # -----------------------------
    # Job Recommendations
    # -----------------------------
    missing_skills = set()
    recommended_roles = []

    with col2:

        st.subheader("Top Job Matches")

        for idx, i in enumerate(I[0]):

            job = jobs.iloc[i]

            recommended_roles.append(job["job_title"])

            similarity = 1 / (1 + D[0][idx])

            match_percent = int(similarity * 100)

            st.markdown(
                f"""
                <div style="
                padding:15px;
                margin-bottom:15px;
                border-radius:12px;
                background:#1E293B;">
                <h4>{job['job_title']}</h4>
                <p>{job['description'][:200]}...</p>
                <b>Match Score:</b> {match_percent}% <br>
                <a href="https://www.google.com/search?q={job['job_title']} jobs" target="_blank">
                Apply Now
                </a>
                </div>
                """,
                unsafe_allow_html=True
            )

            job_skills = extract_skills(job["description"], skills_list)

            for skill in job_skills:
                if skill not in detected_skills:
                    missing_skills.add(skill)


    # -----------------------------
    # Skill Gap
    # -----------------------------
    st.subheader("Skill Gap (Recommended to Learn)")

    if missing_skills:

        for skill in missing_skills:

            st.markdown(
                f"""
                <div style="
                padding:10px;
                margin:5px;
                border-radius:10px;
                background:#7c2d12;">
                {skill}
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        st.write("Your skills match the recommended jobs well.")


    # -----------------------------
    # ATS Score
    # -----------------------------
    required_skills = set()

    for i in I[0]:
        job = jobs.iloc[i]
        job_skills = extract_skills(job["description"], skills_list)
        required_skills.update(job_skills)

    if required_skills:
        skill_match = len(set(detected_skills) & required_skills) / len(required_skills)
    else:
        skill_match = 0

    similarity_score = 1 / (1 + D[0][0])

    resume_length = len(resume_text.split())

    if resume_length > 300:
        quality_score = 1
    elif resume_length > 150:
        quality_score = 0.7
    else:
        quality_score = 0.4

    keyword_coverage = len(detected_skills) / max(len(required_skills), 1)

    ats_score = (
        skill_match * 40 +
        similarity_score * 40 +
        quality_score * 10 +
        keyword_coverage * 10
    )

    ats_score = int(min(ats_score, 100))

    st.subheader("ATS Compatibility Score")

    st.progress(ats_score)

    st.write(f"Match Score: {ats_score}%")


    # -----------------------------
    # AI Feedback
    # -----------------------------
    st.subheader("AI Resume Feedback")

    with st.spinner("Generating AI feedback..."):

        # Helper to call LLM with few-shot prompts
        def generate_feedback(prompt):
            result = llm(
                prompt,
                max_new_tokens=512,
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=2.5,
                early_stopping=True
            )
            text = result[0].get("generated_text", "")
            if isinstance(text, list):
                text = " ".join(text)
            return text.strip()

        # Prepare context strings
        skills_str = ", ".join(detected_skills[:8]) if detected_skills else "no specific skills"
        missing_str = ", ".join(list(missing_skills)[:6]) if missing_skills else "none"
        roles_str = ", ".join(recommended_roles[:3]) if recommended_roles else "software developer"

        # Prompt 1: Strengths (few-shot)
        prompt_strengths = (
            f"You are a career advisor. A candidate applying for {roles_str} roles knows these skills: {skills_str}. "
            f"Their ATS compatibility score is {ats_score}%. "
            f"Write exactly 3 strengths of this candidate with detailed explanation.\n\n"
            f"Example format:\n"
            f"1. Strong programming foundation - Proficiency in multiple languages shows versatility and ability to adapt to different tech stacks.\n"
            f"2. Database expertise - SQL knowledge enables efficient data management and complex query optimization.\n\n"
            f"Now write 3 strengths for this candidate:"
        )
        strengths_text = generate_feedback(prompt_strengths)

        # Prompt 2: Weaknesses (few-shot)
        prompt_weaknesses = (
            f"You are a career advisor. A candidate applying for {roles_str} roles is missing these skills: {missing_str}. "
            f"Their ATS score is {ats_score}%. "
            f"Write exactly 3 weaknesses with detailed explanation of why each matters.\n\n"
            f"Example format:\n"
            f"1. Lack of cloud computing skills - Most modern companies deploy on AWS or Azure, making this a critical gap.\n"
            f"2. No DevOps experience - CI/CD pipelines are essential for modern software delivery workflows.\n\n"
            f"Now write 3 weaknesses for this candidate:"
        )
        weaknesses_text = generate_feedback(prompt_weaknesses)

        # Prompt 3: Skills to learn (few-shot)
        prompt_skills = (
            f"You are a career advisor. A candidate wants to become a {roles_str}. "
            f"They are missing: {missing_str}. "
            f"Recommend exactly 3 skills to learn with explanation of why each skill is important and how to learn it.\n\n"
            f"Example format:\n"
            f"1. Docker - Essential for containerizing applications. Start with Docker official tutorials and build small projects.\n"
            f"2. AWS - Cloud platforms are used by 90% of companies. Get the AWS Cloud Practitioner certification.\n\n"
            f"Now recommend 3 skills for this candidate:"
        )
        skills_text = generate_feedback(prompt_skills)

        # Prompt 4: Resume tips (few-shot)
        prompt_tips = (
            f"You are a resume expert. A {roles_str} candidate knows {skills_str} and has an ATS score of {ats_score}%. "
            f"Give exactly 3 actionable resume improvement tips with specific examples.\n\n"
            f"Example format:\n"
            f"1. Add measurable achievements - Instead of 'worked on backend', write 'Built REST APIs serving 10K daily requests with 99.9% uptime'.\n"
            f"2. Use industry keywords - Include terms like 'Agile', 'microservices', 'scalable systems' to pass ATS filters.\n\n"
            f"Now give 3 tips for this candidate:"
        )
        tips_text = generate_feedback(prompt_tips)

        # Assemble final feedback
        feedback = f"""**✅ Strengths:**\n{strengths_text}\n\n**⚠️ Weaknesses:**\n{weaknesses_text}\n\n**📘 Skills to Learn:**\n{skills_text}\n\n**💡 Resume Improvement Tips:**\n{tips_text}"""

    st.markdown(f"### 📄 AI Resume Review\n\n{feedback}")


    # -----------------------------
    # Download Report
    # -----------------------------
    st.subheader("Download Analysis Report")

    report = f"""
    AI Resume Analyzer Report

    Detected Skills:
    {detected_skills}

    Missing Skills:
    {list(missing_skills)}

    ATS Score:
    {ats_score}

    AI Feedback:
    {feedback}
    """

    st.download_button(
        label="Download Report",
        data=report,
        file_name="resume_analysis.txt"
    )