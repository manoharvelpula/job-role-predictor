import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Job Role Predictor", layout="wide")

# ---------------- HEADER ----------------
st.title("🚀 AI Job Role Fit Analyzer")
st.markdown(
    "Predict the best job role based on your skills using Machine Learning")

st.info("👈 Enter your skills in the sidebar and click 'Analyze'")

# ---------------- DATA ----------------
data = {
    "Role": [
        "Data Analyst", "Data Scientist", "ML Engineer", "AI Engineer",
        "Frontend Developer", "Backend Developer", "Full Stack Developer",
        "DevOps Engineer", "Cybersecurity Analyst", "Software Engineer"
    ],
    "Skills": [
        "python sql excel powerbi statistics visualization",
        "python machine learning statistics pandas numpy data analysis",
        "python deep learning tensorflow pytorch mlops deployment",
        "python ai deep learning nlp computer vision",
        "html css javascript react ui ux frontend",
        "python django flask api backend database",
        "html css javascript python react node fullstack",
        "docker kubernetes aws ci cd linux",
        "network security ethical hacking cryptography",
        "java c++ data structures algorithms problem solving"
    ]
}

df = pd.DataFrame(data)

# ---------------- MODEL ----------------
vectorizer = TfidfVectorizer()
skill_matrix = vectorizer.fit_transform(df["Skills"])

# ---------------- SIDEBAR ----------------
st.sidebar.header("🧠 Enter Your Profile")

user_input = st.sidebar.text_area(
    "Enter your skills (e.g., python sql machine-learning)",
    height=150
)

experience = st.sidebar.slider("Experience (years)", 0, 10, 1)

analyze = st.sidebar.button("🚀 Analyze")

# ---------------- LOGIC ----------------
if analyze:
    if user_input.strip() == "":
        st.warning("Please enter your skills.")
    else:
        try:
            user_vec = vectorizer.transform([user_input.lower()])
            similarity = cosine_similarity(user_vec, skill_matrix)[0]

            best_idx = np.argmax(similarity)
            best_role = df.iloc[best_idx]["Role"]
            best_score = similarity[best_idx]

            score = min(100, int(best_score * 100 + experience * 3))

            required_skills = set(df.iloc[best_idx]["Skills"].split())
            user_skills = set(user_input.lower().split())
            missing = list(required_skills - user_skills)

            # ---------------- RESULTS ----------------
            st.subheader("📊 Results")

            col1, col2, col3 = st.columns(3)
            col1.metric("🎯 Predicted Role", best_role)
            col2.metric("📈 Match Score", f"{score}%")
            col3.metric("⚡ Experience Boost", f"+{experience*3}%")

            # ---------------- CHART ----------------
            st.subheader("📉 Role Comparison")
            chart_df = pd.DataFrame({
                "Role": df["Role"],
                "Score": similarity
            }).sort_values(by="Score", ascending=False)

            st.bar_chart(chart_df.set_index("Role"))

            # ---------------- SKILLS ----------------
            st.subheader("⚠️ Missing Skills")
            if missing:
                st.error(", ".join(missing))
            else:
                st.success("You have all required skills!")

            # ---------------- INSIGHTS ----------------
            st.subheader("💡 Insights")

            if score > 85:
                st.success("Excellent match! You are job-ready.")
            elif score > 65:
                st.warning("Good match. Improve some skills.")
            else:
                st.error("Low match. Work on core skills.")

        except Exception as e:
            st.error(f"Error: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with Streamlit • AI-based Job Role Prediction System")
