import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Career Analyzer", layout="wide")

# ---------------- HEADER ----------------
st.markdown("# 🚀 AI Career Intelligence Dashboard")
st.markdown("### Discover your best career path with AI insights")

# ---------------- DATA ----------------
roles_data = {
    "Data Analyst": ["python", "sql", "excel", "powerbi", "statistics"],
    "Data Scientist": ["python", "machine learning", "statistics", "pandas", "numpy"],
    "ML Engineer": ["python", "deep learning", "tensorflow", "pytorch", "mlops"],
    "AI Engineer": ["python", "nlp", "computer vision", "deep learning"],
    "Frontend Developer": ["html", "css", "javascript", "react"],
    "Backend Developer": ["python", "django", "flask", "api"],
    "Full Stack Developer": ["html", "css", "javascript", "react", "node"],
    "DevOps Engineer": ["docker", "kubernetes", "aws", "linux"],
    "Cybersecurity Analyst": ["network", "security", "cryptography"],
    "Software Engineer": ["java", "c++", "dsa", "algorithms"]
}

df = pd.DataFrame([
    {"Role": role, "Skills": " ".join(skills)}
    for role, skills in roles_data.items()
])

# ---------------- SKILL LIST ----------------
all_skills = sorted(
    list(set(skill for skills in roles_data.values() for skill in skills)))

# ---------------- SIDEBAR ----------------
st.sidebar.header("🧠 Select Your Skills")

selected_skills = st.sidebar.multiselect(
    "Choose your skills",
    all_skills
)

experience = st.sidebar.slider("Experience (years)", 0, 10, 1)

analyze = st.sidebar.button("🚀 Analyze")

# ---------------- MODEL ----------------
vectorizer = TfidfVectorizer()
skill_matrix = vectorizer.fit_transform(df["Skills"])

# ---------------- MAIN ----------------
if analyze:
    if not selected_skills:
        st.warning("Please select at least one skill.")
    else:
        user_input = " ".join(selected_skills)
        user_vec = vectorizer.transform([user_input])
        similarity = cosine_similarity(user_vec, skill_matrix)[0]

        df["Score"] = similarity
        top_roles = df.sort_values(by="Score", ascending=False).head(3)

        st.markdown("## 🎯 Top Career Matches")

        cols = st.columns(3)

        for i, (_, row) in enumerate(top_roles.iterrows()):
            role = row["Role"]
            score = int(row["Score"] * 100 + experience * 2)

            required = set(roles_data[role])
            missing = required - set(selected_skills)

            with cols[i]:
                st.metric(role, f"{score}%")
                st.write("**Missing Skills:**")
                if missing:
                    st.error(", ".join(missing))
                else:
                    st.success("Perfect match!")

        # ---------------- CHART ----------------
        st.markdown("## 📊 Role Comparison")

        chart_df = df.sort_values(by="Score", ascending=False)
        st.bar_chart(chart_df.set_index("Role")["Score"])

        # ---------------- INSIGHTS ----------------
        st.markdown("## 💡 Insights")

        best_role = top_roles.iloc[0]["Role"]

        st.success(f"You are best suited for: **{best_role}**")

        if len(selected_skills) < 3:
            st.warning("Add more skills to improve accuracy.")
        elif len(selected_skills) > 6:
            st.info("You have a diverse skill set. Consider specialization.")

        # ---------------- SUGGESTIONS ----------------
        st.markdown("## 📚 Career Suggestions")

        suggestions = {
            "Data Analyst": "Learn Power BI, Tableau",
            "Data Scientist": "Improve ML & statistics",
            "ML Engineer": "Focus on deployment & MLOps",
            "AI Engineer": "Deep dive into NLP/CV",
            "Frontend Developer": "Learn advanced React",
            "Backend Developer": "Master APIs & databases",
            "Full Stack Developer": "Work on full projects",
            "DevOps Engineer": "Cloud certifications help",
            "Cybersecurity Analyst": "Practice ethical hacking",
            "Software Engineer": "Focus on DSA & system design"
        }

        st.info(suggestions.get(best_role, ""))

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("AI-powered Career Intelligence System • Streamlit App")
