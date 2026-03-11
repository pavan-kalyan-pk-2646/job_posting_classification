import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import random

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Job Intelligence Dashboard",
    page_icon="🚀",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM UI STYLE
# ---------------------------------------------------
st.markdown("""
<style>

.stApp {
background: linear-gradient(135deg,#020617,#0f172a,#1e293b);
color:white;
}

.title {
font-size:65px;
font-weight:900;
text-align:center;
background: linear-gradient(90deg,#22d3ee,#9333ea,#ec4899);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

.glass {
background: rgba(255,255,255,0.05);
backdrop-filter: blur(12px);
border-radius:15px;
padding:25px;
margin-bottom:20px;
border:1px solid rgba(255,255,255,0.08);
}

.stButton>button {
background: linear-gradient(90deg,#22d3ee,#9333ea);
color:white;
border-radius:10px;
height:50px;
font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
model = pickle.load(open("../models/model.pkl","rb"))
vectorizer = pickle.load(open("../models/vectorizer.pkl","rb"))

data = pd.read_csv("../data/job_dataset.csv")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("🚀 Navigation")

page = st.sidebar.radio(
"Select Page",
["🏠 Home","🤖 Job Predictor","📊 Market Analytics","📂 Dataset Explorer"]
)

# ---------------------------------------------------
# HOME PAGE
# ---------------------------------------------------
if page == "🏠 Home":

    st.markdown('<div class="title">AI Job Intelligence System</div>',unsafe_allow_html=True)

    st.markdown(
    """
    <div class="glass">
    AI powered system for <b>Job Posting Classification and Analysis</b>
    using <b>Machine Learning + Natural Language Processing</b>.
    </div>
    """,
    unsafe_allow_html=True
    )

    col1,col2 = st.columns(2)

    col1.metric("📄 Total Jobs",len(data))
    col2.metric("📊 Categories",data["category"].nunique())

    st.markdown("---")

    st.subheader("System Features")

    st.write("""
    • AI Job Category Prediction  
    • Skill Extraction from Job Descriptions  
    • Job Market Analytics Dashboard  
    • Interactive Data Visualization  
    """)

# ---------------------------------------------------
# JOB PREDICTOR
# ---------------------------------------------------
elif page == "🤖 Job Predictor":

    st.markdown('<div class="title">AI Job Predictor</div>',unsafe_allow_html=True)

    st.markdown('<div class="glass">',unsafe_allow_html=True)

    sample_jobs = [
        "Python machine learning deep learning neural networks",
        "Java spring boot backend developer REST API",
        "SEO digital marketing social media campaigns",
        "React javascript frontend developer UI design",
        "SQL data analysis business intelligence dashboards"
    ]

    if st.button("🎲 Generate Sample Job Description"):
        description = random.choice(sample_jobs)
    else:
        description = ""

    description = st.text_area(
        "Paste Job Description",
        value=description,
        height=200
    )

    if st.button("🚀 Predict Category"):

        if description.strip() == "":
            st.warning("Enter job description first")

        else:

            vector = vectorizer.transform([description])
            prediction = model.predict(vector)[0]

            st.success(f"🎯 Predicted Category: {prediction}")

            prob = model.predict_proba(vector)[0]
            confidence = max(prob)*100

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={'text':"Confidence Score"},
                gauge={'axis':{'range':[0,100]}}
            ))

            st.plotly_chart(fig,use_container_width=True)

            fig2 = px.bar(
                x=model.classes_,
                y=prob,
                title="Prediction Probabilities"
            )

            st.plotly_chart(fig2,use_container_width=True)

    st.markdown('</div>',unsafe_allow_html=True)

    # Skill detection
    st.markdown('<div class="glass">',unsafe_allow_html=True)

    st.subheader("🧠 Detected Skills")

    skills = [
        "python","machine learning","deep learning",
        "pandas","numpy","scikit learn",
        "sql","java","react","seo","marketing"
    ]

    found = []

    if description:
        for skill in skills:
            if skill in description.lower():
                found.append(skill)

    if found:

        cols = st.columns(len(found))

        for i,skill in enumerate(found):
            cols[i].metric("Skill",skill)

    else:
        st.info("No major skills detected")

    st.markdown('</div>',unsafe_allow_html=True)

# ---------------------------------------------------
# ANALYTICS DASHBOARD
# ---------------------------------------------------
elif page == "📊 Market Analytics":

    st.markdown('<div class="title">Job Market Analytics</div>',unsafe_allow_html=True)

    st.markdown('<div class="glass">',unsafe_allow_html=True)

    fig = px.histogram(
        data,
        x="category",
        color="category",
        title="Job Category Distribution"
    )

    st.plotly_chart(fig,use_container_width=True)

    st.markdown('</div>',unsafe_allow_html=True)

    # Top skills
    st.markdown('<div class="glass">',unsafe_allow_html=True)

    words = " ".join(data["description"]).lower().split()

    common = Counter(words).most_common(10)

    skills = [i[0] for i in common]
    counts = [i[1] for i in common]

    fig3 = px.bar(
        x=skills,
        y=counts,
        title="Top Skills in Job Market"
    )

    st.plotly_chart(fig3,use_container_width=True)

    st.markdown('</div>',unsafe_allow_html=True)

    # Word cloud
    st.markdown('<div class="glass">',unsafe_allow_html=True)

    text = " ".join(data["description"])

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="black",
        colormap="viridis"
    ).generate(text)

    fig, ax = plt.subplots()

    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig)

    st.markdown('</div>',unsafe_allow_html=True)

# ---------------------------------------------------
# DATASET PAGE
# ---------------------------------------------------
elif page == "📂 Dataset Explorer":

    st.markdown('<div class="title">Dataset Explorer</div>',unsafe_allow_html=True)

    st.markdown('<div class="glass">',unsafe_allow_html=True)

    search = st.text_input("Search Job Description")

    if search:
        results = data[data["description"].str.contains(search,case=False)]
        st.write(results)
    else:
        st.dataframe(data)

    st.write("Dataset Shape:",data.shape)

    if st.checkbox("Show Category Counts"):
        st.write(data["category"].value_counts())

    st.markdown('</div>',unsafe_allow_html=True)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")

st.markdown("""
### 💡 Job Posting Classification and Analysis  
Machine Learning + NLP + Streamlit Dashboard
""")