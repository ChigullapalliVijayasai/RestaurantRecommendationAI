import streamlit as st
import os
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from langchain_openai import ChatOpenAI

# -------------------- Streamlit Page Setup --------------------
st.set_page_config(page_title="🍽️ Restaurant Recommendation AI", page_icon="🍴", layout="centered")
st.title("🍽️ Restaurant Recommendation AI")
st.write("An AI agent that recommends restaurants based on your preferences using LangChain and OpenAI.")

# -------------------- Load API Key --------------------
# Securely stored in Streamlit Cloud (Settings > Secrets)
# In local environment, create a .env file or use `st.secrets`
api_key = st.secrets["OPENAI_API_KEY"]

# -------------------- Load Data --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("restaurants.csv")
    return df

df = load_data()

st.subheader("📊 Sample of Restaurant Data")
st.dataframe(df.head())

# -------------------- User Inputs --------------------
st.subheader("✨ Find Your Ideal Restaurant")

user_query = st.text_input(
    "Describe what you’re looking for (e.g., 'affordable Italian restaurant near city center' or 'romantic dinner spot under 1000 INR')."
)

# -------------------- Create AI Agent --------------------
if user_query:
    with st.spinner("🤖 Thinking..."):
        llm = ChatOpenAI(openai_api_key=api_key, temperature=0.7, model="gpt-4o-mini")

        # Create a LangChain agent connected to the restaurant DataFrame
        agent = create_pandas_dataframe_agent(llm, df, verbose=True)

        # Query the agent
        try:
            response = agent.run(user_query)
            st.success("✅ Recommendation found!")
            st.write(response)
        except Exception as e:
            st.error(f"⚠️ Something went wrong: {e}")

# -------------------- Footer --------------------
st.markdown("---")
st.caption("Built with ❤️ using LangChain, Streamlit, and OpenAI.")


api_key = st.secrets["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"]
