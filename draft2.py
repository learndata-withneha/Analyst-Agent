import streamlit as st
import pandas as pd
# import openai
import os
import re
# from dotenv import load_dotenv
# load_dotenv()
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# Configure OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

# llm_gpt = ChatOpenAI(model="gpt-4o-mini", temperature=0)

import streamlit as st
from langchain_openai import ChatOpenAI

# Load API key from secrets
try:
    llm_gpt = ChatOpenAI(
        api_key=st.secrets["OPENAI_API_KEY"],
        model="gpt-4o-mini",
        temperature=0
    )
    st.success("API key loaded successfully!")
except KeyError:
    st.error("API key missing! Add to Streamlit Secrets.")

def extract_code(llm_response: str) -> str:
    """
    Extracts the Python code from an LLM response.
    Removes markdown code fences if present.
    """
    code = re.sub(r"^```(?:python)?", "", llm_response, flags=re.MULTILINE)
    code = re.sub(r"```$", "", code, flags=re.MULTILINE)
    return code.strip()

def generate_response(prompt):
    response = llm_gpt([
        SystemMessage(content="You are an AI assistant generating insights and dashboards."),
        HumanMessage(content=prompt)
    ])
    return response.content

def generate_insights(df,insight_prompt):
    insights_prompt = f"Analyze the dataset and generate key insights: {df.head().to_string(),insight_prompt}"[:8000]
    return generate_response(insights_prompt)

def generate_visualization_code(user_prompt: str, df) -> str:
    """
    Sends a custom prompt along with the user request to the LLM
    to generate a self-contained Streamlit visualization code snippet.
    """
    custom_prompt = f"""
You are a Python expert. Generate a self-contained Streamlit visualization code snippet that meets the following user request:
\"{user_prompt}\"

Ensure the code:
- Includes all necessary imports.
- Uses df = pd.read_csv(r".\\data.csv")
- Uses column names from {list(df.columns)}
- Creates interactive visualizations with Plotly or other libraries.
- Does NOT include any additional explanation—only output the complete Python code.
"""
    try:
        response = llm_gpt(custom_prompt).content
        if isinstance(response, str):
            code_text = response
        elif hasattr(response, "content"):
            code_text = response.content
        else:
            code_text = str(response)
        return extract_code(code_text)
    except Exception as e:
        st.error(f"Error during code generation: {e}")
        return ""

def main():
    # --- Custom Background Styling ---
    # Using a cute animated agent image from Pixabay
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://cdn.pixabay.com/photo/2017/01/31/21/23/cute-2025157_1280.png");
            background-size: cover;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("LLM-Generated Visualization Dashboard")
    
    # Initialize session state variables
    if "insights" not in st.session_state:
        st.session_state.insights = ""
    if "df" not in st.session_state:
        st.session_state.df = None

    # --- Sidebar Controls ---
    st.sidebar.header("Step 1: Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Save the uploaded file locally.
        with open("data.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success("File saved as data.csv")
        try:
            df = pd.read_csv("data.csv")
            st.session_state.df = df
        except Exception as e:
            st.sidebar.error(f"Error reading CSV file: {e}")
            return
    else:
        st.sidebar.info("Please upload a CSV file to begin.")
        return

    st.sidebar.header("Step 2: Generate Insights")
    insight_prompt = st.sidebar.text_area(
        "Enter your insight request",
        placeholder="E.g., 'Analyze sales trends for the last quarter.'"
    )
    if st.sidebar.button("Generate Insight"):
        st.session_state.insights = generate_insights(st.session_state.df,insight_prompt)

    st.sidebar.header("Step 3: Visualization Request")
    viz_prompt = st.sidebar.text_area(
        "Describe your visualization",
        placeholder="E.g., 'Create a scatter plot of column A vs. column B with a trend line.'"
    )
    generate_viz = st.sidebar.button("Generate Visualization Code")

    # --- Main Page Layout ---
    # Container for data preview and visualization output (charts on top)
    with st.container():
        st.subheader("Data Preview")
        st.write(st.session_state.df.head())
        
        if generate_viz:
            if not viz_prompt.strip():
                st.error("Please enter a visualization request.")
            else:
                st.info("Generating visualization code using the LLM...")
                code = generate_visualization_code(viz_prompt, st.session_state.df)
                if not code:
                    st.error("Failed to generate code. Please try again.")
                else:
                    st.subheader("Generated Visualization Code")
                    st.code(code, language="python")
                    st.subheader("Visualization Output")
                    # Prepare a safe execution context (Note: using exec() can be dangerous in untrusted environments)
                    exec_context = {
                        "st": st,
                        "pd": pd,
                        "df": st.session_state.df,
                        "px": __import__("plotly.express"),
                        "plt": __import__("matplotlib.pyplot"),
                        "sns": __import__("seaborn")
                    }
                    try:
                        exec(code, exec_context)
                    except Exception as exec_error:
                        st.error(f"Error executing generated code: {exec_error}")

    # Container for insights (always visible below the visualization)
    with st.container():
        st.subheader("Data Insights")
        if st.session_state.insights:
            st.write(st.session_state.insights)
        else:
            st.info("Data insights will appear here once generated.")

if __name__ == "__main__":
    main()
