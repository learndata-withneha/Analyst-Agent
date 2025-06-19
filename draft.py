import streamlit as st
import pandas as pd
import os
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Import visualization libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATIONS_AVAILABLE = True
except ImportError as e:
    st.error(f"Visualization libraries not available: {e}")
    VISUALIZATIONS_AVAILABLE = False

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
    st.stop()

def extract_code(llm_response: str) -> str:
    """
    Extracts the Python code from an LLM response.
    Removes markdown code fences if present.
    """
    code = re.sub(r"^```(?:python)?", "", llm_response, flags=re.MULTILINE)
    code = re.sub(r"```$", "", code, flags=re.MULTILINE)
    return code.strip()

def generate_response(prompt):
    response = llm_gpt.invoke([
        SystemMessage(content="You are an AI assistant generating insights and dashboards."),
        HumanMessage(content=prompt)
    ])
    return response.content

def generate_insights(df, insight_prompt):
    insights_prompt = f"Analyze the dataset and generate key insights: {df.head().to_string()}, {insight_prompt}"[:8000]
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
- Uses the dataframe variable 'df' that is already loaded
- Uses column names from {list(df.columns)}
- Creates visualizations with Plotly (preferred) or matplotlib
- Uses st.plotly_chart() for plotly charts or st.pyplot() for matplotlib
- Does NOT include any additional explanation—only output the complete Python code
- Does NOT include imports or data loading - those are already handled

Example format:
fig = px.scatter(df, x='column1', y='column2', title='My Chart')
st.plotly_chart(fig)
"""
    try:
        response = llm_gpt.invoke(custom_prompt)
        
        if hasattr(response, 'content'):
            code_text = response.content
        else:
            code_text = str(response)
            
        return extract_code(code_text)
    except Exception as e:
        st.error(f"Error during code generation: {e}")
        return ""

def main():
    # --- Custom Background Styling ---
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
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.sidebar.success("File loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV file: {e}")
            return
    else:
        st.sidebar.info("Please upload a CSV file to begin.")
        return

    # Check if visualization libraries are available
    if not VISUALIZATIONS_AVAILABLE:
        st.error("Visualization libraries are not installed. Please check your requirements.txt")
        return

    st.sidebar.header("Step 2: Generate Insights")
    insight_prompt = st.sidebar.text_area(
        "Enter your insight request",
        placeholder="E.g., 'Analyze sales trends for the last quarter.'"
    )
    if st.sidebar.button("Generate Insight"):
        with st.spinner("Generating insights..."):
            st.session_state.insights = generate_insights(st.session_state.df, insight_prompt)

    st.sidebar.header("Step 3: Visualization Request")
    viz_prompt = st.sidebar.text_area(
        "Describe your visualization",
        placeholder="E.g., 'Create a scatter plot of column A vs. column B with a trend line.'"
    )
    generate_viz = st.sidebar.button("Generate Visualization Code")

    # --- Main Page Layout ---
    with st.container():
        st.subheader("Data Preview")
        st.write(st.session_state.df.head())
        
        if generate_viz:
            if not viz_prompt.strip():
                st.error("Please enter a visualization request.")
            else:
                with st.spinner("Generating visualization code using the LLM..."):
                    code = generate_visualization_code(viz_prompt, st.session_state.df)
                    
                if not code:
                    st.error("Failed to generate code. Please try again.")
                else:
                    st.subheader("Generated Visualization Code")
                    st.code(code, language="python")
                    st.subheader("Visualization Output")
                    
                    # Create a safer execution environment
                    exec_globals = {
                        "__builtins__": __builtins__,
                        "st": st,
                        "pd": pd,
                        "df": st.session_state.df,
                        "px": px,
                        "go": go,
                        "plt": plt,
                        "sns": sns
                    }
                    
                    try:
                        exec(code, exec_globals)
                    except Exception as exec_error:
                        st.error(f"Error executing generated code: {exec_error}")
                        st.error("Please try a different visualization request or check the generated code.")

    # Container for insights
    with st.container():
        st.subheader("Data Insights")
        if st.session_state.insights:
            st.write(st.session_state.insights)
        else:
            st.info("Data insights will appear here once generated.")

if __name__ == "__main__":
    main()
