import streamlit as st
import pandas as pd
import openai
import os
import re
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.llms import Ollama
from langchain_experimental.tools import PythonAstREPLTool



load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

llm_gpt = ChatOpenAI(model="gpt-4o-mini", temperature=0)




# Configure your OpenAI API key.
# Ensure you set this in Streamlit secrets or as an environment variable.

def extract_code(llm_response: str) -> str:
    """
    Extracts the Python code from an LLM response.
    Removes markdown code fences if present.
    
    Args:
        llm_response (str): The response content from the LLM.
    
    Returns:
        str: Clean Python code.
    """
    # Remove starting and ending markdown code fences if they exist.
    code = re.sub(r"^```(?:python)?", "", llm_response, flags=re.MULTILINE)
    code = re.sub(r"```$", "", code, flags=re.MULTILINE)
    return code.strip()

def generate_visualization_code(user_prompt: str,df) -> str:
    """
    Sends a custom prompt along with the user prompt to the LLM
    to generate Python visualization code.
    
    Args:
        user_prompt (str): The user's visualization request.
    
    Returns:
        str: The generated Python code.
    """
    # Craft a custom prompt with clear instructions.
    custom_prompt = f"""
You are a Python expert. generate a self-contained Streamlit visualization code snippet that meets the following user request:
\"{user_prompt}\"

Ensure the code:
- Includes all necessary imports.
- Uses df=pd.read_csv(r".\supply_chain_data.csv")
--use column names from {df.columns}
- Creates interactive visualizations with Plotly or other libraries.
- Does NOT include any additional explanation—only output the complete Python code.

"""
    try:
        # response = openai.chat.completions.create(
        #     model="gpt-4o",
        #     messages=[{"role": "user", "content": custom_prompt}],
        #     temperature=0  # Lower temperature for more deterministic output.
        # )

        response = llm_gpt(custom_prompt).content
        # Extract the code from the LLM response.
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
    st.title("LLM-Generated Visualization Dashboard")
    
    st.sidebar.header("Step 1: Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    
    st.sidebar.header("Step 2: Enter Visualization Request")
    user_prompt = st.sidebar.text_area(
        "Describe the visualization you'd like to create",
        placeholder="E.g., 'Create a scatter plot of column A vs. column B with a trend line.'"
    )
    
    if not uploaded_file:
        st.info("Please upload a CSV file to begin.")
        return
    
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.write(df.head())
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return
    
    if st.sidebar.button("Generate Visualization Code"):
        if not user_prompt.strip():
            st.error("Please enter a visualization request.")
            return
        
        st.info("Generating visualization code using the LLM...")
        code = generate_visualization_code(user_prompt,df)
        
        if not code:
            st.error("Failed to generate code. Please try again.")
            return
        
        # Display the generated code.
        st.subheader("Generated Visualization Code")
        st.code(code, language="python")
        
        # Prepare the execution context for the generated code.
        # Note: Exposing exec() can be dangerous; only use in trusted environments.
        exec_context = {
            "st": st,
            "pd": pd,
            "df": df,
            # Import common visualization libraries to support generated code.
            "px": __import__("plotly.express"),
            "plt": __import__("matplotlib.pyplot"),
            "sns": __import__("seaborn")
        }
        
        st.subheader("Dashboard Output")
        try:
            # Execute the generated code.
            exec(code, exec_context)
        except Exception as exec_error:
            st.error(f"Error executing generated code: {exec_error}")

if __name__ == "__main__":
    main()
