import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM    
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] =  os.getenv("LANGCHAIN_PROJECT")
prompt=ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant . Please respond to question asked"),
    ("user", "Question:{question}")  
])
# frame work
st.title("Ollama with Langchain")
input_text=st.text_input("Enter your question")
llm=OllamaLLM(model="gemma:2b")
output_pareser=StrOutputParser()
chain=prompt | llm | output_pareser
if input_text:
    response=chain.invoke({"question":input_text})
    st.write(response)

