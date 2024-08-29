import cohere
from pinecone import Pinecone, ServerlessSpec
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

EMBEDDING_MODEL = "embed-english-v3.0"
INDEX_NAME = "cop16-v1"

co = cohere.Client(os.environ.get("COHERE_API_KEY"))
if not co:
    co = st.secrets["COHERE_API_KEY"]

pc = Pinecone(api_key = os.environ.get("PINECONE_API_KEY"))
if not pc:
    co = st.secrets["PINECONE_API_KEY"]

index = pc.Index(INDEX_NAME)

def retrieve_text(query):
    xq = co.embed(
        texts=[query],
        model=EMBEDDING_MODEL,
        input_type='search_query',
        truncate='END'
    ).embeddings

    return index.query(vector=xq, top_k=3, include_metadata=True)

def res_to_colon_separated_string(res):
  result_string = ""
  for idx, match in enumerate(res['matches']):
    metadata = match['metadata']
    result_string  += "Context " + str(idx+1) + ":\n"
    for key, value in metadata.items():
      result_string += f"{key}: {value}\n"
    result_string += "\n"
  return result_string[:-2] 

def get_response(context, query):
    prompt = (f"given the context extracted from a event information document, answer user's question."
              " Always respond with correct date and contact person." 
              " Always respond in English unless the user asks for other language." 
              f"\n\nCONTEXT: {context}. \n\nUSER'S QUESTION: {query}")
    response = co.chat(
    model="command-r-plus",
    message=prompt
    )
    return response.text

def get_answer(query):
    res = retrieve_text(query)
    context = res_to_colon_separated_string(res)
    answer = get_response(context, query)
    return answer


st.title("COP Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = get_answer(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})