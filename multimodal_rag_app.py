# Mohamed YOUSSFI, ENSET, UH2C
import streamlit as st
import os
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
import base64
from chromadb.utils.data_loaders import ImageLoader
from IPython.display import Markdown
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from PIL import Image
import glob

chroma_client = chromadb.PersistentClient(path="vehicles-store-vdb")
image_loader = ImageLoader()
embedding_model = OpenCLIPEmbeddingFunction()
chroma_vdb = chroma_client.get_or_create_collection(
    name="vehicules",
    data_loader=image_loader,
    embedding_function=embedding_model,
)

OPEN_API_KEY = "...."
gpt4o_model = ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=OPEN_API_KEY)
parser = StrOutputParser()
image_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_message}"),
        (
            "user",
            [
                {"type": "text", "text": "{user_query}"},
                {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64,{image_data_1}",
                },
                {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64,{image_data_2}",
                },
            ],
        ),
    ]
)

vision_chain = image_prompt | gpt4o_model | parser

system_message = """
Answer the user's question using the given image context with direct references to the parts of the images provided. 
Use Markdown to give the answer
"""
prompt_inputs = {}
prompt_inputs["system_message"] = system_message


def loadDataIntoVectorStore(folder_name):
    images_ids = []
    images_uris = []
    dataset_folder = folder_name
    for index, file_name in enumerate(sorted(os.listdir(dataset_folder))):
        if file_name.endswith(".jpg"):
            file_path = os.path.join(dataset_folder, file_name)
            images_ids.append(str(index))
            images_uris.append(file_path)
    chroma_vdb.add(ids=images_ids, uris=images_uris)


def search_images(user_question):
    results = chroma_vdb.query(
        query_texts=[user_question], n_results=2, include=["uris", "distances"]
    )
    return results


def askLLM(user_question, images_result):
    prompt_inputs["user_query"] = user_question
    image_path1 = images_result["uris"][0][0]
    image_path2 = images_result["uris"][0][1]
    st.image(image_path1, width=500)
    st.image(image_path2, width=500)
    with open(image_path1, "rb") as image_file:
        image_data1 = image_file.read()
    with open(image_path2, "rb") as image_file:
        image_data2 = image_file.read()
    prompt_inputs["image_data_1"] = base64.b64encode(image_data1).decode("utf-8")
    prompt_inputs["image_data_2"] = base64.b64encode(image_data2).decode("utf-8")
    response = vision_chain.invoke(prompt_inputs)
    return response


def main():
    st.set_page_config(layout="wide")
    st.subheader("Multi modal RAG Chatbot", divider="rainbow")

    # Title of the web app
    st.subheader("Chatbot zone")
    # Sidebar of the web app
    user_question = st.text_input("Ask your question :")
    if user_question:
        results = search_images(user_question)
        response = askLLM(user_question, results)
        st.markdown(response, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
