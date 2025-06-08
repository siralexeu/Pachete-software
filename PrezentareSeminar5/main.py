import streamlit as st

st.set_page_config(page_title="Prezentare Cod - Chat cu PDF", layout="wide")
st.title("📄 Prezentare cod: Chat cu PDF + OpenAI")

st.markdown("""
Aceasta aplicatie iti permite sa interactionezi cu un document PDF folosind un model LLM de la OpenAI.  
Codul este impartit in sectiuni cu explicatii pentru fiecare pas important.
""")

# ========================
# 1. Importuri si setup
# ========================
with st.expander("📦 1. Importuri si setup general"):
    st.code("""
# Librarii standard si externe
import sys
import streamlit as st
import openai
import os
import sqlite3
from dotenv import load_dotenv

# Functii proprii (din fisierele utils)
from utils.chromadb_utils import (
    create_or_get_collection,
    add_documents_to_collection,
    query_collection,
    sanitize_collection_name
)
from utils.pdf_processing import extract_pdf_text, split_text_into_chunks

# Inlocuim sqlite3 cu pysqlite3 pentru compatibilitate mai buna
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
""", language="python")

# ================================
# 2. Chei si configurare OpenAI
# ================================
with st.expander("🔑 2. Incarcare chei si configurare OpenAI"):
    st.code("""
# Incarcam variabilele din fisierul .env,
# Se afla in același director cu fișierul principal Python
load_dotenv()

# Cream clientul OpenAI
client = openai.OpenAI()

# Cheile si setarile din streamlit secrets
# Variabilele vin din secrets.toml (folosit de Streamlit în cloud sau productie)
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
MODEL_NAME = st.secrets["MODEL_NAME"]
PRODUCTION = st.secrets["PRODUCTION"]

# Tipul de chat curent
CHAT_TYPE = "pdf_chat"
""", language="python")

# ============================
# 3. Functia embed_text()
# ============================
with st.expander("🧠 3. Functie: embed_text(text)"):
    st.code("""
# Creeaza un embedding pentru textul dat folosind modelul de la OpenAI
def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding  # Intoarce vectorul de embedding
""", language="python")

# ============================
# 4. Functia main()
# ============================
with st.expander("🧩 4. Functia main() – logica principala"):
    st.code("""
def main():
    # Initializam zona de mesaje daca nu exista
    if f"{CHAT_TYPE}_messages" not in st.session_state:
        st.session_state[f"{CHAT_TYPE}_messages"] = []

    # Legam mesajele la sesiunea curenta
    st.session_state.messages = st.session_state[f"{CHAT_TYPE}_messages"]

    # Verificam daca suntem in productie
    is_production = PRODUCTION == "True"

    # Daca suntem in productie si nu avem cheia API, oprim aplicatia
    if is_production:
        if "api_key" not in st.session_state or not st.session_state.api_key:
            st.error("You do not have a valid API key. Please go back to the main page to enter one.")
            return

    # Setam configurarea paginii
    st.set_page_config(
        page_title="Chat with PDF 📄",
        page_icon="📄",
        layout="wide",
    )

    st.title("Chat with PDF 📄")

    # Alte variabile salvate in sesiune
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = None
    if "previous_file" not in st.session_state:
        st.session_state.previous_file = None
""", language="python")

# ===================================
# 5. Sidebar – Clear chat history
# ===================================
with st.expander("🧹 5. Sidebar – Clear chat"):
    st.code("""
# In sidebar adaugam optiuni de resetare a conversatiei
with st.sidebar:
    st.header("Chat Settings")

    # Buton care sterge toate mesajele curente
    if st.button("Clear chat history"):
        st.session_state[f"{CHAT_TYPE}_messages"] = []
        st.success("Chat history has been cleared!")
        st.rerun()  # Reincarcam pagina
""", language="python")

# =======================================
# 6. Upload PDF si procesare continut
# =======================================
with st.expander("📤 6. Upload si procesare PDF"):
    st.code("""
# Incarcam un fisier PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Daca PDF-ul s-a schimbat, resetam conversatia si colectia
    if uploaded_file.name != st.session_state.previous_file:
        st.session_state.messages = []
        st.session_state.collection_name = None
        st.session_state.previous_file = uploaded_file.name

        with st.spinner("Processing the PDF..."):
            # Extragem textul din PDF
            pdf_text = extract_pdf_text(uploaded_file)

            # Afisam textul daca utilizatorul vrea
            if st.checkbox("Show extracted text from PDF"):
                st.text_area("Extracted Text", pdf_text, height=300)

            # Pregatim colectia pentru ChromaDB
            collection_name = sanitize_collection_name(f"pdf_{uploaded_file.name}")
            st.session_state.collection_name = collection_name
            collection = create_or_get_collection(collection_name)

            # Daca PDF-ul nu a fost deja indexat, il procesam acum
            if not collection.get()["documents"]:
                chunks = split_text_into_chunks(pdf_text)
                embeddings = [embed_text(chunk) for chunk in chunks]
                add_documents_to_collection(collection, chunks, embeddings)
                st.success("The PDF text has been indexed!")
""", language="python")

# =======================================
# 7. Chat cu PDF – afisare si raspunsuri
# =======================================
with st.expander("💬 7. Chat si raspunsuri din PDF"):
    st.code("""
# Afisam toate mesajele din conversatia curenta
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input pentru utilizator (intrebare)
if prompt := st.chat_input("Write a question about the PDF here..."):
    # Salvam mesajul
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state[f"{CHAT_TYPE}_messages"] = st.session_state.messages

    with st.chat_message("user"):
        st.markdown(prompt)

    # Daca avem o colectie deja, cautam informatia
    if st.session_state.collection_name:
        collection = create_or_get_collection(st.session_state.collection_name)
        query_embedding = embed_text(prompt)
        relevant_chunks = query_collection(collection, query_embedding)

        if relevant_chunks:
            # Construim un prompt cu context
            context = "\\n\\n".join(relevant_chunks)
            prompt_with_context = f"Question: {prompt}\\n\\nRelevant content from PDF:\\n{context}\\n\\nAnswer:"

            # Trimitem catre OpenAI si afisam raspunsul live
            with st.chat_message("assistant"):
                with st.spinner("Generating answer..."):
                    response = client.chat.completions.create(
                        model="gpt-4o-mini-2024-07-18",
                        messages=[
                            {"role": "system", "content": "You are an assistant that answers based on the PDF content."},
                            {"role": "user", "content": prompt_with_context},
                        ],
                        stream=True,
                    )

                    message_placeholder = st.empty()
                    full_response = ""

                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")

                    # Afisam raspunsul complet
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.session_state[f"{CHAT_TYPE}_messages"] = st.session_state.messages
        else:
            st.warning("No relevant content found in the PDF for this question.")
""", language="python")

# ✅ Final
#st.success("✅ Prezentare finalizata. Toate sectiunile au fost comentate clar.")
