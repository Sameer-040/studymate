
import streamlit as st
import fitz  # PyMuPDF
import base64
#import pyttsx3
import requests
import numpy as np
import openai
from sentence_transformers import SentenceTransformer, util

# ---------------- CONFIG ----------------
OPENROUTER_API_KEY = "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"
openai.api_key = OPENROUTER_API_KEY
openai.api_base = "https://openrouter.ai/api/v1"

# ---------------- SETUP ----------------
#engine = pyttsx3.init()
#model_embed = SentenceTransformer("all-MiniLM-L6-v2")

#def speak(text):
#    engine.say(text)
#    engine.runAndWait()

# ---------------- PDF HANDLING ----------------
def extract_text_from_pdfs(files):
    all_text = ""
    for file in files:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            all_text += page.get_text()
    return all_text

# ---------------- CHUNKING ----------------
def split_into_chunks(text, chunk_size=300):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return []

    chunks = []
    chunk = ""
    for p in paragraphs:
        if len(chunk) + len(p) < chunk_size:
            chunk += " " + p
        else:
            chunks.append(chunk.strip())
            chunk = p
    if chunk:
        chunks.append(chunk.strip())
    return chunks if chunks else [text[:chunk_size]]

# ---------------- INFERENCE CALLS ----------------
def gpt_answer(question, context):
    prompt = f"Answer the following question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful tutor chatbot."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {e}"

def gpt_summarize(text):
    prompt = f"Summarize the following content:\n{text[:3000]}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {e}"

def gpt_bullets(text):
    prompt = f"Extract 5-10 bullet points from the following notes:\n{text[:3000]}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {e}"

# ---------------- SMART RETRIEVAL ----------------
def find_best_chunk(chunks, question):
    if not chunks:
        return ""
    chunk_embeddings = model_embed.encode(chunks, convert_to_tensor=True)
    question_embedding = model_embed.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, chunk_embeddings)[0]
    top_idx = int(np.argmax(scores))
    return chunks[top_idx]

# ---------------- UI ----------------
st.set_page_config(page_title="StudyMate AI", layout="wide")
st.title("📘 StudyMate GPT-Powered Chatbot")
st.markdown("Ask questions, get summaries, extract bullet points from PDFs using GPT-3.5.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_answer" not in st.session_state:
    st.session_state.current_answer = ""

uploaded_files = st.file_uploader("📄 Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    full_text = extract_text_from_pdfs(uploaded_files)
    chunks = split_into_chunks(full_text)

    col1, col2 = st.columns([2, 1])

    with col1:
        question = st.text_input("💬 Ask a question from your notes:")
        if question:
            with st.spinner("Finding context and calling GPT..."):
                best_chunk = find_best_chunk(chunks, question)
                answer = gpt_answer(question, best_chunk)
                st.session_state.chat_history.append((question, answer))
                st.session_state.current_answer = answer

            st.success("✅ Answer ready!")
            st.markdown(f"**Answer:** {answer}")
            #if st.button("🔊 Read Aloud"):
             #   speak(answer)

    with col2:
        st.markdown("### 📚 Tools")
        if st.button("📝 Summarize My Notes"):
            with st.spinner("Summarizing PDF content..."):
                summary = gpt_summarize(full_text)
                st.markdown("#### ✨ Summary")
                st.write(summary)

        if st.button("📌 Extract Key Points"):
            with st.spinner("Extracting bullet points..."):
                bullets = gpt_bullets(full_text)
                st.markdown("#### 🔑 Key Points")
                st.markdown(bullets)

        if st.button("📤 Export Chat to .txt"):
            chat_log = ""
            for q, a in st.session_state.chat_history:
                chat_log += f"You: {q}\nBot: {a}\n\n"
            b64 = base64.b64encode(chat_log.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="chat_history.txt">Download Chat History</a>'
            st.markdown(href, unsafe_allow_html=True)

        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history.clear()
            st.success("Chat history deleted.")

    with st.expander("📜 Chat History"):
        for q, a in st.session_state.chat_history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")
            st.markdown("---")
else:
    st.info("👆 Please upload PDF files to begin.")
