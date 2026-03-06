import json
import os
import re

import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

FAISS_INDEX_DIR = "faiss_index"
METADATA_FILENAME = "metadata.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 4

REFUSAL_MESSAGE = (
    "I can only provide factual information about mutual fund schemes and cannot provide investment advice."
)
AMFI_EDUCATION_LINK = "https://www.amfiindia.com/investor-corner/investor-education"
DISCLAIMER = (
    "This assistant provides factual information from official public sources only. No investment advice."
)


def load_index_and_metadata():
    index_path = os.path.join(FAISS_INDEX_DIR, "index.faiss")
    metadata_path = os.path.join(FAISS_INDEX_DIR, METADATA_FILENAME)

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        return None, None, None

    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)

    model = SentenceTransformer(EMBEDDING_MODEL)
    return index, metadata_list, model


@st.cache_resource
def get_search_resources():
    return load_index_and_metadata()


def contains_pii(text: str) -> bool:
    lower = text.lower()

    pii_keywords = [
        "pan", "aadhaar", "aadhar", "otp",
        "account number", "bank account",
        "credit card", "debit card",
        "phone number", "mobile number",
    ]

    if any(k in lower for k in pii_keywords):
        return True

    # PAN pattern
    if re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", text.upper()):
        return True

    # email pattern
    if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
        return True

    # phone numbers (10 digits)
    if re.search(r"\b\d{10}\b", text):
        return True

    return False


def is_advice_question(text: str) -> bool:
    lower = text.lower()
    phrases = [
        "should i invest", "should i buy", "should i sell",
        "which fund is better", "which is better", "best fund",
        "top fund", "recommend a fund", "recommendation",
        "suggest a fund", "what should i invest in",
        "where should i invest", "buy or sell",
        "expected return", "future return", "forecast",
        "projected return", "good investment", "bad investment",
        "compare this fund", "comparison", " vs ", " versus ",
    ]
    return any(p in lower for p in phrases)


def first_sentences(text: str, max_sentences: int = 3) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    parts = [p.strip() for p in parts if p.strip()]

    if not parts:
        trimmed = text.strip()[:400]
        return trimmed + ("..." if len(text.strip()) > 400 else "")

    out = " ".join(parts[:max_sentences]).strip()
    if out and out[-1] not in ".!?":
        out += "."
    return out


def answer_from_retrieval(question: str, index, metadata_list, model):
    if index is None or metadata_list is None or model is None:
        return (
            "I do not have sufficient factual information from the available public sources to answer this question.",
            None,
            None,
        )

    q_embedding = model.encode([question]).astype("float32")
    scores, indices = index.search(np.array(q_embedding), min(TOP_K, len(metadata_list)))

    if indices.size == 0 or indices[0][0] < 0:
        return (
            "I do not have sufficient factual information from the available public sources to answer this question.",
            None,
            None,
        )

    best_idx = int(indices[0][0])
    chunk = metadata_list[best_idx]

    text = chunk.get("text", "")
    title = chunk.get("title", "Source")
    source = chunk.get("source", "")

    answer = first_sentences(text, max_sentences=3)
    if not answer:
        answer = "I do not have sufficient factual information from the available public sources to answer this question."

    answer = f"{answer} Last updated from sources."
    return answer, source or None, title or None


def ask_question(question, index, metadata_list, model):
    if contains_pii(question):
        return {
            "content": "Please do not share personal financial information.",
            "source_url": None,
            "source_title": None,
        }

    if is_advice_question(question):
        return {
            "content": REFUSAL_MESSAGE,
            "source_url": AMFI_EDUCATION_LINK,
            "source_title": "AMFI Investor Education",
        }

    answer, source_url, source_title = answer_from_retrieval(
        question, index, metadata_list, model
    )
    return {
        "content": answer,
        "source_url": source_url,
        "source_title": source_title,
    }


def main():
    st.set_page_config(page_title="Mutual Fund Facts Assistant", page_icon="📘", layout="centered")

    st.title("Mutual Fund Facts Assistant")
    st.caption("Facts only. No investment advice.")
    st.info("This assistant answers factual mutual fund questions using official public sources only.")

    with st.sidebar:
        st.subheader("About")
        st.write("Answers factual questions about mutual funds, such as expense ratio, exit load, lock-in, benchmark, riskometer, and statement download.")
        st.subheader("Not supported")
        st.write("No investment advice, recommendations, comparisons, or return predictions.")
        st.caption(DISCLAIMER)

    index, metadata_list, model = get_search_resources()
    if index is None or metadata_list is None or model is None:
        st.error("FAISS index not found. Run `python ingest.py` first.")
        st.stop()

    st.write("**Example questions**")
    c1, c2, c3 = st.columns(3)

    if c1.button("Expense ratio of SBI Bluechip Fund?"):
        st.session_state["pending_question"] = "What is the expense ratio of SBI Bluechip Fund?"
    if c2.button("Lock-in period of ELSS funds?"):
        st.session_state["pending_question"] = "What is the lock-in period of ELSS funds?"
    if c3.button("Download capital gains statement?"):
        st.session_state["pending_question"] = "How can I download my capital gains statement?"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pending_question" in st.session_state and st.session_state["pending_question"]:
        question = st.session_state["pending_question"]
        st.session_state["pending_question"] = None

        st.session_state.messages.append({"role": "user", "content": question})
        result = ask_question(question, index, metadata_list, model)
        st.session_state.messages.append({"role": "assistant", **result})

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("source_url"):
                st.markdown(f"**Source:** [{msg.get('source_title', 'Link')}]({msg['source_url']})")

    user_input = st.chat_input("Ask a factual question about mutual funds...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        result = ask_question(user_input, index, metadata_list, model)
        st.session_state.messages.append({"role": "assistant", **result})

        st.rerun()


if __name__ == "__main__":
    main()