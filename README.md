# mf-faq-assistant


# Mutual Fund Facts Assistant

A small RAG-based FAQ assistant that answers factual mutual fund questions using official public sources.

## Scope

Platform chosen: Groww  
AMC used for scheme data: SBI Mutual Fund

Schemes referenced:
- SBI Bluechip Fund (Large Cap)
- SBI Flexicap Fund
- SBI Small Cap Fund
- SBI Long Term Equity Fund (ELSS)
- SBI Magnum Midcap Fund

## Features

- Answers factual questions about mutual funds
- Uses official public sources (AMC / AMFI / SEBI)
- Provides one citation link with every answer
- Refuses investment advice questions
- Detects and blocks personal financial information (PAN, Aadhaar, OTP etc.)
- Uses a small Retrieval-Augmented Generation (RAG) system

## Tech Stack

- Python
- Streamlit
- FAISS
- Sentence Transformers (all-MiniLM-L6-v2)
- BeautifulSoup
- PyPDF

## Setup

Install dependencies:
