# 🏥 Medexa AI: Smart Clinical Assistant

<img width="2912" height="1830" alt="medexa-ai onrender com_" src="https://github.com/user-attachments/assets/92d062d4-2e1b-4869-ab14-198162a8c421" />

**Medexa AI** is a multi-modal, full-stack healthcare suite designed to assist both patients and medical professionals. By integrating advanced Retrieval-Augmented Generation (RAG) with vision-capable LLMs, Medexa provides instant lab report analysis, medication safety checks, and automated clinical scribing.

---

## 🌟 Key Features

* **🤖 Smart AI Assistant (RAG):** Context-aware medical chat powered by Pinecone vector storage and Google Gemini.
* **👁️ Lab Lens:** Upload blood reports or scans for instant, simplified AI analysis.
* **📄 AI Med-Scribe:** Converts rough clinical notes into professional, structured PDF medical reports.
* **💊 Safety Checker:** Scans medication labels for potential drug-drug interactions.
* **📉 Risk Predictor:** Analytical engine to estimate health risks based on vitals and lifestyle.
* **🚨 Emergency Gate:** Non-AI keyword monitoring to instantly trigger emergency protocols for critical symptoms.

---

## 🏗️ Technical Architecture



The system utilizes a **Triple Fallback Model System** to ensure high availability and reliability:

1. **Primary Model:** Google Gemini 2.0 Flash (Fastest, optimized for RAG)
2. **Secondary Model:** Meta Llama 3.2 Vision (Fallback for image analysis)
3. **Tertiary Model:** Mistral Pixtral 12B (General purpose fallback)

---

## 🛠️ Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend** | Python, Flask, Gunicorn |
| **AI Orchestration** | LangChain, LangSmith |
| **Vector Database** | Pinecone |
| **Embeddings** | HuggingFace (`sentence-transformers`) |
| **LLMs** | OpenRouter (Gemini, Llama 3.2, Mistral) |
| **PDF Generation** | FPDF2 |

---

## ⚖️ Disclaimer

> **IMPORTANT:** This application is an AI-powered tool for educational and assistance purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

**Developed with ❤️ by Gautam Tiwari**
