import os
import base64
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

from src.helper import download_hugging_face_embeddings
from src.prompt import (
    contextualize_q_prompt,
    qa_prompt,
    LAB_SYSTEM_PROMPT,
    build_risk_prompt
)

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

embeddings = download_hugging_face_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot",
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_kwargs={"k": 3})

primary_model = ChatOpenAI(
    model_name="google/gemma-3-27b-it:free",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=OPENROUTER_API_KEY,
    temperature=0.3
)

backup_model = ChatOpenAI(
    model_name="qwen/qwen-2.5-vl-72b-instruct:free",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=OPENROUTER_API_KEY,
    temperature=0.3
)

chat_model = primary_model.with_fallbacks([backup_model])

history_aware_retriever = create_history_aware_retriever(
    chat_model, retriever, contextualize_q_prompt
)

question_answer_chain = create_stuff_documents_chain(
    chat_model, qa_prompt
)

rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain
)

chat_history = []

def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode("utf-8")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat")
def chat_page():
    return render_template("chat.html")

@app.route("/lab")
def lab_page():
    return render_template("lab.html")

@app.route("/risk")
def risk_page():
    return render_template("risk.html")

@app.route("/emergency")
def emergency_page():
    return render_template("emergency.html")

@app.route("/get", methods=["POST"])
def chat():
    global chat_history

    user_input = request.form.get("msg", "").strip()
    image_file = request.files.get("image")

    # Emergency keyword gate (fast, no AI)
    emergency_keywords = [
        "chest pain",
        "choking",
        "stroke",
        "bleeding",
        "unconscious",
        "difficulty breathing",
        "can't breathe",
        "shortness of breath"
    ]

    if user_input and any(k in user_input.lower() for k in emergency_keywords):
        return "TRIGGER_EMERGENCY"

    try:
        # Vision path
        if image_file and image_file.filename:
            base64_image = encode_image(image_file)
            content = [
                {"type": "text", "text": user_input or "Analyze this image"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
            response = chat_model.invoke([HumanMessage(content=content)])
            answer = response.content

        # Text RAG path
        else:
            response = rag_chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            answer = response["answer"]

        # AI emergency fallback
        if answer.lower().startswith("emergency"):
            return "TRIGGER_EMERGENCY"

        chat_history.extend([
            HumanMessage(content=user_input or "Image uploaded"),
            AIMessage(content=answer)
        ])
        chat_history = chat_history[-6:]

        return str(answer)

    except Exception:
        return "Error processing request.", 500


@app.route("/analyze_report", methods=["POST"])
def analyze_report():
    image_file = request.files.get("image")
    if not image_file:
        return "No report provided.", 400

    base64_image = encode_image(image_file)
    content = [
        {"type": "text", "text": LAB_SYSTEM_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ]

    response = chat_model.invoke([HumanMessage(content=content)])
    return str(response.content)

@app.route("/predict_risk", methods=["POST"])
def predict_risk():
    data = request.json

    age = int(data["age"])
    bp = int(data["bp"])
    chol = int(data["chol"])
    smoker = data["smoker"]

    risk = 0
    if age > 50: risk += 5
    if bp > 140: risk += 7
    if chol > 240: risk += 5
    if smoker == "yes": risk += 8

    risk_score = min(risk + 2, 35)

    risk_prompt = build_risk_prompt(age, bp, chol, smoker, risk_score)

    response = chat_model.invoke([
        HumanMessage(content=risk_prompt)
    ])

    return jsonify({
        "score": risk_score,
        "insight": response.content
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
