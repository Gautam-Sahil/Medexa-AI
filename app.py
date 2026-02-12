import os
import base64
import json
import re
from io import BytesIO
from flask import Flask, render_template, request, jsonify, send_file
from fpdf import FPDF
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
    INTERACTION_CHECK_PROMPT,
    REPORT_GENERATOR_PROMPT,
    build_risk_prompt
)

app = Flask(__name__, static_folder="static", template_folder="templates")
load_dotenv()

# --- API KEYS ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- CORE AI SETUP (RAG) ---
embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot",
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_kwargs={"k": 3})

# --- TRIPLE MODEL FALLBACK SYSTEM ---
model_kwargs = {
    "openai_api_base": "https://openrouter.ai/api/v1",
    "openai_api_key": OPENROUTER_API_KEY,
    "temperature": 0.3
}

primary_model = ChatOpenAI(
    model_name="openrouter/free",
    **model_kwargs
)

secondary_model = ChatOpenAI(
    model_name="mistralai/mixtral-8x7b-instruct:free",
    **model_kwargs
)

tertiary_model = ChatOpenAI(
    model_name="openchat/openchat-3.5-0106:free",
    **model_kwargs
)

chat_model = primary_model.with_fallbacks([secondary_model, tertiary_model])


# --- RAG CHAIN SETUP ---
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

# --- NAVIGATION ROUTES ---

@app.route("/")
def home(): return render_template("index.html")

@app.route("/chat")
def chat_page(): return render_template("chat.html")

@app.route("/lab")
def lab_page(): return render_template("lab.html")

@app.route("/risk")
def risk_page(): return render_template("risk.html")

@app.route("/emergency")
def emergency_page(): return render_template("emergency.html")

@app.route("/med_interact")
def med_interact_page(): return render_template("med_interact.html")

@app.route("/scribe")
def scribe_page(): return render_template("scribe.html")

# --- FEATURE 1: MEDICAL ASSISTANT (RAG) ---
@app.route("/get", methods=["POST"])
def chat():
    global chat_history
    user_input = request.form.get("msg", "").strip()
    image_file = request.files.get("image")

    emergency_keywords = ["chest pain", "choking", "stroke", "bleeding", "unconscious", "can't breathe"]
    if user_input and any(k in user_input.lower() for k in emergency_keywords):
        return "TRIGGER_EMERGENCY"

    try:
        if image_file and image_file.filename:
            base64_image = encode_image(image_file)
            content = [
                {"type": "text", "text": user_input or "Analyze this image"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
            response = chat_model.invoke([HumanMessage(content=content)])
            answer = response.content
        else:
            response = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
            answer = response["answer"]

        chat_history.extend([HumanMessage(content=user_input or "Image sent"), AIMessage(content=answer)])
        chat_history = chat_history[-6:]
        return str(answer)
    except Exception as e:
        print(f"Chat Error: {str(e)}")
        return "The AI assistant is temporarily busy. Please try again.", 500

# --- FEATURE 2: LAB LENS (REPORT ANALYSIS) ---
@app.route("/analyze_report", methods=["POST"])
def analyze_report():
    image_file = request.files.get("image")
    if not image_file: return "No report provided.", 400

    try:
        base64_image = encode_image(image_file)
        content = [
            {"type": "text", "text": LAB_SYSTEM_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
        response = chat_model.invoke([HumanMessage(content=content)])
        return str(response.content)
    except Exception as e:
        print(f"Lab Lens Error: {str(e)}")
        return "Service temporarily busy. Please wait 30 seconds and retry.", 429

# --- FEATURE 3: RISK PREDICTOR ---
@app.route("/predict_risk", methods=["POST"])
def predict_risk():
    data = request.json
    age, bp, chol, smoker = int(data["age"]), int(data["bp"]), int(data["chol"]), data["smoker"]

    risk = 0
    if age > 50: risk += 5
    if bp > 140: risk += 7
    if chol > 240: risk += 5
    if smoker == "yes": risk += 8

    risk_score = min(risk + 2, 35)
    risk_prompt = build_risk_prompt(age, bp, chol, smoker, risk_score)
    response = chat_model.invoke([HumanMessage(content=risk_prompt)])

    return jsonify({"score": risk_score, "insight": response.content})

# --- FEATURE 4: MED-INTERACT (SAFETY CHECKER) ---
@app.route("/check_interactions", methods=["POST"])
def check_interactions():
    user_input = request.form.get("msg", "").strip()
    image_file = request.files.get("image")

    try:
        if image_file and image_file.filename:
            base64_image = encode_image(image_file)
            content = [
                {"type": "text", "text": f"{INTERACTION_CHECK_PROMPT}\nInput: {user_input or 'Analyze meds'}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
            response = chat_model.invoke([HumanMessage(content=content)])
        else:
            full_prompt = INTERACTION_CHECK_PROMPT.format(medication_data=user_input)
            response = chat_model.invoke([HumanMessage(content=full_prompt)])
        return jsonify({"result": response.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500 

# --- FEATURE 5: MED-SCRIBE (PDF GENERATOR) ---
@app.route("/generate_pdf_report", methods=["POST"])
def generate_pdf_report():
    notes = request.form.get("notes")
    if not notes: return "No notes provided.", 400

    try:
        ai_response = chat_model.invoke([HumanMessage(content=REPORT_GENERATOR_PROMPT.format(clinical_notes=notes))])
        json_match = re.search(r'\{.*\}', ai_response.content, re.DOTALL)
        
        if json_match:
            report_data = json.loads(json_match.group())
        else:
            return "AI failed to format notes. Try again.", 500

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", 'B', 22)
        pdf.set_text_color(63, 102, 241) 
        pdf.cell(0, 15, "MEDEXA SMART CLINIC", align='C', new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("Helvetica", size=10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 5, "Digital Health Record | Automated AI Scribe", align='C', new_x="LMARGIN", new_y="NEXT")
        pdf.ln(15)

        sections = [
            ("Diagnosis", report_data.get('diagnosis', 'N/A')),
            ("Clinical Summary", report_data.get('summary', 'N/A')),
            ("Prescription & Dosage", report_data.get('medications', [])),
            ("Doctor's Advice", report_data.get('advice', 'N/A')),
            ("Follow-up Plan", report_data.get('follow_up', 'N/A'))
        ]

        for title, content in sections:
            pdf.set_font("Helvetica", 'B', 12)
            pdf.set_text_color(30, 41, 59)
            pdf.cell(0, 10, f"{title}:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=11)
            pdf.set_text_color(51, 65, 85)
            
            if isinstance(content, list):
                for item in content:
                    pdf.multi_cell(0, 7, f"- {item}")
            else:
                pdf.multi_cell(0, 7, str(content))
            pdf.ln(5)

        pdf_bytes = pdf.output()
        return send_file(BytesIO(pdf_bytes), download_name="Medical_Report.pdf", as_attachment=True, mimetype='application/pdf')
    except Exception as e:
        print(f"Scribe Error: {e}")
        return f"Failed to generate report: {str(e)}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
