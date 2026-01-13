from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Contextualizer
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question. Do NOT answer the question."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Main medical QA
system_prompt = (
    "You are a professional Medical Assistant. "
    "Use the provided context to answer accurately. "
    "If you don't know, say you don't know. "
    "Keep answers concise and professional.\n\n"
    "**EMERGENCY:** If symptoms sound life-threatening, "
    "lead with: 'PLEASE CALL 911 IMMEDIATELY.'\n\n"
    "Context:\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Lab report analysis
LAB_SYSTEM_PROMPT = (
    "You are a Senior Laboratory Pathologist AI.\n"
    "1. Extract every test name, value, and reference range.\n"
    "2. Mark abnormal values with ⚠️ ABNORMAL.\n"
    "3. Explain each test in one simple sentence.\n"
    "4. Give a 3-sentence health summary.\n"
    "5. End with: 'Note: This is an AI analysis. Please confirm results with your doctor.'"
)

# Risk explanation prompt
def build_risk_prompt(age, bp, chol, smoker, risk_score):
    return (
        f"The patient has a 10-year cardiovascular risk score of {risk_score}%. "
        f"Age: {age}, BP: {bp}, Cholesterol: {chol}, Smoker: {smoker}. "
        "Provide a professional clinical interpretation. "
        "Format as follows:\n"
        "## Risk Interpretation\n(Briefly explain what the score means)\n\n"
        "**Lifestyle Recommendation:** (Provide one specific tip)\n\n"
        "Do NOT include long disclaimers."
    )
