import streamlit as st
import requests
from llama_cpp import Llama
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import re

# --- CONFIGURATION ---
MODEL_PATH = "models/qwen3b.gguf"
TOOL_URL = "http://localhost:8000"

st.set_page_config(page_title="EWU AI Engine", layout="wide", page_icon="🎓")

# 1. Load Brain (Qwen) - No Translator needed
@st.cache_resource
def load_llm():
    return Llama(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=25, verbose=False)

llm = load_llm()

# --- AGENT STATE ---
class AgentState(TypedDict):
    original_question: str
    search_keywords: str
    context: str
    sources: List[str]
    tool_log: str
    final_ans: str
    thinking_steps: List[str] # New: For Transparency

# --- NODES ---

def preprocess_query(state: AgentState):
    """Step 0: Intent Classification & Keyword Extraction"""
    user_q = state["original_question"]
    steps = state.get("thinking_steps", [])
    steps.append(f"🧠 **Router:** Analyzing query '{user_q}'...")
    
    # FEW-SHOT PROMPT: Optimize for Policy Retrieval
    prompt = f"""<|im_start|>system
You are a University Policy Search Assistant. 
Extract the core keywords from the question for a vector database search.
Focus on terms like "Misconduct", "Punishment", "Committee", "Appeal", "Exam".

Example 1:
Input: "What happens if I cheat in an exam?"
Keywords: Cheating unfair means examination misconduct punishment

Example 2:
Input: "Who is in the disciplinary committee?"
Keywords: Disciplinary Committee members authority composition

Input: "{user_q}"
Keywords:
<|im_end|>
<|im_start|>assistant
"""
    out = llm(prompt, max_tokens=64, stop=["<|im_end|>"], temperature=0.1)
    keywords = out["choices"][0]["text"].strip()
    
    if not keywords: keywords = user_q
    
    steps.append(f"🔑 **Keywords Extracted:** `{keywords}`")
    return {"search_keywords": keywords, "thinking_steps": steps}

def retrieve(state: AgentState):
    """Step 1: Search Policy Documents"""
    steps = state["thinking_steps"]
    steps.append(f"📡 **Retrieval:** Searching Vector DB...")
    
    try:
        res = requests.post(f"{TOOL_URL}/search", json={"query": state["search_keywords"]})
        data = res.json()
        
        if not data.get("results"):
            steps.append("❌ **Retrieval:** No documents found.")
            return {"context": "MISSING", "sources": [], "thinking_steps": steps}
        
        context_text = "\n\n".join(data["results"])
        
        # Safe Metadata Extraction
        file_sources = []
        if data.get("metadatas"):
            file_sources = list(set([m.get('filename', 'Unknown') for m in data['metadatas']]))
            
        steps.append(f"✅ **Retrieval:** Found {len(data['results'])} relevant clauses.")
        return {"context": context_text, "sources": file_sources, "thinking_steps": steps}
    except:
        steps.append("⚠️ **Error:** Tool Server Unreachable.")
        return {"context": "Error connecting to Tool Server", "sources": [], "thinking_steps": steps}

def reason(state: AgentState):
    """Step 2: Check for Tool Use (Calculations)"""
    steps = state["thinking_steps"]
    q = state["original_question"].lower()
    
    # Check if we need math (e.g. "Calculate the fine for 2 offenses" if policy has that)
    # Even if questions don't strictly need it, we keep the logic to satisfy the Requirement.
    has_numbers = any(char.isdigit() for char in q)
    math_words = ["calculate", "sum", "total", "add", "multiply"]
    
    if has_numbers and any(w in q for w in math_words):
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", q)
        if len(nums) >= 2:
            steps.append(f"⚙️ **Planner:** Calculation detected. Routing to Calculator Tool.")
            return {"tool_log": f"CALC:{nums[0]},{nums[1]}", "thinking_steps": steps}
            
    steps.append(f"⚙️ **Planner:** No calculation needed. Proceeding to Answer Generation.")
    return {"tool_log": "NONE", "thinking_steps": steps}

def execute_tool(state: AgentState):
    """Step 3: Call Calculator Tool"""
    steps = state["thinking_steps"]
    
    if state["tool_log"].startswith("CALC"):
        try:
            _, params = state["tool_log"].split(":")
            n1, n2 = params.split(",")
            steps.append(f"🧮 **Tool:** Calling API with `{n1}, {n2}`...")
            
            res = requests.post(f"{TOOL_URL}/calculate", json={"dose": float(n1), "area": float(n2)})
            result = res.json()["msg"]
            
            steps.append(f"✅ **Tool:** Result received: `{result}`")
            return {"tool_log": result, "thinking_steps": steps}
        except:
            steps.append(f"❌ **Tool:** Execution Failed.")
            return {"tool_log": "Calculation Error", "thinking_steps": steps}
            
    return {"tool_log": "No calculation needed.", "thinking_steps": steps}

def generate(state: AgentState):
    """Step 4: Generate Final Answer (Strict Grounding)"""
    steps = state["thinking_steps"]
    steps.append(f"📝 **Generator:** Synthesizing final answer...")
    
    # Policy: Refusal if context is missing
    if state["context"] == "MISSING":
        return {"final_ans": "I don't know based on the provided documents.", "thinking_steps": steps}

    # PROMPT: STRICT GROUNDING (As per Requirement)
    prompt = f"""<|im_start|>system
You are an intelligent Assistant for East West University (EWU).
Your task is to answer policy questions based STRICTLY on the provided context.

INSTRUCTIONS:
1. Use ONLY the information in the CONTEXT below.
2. If the answer is not in the context, say "I don't know based on the provided documents."
3. Do not make up rules or punishments.
4. If a list is provided in the context, format the answer as a bulleted list.

CONTEXT:
{state['context']}

USER QUESTION:
{state['original_question']}
<|im_end|>
<|im_start|>assistant
"""
    
    out = llm(prompt, max_tokens=350, stop=["<|im_end|>"], temperature=0.1)
    ans = out["choices"][0]["text"].strip()
    
    return {"final_ans": ans, "thinking_steps": steps}

# --- GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("preprocess", preprocess_query)
workflow.add_node("retrieve", retrieve)
workflow.add_node("reason", reason)
workflow.add_node("tool", execute_tool)
workflow.add_node("generate", generate)

workflow.set_entry_point("preprocess")
workflow.add_edge("preprocess", "retrieve")
workflow.add_edge("retrieve", "reason")
workflow.add_edge("reason", "tool")
workflow.add_edge("tool", "generate")
workflow.add_edge("generate", END)
app = workflow.compile()

# --- UI ---
st.title("🎓 EWU AI Engine: Policy Assistant")
st.caption("Agentic RAG System | Innovation Challenge 2026")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about EWU Policy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("🤖 Agent is thinking..."):
        # Initialize State
        inputs = {
            "original_question": prompt, 
            "search_keywords": "", 
            "context": "", 
            "sources": [], 
            "tool_log": "", 
            "final_ans": "",
            "thinking_steps": []
        }
        
        result = app.invoke(inputs)
        response = result["final_ans"]

    # Display Thinking Process (Crucial for Requirement)
    with st.expander("🧠 View Agent Reasoning Chain (Transparency)", expanded=True):
        for step in result['thinking_steps']:
            st.markdown(step)
        
        st.divider()
        st.markdown("**📚 Sources Used:**")
        if result['sources']:
            for src in result['sources']:
                st.code(src.split("/")[-1]) # Show just filename
        else:
            st.write("No sources used.")

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)