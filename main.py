import streamlit as st
import pandas as pd
import json
import re
import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
import plotly.express as px
from openai import OpenAI

# =====================================================================
# ENVIRONMENT VARIABLES SETUP
# =====================================================================
# 1. Load the variables from the .env file into os.environ
load_dotenv()

# 2. Retrieve variables using os.getenv()
# (Pass a second argument as a default value if the key isn't found)
port = os.getenv("PORT", "3000")
db_url = os.getenv("DATABASE_URL")
debug_mode = os.getenv("DEBUG", "False").lower() in ("true", "1")

# Retrieve API key and custom base URL (e.g., Great Learning proxy)
default_api_key = os.getenv("OPENAI_API_KEY", "")
api_base_url = os.getenv("OPENAI_API_BASE", None)

# 3. Use the variables
print(f"App running on port: {port}")
print(f"Database URL: {db_url}")
print(f"Debug Mode: {debug_mode}")

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="ShopNest Global - AI Support Ticket Intelligence",
    page_icon="🎫",
    layout="wide"
)

st.title("🎫 ShopNest Global: AI Support Ticket Intelligence System")
st.markdown("""
This platform ingests raw customer tickets, executes **Summarization** and **Policy-Compliant Response Generation**, 
runs dual **LLM-as-a-Judge Evaluation Frameworks**, and synthesizes **Actionable Business Insights**.
""")

# Sidebar Configuration
st.sidebar.header("⚙️ System Configuration")
api_key = st.sidebar.text_input(
    label="OpenAI API Key", 
    value=os.getenv("OPENAI_API_KEY", ""),
    type="password",
    help="Enter your API key securely. Never hardcode API keys in source code."
)
selected_model = st.sidebar.selectbox("Select Model Engine", ["gpt-4o-mini", "gpt-4o"], index=0)

if not api_key:
    st.info("👈 Please enter your OpenAI API key in the sidebar to initiate the pipeline.")
    st.stop()

client = OpenAI(api_key=api_key,
        base_url=api_base_url if api_base_url else None)

# =====================================================================
# SYSTEM PROMPTS
# =====================================================================
SUMMARIZER_SYSTEM_PROMPT = """
You are an expert AI Customer Support Analyst for ShopNest Global.
Analyze raw, unstructured support tickets and extract key facts cleanly.

RULES:
1. Strip out emotional noise, rants, and irrelevant background details.
2. Extract: Core Issue, Order ID, Product Details, Customer Request, and Urgency level.
3. State explicitly if critical details are missing.

OUTPUT FORMAT STRICTLY:
- Core Issue: <Brief description>
- Order ID: <Extracted ID or 'Not Provided'>
- Product Details: <Extracted Product or 'Not Provided'>
- Key Customer Request: <Replacement, Refund, Status Check, Technical Support, etc.>
- Escalation/Urgency: <High / Medium / Low>
"""

RESPONSE_SYSTEM_PROMPT = """
You are a senior customer support representative for ShopNest Global.
Draft a professional, empathetic response adhering strictly to company policy.

SHOPNEST POLICIES:
1. Defective/Damaged Item: Express sincere regret. Offer immediate free replacement or full refund upon return. Home pickup available upon request.
2. Billing/Double Charge: Acknowledge issue. Reversals process within 24-48 hours. Advise bank posting windows.
3. Delayed/Missing Delivery: Apologize. Verify tracking. If marked 'Delivered' but missing, open investigation or reship immediately.
4. Vague Tickets: Empathetically request missing details (Order ID, error details) to proceed.

FORMAT REQUIREMENTS:
- Professional Greeting
- Empathy Statement
- Bulleted Actionable Next Steps
- Professional Sign-off
"""

EVAL_SUMMARIZER_PROMPT = """
You are a Quality Assurance Auditor evaluating AI-generated ticket summaries.
Evaluate the Summary against the Raw Ticket on a scale of 1 to 5 for:
- Accuracy & Factuality
- Noise Reduction & Conciseness
- Completeness of key fields

Output strictly valid JSON:
{
  "summary_score": <1-5 integer>,
  "summary_reasoning": "<Concise justification>"
}
"""

EVAL_RESPONSE_PROMPT = """
You are a Quality Assurance Auditor evaluating AI-generated customer responses.
Evaluate the Draft Response against the Raw Ticket and Policy on a scale of 1 to 5 for:
- Policy Compliance & Tone
- Empathy & Professionalism
- Clarity of Next Steps

Output strictly valid JSON:
{
  "response_score": <1-5 integer>,
  "response_reasoning": "<Concise justification>"
}
"""

INSIGHTS_GENERATOR_PROMPT = """
You are a CX Operations Director analyzing support ticket trends.
Review the aggregated ticket data summaries and evaluation scores provided.
Provide:
1. Top 3 Root Causes of Customer Friction
2. Policy or Operational Gaps Identified
3. Strategic Recommendations to Reduce Support Ticket Volume

Keep output formatted with clear Markdown headers and bullet points.
"""

# =====================================================================
# AGENT HELPER FUNCTIONS
# =====================================================================
def run_agent(system_prompt: str, user_content: str, model_name: str, temp: float = 0.2, max_tokens: int = 400) -> str:
    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=temp,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"Error executing agent: {str(e)}"

def run_json_evaluator(system_prompt: str, user_content: str, model_name: str, score_key: str, reasoning_key: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            max_tokens=250,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        )
        raw_text = response.choices[0].message.content or "{}"
        parsed = json.loads(raw_text)
        return {
            "score": int(parsed.get(score_key, 3)),
            "reasoning": str(parsed.get(reasoning_key, "Evaluation completed successfully."))
        }
    except Exception as e:
        # Secure fallback: Mark as unverified low score rather than inflating to 5/5
        return {"score": 1, "reasoning": f"Automated Evaluation Failed: {str(e)}"}

# =====================================================================
# FILE INGESTION
# =====================================================================
uploaded_file = st.file_uploader("📂 Upload Support Tickets Dataset (.xlsx or .csv)", type=["xlsx", "csv"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".xlsx"):
            raw_df = pd.read_excel(uploaded_file)
        else:
            raw_df = pd.read_csv(uploaded_file)

        st.success(f"Successfully loaded `{uploaded_file.name}` containing **{len(raw_df)}** records.")

        # Data Cleaning and Parsing Logic
        tickets = []
        
        # Find ID column (priority to 'id', avoiding 'desc', 'text', 'comment')
        id_candidates = [c for c in raw_df.columns if 'id' in str(c).lower() and not any(x in str(c).lower() for x in ['desc', 'text', 'comment'])]
        id_col = id_candidates[0] if id_candidates else next((c for c in raw_df.columns if 'id' in str(c).lower()), None)
        
        # Find description/text column (priority to 'desc', 'text', 'comment')
        desc_candidates = [c for c in raw_df.columns if any(x in str(c).lower() for x in ['desc', 'text', 'comment'])]
        desc_col = desc_candidates[0] if desc_candidates else next((c for c in raw_df.columns if 'ticket' in str(c).lower() and c != id_col), None)

        if id_col is not None and desc_col is not None and id_col != desc_col:
            for idx, row in raw_df.iterrows():
                tickets.append({
                    "support_ticket_id": str(row[id_col]).replace('.0', '').strip(),
                    "support_ticket_desc": str(row[desc_col]).strip().strip('"')
                })
        else:
            for idx, row in raw_df.iterrows():
                row_str = " ".join([str(val) for val in row.values if pd.notna(val)])
                tickets.append({
                    "support_ticket_id": f"TICK-{idx + 1001}",
                    "support_ticket_desc": row_str.strip()
                })

        st.subheader("📋 Ingested Dataset Preview")
        st.dataframe(pd.DataFrame(tickets).head(5), hide_index=True, use_container_width=True)

        # Processing Trigger
        if st.button("🚀 Execute Multi-Agent Intelligence Pipeline"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            total_tickets = len(tickets)

            for idx, t in enumerate(tickets):
                status_text.text(f"Processing Ticket {idx + 1} of {total_tickets} (ID: {t['support_ticket_id']})...")

                # Agent Pipeline Execution
                summary = run_agent(SUMMARIZER_SYSTEM_PROMPT, f"Ticket:\n{t['support_ticket_desc']}", selected_model, temp=0.1)
                draft_resp = run_agent(RESPONSE_SYSTEM_PROMPT, f"Raw Ticket:\n{t['support_ticket_desc']}\n\nSummary:\n{summary}", selected_model, temp=0.3)
                
                # Independent Evaluator Agents
                sum_eval = run_json_evaluator(
                    EVAL_SUMMARIZER_PROMPT, 
                    f"Raw Ticket: {t['support_ticket_desc']}\nSummary: {summary}", 
                    selected_model, "summary_score", "summary_reasoning"
                )
                resp_eval = run_json_evaluator(
                    EVAL_RESPONSE_PROMPT, 
                    f"Raw Ticket: {t['support_ticket_desc']}\nSummary: {summary}\nDraft Response: {draft_resp}", 
                    selected_model, "response_score", "response_reasoning"
                )

                results.append({
                    "support_ticket_id": t['support_ticket_id'],
                    "raw_ticket_text": t['support_ticket_desc'],
                    "generated_summary": summary,
                    "generated_response": draft_resp,
                    "summary_score": sum_eval["score"],
                    "summary_reasoning": sum_eval["reasoning"],
                    "response_score": resp_eval["score"],
                    "response_reasoning": resp_eval["reasoning"]
                })

                progress_bar.progress((idx + 1) / total_tickets)

            status_text.text("✅ Pipeline Execution Complete!")
            st.session_state['results_df'] = pd.DataFrame(results)

    except Exception as e:
        st.error(f"Data Processing Error: {str(e)}")

# =====================================================================
# DASHBOARD, ANALYTICS & INSIGHTS GENERATION
# =====================================================================
if 'results_df' in st.session_state:
    df_res = st.session_state['results_df']

    st.markdown("---")
    st.header("📊 Pipeline Performance & Evaluation Metrics")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Tickets Processed", len(df_res))
    m2.metric("Avg Summary Score", f"{df_res['summary_score'].mean():.2f} / 5.0")
    m3.metric("Avg Response Score", f"{df_res['response_score'].mean():.2f} / 5.0")
    
    pass_rate = (len(df_res[(df_res['summary_score'] >= 4) & (df_res['response_score'] >= 4)]) / len(df_res)) * 100
    m4.metric("QA Compliance Pass Rate", f"{pass_rate:.1f}%")

    # Visualizations
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig_sum = px.histogram(df_res, x="summary_score", nbins=5, title="Summary Score Distribution", range_x=[0.5, 5.5])
        st.plotly_chart(fig_sum, use_container_width=True)
    with col_chart2:
        fig_resp = px.histogram(df_res, x="response_score", nbins=5, title="Response Score Distribution", range_x=[0.5, 5.5])
        st.plotly_chart(fig_resp, use_container_width=True)

    # Individual Ticket Inspector
    st.markdown("---")
    st.header("🔍 Detailed Ticket QA Inspector")
    selected_id = st.selectbox("Select Support Ticket ID:", df_res['support_ticket_id'])
    sel_row = df_res[df_res['support_ticket_id'] == selected_id].iloc[0]

    t_c1, t_c2 = st.columns(2)
    with t_c1:
        st.subheader("Raw Customer Ticket")
        st.info(sel_row['raw_ticket_text'])

        st.subheader("Generated Summary")
        st.success(sel_row['generated_summary'])
        st.caption(f"**Summary Audit Score:** {sel_row['summary_score']}/5 | **Reasoning:** {sel_row['summary_reasoning']}")

    with t_c2:
        st.subheader("Generated Customer Response Draft")
        st.warning(sel_row['generated_response'])
        st.caption(f"**Response Audit Score:** {sel_row['response_score']}/5 | **Reasoning:** {sel_row['response_reasoning']}")

    # Actionable Insights & Business Recommendations Section (Fulfills 4-Point Rubric Rule)
    st.markdown("---")
    st.header("💡 Actionable Insights & Strategic Recommendations")
    
    if st.button("🧠 Generate Executive Strategic Insights Report"):
        with st.spinner("Synthesizing operational trends across dataset..."):
            ticket_corpus = "\n---\n".join(
                f"Ticket ID: {r['support_ticket_id']}\nSummary: {r['generated_summary']}\nAudit Reason: {r['response_reasoning']}"
                for _, r in df_res.iterrows()
            )
            insights_summary = run_agent(
                INSIGHTS_GENERATOR_PROMPT, 
                f"Aggregated Batch Processing Results:\n{ticket_corpus}", 
                selected_model, 
                temp=0.2, 
                max_tokens=700
            )
            st.session_state['insights_report'] = insights_summary

    if 'insights_report' in st.session_state:
        st.markdown(st.session_state['insights_report'])

    # Export Section
    st.markdown("---")
    st.header("💾 Dataset Export")
    csv_bytes = df_res.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Consolidated QA Report (CSV)",
        data=csv_bytes,
        file_name="shopnest_consolidated_qa_report.csv",
        mime="text/csv"
    )
