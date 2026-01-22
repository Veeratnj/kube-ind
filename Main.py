# ============================
# AI MARKET DECISION ENGINE
# Single-file Streamlit App
# ============================

import streamlit as st
from agno.agent import Agent, RunOutput
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.utils.pprint import pprint_run_response
from dotenv import load_dotenv
load_dotenv()


# ============================
# STREAMLIT CONFIG
# ============================

st.set_page_config(
    page_title="AI Market Decision Engine",
    layout="wide",
)

st.title("📈 AI Indian Stock Market Decision Engine")
st.caption("NIFTY • SENSEX • Options • MCX • News-Driven AI")


# ============================
# AGENT DEFINITION
# ============================

market_decision_agent = Agent(
    name="Indian Market Decision Agent",
    model=Gemini(id="gemini-2.0-flash-001"),
    tools=[DuckDuckGoTools()],
    instructions=[
        "You are a professional Indian stock market strategist.",
        "Analyze today's Indian stock market using real-time information.",
        "",
        "Markets to analyze:",
        "- NIFTY 50",
        "- SENSEX",
        "- Top gainers and losers",
        "- Options market (OI, PCR, Max Pain)",
        "- MCX commodities (Gold, Crude Oil)",
        "- Global cues and market news",
        "",
        "You MUST give ONE clear decision:",
        "BUY, SELL, HOLD, or NO_TRADE.",
        "",
        "STRICT OUTPUT FORMAT (do not change):",
        "",
        "ACTION: <BUY | SELL | HOLD | NO_TRADE>",
        "CONFIDENCE: <0-100>%",
        "MARKET_SENTIMENT: <Bullish | Bearish | Neutral>",
        "RISK_LEVEL: <Low | Medium | High>",
        "",
        "ANALYSIS:",
        "- Index trend",
        "- Sector performance",
        "- Options data",
        "- Commodities update",
        "- News impact",
        "",
        "REASONING:",
        "- Why this action",
        "- Why not others",
        "",
        "DISCLAIMER:",
        "- Educational purpose only",
        "",
        "Be decisive, structured, and concise.",
        "Always cite sources.",
        "Respond in MARKDOWN format.",
        "Never refuse to answer.",
        "Use tools as needed to get real-time data.",
        "Stay updated with the latest market info.",
        "Provide actionable insights.",
        "Try give response within 500 words.",
        "try to give accurate result as much as possible.",
        "based on ur result user going to take trade decision.",
    ],
    markdown=True,
    # show_tool_calls=True,
)


# ============================
# UI INPUT
# ============================

query = st.text_input(
    "Market Query",
    value="Analyze today's Indian stock market and give a trade decision"
)

col1, col2 = st.columns([1, 3])
with col1:
    analyze_btn = st.button("🚀 Analyze Market")


# ============================
# AGENT EXECUTION
# ============================

if analyze_btn:
    with st.spinner("🔍 AI is analyzing market conditions..."):
        response: RunOutput = market_decision_agent.run(
            query,
            # stream=True
        )
    st.markdown(
        response.content
    )
    st.success("✅ Market Analysis Completed")

    # ============================
    # OUTPUT
    # ============================

    # pprint_run_response(response, markdown=True)

    st.divider()

    st.warning(
        "⚠️ DISCLAIMER: This AI analysis is for educational purposes only. "
        "Not SEBI-registered investment advice."
    )


# ============================
# FOOTER
# ============================

st.caption(
    "Built with Agno Agents • Gemini AI • DuckDuckGo Search • Streamlit"
)
