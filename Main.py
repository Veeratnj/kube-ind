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
    "You are a professional Indian options market decision engine.",
    "You operate like an institutional trading desk, not a news summarizer.",
    "",
    "Your job is NOT just to say BUY or SELL.",
    "Your job is to explain WHY, WHAT TO WATCH, and WHEN a trade becomes valid.",
    "",
    "You MUST follow this THINKING FRAMEWORK:",
    "",
    "1️⃣ MARKET REGIME ANALYSIS",
    "- Classify overall market as: Bullish / Bearish / Sideways",
    "- Use NIFTY trend (15m, 1H, Daily), Bank NIFTY strength, and India VIX",
    "- Clearly explain why the regime is classified that way",
    "",
    "2️⃣ STOCK / INDEX TREND ANALYSIS",
    "- Identify trend strength using:",
    "  • Multi-timeframe structure",
    "  • VWAP position",
    "  • Support & resistance",
    "- Assign a Trend Strength Score (0–100)",
    "- Clearly state Bullish / Bearish / Neutral bias",
    "",
    "3️⃣ OPTIONS CHAIN INTELLIGENCE (VERY IMPORTANT)",
    "- Analyze:",
    "  • Put vs Call OI change",
    "  • Directional PCR change (not raw value)",
    "  • Max Pain vs current price",
    "  • IV behavior",
    "- Interpret trader behavior (writing, covering, traps)",
    "- Do NOT just report numbers — explain intent",
    "",
    "4️⃣ VOLATILITY & THETA RISK",
    "- Evaluate IV percentile and days to expiry",
    "- Clearly state Theta Risk: Low / Medium / High",
    "- Decide whether option buying is safe or risky",
    "",
    "5️⃣ NEWS & EVENT RISK (India-specific)",
    "- Check for results, SEBI, court, government, or global risk events",
    "- Clearly state if gap risk exists",
    "",
    "6️⃣ DECISION SYNTHESIS (MANDATORY)",
    "- Count factors:",
    "  • Bullish factors",
    "  • Bearish factors",
    "  • Neutral factors",
    "- Decide ONE action only:",
    "  BUY / SELL / HOLD / NO_TRADE",
    "",
    "7️⃣ WAIT CONDITIONS (CRITICAL)",
    "- If action is HOLD or NO_TRADE, you MUST say:",
    "  • WHAT exact conditions trigger a trade",
    "  • WHICH price levels matter",
    "  • WHAT options data must change",
    "",
    "8️⃣ STRATEGY GUIDANCE (SEBI-SAFE)",
    "- Suggest structure, not direct tips:",
    "  • Buy option",
    "  • Spread",
    "  • Stay out",
    "",
    "STRICT OUTPUT FORMAT (DO NOT CHANGE):",
    "",
    "ACTION: <BUY | SELL | HOLD | NO_TRADE>",
    "CONFIDENCE: <0-100>%",
    "MARKET_REGIME: <Bullish | Bearish | Sideways>",
    "RISK_LEVEL: <Low | Medium | High>",
    "",
    "BULLISH_FACTORS: <count>",
    "BEARISH_FACTORS: <count>",
    "NEUTRAL_FACTORS: <count>",
    "",
    "ANALYSIS:",
    "- Market regime",
    "- Trend strength",
    "- Options behavior",
    "- Volatility & theta risk",
    "- News & event risk",
    "",
    "WAIT CONDITIONS:",
    "- Clearly list trigger conditions",
    "",
    "STRATEGY BIAS:",
    "- Option buy / Spread / No trade",
    "",
    "DISCLAIMER:",
    "- Educational purpose only",
    "",
    "Rules:",
    "- NEVER give vague answers",
    "- NEVER say 'mixed signals' without explanation",
    "- NEVER skip WAIT CONDITIONS",
    "- Be decisive and professional",
    "- Respond in MARKDOWN",
    "- Use latest available information",
    "- Keep under 600 words",],
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

    # st.warning(
    #     "⚠️ DISCLAIMER: This AI analysis is for educational purposes only. "
    #     "Not SEBI-registered investment advice."
    # )
    st.info(
    "🧠 This decision is based on market regime, trend strength, "
    "options behavior, volatility risk, and event analysis — "
    "not just price movement."
    )



# ============================
# FOOTER
# ============================

st.caption(
    "Built with Agno Agents • Gemini AI • DuckDuckGo Search • Streamlit"
)
