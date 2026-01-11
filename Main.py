import streamlit as st
from typing import Optional, Literal
from pydantic import BaseModel, Field
from agno.agent import Agent
# from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------------
# Streamlit Page Config
# ---------------------------------
st.set_page_config(
    page_title="NSE Index Options Analyst",
    layout="centered"
)

st.title("📊 NSE Index Options Analyst")
st.caption("AI-powered technical analysis for NIFTY & Bank Nifty options")

# ---------------------------------
# Output Schema
# ---------------------------------
class OptionsRecommendation(BaseModel):
    summary: str = Field(..., description="Brief technical analysis summary")
    strike_price: float = Field(..., description="Recommended option strike price")
    entry_type: Optional[Literal["BUY", "SELL"]] = Field(
        default=None,
        description="Trade signal or None if market is sideways"
    )

# ---------------------------------
# Initialize Agent (cached)
# ---------------------------------
@st.cache_resource
def load_agent():
    return Agent(
        name="Index Options Analyst",
        model=Gemini(id="gemini-2.0-flash"),
        tools=[
            YFinanceTools(
                # stock_price=True,
                # technical_indicators=True,
                # historical_prices=True,
            )
        ],
        description=(
            "Expert NSE index options analyst focused on NIFTY and Bank Nifty."
        ),
        instructions=[
            "Analyze NIFTY (^NSEI) and Bank Nifty using trend, RSI, MACD, EMA.",
            "Identify bullish, bearish, or sideways market structure.",
            "For CALL options → bullish trend only.",
            "For PUT options → bearish trend only.",
            "ATM strikes must be rounded to nearest 50 (NIFTY) or 100 (Bank Nifty).",
            "Avoid trade if market is sideways or signals conflict.",
            "Always include a risk disclaimer.",
        ],
        output_schema=OptionsRecommendation,
        markdown=True,
    )

agent = load_agent()

# ---------------------------------
# UI Controls
# ---------------------------------
index = st.selectbox(
    "Select Index",
    ["NIFTY 50", "BANK NIFTY"]
)

option_type = st.selectbox(
    "Select Option Type",
    ["CALL", "PUT"]
)

analyze_btn = st.button("🔍 Analyze Market")

# ---------------------------------
# Run Analysis
# ---------------------------------
if analyze_btn:
    with st.spinner("Analyzing market data..."):
        try:
            prompt = (
                f"Analyze {index} and recommend a {option_type} option strike price"
            )

            response = agent.run(prompt, stream=False)

            if response and response.content:
                st.success("Analysis Complete")

                st.markdown("### 📈 Market Summary")
                st.write(response.content["summary"])

                st.markdown("### 🎯 Trade Recommendation")
                st.metric(
                    label="Entry Signal",
                    value=response.content["entry_type"] or "NO TRADE"
                )

                st.metric(
                    label="Recommended Strike Price",
                    value=response.content["strike_price"]
                )

                st.markdown("---")
                st.warning(
                    "⚠️ **Disclaimer:** This is AI-generated analysis for educational "
                    "purposes only. Not financial advice."
                )
            else:
                st.error("No response received from the agent.")

        except Exception as e:
            st.error(f"Error: {str(e)}")
