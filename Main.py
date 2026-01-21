import streamlit as st
from typing import Optional, Literal
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

# ---------------------------------
# Load environment variables
# ---------------------------------
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
# Technical Indicators Dictionary
# ---------------------------------
INDICATORS = {
    "🟢 Candlestick - Bullish": [
        "Bullish Abandoned Baby",
        "Bullish Engulfing Pattern",
        "Bullish Harami",
        "Bullish Harami Cross",
        "Bullish Kicking",
        "Dragonfly Doji",
        "Hammer",
        "Inverted Hammer",
        "Morning Star",
        "Piercing Line",
        "Three White Soldiers",
        "Upside Tasuki Gap",
        "White Marubozu"
    ],
    "🔴 Candlestick - Bearish": [
        "Abandoned Baby Top",
        "Bearish Engulfing",
        "Bearish Harami",
        "Bearish Harami Cross",
        "Black Marubozu",
        "Dark Cloud Cover",
        "Downside Tasuki Gap",
        "Hanging Man",
        "Identical Three Crows",
        "Shooting Star"
    ],
    "🟢 Moving Average - Bullish": [
        "Golden Cross (50D SMA > 200D SMA)",
        "Positive Breakouts – Long Trend (LTP > 200D SMA)",
        "Positive Breakouts – Medium Trend (LTP > 50D SMA)",
        "Positive Breakouts – Short Trend (LTP > 30D SMA)"
    ],
    "🔴 Moving Average - Bearish": [
        "Death Cross (50D SMA < 200D SMA)",
        "Negative Breakouts – Long Trend (LTP < 200D SMA)",
        "Negative Breakouts – Medium Trend (LTP < 50D SMA)",
        "Negative Breakouts – Short Trend (LTP < 30D SMA)"
    ],
    "🟢 Technical - Bullish": [
        "LTP > 20% from Week Low",
        "MACD Crosses Above 0 Line",
        "MACD Crossover Above",
        "MFI Overbought",
        "RSI and MFI Overbought",
        "RSI Bullish",
        "SMA Crossover (30D SMA > 200D SMA)"
    ],
    "🔴 Technical - Bearish": [
        "MACD Crosses Below 0 Line",
        "MACD Crossover Below",
        "MFI Oversold",
        "RSI and MFI Oversold",
        "RSI Bearish"
    ]
}

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
# Build comprehensive indicators list for prompt
# ---------------------------------
def build_indicators_text():
    """Build formatted text of all indicators organized by category"""
    indicators_text = "\n\nAnalyze using these technical indicators:\n\n"
    
    for category, indicators in INDICATORS.items():
        indicators_text += f"{category}:\n"
        for indicator in indicators:
            indicators_text += f"  - {indicator}\n"
        indicators_text += "\n"
    
    return indicators_text

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

# ✅ Custom Question Input
st.markdown("### ✍️ Ask your own question (optional)")
custom_question = st.text_area(
    "Custom market question",
    placeholder="Eg: Is Bank Nifty bullish today? Should I buy 46000 CE?",
    height=90,
    help="Leave empty for standard analysis with all indicators"
)

analyze_btn = st.button("🔍 Analyze Market")

# ---------------------------------
# Run Analysis
# ---------------------------------
if analyze_btn:
    with st.spinner("Analyzing market data..."):
        try:
            # ✅ Build prompt with all indicators included
            indicators_text = build_indicators_text()
            
            if custom_question.strip():
                prompt = custom_question.strip() + indicators_text
            else:
                prompt = f"Analyze {index} and recommend a {option_type} option strike price.{indicators_text}"

            # Show question sent to AI
            st.markdown("### 🧠 Question Sent to AI")
            with st.expander("View Full Prompt"):
                st.code(prompt)

            response = agent.run(prompt, stream=False)

            if response and response.content:
                st.success("Analysis Complete")

                st.markdown("### 📈 Market Summary")
                st.write(response.content.summary)

                st.markdown("### 🎯 Trade Recommendation")
                st.metric(
                    label="Entry Signal",
                    value=response.content.entry_type or "NO TRADE"
                )

                st.metric(
                    label="Recommended Strike Price",
                    value=response.content.strike_price
                )

                st.markdown("---")
                with st.expander("⚠️ Risk Disclaimer"):
                    st.write(
                        """
                        This AI-generated analysis is for **educational purposes only**.
                        Options trading involves significant risk.
                        Do not use this as financial advice.
                        """
                    )
            else:
                st.error("No response received from the agent.")

        except Exception as e:
            st.error(f"Error: {str(e)}")