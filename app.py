import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools import Toolkit
from agno.tools.duckduckgo import DuckDuckGoTools
from nsepython import nse_get_index_quote, nse_eq, nse_optionchain_scrapper
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# NSE TOOLKIT
# ---------------------------
class NSETools(Toolkit):
    def __init__(self, **kwargs):
        tools = [
            self.get_nse_index_price,
            self.get_nse_equity_quote,
            self.get_nse_option_chain,
            self.get_historical_ohlc,
        ]
        super().__init__(name="nse_tools", tools=tools, **kwargs)

    def get_nse_index_price(self, index_name: str) -> dict:
        try:
            data = nse_get_index_quote(index_name)
            return {
                "index": index_name,
                "last_price": data.get("last"),
                "change": data.get("change"),
                "percent_change": data.get("percChange"),
                "time": data.get("lastUpdateTime")
            }
        except Exception as e:
            return {"error": str(e)}

    def get_nse_equity_quote(self, symbol: str) -> dict:
        try:
            data = nse_eq(symbol)
            return {
                "symbol": symbol,
                "last_price": data.get("priceInfo", {}).get("lastPrice"),
                "open": data.get("priceInfo", {}).get("open"),
                "high": data.get("priceInfo", {}).get("intraDayHighLow", {}).get("max"),
                "low": data.get("priceInfo", {}).get("intraDayHighLow", {}).get("min"),
                "volume": data.get("securityWiseDP", {}).get("quantityTraded")
            }
        except Exception as e:
            return {"error": str(e)}

    def get_nse_option_chain(self, symbol: str) -> dict:
        try:
            oc = nse_optionchain_scrapper(symbol)
            return {
                "symbol": symbol,
                "spot_price": oc["records"]["underlyingValue"],
                "expiry_dates": oc["records"]["expiryDates"],
            }
        except Exception as e:
            return {"error": str(e)}

    def get_historical_ohlc(self, symbol: str, interval: str = "5m", period: str = "5d") -> list:
        try:
            df = yf.download(symbol + ".NS", interval=interval, period=period)
            return df.tail(10).reset_index().to_dict(orient="records")
        except Exception as e:
            return [{"error": str(e)}]

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="NSE AI Trading Engine", layout="wide")

st.title("📈 NSE AI Trading Decision Engine")

tab1, tab2 = st.tabs(["Index Analysis", "Stock + Options"])

# ---------------------------
# TAB 1: INDEX ANALYSIS
# ---------------------------
with tab1:
    st.subheader("NIFTY / BANKNIFTY Analysis")

    index = st.selectbox("Select Index", ["NIFTY 50", "BANKNIFTY"])

    if st.button("Analyze Index"):
        with st.spinner("Fetching market data..."):

            agent = Agent(
                model=Gemini(id="gemini-2.0-flash"),
                tools=[NSETools(), DuckDuckGoTools()],
                markdown=True,
                instructions=[
                    "You are a professional Indian options market decision engine.",
                    "Give structured, institutional-grade analysis.",
                    "STRICT OUTPUT FORMAT:",
                    "ACTION: <BUY | SELL | HOLD | NO_TRADE>",
                    "CONFIDENCE: <0-100>%",
                    "MARKET_REGIME: <Bullish | Bearish | Sideways>",
                    "RISK_LEVEL: <Low | Medium | High>",
                    "ANALYSIS: concise but sharp",
                    "WAIT CONDITIONS: list triggers",
                    "STRATEGY BIAS: Option buy / Spread / No trade",
                ],
            )

            query = f"Analyze {index} with full market regime, options insight, and risk view."
            result = agent.run(query)

        st.markdown(result)

# ---------------------------
# TAB 2: STOCK + OPTIONS
# ---------------------------
with tab2:
    st.subheader("Stock & Option Chain AI Analysis")

    stock = st.text_input("Enter Stock (e.g., RELIANCE, INFY, TCS)")

    if st.button("Analyze Stock"):
        if not stock:
            st.warning("Enter a stock symbol first!")
        else:
            with st.spinner("Running AI analysis..."):

                agent = Agent(
                    model=Gemini(id="gemini-2.0-flash"),
                    tools=[NSETools(), DuckDuckGoTools()],
                    markdown=True,
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
                        "Rules:",
                        "- NEVER give vague answers",
                        "- NEVER say 'mixed signals' without explanation",
                        "- NEVER skip WAIT CONDITIONS",
                        "- Be decisive and professional",
                        "- Respond in MARKDOWN",
                        "- Use latest available information",
                        "- Keep under 600 words",
                    ],
                )


                query = f"Analyze {stock} including option chain and trend."
                result = agent.run(query)

            st.markdown(result)
