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
# STREAMLIT CHAT UI
# ---------------------------
st.set_page_config(page_title="NSE AI Trading Engine", layout="wide")

st.title("📈 NSE AI Trading Decision Chatbot")

st.markdown("""
### Type your question like:
- *Analyze NIFTY 50*
- *Analyze BANKNIFTY with options*
- *Analyze RELIANCE including option chain*
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box (MULTILINE like you wanted)
user_query = st.chat_input("Ask anything about NIFTY, BANKNIFTY, or any stock...")

if user_query:

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking like an institutional desk..."):

            agent = Agent(
                model=Gemini(id="gemini-2.0-flash"),
                tools=[NSETools(), DuckDuckGoTools()],
                markdown=True,
                instructions=[
                    "You are a professional Indian options market decision engine.",
                    "You operate like an institutional trading desk, not a news summarizer.",
                    "Your job is NOT just to say BUY or SELL.",
                    "Your job is to explain WHY, WHAT TO WATCH, and WHEN a trade becomes valid.",
                    "",
                    "You MUST follow this THINKING FRAMEWORK:",
                    "1️⃣ MARKET REGIME ANALYSIS",
                    "2️⃣ STOCK / INDEX TREND ANALYSIS",
                    "3️⃣ OPTIONS CHAIN INTELLIGENCE",
                    "4️⃣ VOLATILITY & THETA RISK",
                    "5️⃣ NEWS & EVENT RISK",
                    "6️⃣ DECISION SYNTHESIS",
                    "7️⃣ WAIT CONDITIONS",
                    "8️⃣ STRATEGY GUIDANCE",
                    "",
                    "STRICT OUTPUT FORMAT:",
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
                    "- NEVER skip WAIT CONDITIONS",
                    "- Be decisive and professional",
                    "- Respond in MARKDOWN",
                    "- Use latest available information",
                    "- Keep under 600 words",
                ],
            )

            result = agent.run(f"user query is {user_query}. Provide a detailed trading decision using our tools & market knowledge.")

        st.markdown(result.content)
        st.session_state.messages.append({"role": "assistant", "content": result.content})
