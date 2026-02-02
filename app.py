from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools import Toolkit
from agno.tools.duckduckgo import DuckDuckGoTools
from nsepython import nse_get_index_quote, nse_eq, nse_optionchain_scrapper
import yfinance as yf
from dotenv import load_dotenv
load_dotenv()




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
            return {"error": f"Failed to fetch index price for {index_name}: {e}"}

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
            return {"error": f"Failed to fetch equity quote for {symbol}: {e}"}

    def get_nse_option_chain(self, symbol: str) -> dict:
        try:
            oc = nse_optionchain_scrapper(symbol)
            return {
                "symbol": symbol,
                "spot_price": oc["records"]["underlyingValue"],
                "expiry_dates": oc["records"]["expiryDates"],
                "data": oc["records"]["data"]
            }
        except Exception as e:
            return {"error": f"Failed to fetch option chain for {symbol}: {e}"}

    def get_historical_ohlc(self, symbol: str, interval: str = "5m", period: str = "5d") -> list:
        try:
            df = yf.download(symbol, interval=interval, period=period)
            return df.reset_index().to_dict(orient="records")
        except Exception as e:
            return [{"error": f"Failed to fetch historical OHLC for {symbol}: {e}"}]

# ---------------------------
# Use with Gemini Agent
# ---------------------------
agent = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    tools=[NSETools(), DuckDuckGoTools()],
    markdown=True,
    instructions=[
        "You are a professional Indian options market decision engine.",
        "You operate like an institutional trading desk, not a news summarizer.",
        "Your job is NOT just to say BUY or SELL.",
        "You MUST follow this THINKING FRAMEWORK:",
        "1️⃣ MARKET REGIME ANALYSIS - classify as Bullish / Bearish / Sideways",
        "2️⃣ STOCK / INDEX TREND ANALYSIS - multi-timeframe, VWAP, S/R, Trend Strength Score",
        "3️⃣ OPTIONS CHAIN INTELLIGENCE - OI, PCR changes, Max Pain, IV behavior",
        "4️⃣ VOLATILITY & THETA RISK - IV percentile, expiry, Theta Risk",
        "5️⃣ NEWS & EVENT RISK - results, SEBI, court, government, global events",
        "6️⃣ DECISION SYNTHESIS - ONE action only: BUY / SELL / HOLD / NO_TRADE",
        "7️⃣ WAIT CONDITIONS - trigger conditions if HOLD/NO_TRADE",
        "8️⃣ STRATEGY GUIDANCE - suggest structure, not direct tips",
        "STRICT OUTPUT FORMAT:",
        "ACTION: <BUY | SELL | HOLD | NO_TRADE>",
        "CONFIDENCE: <0-100>%",
        "MARKET_REGIME: <Bullish | Bearish | Sideways>",
        "RISK_LEVEL: <Low | Medium | High>",
        "BULLISH_FACTORS: <count>",
        "BEARISH_FACTORS: <count>",
        "NEUTRAL_FACTORS: <count>",
        "ANALYSIS: - Market regime, Trend, Options behavior, Volatility & theta risk, News & event risk",
        "WAIT CONDITIONS: - Clearly list trigger conditions",
        "STRATEGY BIAS: - Option buy / Spread / No trade",
        "Rules: NEVER give vague answers, NEVER skip WAIT CONDITIONS, Be decisive, Respond in MARKDOWN, Keep under 600 words",
    ],
)

# ---------------------------
# Example usage
# ---------------------------
result=agent.run("What's the current NIFTY 50 price?")
print(result)



