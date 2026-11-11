MARKET_GROUPS = {
    "Deutschland": {
        "DAX": "^GDAXI", "SAP": "SAP.DE", "Siemens": "SIE.DE", "Allianz": "ALV.DE",
        "Deutsche Telekom": "DTE.DE", "Airbus": "AIR.DE", "Rheinmetall": "RHM.DE",
        "Mercedes-Benz": "MBG.DE", "BMW": "BMW.DE", "BASF": "BAS.DE",
        "Infineon": "IFX.DE", "Deutsche Bank": "DBK.DE", "Commerzbank": "CBK.DE",
        "Adidas": "ADS.DE", "Vonovia": "VNA.DE", "DHL Group": "DHL.DE",
    },
    "US Tech": {
        "Microsoft": "MSFT", "Apple": "AAPL", "Nvidia": "NVDA", "Amazon": "AMZN",
        "Meta": "META", "Alphabet": "GOOGL", "Tesla": "TSLA", "AMD": "AMD",
        "Broadcom": "AVGO", "Netflix": "NFLX", "Palantir": "PLTR", "Micron": "MU",
        "Qualcomm": "QCOM", "Intel": "INTC", "Adobe": "ADBE", "Salesforce": "CRM",
    },
    "Indizes": {
        "DAX Index": "^GDAXI", "Nasdaq 100": "^NDX", "S&P 500": "^GSPC",
        "Dow Jones": "^DJI", "Euro Stoxx 50": "^STOXX50E", "Russell 2000": "^RUT",
    },
    "Metalle": {
        "Gold": "GC=F", "Silber": "SI=F", "Kupfer": "HG=F", "Platin": "PL=F",
    },
    "Energie": {
        "WTI Öl": "CL=F", "Brent Öl": "BZ=F", "Natural Gas": "NG=F", "Heizöl": "HO=F",
    },
    "Forex": {
        "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X",
        "EUR/GBP": "EURGBP=X", "USD/CHF": "CHF=X", "AUD/USD": "AUDUSD=X",
        "USD/CAD": "CAD=X", "NZD/USD": "NZDUSD=X",
    },
}


def build_universe(groups):
    result = {}
    for group in groups:
        for name, ticker in MARKET_GROUPS[group].items():
            # Doppelte Ticker konsolidieren, Anzeigename der ersten Gruppe behalten.
            if ticker not in {v["ticker"] for v in result.values()}:
                result[f"{group} | {name}"] = {"group": group, "name": name, "ticker": ticker}
    return result
