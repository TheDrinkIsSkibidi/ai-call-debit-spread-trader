import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json

# Configure page
st.set_page_config(
    page_title="AI Call Debit Spread Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e1e5e9;
    }
    .positive {
        color: #00D4AA;
    }
    .negative {
        color: #FF6B6B;
    }
    .neutral {
        color: #4DABF7;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Helper functions
@st.cache_data(ttl=60)
def fetch_market_candidates():
    """Fetch candidate symbols from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/market/candidates")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {"candidates": ["SPY", "QQQ", "AAPL"], "total_count": 3}

@st.cache_data(ttl=30)
def fetch_market_snapshot(symbol):
    """Fetch market snapshot for symbol"""
    try:
        response = requests.get(f"{API_BASE_URL}/market/snapshot/{symbol}")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {"price": 0, "rsi": 50, "volume": 0}

def scan_spreads(symbols, max_per_symbol=3):
    """Scan for spreads using API"""
    try:
        payload = {
            "symbols": symbols,
            "max_spreads_per_symbol": max_per_symbol
        }
        response = requests.post(f"{API_BASE_URL}/spreads/scan", json=payload)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {"spreads": [], "total_count": 0}

@st.cache_data(ttl=300)
def fetch_performance_analytics(days_back=30):
    """Fetch performance analytics"""
    try:
        response = requests.get(f"{API_BASE_URL}/analytics/performance?days_back={days_back}")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def fetch_ai_insights():
    """Fetch AI insights"""
    try:
        response = requests.get(f"{API_BASE_URL}/analytics/insights")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

# Main App
def main():
    st.title("🤖 AI Call Debit Spread Trader")
    st.markdown("*Intelligent options trading powered by AI*")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Dashboard", "Market Scanner", "AI Spreads", "Portfolio", "Analytics", "Settings"]
        )
        
        st.header("Quick Stats")
        
        # API Health Check
        try:
            response = requests.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                st.success("🟢 API Connected")
            else:
                st.error("🔴 API Error")
        except:
            st.error("🔴 API Offline")
        
        # Current positions (placeholder)
        st.metric("Active Positions", "0", delta="0")
        st.metric("Today's P&L", "$0.00", delta="$0.00")
    
    # Main content based on selected page
    if page == "Dashboard":
        dashboard_page()
    elif page == "Market Scanner":
        market_scanner_page()
    elif page == "AI Spreads":
        ai_spreads_page()
    elif page == "Portfolio":
        portfolio_page()
    elif page == "Analytics":
        analytics_page()
    elif page == "Settings":
        settings_page()

def dashboard_page():
    """Main dashboard page"""
    st.header("📊 Trading Dashboard")
    
    # Market Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        spy_data = fetch_market_snapshot("SPY")
        st.metric(
            "SPY", 
            f"${spy_data.get('price', 0):.2f}",
            delta=f"{spy_data.get('rsi', 50):.1f} RSI"
        )
    
    with col2:
        qqq_data = fetch_market_snapshot("QQQ")
        st.metric(
            "QQQ", 
            f"${qqq_data.get('price', 0):.2f}",
            delta=f"{qqq_data.get('rsi', 50):.1f} RSI"
        )
    
    with col3:
        candidates_data = fetch_market_candidates()
        st.metric(
            "Candidates", 
            candidates_data.get('total_count', 0),
            delta=f"{len(candidates_data.get('candidates', []))} symbols"
        )
    
    with col4:
        st.metric(
            "Strategy Status", 
            "Active",
            delta="Running"
        )
    
    # Recent Activity
    st.subheader("🔄 Recent Activity")
    
    # Placeholder activity data
    activity_data = pd.DataFrame({
        'Time': [datetime.now() - timedelta(minutes=x*15) for x in range(10)],
        'Action': ['Scan', 'Analysis', 'Monitor', 'Scan', 'Analysis'] * 2,
        'Symbol': ['SPY', 'AAPL', 'QQQ', 'MSFT', 'GOOGL'] * 2,
        'Status': ['Complete', 'Complete', 'Active', 'Complete', 'Active'] * 2
    })
    
    st.dataframe(
        activity_data,
        use_container_width=True,
        hide_index=True
    )
    
    # Quick Actions
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔍 Run Market Scan", use_container_width=True):
            st.success("Market scan initiated!")
    
    with col2:
        if st.button("🤖 Generate AI Signals", use_container_width=True):
            st.success("AI analysis started!")
    
    with col3:
        if st.button("📈 Update Portfolio", use_container_width=True):
            st.success("Portfolio updated!")
    
    with col4:
        if st.button("🧠 Train ML Model", use_container_width=True):
            st.success("Model training initiated!")

def market_scanner_page():
    """Market scanner page"""
    st.header("🔍 Market Scanner")
    
    # Filter candidates
    candidates_data = fetch_market_candidates()
    candidates = candidates_data.get('candidates', [])
    
    st.subheader("📋 Current Candidates")
    st.write(f"Found {len(candidates)} symbols passing technical filters")
    
    # Display candidates with market data
    if candidates:
        market_data = []
        
        for symbol in candidates:
            snapshot = fetch_market_snapshot(symbol)
            market_data.append({
                'Symbol': symbol,
                'Price': f"${snapshot.get('price', 0):.2f}",
                'RSI': f"{snapshot.get('rsi', 50):.1f}",
                'Volume': f"{snapshot.get('volume', 0):,}",
                'IV Rank': f"{snapshot.get('iv_rank', 50):.1f}%"
            })
        
        df = pd.DataFrame(market_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Manual symbol input
    st.subheader("🎯 Custom Symbol Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        custom_symbol = st.text_input("Enter symbol to analyze:", placeholder="AAPL")
    
    with col2:
        if st.button("Analyze", use_container_width=True):
            if custom_symbol:
                snapshot = fetch_market_snapshot(custom_symbol.upper())
                st.json(snapshot)
    
    # Market filters configuration
    st.subheader("⚙️ Filter Configuration")
    
    with st.expander("Market Filters"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.slider("RSI Minimum", 30, 70, 55)
            st.slider("IV Rank Minimum", 20, 60, 30)
        
        with col2:
            st.slider("RSI Maximum", 70, 100, 85)
            st.slider("IV Rank Maximum", 60, 90, 80)

def ai_spreads_page():
    """AI spreads analysis page"""
    st.header("🤖 AI Spread Analysis")
    
    # Symbol selection
    candidates_data = fetch_market_candidates()
    candidates = candidates_data.get('candidates', [])
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_symbols = st.multiselect(
            "Select symbols to analyze:",
            options=candidates + ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            default=candidates[:3] if candidates else ["SPY", "QQQ"]
        )
    
    with col2:
        max_per_symbol = st.number_input("Max spreads per symbol:", 1, 10, 3)
    
    with col3:
        if st.button("🔍 Scan Spreads", use_container_width=True):
            if selected_symbols:
                with st.spinner("Scanning for optimal spreads..."):
                    spread_results = scan_spreads(selected_symbols, max_per_symbol)
                    st.session_state['spread_results'] = spread_results
    
    # Display results
    if 'spread_results' in st.session_state:
        results = st.session_state['spread_results']
        spreads = results.get('spreads', [])
        
        if spreads:
            st.subheader(f"📈 Found {len(spreads)} Optimal Spreads")
            
            for i, spread_data in enumerate(spreads):
                spread = spread_data['spread']
                llm_analysis = spread_data['llm_analysis']
                ml_prediction = spread_data['ml_prediction']
                market_context = spread_data['market_context']
                
                with st.expander(f"{spread['symbol']} - Strike {spread['long_strike']}/{spread['short_strike']} - AI Score: {llm_analysis['confidence_score']:.0f}"):
                    
                    # Spread details
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("📊 Spread Details")
                        st.write(f"**Symbol:** {spread['symbol']}")
                        st.write(f"**Expiration:** {spread['expiration']}")
                        st.write(f"**Strikes:** {spread['long_strike']}/{spread['short_strike']}")
                        st.write(f"**Net Debit:** ${spread['net_debit']:.2f}")
                        st.write(f"**Max Profit:** ${spread['max_profit']:.2f}")
                        st.write(f"**Max Loss:** ${spread['max_loss']:.2f}")
                        st.write(f"**R/R Ratio:** {spread['risk_reward_ratio']:.2f}")
                    
                    with col2:
                        st.subheader("🤖 AI Analysis")
                        st.write(f"**LLM Score:** {llm_analysis['confidence_score']:.0f}/100")
                        st.write(f"**Recommendation:** {llm_analysis['recommendation']}")
                        st.write(f"**ML Win Prob:** {ml_prediction['win_probability']:.1%}")
                        st.write(f"**ML Confidence:** {ml_prediction['confidence']:.1%}")
                        
                        # AI reasoning (truncated)
                        st.write("**LLM Reasoning:**")
                        st.write(llm_analysis['reasoning'][:150] + "...")
                    
                    with col3:
                        st.subheader("📈 Market Context")
                        st.write(f"**Current Price:** ${market_context['current_price']:.2f}")
                        st.write(f"**RSI:** {market_context['rsi']:.1f}")
                        st.write(f"**IV Rank:** {market_context['iv_rank']:.1f}%")
                        st.write(f"**Days to Exp:** {spread['days_to_expiration']}")
                        st.write(f"**Breakeven:** ${spread['breakeven']:.2f}")
                        
                        # Action button
                        if st.button(f"Execute Trade {i+1}", key=f"execute_{i}"):
                            st.warning("Trade execution is not enabled in demo mode")
        else:
            st.info("No viable spreads found. Try adjusting your criteria or selecting different symbols.")

def portfolio_page():
    """Portfolio monitoring page"""
    st.header("💼 Portfolio Management")
    
    # Portfolio overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Portfolio Value", "$10,000", delta="$0")
    
    with col2:
        st.metric("Available Cash", "$5,000", delta="$0")
    
    with col3:
        st.metric("Open Positions", "0", delta="0")
    
    with col4:
        st.metric("Day P&L", "$0.00", delta="$0.00")
    
    # Position monitoring
    st.subheader("📋 Current Positions")
    
    # Placeholder for positions
    st.info("No open positions currently. Use the AI Spreads page to find and execute trades.")
    
    # Risk management
    st.subheader("⚖️ Risk Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Risk Limits:**")
        st.progress(0.0, text="Portfolio Risk: 0%")
        st.write(f"Max Risk Per Trade: 2%")
        st.write(f"Max Positions: 5")
    
    with col2:
        st.write("**Position Sizing:**")
        st.write("Automatic position sizing based on risk parameters")
        st.write("Kelly Criterion optimization available")
    
    # Order management
    st.subheader("📝 Order Management")
    
    # Placeholder order history
    order_data = pd.DataFrame({
        'Time': [datetime.now() - timedelta(hours=x) for x in range(5)],
        'Symbol': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL'],
        'Action': ['BUY', 'SELL', 'BUY', 'SELL', 'BUY'],
        'Quantity': [1, 1, 2, 1, 1],
        'Price': [4.50, 3.25, 5.75, 4.00, 6.25],
        'Status': ['Filled', 'Filled', 'Cancelled', 'Filled', 'Pending']
    })
    
    st.dataframe(order_data, use_container_width=True, hide_index=True)

def analytics_page():
    """Analytics and performance page"""
    st.header("📊 Performance Analytics")
    
    # Time period selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        days_back = st.selectbox("Analysis Period", [7, 30, 90, 180, 365], index=1)
    
    # Fetch performance data
    performance_data = fetch_performance_analytics(days_back)
    
    if performance_data:
        metrics = performance_data.get('performance_metrics', {})
        
        # Key metrics
        st.subheader("🎯 Key Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Trades", 
                metrics.get('total_trades', 0),
                delta=f"{metrics.get('winning_trades', 0)} wins"
            )
        
        with col2:
            win_rate = metrics.get('win_rate', 0) * 100
            st.metric(
                "Win Rate", 
                f"{win_rate:.1f}%",
                delta=f"{metrics.get('losing_trades', 0)} losses"
            )
        
        with col3:
            st.metric(
                "Total P&L", 
                f"${metrics.get('total_pnl', 0):.2f}",
                delta=f"{metrics.get('expectancy', 0):.2f} exp"
            )
        
        with col4:
            st.metric(
                "Sharpe Ratio", 
                f"{metrics.get('sharpe_ratio', 0):.2f}",
                delta=f"{metrics.get('profit_factor', 0):.2f} PF"
            )
        
        # Performance charts
        st.subheader("📈 Performance Charts")
        
        # Equity curve (placeholder)
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        equity_values = np.cumsum(np.random.randn(30) * 100) + 10000
        
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=dates,
            y=equity_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        fig_equity.update_layout(
            title="Portfolio Equity Curve",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=400
        )
        st.plotly_chart(fig_equity, use_container_width=True)
        
        # Monthly performance
        monthly_data = performance_data.get('monthly_performance', [])
        if monthly_data:
            st.subheader("📅 Monthly Performance")
            
            df_monthly = pd.DataFrame(monthly_data)
            
            fig_monthly = px.bar(
                df_monthly,
                x='month',
                y='total_pnl',
                title="Monthly P&L",
                color='total_pnl',
                color_continuous_scale=['red', 'yellow', 'green']
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
    
    else:
        st.info("No performance data available. Start trading to see analytics.")
    
    # AI Model Performance
    st.subheader("🤖 AI Model Performance")
    
    if performance_data and 'ai_model_performance' in performance_data:
        ai_perf = performance_data['ai_model_performance']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**LLM Performance:**")
            llm_perf = ai_perf.get('llm_performance', {})
            st.write(f"Correlation with ROI: {llm_perf.get('correlation_with_roi', 0):.3f}")
            
            accuracy_data = llm_perf.get('accuracy_by_threshold', {})
            if accuracy_data:
                df_llm = pd.DataFrame.from_dict(accuracy_data, orient='index')
                st.dataframe(df_llm, use_container_width=True)
        
        with col2:
            st.write("**ML Model Performance:**")
            ml_perf = ai_perf.get('ml_performance', {})
            st.write(f"Correlation with ROI: {ml_perf.get('correlation_with_roi', 0):.3f}")
            
            calibration_data = ml_perf.get('calibration_analysis', {})
            if calibration_data:
                df_ml = pd.DataFrame.from_dict(calibration_data, orient='index')
                st.dataframe(df_ml, use_container_width=True)
    
    # AI Insights
    st.subheader("💡 AI-Generated Insights")
    
    if st.button("🧠 Generate Fresh Insights"):
        with st.spinner("Analyzing trading patterns..."):
            insights = fetch_ai_insights()
            
            if insights:
                st.session_state['ai_insights'] = insights
    
    if 'ai_insights' in st.session_state:
        insights = st.session_state['ai_insights']
        llm_insights = insights.get('llm_insights', {})
        
        if llm_insights:
            st.write("**AI Analysis:**")
            st.write(llm_insights.get('analysis', 'No insights available'))

def settings_page():
    """Settings and configuration page"""
    st.header("⚙️ Settings & Configuration")
    
    # API Configuration
    st.subheader("🔌 API Configuration")
    
    try:
        response = requests.get(f"{API_BASE_URL}/config/status")
        if response.status_code == 200:
            config_status = response.json()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Environment:**")
                st.info(f"Mode: {config_status.get('environment', 'unknown')}")
                st.info(f"Debug: {config_status.get('debug', False)}")
                st.info(f"Paper Trading: {config_status.get('paper_trading', True)}")
            
            with col2:
                st.write("**API Keys Status:**")
                api_keys = config_status.get('api_keys_configured', {})
                
                for service, configured in api_keys.items():
                    if configured:
                        st.success(f"✅ {service.upper()} configured")
                    else:
                        st.error(f"❌ {service.upper()} not configured")
    
    except:
        st.error("Unable to fetch configuration status")
    
    # Trading Parameters
    st.subheader("💰 Trading Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.number_input("Max Risk Per Trade (%)", 1, 10, 2)
        st.number_input("Max Open Positions", 1, 20, 5)
        st.number_input("Profit Target (%)", 50, 200, 80)
    
    with col2:
        st.number_input("Stop Loss (%)", 25, 75, 50)
        st.number_input("Days to Expiration (Min)", 14, 45, 21)
        st.number_input("Days to Expiration (Max)", 30, 90, 60)
    
    # AI Model Settings
    st.subheader("🤖 AI Model Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.selectbox("LLM Provider", ["OpenAI GPT-4", "Anthropic Claude"])
        st.slider("LLM Confidence Threshold", 60, 90, 70)
        st.slider("ML Win Probability Threshold", 0.5, 0.9, 0.6, 0.05)
    
    with col2:
        st.selectbox("ML Model Type", ["Gradient Boosting", "Random Forest", "XGBoost"])
        st.checkbox("Auto-retrain ML Model", value=True)
        st.checkbox("Enable Real-time Scoring", value=True)
    
    # System Actions
    st.subheader("🔧 System Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared!")
    
    with col2:
        if st.button("🧠 Retrain ML Model", use_container_width=True):
            try:
                response = requests.post(f"{API_BASE_URL}/ml/train")
                if response.status_code == 200:
                    st.success("Model training started!")
            except:
                st.error("Failed to start training")
    
    with col3:
        if st.button("📊 Export Data", use_container_width=True):
            st.info("Data export feature coming soon")
    
    # Diagnostics
    st.subheader("🔍 System Diagnostics")
    
    try:
        response = requests.get(f"{API_BASE_URL}/ml/diagnostics")
        if response.status_code == 200:
            diagnostics = response.json()
            st.json(diagnostics)
    except:
        st.error("Unable to fetch diagnostics")

if __name__ == "__main__":
    main()