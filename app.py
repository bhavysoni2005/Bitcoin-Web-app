import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Bitcoin Trading Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= DARK THEME & CUSTOM CSS =================
st.markdown("""
<style>
* {
    margin: 0;
    padding: 0;
}

body, .stApp {
    background-color: #0e1117;
    color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.main {
    background-color: #0e1117;
}

.stSidebar {
    background-color: #161b22;
}

.metric-card {
    background: linear-gradient(135deg, #1c1f26 0%, #161b22 100%);
    padding: 25px;
    border-radius: 12px;
    border: 1px solid #30363d;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.metric-card:hover {
    transform: translateY(-5px);
    border-color: #00d9ff;
    box-shadow: 0 10px 20px rgba(0,217,255,0.15);
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 30px;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 2px solid #00d9ff;
    color: #00d9ff;
}

.signal-card-buy {
    background: linear-gradient(135deg, rgba(0,200,83,0.1) 0%, rgba(0,200,83,0.05) 100%);
    border: 2px solid #00c853;
    padding: 25px;
    border-radius: 12px;
    text-align: center;
}

.signal-card-sell {
    background: linear-gradient(135deg, rgba(255,82,82,0.1) 0%, rgba(255,82,82,0.05) 100%);
    border: 2px solid #ff5252;
    padding: 25px;
    border-radius: 12px;
    text-align: center;
}

.signal-card-hold {
    background: linear-gradient(135deg, rgba(139,148,158,0.1) 0%, rgba(139,148,158,0.05) 100%);
    border: 2px solid #8b949e;
    padding: 25px;
    border-radius: 12px;
    text-align: center;
}

.disclaimer {
    background: #161b22;
    border-left: 4px solid #f0883e;
    padding: 15px;
    margin-top: 30px;
    border-radius: 8px;
    font-size: 12px;
    color: #8b949e;
}

.footer {
    text-align: center;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid #30363d;
    color: #8b949e;
    font-size: 13px;
}

.footer p {
    margin: 8px 0;
    line-height: 1.6;
}

.footer p:nth-child(2) {
    color: #f0883e;
    font-weight: 600;
    font-size: 14px;
}

.stMetric {
    background: transparent;
}

h1, h2, h3, h4, h5, h6 {
    color: #e6edf3;
}

.stButton > button {
    background-color: #238636;
    color: white;
    border-radius: 6px;
    border: 1px solid #238636;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background-color: #2ea043;
    border-color: #00d9ff;
}

.stSelectbox, .stSlider, .stNumberInput {
    color: #e6edf3;
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #161b22;
}

::-webkit-scrollbar-thumb {
    background: #30363d;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #00d9ff;
}
</style>
""", unsafe_allow_html=True)

# ================= DATA FETCHING =================
@st.cache_data(ttl=3600)
def fetch_data(days):
    """Fetch Bitcoin data from yfinance with proper error handling."""
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        df = yf.download("BTC-USD", start=start, end=end, progress=False)
        
        if df.empty:
            return None
        
        df.reset_index(inplace=True)
        
        # Convert only columns that exist
        dtype_map = {}
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
            if col in df.columns:
                dtype_map[col] = 'float64'
        
        if dtype_map:
            df = df.astype(dtype_map)
        
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# ================= TECHNICAL INDICATORS =================
def calculate_indicators(data):
    """Calculate technical indicators with proper numpy handling."""
    df = data.copy()
    
    try:
        # Moving Averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # Volatility (Rolling Std)
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # RSI (14)
        df['RSI'] = calculate_rsi(df['Close'], period=14)
        
        # MACD
        macd_result = calculate_macd(df['Close'])
        df['MACD'] = macd_result['macd']
        df['Signal_Line'] = macd_result['signal_line']
        df['MACD_Histogram'] = macd_result['histogram']
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    except Exception as e:
        st.warning(f"Error calculating indicators: {e}")
        return data

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    
    return {
        'macd': macd,
        'signal_line': signal_line,
        'histogram': histogram
    }

# ================= FEATURE ENGINEERING =================
def prepare_features(data, window=10):
    """Prepare features for ML model with proper type handling."""
    try:
        close = data['Close'].values.astype(np.float64)
        
        # Handle volume column - use 1.0 if not available
        if 'Volume' in data.columns:
            volume = data['Volume'].values.astype(np.float64)
        else:
            volume = np.ones(len(close), dtype=np.float64)
        
        X, y = [], []
        
        for i in range(len(close) - window - 1):
            cw = close[i:i+window]
            vol = volume[i:i+window]
            
            # Use .item() to ensure proper conversion from numpy scalar to Python scalar
            mean_price = float(np.mean(cw).item() if hasattr(np.mean(cw), 'item') else np.mean(cw))
            std_price = float(np.std(cw).item() if hasattr(np.std(cw), 'item') else np.std(cw))
            max_price = float(np.max(cw).item() if hasattr(np.max(cw), 'item') else np.max(cw))
            min_price = float(np.min(cw).item() if hasattr(np.min(cw), 'item') else np.min(cw))
            mean_vol = float(np.mean(vol).item() if hasattr(np.mean(vol), 'item') else np.mean(vol))
            
            # Percentage change - use .item() for scalar conversion
            start_price = float(cw[0].item() if hasattr(cw[0], 'item') else cw[0])
            end_price = float(close[i+window].item() if hasattr(close[i+window], 'item') else close[i+window])
            
            pct_change = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0.0
            
            X.append([mean_price, std_price, max_price, min_price, mean_vol, pct_change])
            y.append(end_price)
        
        if len(X) == 0:
            return None, None
        
        return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)
    except Exception as e:
        st.warning(f"Error preparing features: {e}")
        return None, None

# ================= MODEL TRAINING =================
def train_model(X, y):
    """Train RandomForest model and return metrics."""
    try:
        if X is None or len(X) < 50:
            return None, None, None, None
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics - ensure proper conversion to Python scalars
        mse_value = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse_value).item() if hasattr(np.sqrt(mse_value), 'item') else np.sqrt(mse_value))
        r2_value = r2_score(y_test, y_pred)
        r2 = float(r2_value.item() if hasattr(r2_value, 'item') else r2_value)
        
        train_samples = int(X_train.shape[0])
        
        return model, rmse, r2, train_samples
    except Exception as e:
        st.warning(f"Error training model: {e}")
        return None, None, None, None

# ================= PREDICTION =================
def predict_prices(model, data, days=7):
    """Predict future prices with proper type handling."""
    try:
        if model is None:
            return None
        
        # Convert Close values to list of floats - handle numpy array properly
        close_values = data['Close'].values.astype(np.float64)
        close = [float(v) for v in close_values.flat]  # .flat flattens multidimensional arrays
        
        if 'Volume' in data.columns:
            volume_values = data['Volume'].values.astype(np.float64)
            volume = [float(v) for v in volume_values.flat]
        else:
            volume = [1.0] * len(close)
        
        predictions = []
        
        for _ in range(days):
            # Get last 10 values as lists
            cw = close[-10:]
            vol = volume[-10:]
            
            # Create numpy array for features explicitly
            features_array = np.array([
                float(np.mean(cw)),
                float(np.std(cw)),
                float(np.max(cw)),
                float(np.min(cw)),
                float(np.mean(vol)),
                ((float(close[-1]) - float(cw[0])) / float(cw[0]) * 100) if float(cw[0]) != 0 else 0.0
            ], dtype=np.float64)
            
            # Make prediction and convert result explicitly
            pred_result = model.predict(features_array.reshape(1, -1))[0]
            
            # Ensure we get a Python float, not numpy scalar
            if hasattr(pred_result, 'item'):
                pred = float(pred_result.item())
            else:
                pred = float(pred_result)
            
            predictions.append(pred)
            close.append(pred)
            volume.append(volume[-1])
        
        return predictions
    except Exception as e:
        st.warning(f"Error predicting prices: {e}")
        import traceback
        st.warning(f"Traceback: {traceback.format_exc()}")
        return None

# ================= TRADING SIGNALS =================
def generate_signals(data):
    """Generate trading signals using multiple indicators."""
    try:
        df = data.copy()
        df['Signal'] = 0
        
        # MA Crossover Signal
        ma_signal = np.where(df['MA20'] > df['MA50'], 1, -1)
        ma_signal = np.where(df['MA20'].isna() | df['MA50'].isna(), 0, ma_signal)
        
        # RSI Signal
        rsi_signal = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
        rsi_signal = np.where(df['RSI'].isna(), 0, rsi_signal)
        
        # MACD Signal
        macd_signal = np.where(
            (df['MACD'] > df['Signal_Line']) & (df['MACD'].shift(1) <= df['Signal_Line'].shift(1)),
            1,
            np.where(
                (df['MACD'] < df['Signal_Line']) & (df['MACD'].shift(1) >= df['Signal_Line'].shift(1)),
                -1,
                0
            )
        )
        macd_signal = np.where(df['MACD'].isna() | df['Signal_Line'].isna(), 0, macd_signal)
        
        # Combine signals (majority voting)
        combined = ma_signal + rsi_signal + macd_signal
        df['Signal'] = np.where(combined > 0.5, 1, np.where(combined < -0.5, -1, 0))
        
        return df
    except Exception as e:
        st.warning(f"Error generating signals: {e}")
        return data

# ================= MAIN APP =================
def main():
    # Header
    st.markdown(
        "<h1 style='text-align:center; color:#00d9ff; font-size:48px; margin-bottom:10px;'>₿ Bitcoin Trading Dashboard</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; color:#8b949e; font-size:14px;'>Advanced ML-Powered Price Prediction & Trading Signals</p>",
        unsafe_allow_html=True
    )
    
    st.divider()
    
    # Sidebar Controls
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        days_history = st.slider("Historical Data (days)", 30, 365, 180, step=30)
        prediction_days = st.slider("Prediction Days", 1, 30, 7)
        
        st.divider()
        st.markdown("### 📊 Data Info")
        
        # Fetch Data
        data = fetch_data(days_history)
        
        if data is None:
            st.error("Failed to fetch data")
            return
        
        st.info(f"📅 Data points: {len(data)}\n\n⏱️ Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Calculate Indicators
    data = calculate_indicators(data)
    data = generate_signals(data)
    
    if data is None or len(data) < 50:
        st.error("Not enough data for analysis")
        return
    
    # Get Latest Values
    current_price = float(data['Close'].iloc[-1])
    prev_price = float(data['Close'].iloc[-2])
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price * 100) if prev_price != 0 else 0.0
    volatility = float(data['Volatility'].iloc[-1]) if data['Volatility'].iloc[-1] > 0 else 0.0
    rsi_value = float(data['RSI'].iloc[-1]) if not pd.isna(data['RSI'].iloc[-1]) else 50.0
    
    # Metrics Row
    st.markdown("<div class='section-title'>📈 Current Metrics</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Current Price</h4>
            <h2 style='color:#00d9ff;'>${current_price:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        color = "#00c853" if price_change >= 0 else "#ff5252"
        st.markdown(f"""
        <div class='metric-card'>
            <h4>24h Change</h4>
            <h2 style='color:{color};'>{price_change:+.2f} ({price_change_pct:+.2f}%)</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Volatility (20d)</h4>
            <h2 style='color:#f0883e;'>${volatility:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        rsi_color = "#00c853" if rsi_value < 30 else "#ff5252" if rsi_value > 70 else "#8b949e"
        st.markdown(f"""
        <div class='metric-card'>
            <h4>RSI (14)</h4>
            <h2 style='color:{rsi_color};'>{rsi_value:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Trading Signal
    st.markdown("<div class='section-title'>🎯 Trading Signal</div>", unsafe_allow_html=True)
    
    latest_signal = int(data['Signal'].iloc[-1])
    
    if latest_signal == 1:
        signal_text = "BUY 🟢"
        signal_color = "#00c853"
        css_class = "signal-card-buy"
    elif latest_signal == -1:
        signal_text = "SELL 🔴"
        signal_color = "#ff5252"
        css_class = "signal-card-sell"
    else:
        signal_text = "HOLD ⏸️"
        signal_color = "#8b949e"
        css_class = "signal-card-hold"
    
    st.markdown(f"""
    <div class='{css_class}'>
        <h3 style='color:{signal_color}; margin-bottom:10px;'>{signal_text}</h3>
        <p style='color:#8b949e; font-size:12px;'>Based on MA Crossover, RSI, and MACD indicators</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Charts Section
    st.markdown("<div class='section-title'>📊 Price Chart & Indicators</div>", unsafe_allow_html=True)
    
    # Candlestick Chart
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=data['Date'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='BTC Price',
        increasing_line_color='#00c853',
        increasing_fillcolor='#00c853',
        decreasing_line_color='#ff5252',
        decreasing_fillcolor='#ff5252'
    ))
    
    # Moving Averages
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['MA20'],
        name='MA20',
        line=dict(color='#ffa500', width=2),
        hovertemplate='<b>MA20</b><br>%{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['MA50'],
        name='MA50',
        line=dict(color='#1e90ff', width=2),
        hovertemplate='<b>MA50</b><br>%{y:,.0f}<extra></extra>'
    ))
    
    # Buy Signals
    buy_signals = data[data['Signal'] == 1]
    if len(buy_signals) > 0:
        fig.add_trace(go.Scatter(
            x=buy_signals['Date'],
            y=buy_signals['Close'],
            mode='markers',
            marker=dict(symbol='triangle-up', color='#00c853', size=12, line=dict(color='#00c853', width=2)),
            name='BUY Signal',
            hovertemplate='<b>BUY</b><br>%{y:,.0f}<extra></extra>'
        ))
    
    # Sell Signals
    sell_signals = data[data['Signal'] == -1]
    if len(sell_signals) > 0:
        fig.add_trace(go.Scatter(
            x=sell_signals['Date'],
            y=sell_signals['Close'],
            mode='markers',
            marker=dict(symbol='triangle-down', color='#ff5252', size=12, line=dict(color='#ff5252', width=2)),
            name='SELL Signal',
            hovertemplate='<b>SELL</b><br>%{y:,.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        template="plotly_dark",
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='#0e1117',
        plot_bgcolor='#161b22',
        font=dict(color='#e6edf3', size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # RSI Chart
    fig_rsi = go.Figure()
    
    fig_rsi.add_trace(go.Scatter(
        x=data['Date'],
        y=data['RSI'],
        name='RSI (14)',
        line=dict(color='#00d9ff', width=2),
        hovertemplate='<b>RSI</b><br>%{y:.1f}<extra></extra>'
    ))
    
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    
    fig_rsi.update_layout(
        template="plotly_dark",
        height=300,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='#0e1117',
        plot_bgcolor='#161b22',
        font=dict(color='#e6edf3', size=12),
        yaxis=dict(range=[0, 100])
    )
    
    st.plotly_chart(fig_rsi, use_container_width=True)
    
    # MACD Chart
    fig_macd = go.Figure()
    
    fig_macd.add_trace(go.Scatter(
        x=data['Date'],
        y=data['MACD'],
        name='MACD',
        line=dict(color='#00d9ff', width=2),
        hovertemplate='<b>MACD</b><br>%{y:.4f}<extra></extra>'
    ))
    
    fig_macd.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Signal_Line'],
        name='Signal Line',
        line=dict(color='#ffa500', width=2),
        hovertemplate='<b>Signal</b><br>%{y:.4f}<extra></extra>'
    ))
    
    fig_macd.add_trace(go.Bar(
        x=data['Date'],
        y=data['MACD_Histogram'],
        name='Histogram',
        marker=dict(color=['#00c853' if x > 0 else '#ff5252' for x in data['MACD_Histogram']]),
        hovertemplate='<b>Histogram</b><br>%{y:.4f}<extra></extra>'
    ))
    
    fig_macd.update_layout(
        template="plotly_dark",
        height=300,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='#0e1117',
        plot_bgcolor='#161b22',
        font=dict(color='#e6edf3', size=12)
    )
    
    st.plotly_chart(fig_macd, use_container_width=True)
    
    st.divider()
    
    # Machine Learning Section
    st.markdown("<div class='section-title'>🤖 Machine Learning Model</div>", unsafe_allow_html=True)
    
    with st.spinner("Training model..."):
        X, y = prepare_features(data)
        
        if X is not None:
            model, rmse, r2, train_samples = train_model(X, y)
            
            if model is not None:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h4>Model Type</h4>
                        <h3>RandomForest</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h4>RMSE</h4>
                        <h3>${rmse:,.2f}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h4>R² Score</h4>
                        <h3>{r2:.4f}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.info(f"✅ Model trained on {train_samples} samples with 100 estimators")
                
                st.divider()
                
                # Prediction Section
                st.markdown("<div class='section-title'>🔮 Price Prediction</div>", unsafe_allow_html=True)
                
                with st.spinner("Generating predictions..."):
                    predictions = predict_prices(model, data, prediction_days)
                    
                    if predictions is not None and len(predictions) > 0:
                        try:
                            # ============= DATA PREPARATION =============
                            # Work with clean data - drop NaN values
                            clean_data = data[['Date', 'Close']].dropna()
                            
                            # Convert dates to datetime
                            hist_dates = pd.to_datetime(clean_data['Date']).values
                            hist_prices = clean_data['Close'].values.astype(np.float64)
                            
                            # Generate future dates - use the last date from original data
                            last_date = pd.to_datetime(data['Date'].iloc[-1])
                            future_dates = [last_date + timedelta(days=i+1) for i in range(len(predictions))]
                            
                            # Ensure predictions are valid floats and filter out NaN
                            pred_prices = [float(p) for p in predictions if p is not None]
                            
                            # Adjust future_dates to match predictions length
                            future_dates = future_dates[:len(pred_prices)]
                            
                            # ============= STATISTICS =============
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric(
                                    "Current Price",
                                    f"${current_price:,.0f}",
                                    f"{price_change_pct:+.2f}%"
                                )
                            
                            with col2:
                                predicted_avg = float(np.mean(pred_prices)) if pred_prices else current_price
                                st.metric(
                                    "Avg Predicted",
                                    f"${predicted_avg:,.0f}",
                                    f"{((predicted_avg / current_price - 1) * 100):+.2f}%"
                                )
                            
                            with col3:
                                predicted_high = float(np.max(pred_prices)) if pred_prices else current_price
                                st.metric(
                                    "Highest Predicted",
                                    f"${predicted_high:,.0f}",
                                    f"{((predicted_high / current_price - 1) * 100):+.2f}%"
                                )
                            
                            with col4:
                                predicted_low = float(np.min(pred_prices)) if pred_prices else current_price
                                st.metric(
                                    "Lowest Predicted",
                                    f"${predicted_low:,.0f}",
                                    f"{((predicted_low / current_price - 1) * 100):+.2f}%"
                                )
                            
                            # ============= PREDICTION TABLE =============
                            try:
                                pred_df = pd.DataFrame({
                                    'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
                                    'Predicted Price': [f"${p:,.2f}" for p in pred_prices],
                                    'Change from Today': [f"{((p / current_price - 1) * 100):+.2f}%" if current_price > 0 else "N/A" for p in pred_prices]
                                })
                                
                                st.markdown("**📋 Detailed Predictions:**")
                                st.dataframe(pred_df, use_container_width=True, hide_index=True)
                            except Exception as table_err:
                                st.warning(f"Could not display prediction table: {table_err}")
                                
                        except Exception as chart_err:
                            st.error(f"Error creating prediction chart: {chart_err}")
                            import traceback
                            st.text(traceback.format_exc())
                    else:
                        st.warning("⚠️ Could not generate predictions. Please try refreshing the page.")
    
    st.divider()
    
    # Disclaimer and Footer
    st.markdown("""
    <div class='disclaimer'>
        <strong>⚠️ Disclaimer:</strong> This dashboard is for educational and informational purposes only. 
        It is not financial advice. Cryptocurrency trading is highly risky. Always conduct your own research and 
        consult with a financial advisor before making investment decisions. Past performance does not guarantee 
        future results.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='footer'>
        <p>₿ Bitcoin Trading Dashboard | Powered by Streamlit, Scikit-Learn, and Plotly</p>
        <p>Only for educational use. Not financial advice.</p>
        <p>© 2026 | Built with ❤️ for traders and analysts</p>
        <p style='margin-top:15px; color:#30363d;'>
            📧 Contact | 🔗 <a href='https://github.com' target='_blank'>GitHub</a> | 
            💼 <a href='https://linkedin.com' target='_blank'>LinkedIn</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
