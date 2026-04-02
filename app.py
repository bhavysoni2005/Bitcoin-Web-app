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
import time

warnings.filterwarnings('ignore')

# ================= HELPER FUNCTIONS =================
# FIXED: Add safe_float() helper to handle type conversions safely
def safe_float(value, default=0.0):
    """
    Safely convert any value to Python float.
    Handles Series, numpy scalars, None, NaN, etc.
    """
    try:
        # If it's a pandas Series, extract scalar
        if isinstance(value, pd.Series):
            if len(value) == 0:
                return default
            value = value.iloc[0]
        
        # If it's a numpy array, extract first element
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            value = value.flat[0]
        
        # If it has .item() method (numpy scalar), use it
        if hasattr(value, 'item'):
            value = value.item()
        
        # Handle None and NaN
        if value is None:
            return default
        if pd.isna(value):
            return default
        
        # Convert to float
        return float(value)
    except (TypeError, ValueError, AttributeError):
        return default

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
    padding: 20px;
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
    margin-top: 50px;
    margin-bottom: 50px;
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
def fetch_data(days, max_retries=3, backoff_factor=2):
    """Fetch Bitcoin data from yfinance with retry logic and proper error handling."""
    
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            
            df = yf.download(
                "BTC-USD",
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False
            )

            # ✅ Check empty data
            if df is None or df.empty:
                st.warning("No data retrieved. Retrying...")
                retry_count += 1
                time.sleep(backoff_factor ** retry_count)
                continue

            # ✅ FIX: Flatten multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # ✅ Reset index if needed
            if 'Date' not in df.columns:
                df.reset_index(inplace=True)

            # ✅ FIX: Ensure Close column exists
            if 'Close' not in df.columns:
                st.error(f"Missing 'Close' column. Columns found: {list(df.columns)}")
                return None

            # ✅ Ensure Close is 1D
            if isinstance(df['Close'], pd.DataFrame):
                df['Close'] = df['Close'].iloc[:, 0]

            # ✅ Convert numeric columns
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # ✅ Drop NaN Close values
            df = df.dropna(subset=['Close'])

            if len(df) < 20:
                st.warning(f"Not enough data ({len(df)}). Retrying...")
                retry_count += 1
                time.sleep(backoff_factor ** retry_count)
                continue

            return df

        # ❌ REMOVED broken yf.utils.TickerMissingError
        except Exception as e:
            last_error = e
            retry_count += 1

            if retry_count < max_retries:
                wait_time = backoff_factor ** retry_count
                st.warning(f"API error ({retry_count}/{max_retries}): {str(e)}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                st.error(f"Failed after {max_retries} retries: {last_error}")
                return None

    return None
# ================= TECHNICAL INDICATORS =================
# FIXED: Add better error handling and ensure proper NaN handling
def calculate_indicators(data):
    """Calculate technical indicators with proper numpy handling."""
    if data is None or data.empty:
        return None
    
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
        
        # FIXED: Use proper NaN handling (fillna with method parameter deprecated in newer pandas)
        df = df.bfill().ffill()  # backward fill then forward fill
        
        return df
    except Exception as e:
        st.warning(f"Error calculating indicators: {e}")
        return data

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index with proper error handling."""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # FIXED: Avoid division by zero
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        st.warning(f"Error calculating RSI: {e}")
        return prices * 0 + 50  # Return neutral RSI

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator with error handling."""
    try:
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
    except Exception as e:
        st.warning(f"Error calculating MACD: {e}")
        return {
            'macd': prices * 0,
            'signal_line': prices * 0,
            'histogram': prices * 0
        }

# ================= FEATURE ENGINEERING =================
# FIXED: Use safe_float() for proper type conversions and handle edge cases
def prepare_features(data, window=10):
    """Prepare features for ML model with proper type handling."""
    if data is None or data.empty:
        return None, None
    
    try:
        # FIXED: Ensure Close is 1D array
        close_values = data['Close'].values
        if isinstance(close_values, pd.DataFrame):
            close_values = close_values.iloc[:, 0].values
        close = np.array([safe_float(v) for v in close_values.flat], dtype=np.float64)
        
        # FIXED: Handle Volume column - use 1.0 if not available
        if 'Volume' in data.columns:
            volume_values = data['Volume'].values
            if isinstance(volume_values, pd.DataFrame):
                volume_values = volume_values.iloc[:, 0].values
            volume = np.array([safe_float(v) for v in volume_values.flat], dtype=np.float64)
        else:
            volume = np.ones(len(close), dtype=np.float64)
        
        # FIXED: Remove NaN/Inf values
        valid_idx = np.isfinite(close) & np.isfinite(volume)
        close = close[valid_idx]
        volume = volume[valid_idx]
        
        if len(close) < window + 2:
            return None, None
        
        X, y = [], []
        
        for i in range(len(close) - window - 1):
            cw = close[i:i+window]
            vol = volume[i:i+window]
            
            # FIXED: Use safe_float() for all conversions
            mean_price = safe_float(np.mean(cw))
            std_price = safe_float(np.std(cw))
            max_price = safe_float(np.max(cw))
            min_price = safe_float(np.min(cw))
            mean_vol = safe_float(np.mean(vol))
            
            # Percentage change using safe_float
            start_price = safe_float(cw[0])
            end_price = safe_float(close[i+window])
            pct_change = ((end_price - start_price) / max(start_price, 1e-10) * 100)
            
            X.append([mean_price, std_price, max_price, min_price, mean_vol, pct_change])
            y.append(end_price)
        
        if len(X) == 0:
            return None, None
        
        return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)
    except Exception as e:
        st.warning(f"Error preparing features: {e}")
        return None, None

# ================= MODEL TRAINING =================
# FIXED: Add better error handling and use safe_float for metrics
def train_model(X, y):
    """Train RandomForest model and return metrics."""
    try:
        if X is None or y is None or len(X) < 50:
            return None, None, None, None
        
        # FIXED: Remove NaN/Inf values from training data
        valid_idx = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]
        
        if len(X_clean) < 50:
            st.warning(f"Not enough valid training samples: {len(X_clean)}")
            return None, None, None, None
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, 
            test_size=0.2, 
            random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=15, 
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # FIXED: Ensure proper conversion to Python scalars using safe_float
        mse_value = mean_squared_error(y_test, y_pred)
        rmse = safe_float(np.sqrt(mse_value))
        r2_value = r2_score(y_test, y_pred)
        r2 = safe_float(r2_value)
        train_samples = int(X_train.shape[0])
        
        return model, rmse, r2, train_samples
    except Exception as e:
        st.warning(f"Error training model: {e}")
        return None, None, None, None

# ================= PREDICTION =================
# FIXED: Use safe_float for all conversions and handle edge cases
def predict_prices(model, data, days=7):
    """Predict future prices with proper type handling."""
    if model is None or data is None or data.empty:
        return None
    
    try:
        # FIXED: Ensure Close is 1D array
        close_values = data['Close'].values
        if isinstance(close_values, pd.DataFrame):
            close_values = close_values.iloc[:, 0].values
        close = np.array([safe_float(v) for v in close_values.flat], dtype=np.float64)
        
        # FIXED: Handle Volume column safely
        if 'Volume' in data.columns:
            volume_values = data['Volume'].values
            if isinstance(volume_values, pd.DataFrame):
                volume_values = volume_values.iloc[:, 0].values
            volume = np.array([safe_float(v, 1.0) for v in volume_values.flat], dtype=np.float64)
        else:
            volume = np.ones(len(close), dtype=np.float64)
        
        # FIXED: Convert to Python lists for manipulation
        close = close.tolist()
        volume = volume.tolist()
        
        if len(close) < 10:
            st.warning("Not enough historical data for prediction")
            return None
        
        predictions = []
        
        for _ in range(days):
            # FIXED: Get last 10 values safely
            cw = close[-10:] if len(close) >= 10 else close
            vol = volume[-10:] if len(volume) >= 10 else volume
            
            # FIXED: Create features with safe_float and use max() to prevent division by zero
            features_array = np.array([
                safe_float(np.mean(cw)),
                safe_float(np.std(cw)),
                safe_float(np.max(cw)),
                safe_float(np.min(cw)),
                safe_float(np.mean(vol)),
                (safe_float(close[-1]) - safe_float(cw[0])) / max(safe_float(cw[0]), 1e-10) * 100
            ], dtype=np.float64)
            
            # FIXED: Make prediction and convert result explicitly
            pred_result = model.predict(features_array.reshape(1, -1))[0]
            pred = safe_float(pred_result)
            
            # FIXED: Validate prediction is reasonable (not NaN or extreme)
            if not np.isfinite(pred) or pred <= 0:
                st.warning("Invalid prediction generated")
                return None
            
            predictions.append(pred)
            close.append(pred)
            volume.append(volume[-1] if volume else 1.0)
        
        return predictions
    except Exception as e:
        st.warning(f"Error predicting prices: {e}")
        import traceback
        st.warning(f"Traceback: {traceback.format_exc()}")
        return None

# ================= TRADING SIGNALS =================
# FIXED: Add better error handling for signal generation
def generate_signals(data):
    """Generate trading signals using multiple indicators."""
    if data is None or data.empty:
        return None
    
    try:
        df = data.copy()
        df['Signal'] = 0
        
        # FIXED: Add NaN checks for MA signals
        ma_signal = np.where(df['MA20'].notna() & df['MA50'].notna() & (df['MA20'] > df['MA50']), 1, -1)
        ma_signal = np.where(df['MA20'].isna() | df['MA50'].isna(), 0, ma_signal)
        
        # FIXED: Add NaN checks for RSI signals
        rsi_signal = np.where(df['RSI'].notna() & (df['RSI'] < 30), 1, 
                             np.where(df['RSI'].notna() & (df['RSI'] > 70), -1, 0))
        rsi_signal = np.where(df['RSI'].isna(), 0, rsi_signal)
        
        # FIXED: Add NaN checks for MACD signals
        mask_buy = (df['MACD'].notna() & df['Signal_Line'].notna() & 
                    (df['MACD'] > df['Signal_Line']) & 
                    (df['MACD'].shift(1).notna() & df['Signal_Line'].shift(1).notna()) &
                    (df['MACD'].shift(1) <= df['Signal_Line'].shift(1)))
        
        mask_sell = (df['MACD'].notna() & df['Signal_Line'].notna() & 
                     (df['MACD'] < df['Signal_Line']) & 
                     (df['MACD'].shift(1).notna() & df['Signal_Line'].shift(1).notna()) &
                     (df['MACD'].shift(1) >= df['Signal_Line'].shift(1)))
        
        macd_signal = np.where(mask_buy, 1, np.where(mask_sell, -1, 0))
        
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
        
        # FIXED: Fetch Data with proper error handling
        data = fetch_data(days_history)
        
        if data is None or data.empty:
            st.error("❌ Failed to fetch data. Please refresh the page or try again later.")
            st.info("This may be due to Yahoo Finance rate limiting. The app will retry automatically.")
            return
        
        st.info(f"📅 Data points: {len(data)}\n\n⏱️ Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # FIXED: Calculate Indicators with proper error handling
    data = calculate_indicators(data)
    
    if data is None or data.empty:
        st.error("Failed to calculate indicators")
        return
    
    data = generate_signals(data)
    
    if data is None or data.empty or len(data) < 50:
        st.error("Not enough data for analysis (minimum 50 data points required)")
        return
    
    # FIXED: Get Latest Values using safe_float to handle Series/scalar issues
    try:
        # FIXED: Use safe_float with .iloc to prevent Series conversion issues
        current_price = safe_float(data['Close'].iloc[-1], 0)
        prev_price = safe_float(data['Close'].iloc[-2], current_price)
        
        if current_price <= 0 or prev_price <= 0:
            st.error("Invalid price data")
            return
        
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price * 100) if prev_price != 0 else 0.0
        volatility = safe_float(data['Volatility'].iloc[-1], 0.0)
        volatility = max(volatility, 0.0)  # Ensure non-negative
        rsi_value = safe_float(data['RSI'].iloc[-1], 50.0)
        rsi_value = np.clip(rsi_value, 0, 100)  # Ensure RSI is 0-100
        
    except Exception as e:
        st.error(f"Error extracting current metrics: {e}")
        return
    
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
    
    # FIXED: Use safe_float for Signal value extraction
    try:
        latest_signal = int(safe_float(data['Signal'].iloc[-1], 0))
    except Exception:
        latest_signal = 0
    
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
    
    # FIXED: Add error handling for chart generation
    try:
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
    except Exception as e:
        st.error(f"Error generating price chart: {e}")
    
    # RSI Chart
    try:
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
    except Exception as e:
        st.error(f"Error generating RSI chart: {e}")
    
    # MACD Chart
    try:
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
    except Exception as e:
        st.error(f"Error generating MACD chart: {e}")
    
    st.divider()
    
    # Machine Learning Section
    st.markdown("<div class='section-title'>🤖 Machine Learning Model</div>", unsafe_allow_html=True)
    
    with st.spinner("Training model..."):
        X, y = prepare_features(data)
        
        if X is not None and y is not None:
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
                            # FIXED: Work with clean data - drop NaN values
                            clean_data = data[['Date', 'Close']].dropna()
                            
                            if clean_data.empty:
                                st.warning("No clean data available for predictions")
                            else:
                                # FIXED: Convert dates to datetime
                                hist_dates = pd.to_datetime(clean_data['Date']).values
                                hist_prices = np.array([safe_float(p) for p in clean_data['Close'].values], dtype=np.float64)
                                
                                # Generate future dates - use the last date from original data
                                last_date = pd.to_datetime(safe_float(data['Date'].iloc[-1]) if isinstance(data['Date'].iloc[-1], (int, float)) else data['Date'].iloc[-1])
                                future_dates = [last_date + timedelta(days=i+1) for i in range(len(predictions))]
                                
                                # FIXED: Ensure predictions are valid floats and filter out NaN/Inf
                                pred_prices = [float(p) for p in predictions if p is not None and np.isfinite(p)]
                                
                                if not pred_prices:
                                    st.warning("No valid predictions generated")
                                else:
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
                                            f"{((predicted_avg / max(current_price, 1e-10) - 1) * 100):+.2f}%"
                                        )
                                    
                                    with col3:
                                        predicted_high = float(np.max(pred_prices)) if pred_prices else current_price
                                        st.metric(
                                            "Highest Predicted",
                                            f"${predicted_high:,.0f}",
                                            f"{((predicted_high / max(current_price, 1e-10) - 1) * 100):+.2f}%"
                                        )
                                    
                                    with col4:
                                        predicted_low = float(np.min(pred_prices)) if pred_prices else current_price
                                        st.metric(
                                            "Lowest Predicted",
                                            f"${predicted_low:,.0f}",
                                            f"{((predicted_low / max(current_price, 1e-10) - 1) * 100):+.2f}%"
                                        )
                                    
                                    # ============= PREDICTION TABLE =============
                                    try:
                                        pred_df = pd.DataFrame({
                                            'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
                                            'Predicted Price': [f"${p:,.2f}" for p in pred_prices],
                                            'Change from Today': [f"{((p / max(current_price, 1e-10) - 1) * 100):+.2f}%" for p in pred_prices]
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
        else:
            st.warning("⚠️ Unable to prepare features for model training. Please ensure you have sufficient data.")
    
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
