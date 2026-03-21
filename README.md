🚀 Bitcoin Price Prediction Dashboard

A machine learning-powered Streamlit web application for real-time Bitcoin price analysis and future price prediction using historical market data.


📌 Overview
--This project provides an interactive trading-style dashboard that:
-Fetches real-time Bitcoin data
-Performs technical analysis
-Predicts future prices using Machine Learning
-Displays results with modern dark UI and interactive charts


✨ Features
--📊 Data & Analysis
-Real-time Bitcoin data using yfinance
-Historical data visualization
-Moving Averages (MA20, MA50)
-Volatility calculation

--🤖 Machine Learning
-Model: Random Forest Regressor
-Rolling window feature engineering
-Future price prediction (1–30 days)
-Handles edge cases (NaN, scaling, data issues)

--📈 Visualization
-Interactive Plotly charts
-Historical vs Predicted price comparison
-Dark-themed trading dashboard
-Smooth UI with hover effects

--🎨 UI/UX
-Fully responsive Streamlit UI
-Dark mode (trading-style theme)
-Animated metric cards
-Sidebar controls

--👨‍💻 Portfolio Integration
-Resume download button
-GitHub stats integration
-LinkedIn, GitHub, Portfolio links
-Professional footer section

🛠️ Tech Stack
-Frontend/UI: Streamlit
-Data Processing: Pandas, NumPy
-Visualization: Plotly
-Machine Learning: Scikit-learn (Random Forest)
-Data Source: Yahoo Finance (yfinance)

📂 Project Structure
 📦 Bitcoin-Price-Predictor
 ┣ 📜 app.py
 ┣ 📜 requirements.txt
 ┣ 📄 resume.pdf (optional)
 ┗ 📄 README.md

⚙️ Installation
  1️⃣ Clone the Repository
     git clone https://github.com/your-username/bitcoin-price-predictor.git
     cd bitcoin-price-predictor
  2️⃣ Install Dependencies
     pip install -r requirements.txt
  3️⃣ Run the App
     streamlit run app.py

📊 How It Works
  Fetches historical Bitcoin price data
  Applies feature engineering (rolling window, indicators)
  Trains ML model on historical patterns
  Predicts future prices iteratively
  Displays results in interactive dashboard

📈 Sample Output
  📉 Historical price trends
  🔮 Predicted future prices
  📊 Market metrics (price, volatility, change)

⚠️ Disclaimer
  This project is for educational purposes only.It does NOT provide financial advice.Cryptocurrency    markets are highly volatile.

👨‍💻 Author
Bhavy Soni

  🔗 GitHub: https://github.com/bhavysoni2005
  💼 LinkedIn: https://www.linkedin.com/in/bhavy-soni-b3746b316
  🌐 Portfolio: https://bhavysoni.netlify.app
  📧 Email: bhavysoni2005@gmail.com

🌟 Future Improvements
  LSTM / Deep Learning model
  Live trading signals (Buy/Sell)
  API integration (Binance)
  Deployment on Streamlit Cloud
  Advanced indicators (RSI, MACD)

⭐ Support
If you like this project:
 ⭐ Star the repo
 🍴 Fork it
 🛠️ Contribute

 
📜 License
  This project is open-source and available under the MIT License.
