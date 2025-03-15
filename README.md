# Apple Stock Anomaly Detection – Identifying Pump-and-Dump Cases

## Project Overview
This project focuses on **anomaly detection in Apple (AAPL) stock prices**, specifically to identify **pump-and-dump cases** or unusual trading patterns. By leveraging **unsupervised learning techniques**, we highlight potential stock manipulations using multiple statistical and AI-based approaches.

**Disclaimer:** This is a **student project** and is not meant for real-world financial decision-making. While it demonstrates anomaly detection techniques, it does not catch all anomalies accurately based on real-world data.

## Methodology
### Data Collection & Preprocessing
- Collected **10+ years of Apple stock data** from Yahoo Finance using `yfinance`.
- Cleaned data, handled missing values, and converted necessary columns.
- Conducted **Exploratory Data Analysis (EDA)** with trend visualizations.

### Technical Indicators for Trend Analysis
- **Moving Averages**: SMA (20-day, 50-day) & EMA (20-day) to detect trends.
- **Volatility Analysis**: Calculated **daily returns** and 20-day rolling volatility.
- **Bollinger Bands**: Identified overbought/oversold conditions.

### Anomaly Detection Techniques & Why They Were Used
This project explores multiple methods to detect stock price anomalies:

1️⃣ **Z-Score Method**: A simple statistical method that flags anomalies when data points deviate more than **3 standard deviations** from the mean. It's useful for quick detection but struggles with non-Gaussian distributions.

2️⃣ **Isolation Forest**: A tree-based model that isolates anomalies by splitting data points in a random manner. It works well for **high-dimensional data** and is widely used in financial fraud detection.

3️⃣ **Local Outlier Factor (LOF)**: This method identifies anomalies based on how isolated a data point is compared to its neighbors. It is effective in **detecting local deviations**, such as sudden price spikes.

4️⃣ **DBSCAN (Density-Based Clustering)**: Finds anomalies by looking at data density. Data points that don’t belong to any dense cluster are flagged as anomalies. This is useful for **detecting irregular trading patterns**.

5️⃣ **Hybrid Model (IF + LOF)**: Combines **Isolation Forest and LOF** to flag anomalies detected by both methods, improving accuracy by reducing false positives.

### AI Model – LSTM & Autoencoder for Deep Learning Anomaly Detection
To explore more advanced anomaly detection, we implemented an **LSTM Autoencoder**:

🔹 **LSTM (Long Short-Term Memory)**: A type of recurrent neural network (RNN) that captures temporal dependencies in stock prices. It helps in predicting stock price behavior over time.

🔹 **Autoencoder for Anomaly Detection**:
   - The model learns to **reconstruct stock price patterns**.
   - If the reconstruction error is **too high**, it signals an anomaly.
   - We applied **Median Absolute Deviation (MAD) and normalized thresholding** to classify anomalies.

🔹 **LSTM Code Structure:**
   - **Preprocessing:** Data is normalized using `MinMaxScaler`.
   - **Sequence Creation:** A time window of **30 days** is used to feed stock price movements into the model.
   - **LSTM Autoencoder Architecture:**
     - **Encoder:** Two LSTM layers compress data into a smaller representation.
     - **Decoder:** Two LSTM layers reconstruct the original data.
     - **Reconstruction Error:** Anomalies are flagged if the difference between actual and predicted prices is too large.
   - **Loss Function:** Mean Squared Error (MSE) is used to measure reconstruction accuracy.

### Validation: Real Stock Events
- Matched detected anomalies with **historical stock events** (earnings reports, product launches, regulatory changes, market crashes).
- Used **Yahoo Finance news data** for validation.

## How to Run the Project
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- Jupyter Notebook / Google Colab
- Required libraries:
  ```bash
  pip install pandas numpy matplotlib seaborn yfinance scipy scikit-learn tensorflow keras
  ```
  Additional libraries used:
  - `matplotlib.dates` for date formatting
  - `scipy.stats` for statistical calculations (Z-Score, MAD)
  - `sklearn.ensemble` for Isolation Forest
  - `sklearn.neighbors` for Local Outlier Factor
  - `sklearn.preprocessing` for data scaling
  - `sklearn.cluster` for DBSCAN

### Installation & Execution
1. Clone the repository:
   ```bash
   git clone https://github.com/SS8816/Apple-Stock-Anomaly-Detection-Identifying-Pump-and-Dump-Cases.git
   cd apple-stock-anomaly-detection
   ```
2. Install dependencies manually
   ```bash
   pip install pandas numpy matplotlib seaborn yfinance scipy scikit-learn tensorflow keras
   ```
3. Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook
   ```
   OR
   ```bash
   python anomaly_detection.py
   ```

## Results & Insights
- Detected anomalies in stock prices and volume patterns.
- Identified **potential pump-and-dump activities**.
- Compared **traditional statistical methods vs. AI-based approaches**.

---
📌 *Feel free to fork, contribute, or drop your feedback!* ⭐


