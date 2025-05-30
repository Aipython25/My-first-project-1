# My-first-project-1
Aipython25/My-first-project
Google colab work to stock price prediction 

# Step 1: ज़रूरी लाइब्रेरी इम्पोर्ट करें
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Step 2: स्टॉक डेटा डाउनलोड करें (Apple का डेटा)
data = yf.download('AAPL', start='2022-01-01', end='2023-01-01')
data = data[['Close']]  # सिर्फ Close प्राइस

# Step 3: डेटा को प्रेपेयर करें
data['Prediction'] = data[['Close']].shift(-1)  # अगला दिन प्रेडिक्ट करना है
X = data.drop(['Prediction'], axis=1)[:-1]
y = data['Prediction'][:-1]

# Step 4: मॉडल बनाएं और ट्रेन करें
model = LinearRegression()
model.fit(X, y)

# Step 5: Prediction करें (अगले दिन का प्राइस)
future_price = model.predict(X[-1:].values.reshape(1, -1))
print("📈 Next day predicted price:", round(future_price[0], 2))

# Step 6: Visualization (ग्राफ बनाएं)
plt.figure(figsize=(10,5))
plt.plot(data['Close'], label='Actual Price')
plt.title("Stock Price History (AAPL)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.show()