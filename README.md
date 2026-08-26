# ✈️ AeroGuardian

### Aircraft Remaining Useful Life Prediction

AeroGuardian is a predictive-maintenance system that predicts the **Remaining Useful Life (RUL)** of an aircraft engine using historical sensor data.

---

## 🎯 Problem Statement

Predict how many operating cycles an aircraft engine has left before failure based on its recent sensor measurements.

**Problem Type:** Supervised Deep Learning Regression

---

## 📊 Dataset

**NASA C-MAPSS FD001** — simulated turbofan engine run-to-failure sensor data.

---

## 🧠 Model

**CNN + LSTM + Attention**

The model takes the latest **30 engine cycles** and predicts the engine's remaining useful life.

### CNN
Extracts **local patterns and short-term changes** from the sensor sequence.

### LSTM
Learns **how the engine condition changes over time** and captures temporal dependencies.

### Attention
Identifies **which timesteps in the 30-cycle sequence are more important** for the RUL prediction.

### Dense Layer
Converts the learned representation into the final **continuous RUL value**.
---

## 🔄 Process

