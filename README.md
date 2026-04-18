# Logistic Regression from Scratch (Python)

This repository contains a full implementation of **Logistic Regression** built from the ground up using Python.
It does not rely on machine learning libraries like `scikit-learn`, making it ideal for learning how the algorithm works internally.

---

## 🚀 Features

* Logistic regression using **gradient descent**
* **Binary classification**
* **Cross-entropy loss function**
* **L2 regularization (Ridge)**
* **Z-score feature normalization**
* Training visualization with cost vs iterations
* Accuracy evaluation on training data

---

## 📂 Dataset

The model expects an Excel file:

```
logistic_regression_training_data.xlsx
```

With a sheet named:

```
Subscription_Data
```

Target column:

```
Purchased_Premium
```

All other columns are treated as input features.

---

## 🧠 How It Works

### 1. Data Loading

* Uses `pandas` to load data from Excel
* Splits into features (`X`) and labels (`y`)

### 2. Feature Scaling

* Applies **Z-score normalization**:

[
x' = \frac{x - \mu}{\sigma}
]

### 3. Hypothesis Function

The model computes:

[
h(x) = \sigma(w \cdot x + b)
]

Where:

* ( \sigma ) is the sigmoid function
* ( w ) = weights
* ( b ) = bias

---

### 4. Cost Function

Uses **log loss (cross-entropy)** with L2 regularization:

[
J(w, b) = \frac{1}{m} \sum \left[-y \log(h(x)) - (1-y)\log(1-h(x))\right] + \frac{\lambda}{2m} \sum w^2
]

---

### 5. Gradient Descent

Weights and bias are updated using:

[
w := w - \alpha \cdot \frac{\partial J}{\partial w}
]
[
b := b - \alpha \cdot \frac{\partial J}{\partial b}
]

---

### 6. Prediction

Outputs:

* Probability: `0 → 1`
* Class:

```python
1 if probability >= 0.5 else 0
```

---

## 📊 Training Output

* Displays a plot of **cost vs iterations**
* Prints training accuracy:

```
total predictions hit X/N
```

---

## ⚙️ Parameters

You can tune:

```python
alpha = 0.01   # learning rate
delta = 0.7    # regularization strength (lambda)
iterations = 4000
```

---

## ▶️ How to Run

1. Install dependencies:

```bash
pip install numpy pandas matplotlib openpyxl
```

2. Place dataset file in the root directory

3. Run:

```bash
python your_script.py
```

---

## 📈 Example Output

```
total predictions hit 85/100
expected is 1, probability is 0.82 and prediction is 1
```

---

## ⚠️ Notes

* This implementation is **loop-based** (not vectorized), prioritizing clarity over performance
* Accuracy is evaluated on the **training set only**
* For real-world usage, consider:

  * Train/test split
  * Vectorization with NumPy
  * Comparison with `scikit-learn`

---

## 📚 Learning Goals

This project is designed to help understand:

* How logistic regression works internally
* How gradient descent updates parameters
* The role of regularization
* Numerical stability in loss functions

---

## 📄 License

MIT License
