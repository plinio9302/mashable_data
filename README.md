# Mashable News Popularity — Tree-Based ML Case Study

**Course:** MATH 373 — Intro to Machine Learning · University of San Francisco  
**Author:** Plinio Durango  
**Dataset:** [Mashable Online News Popularity](https://archive.ics.uci.edu/dataset/332/online+news+popularity) + SpamAssassin Public Corpus

---

## 📖 Overview

This repository contains two end-to-end machine learning projects built around the **Case Study: Tree-Based Methods** (James D. Wilson, USF). Both projects follow the same pipeline: **load → preprocess → engineer features → tune → evaluate → compare**.

The core theme is **ensemble methods** — showing how progressively more powerful tree-based models reduce prediction error.

---

## 📁 Repository Structure

```
mashable_data/
│
├── Data/                              # Raw datasets
├── Ensemble_Methods_Case_Study.pdf    # Course case study reference (Wilson, USF)
├── classification_tree.ipynb          # Project 1: News popularity prediction
├── Spam Classifier.ipynb              # Project 2: Spam email detection
└── README.md
```

---

## 🗂️ Projects

### 1. Classification Tree — Mashable News Popularity

**Notebook:** `classification_tree.ipynb`

**Goal:** Predict whether a Mashable news article will become **"popular"** (above-median share count) using 58 article features.

**Pipeline:**
| Step | Detail |
|------|--------|
| Data | 39,644 articles, 58 features, binary target (`popular`) |
| Preprocessing | StandardScaler, drop non-predictive cols (`url`, `timedelta`, `shares`) |
| Models | Decision Tree → Bagging → Random Forest |
| Tuning | GridSearchCV / RandomizedSearchCV (5-fold CV) |
| Evaluation | Accuracy, F1, Confusion Matrix, ROC-AUC |

**Key Results:**

| Model | Test Accuracy |
|-------|--------------|
| Decision Tree (unpruned) | ~70% |
| Decision Tree (tuned) | ~67% |
| Bagging | ~69% |
| Random Forest (tuned) | **~72%** |

**Skills demonstrated:** decision boundary visualization, Gini impurity, bias-variance tradeoff, PCA, feature importance, cross-validation.

---

### 2. Spam Email Classifier

**Notebook:** `Spam Classifier.ipynb`

**Goal:** Classify real `.eml` email files as **spam (1)** or **ham (0)** using tree-based ensemble methods on top of TF-IDF text features.

**Pipeline:**
| Step | Detail |
|------|--------|
| Data | SpamAssassin Public Corpus — `.eml` files parsed with Python's `email` library |
| Features | Subject + body → TF-IDF (5,000 most informative words, English stop words removed) |
| Models | Decision Tree → Bagging → Random Forest |
| Tuning | GridSearchCV (5-fold CV) for both Decision Tree and Random Forest |
| Evaluation | Accuracy, Precision, Recall, F1, MSPE |

**Key Results:**

| Model | Accuracy | MSPE |
|-------|----------|------|
| Decision Tree | ~96.7% | 0.038 |
| Bagging (100 trees) | ~97.3% | 0.027 |
| Random Forest (100 trees) | **~97.5%** | **0.025** |

**Top spam keywords:** `href`, `wrote`, `border`, `color`, `spam`, `price`, `receive`, `money`

**Skills demonstrated:** NLP feature extraction, TF-IDF vectorization, class imbalance analysis, ensemble methods, feature importance averaging across bootstrap trees.

---

## 🧠 Key Concepts Covered

- **Decision Trees** — Gini impurity, recursive binary splitting, pruning via `max_depth` / `min_samples_leaf`
- **Bagging** — Bootstrap aggregation, variance reduction, out-of-bag error
- **Random Forest** — Feature subsampling at each split, de-correlated trees, `sqrt(n_features)` heuristic
- **TF-IDF** — Term frequency–inverse document frequency for text-to-numeric conversion
- **GridSearchCV** — Exhaustive hyperparameter search with cross-validation
- **Bias-Variance Tradeoff** — How ensemble size and depth control model complexity

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| pandas / numpy | Data manipulation |
| scikit-learn | ML models, preprocessing, metrics |
| matplotlib / seaborn | Visualization |
| email (stdlib) | Parsing `.eml` files |
| Jupyter Notebook | Development environment |

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/plinio9302/mashable_data.git
cd mashable_data

# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn jupyter

# Launch Jupyter
jupyter notebook
```

> **Note:** The Spam Classifier notebook expects the SpamAssassin `.eml` dataset in `~/Downloads/TRAINING` and `~/Downloads/TESTING`. Download it from the [SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/).

---

## 📊 Reference

Wilson, J.D. (2024). *Case Study: Tree-Based Methods.* MATH 373 — Intro to Machine Learning. University of San Francisco.

---

*Part of the [ml-portfolio](https://github.com/plinio9302/ml-portfolio) · Plinio Durango*
