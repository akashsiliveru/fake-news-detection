<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/NLP-TF--IDF-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Accuracy-96.26%25-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<h1 align="center">🕵️ Fake News Detection</h1>
<p align="center">
  A machine learning pipeline to classify news articles as <strong>Real</strong> or <strong>Fake</strong> using NLP and Multinomial Naive Bayes — achieving <strong>96.26% accuracy</strong> after hyperparameter tuning.
</p>


##  Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Text Preprocessing](#text-preprocessing)
  - [Model Training](#model-training)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Installation](#installation)
- 
- [Usage](#usage)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)


## Overview

Misinformation and fake news are a growing challenge in today's digital media landscape. This project builds a **binary text classifier** that can identify whether a news article is real or fabricated. The pipeline covers:

- Full **Exploratory Data Analysis (EDA)** with statistical testing
- **NLP-based text preprocessing** (stopword removal, URL/punctuation stripping)
- **TF-IDF vectorization** with Multinomial Naive Bayes classification
- **GridSearchCV hyperparameter tuning** to maximize accuracy


##  Dataset

The dataset consists of two CSV files:

| File | Label | Records |
|------|-------|---------|
| `True.csv` | `1` — Real news | ~21,417 |
| `Fake.csv` | `0` — Fake news | ~23,481 |

**Features:**
- `title` — Headline of the article
- `text` — Full article body
- `subject` — Topic category (e.g., `politicsNews`, `worldnews`, `News`, `left-news`)
- `date` — Publication date
- `label` — Ground truth (1 = Real, 0 = Fake)

> The `title` and `text` columns are concatenated to form a single `new_text` feature for modelling.

---

##  Project Structure

```
fake-news-detection/
│
├── True.csv                                    # Real news dataset
├── Fake.csv                                    # Fake news dataset
├── true_sample.xls                             # Sample of real news records
├── fake_sample.xls                             # Sample of fake news records
│
├── EDA_Fake_or_True_News_.ipynb                # Exploratory data analysis notebook
├── Multinomial_NB_on_Fake_are_True_News_.ipynb # Model training & evaluation notebook
│
└── README.md
```


##  Methodology

### Exploratory Data Analysis

The EDA notebook (`EDA_Fake_or_True_News_.ipynb`) investigates the textual characteristics of the dataset:

- **Character count distribution** — Highly right-skewed; log transformation applied to achieve near-normality
- **Word count & average word length** — Engineered as features for analysis
- **Normality testing** — KS test and Anderson-Darling test applied to validate distributional assumptions
- **Box plots** — Comparison of character counts across fake vs. real labels reveals structural differences in article length

Key finding: Article character counts follow a **log-normal distribution**. After log transformation, skewness is significantly reduced, making the data more suitable for downstream modelling.

### Text Preprocessing

The following preprocessing steps are applied to the combined `new_text` column:

1. **Lowercasing** — Normalize all text to lowercase
2. **URL removal** — Strip `http://` and `www.` links using regex
3. **Punctuation removal** — Remove all non-alphanumeric characters
4. **Whitespace normalization** — Collapse multiple spaces into a single space
5. **Stopword removal** — Remove English stopwords via NLTK

### Model Training

- **Vectorization:** TF-IDF (`TfidfVectorizer` with `stop_words='english'`)
- **Train/Test Split:** 80/20 stratified split (`random_state=42`)
- **Classifier:** `MultinomialNB` from scikit-learn

> **Note:** Converting the TF-IDF sparse matrix to a dense array would require ~54.6 GB RAM. The sparse representation is retained throughout.

### Hyperparameter Tuning

`GridSearchCV` with 5-fold cross-validation was used to tune:

| Parameter | Values Searched |
|-----------|----------------|
| `alpha` (Laplace smoothing) | `[0.01, 0.1, 0.5, 1.0, 2.0]` |
| `fit_prior` | `[True, False]` |

**Best Parameters:** `alpha=0.01`, `fit_prior=False`


## Results

### Baseline Model

| Metric | Value |
|--------|-------|
| **Accuracy** | **94.39%** |

**Confusion Matrix:**

|  | Predicted Fake | Predicted Real |
|--|---------------|----------------|
| **Actual Fake** | 4427 | 269 |
| **Actual Real** | 234 | 4050 |


### Tuned Model (GridSearchCV)

| Metric | Value |
|--------|-------|
| **Accuracy** | **96.26%** |

**Confusion Matrix:**

|  | Predicted Fake | Predicted Real |
|--|---------------|----------------|
| **Actual Fake** | 4538 | 158 |
| **Actual Real** | 178 | 4106 |

Hyperparameter tuning improved accuracy by **+1.87 percentage points** and reduced both false positives and false negatives meaningfully.


##  Installation

1. **Clone the repository**

```bash
git clone https://github.com/akashsiliveru/fake-news-detection.git
cd fake-news-detection
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn scipy
```

4. **Download NLTK data**

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```


## 🚀 Usage

Open the notebooks in order:

```bash
# Step 1 — Exploratory Data Analysis
jupyter notebook EDA_Fake_or_True_News_.ipynb

# Step 2 — Model Training and Evaluation
jupyter notebook Multinomial_NB_on_Fake_are_True_News_.ipynb
```

Make sure `True.csv` and `Fake.csv` are in the same directory as the notebooks before running.


## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.8+** | Core language |
| **Pandas / NumPy** | Data manipulation |
| **NLTK** | Stopword removal, stemming, lemmatization |
| **scikit-learn** | TF-IDF, Naive Bayes, GridSearchCV |
| **Matplotlib / Seaborn** | Visualizations |
| **SciPy** | Statistical testing (KS test, Anderson-Darling) |

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request


<p align="center">Made with ❤️ by <a href="https://github.com/akashsiliveru">Akash Siliveru</a></p>
