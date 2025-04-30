# Behavioral Cues and Adversarial Robustness in Phishing Email Detection

### Final project for Duke University's IDS 705: Machine Learning course.

**Team Capybara:** Ailina Aniwan | Jay Liu | Eric Ortega Rodriguez | Tursunai Turumbekova

## 📌 Overview

This project explores phishing email classification using both traditional machine learning and deep learning techniques, aiming to evaluate model accuracy, interpretability, robustness, and feature enhancement strategies.

## 📄 Abstract

Phishing emails remain a persistent and evolving cybersecurity threat, often exploiting linguistic ambiguity and psychological manipulation to bypass conventional filters. This project explores phishing detection through the lens of supervised learning by analyzing four key dimensions: model architecture, text preprocessing, adversarial robustness, and behavioral feature engineering. We compare traditional machine learning models (Naïve Bayes, Logistic Regression, XGBoost) with deep learning approaches (BERT), and assess how incorporating structured metadata and psychological deception cues influences classification performance.

Our findings indicate that BERT delivers consistently high performance with minimal parameter tuning, while XGBoost combined with engineered features achieves the highest overall accuracy. Importantly, we demonstrate that reducing false negatives—even marginally—can significantly improve security in high-risk environments. Lastly, we show that integrating behavioral deception features enhances performance in lightweight models, offering a practical and interpretable alternative or complement to deep content-based approaches.



## 📂 Repository Structure

- **`data/`**: Contains the TREC 2007 Public Spam Corpus dataset.
- **`notebooks/`**: Jupyter notebooks: model training and comparison Experiment 1, data preprocessing pipelines Experiment 2, adverserial noise model evaluation Experiment 3, and Social Deception features evaluation Experiment 4
- **`report and slides/`**: Stored outputs including model performance summary
- **`requirements.txt`**: List of Python dependencies required to run the project.
- **`README.md`**

```
├── data/
│   ├── raw/                     # Raw TREC 2007 Public Spam Corpus
│   └── cleaning/                # Data cleaning notebook and cleaned CSV
├── notebooks/                   # Experiments 1, 2, 3, 4
│   ├── 3_experiment_noise_bert/ # Experiment 3
│   └── results                  # Results from each run
├── report and slides/
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## 📚 Data

We utilized the **2007 TREC Public Spam Corpus**, a publicly available dataset hosted on Zenodo, as curated by Champa et al. (2024b). The dataset includes **52,713 labeled emails**, each annotated as either:

- **Phishing (label = 1)**  
- **Not phishing (label = 0)**

Each email entry contains rich metadata fields such as:

- Sender and receiver addresses  
- Subject line  
- Email body  
- URL counts  
- Timestamp information

This structure enabled us to experiment with both **pure-text approaches** (e.g., BERT, TF-IDF + Logistic Regression) and **hybrid models** incorporating behavioral or structural features.

> ⚠️ **Limitations**:  
> While the dataset offers a realistic view of email-based phishing strategies as of 2007, phishing tactics have evolved significantly. Modern attacks now incorporate AI-generated content, targeted spear-phishing, and advanced evasion strategies—often personalized for specific victims (IBM X-Force Threat Intelligence Index, 2024). Thus, while our models perform well on this corpus, real-world generalization may require retraining on newer datasets.

As shown in Figure 1 of our notebook, the dataset exhibits **moderate class imbalance**, with phishing emails making up approximately **54%** of all entries. This imbalance underscores the need to optimize for **recall**, as false negatives in phishing detection carry significant security risks.



## 🧪 Models Implemented

The project explores and compares the following models:

### Traditional Machine Learning Models

- **Logistic Regression**
- **Naïve Bayes**
- **Random Forest**
- **XGBoost**

### Deep Learning Models

- **BERT (Bidirectional Encoder Representations from Transformers)**

## 📊 Evaluation Metrics

To assess model performance, we utilize:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

These metrics provide a comprehensive view of each model's effectiveness in classifying phishing emails.

## 📈 Key Findings

- BERT consistently outperforms all models with minimal tuning and strong generalization.

- XGBoost achieves the highest overall accuracy when enhanced with structured and behavioral features.

- Behavioral deception cues (e.g., urgency, fear) enhance performance in lightweight models like Logistic Regression.

- Adversarial perturbations challenge BERT’s robustness, motivating further research in adversarial training. 

## ⚙️ Setup & Reproducibility

This repository contains a descriptive and reproducible setup to replicate all results.

### ✅ Prerequisites

- Python 3.7+
- pip or virtualenv

### 📦 Install Dependencies

```bash
git clone https://github.com/ericiortega/ids705-phishing-detection.git
cd ids705-phishing-detection
pip install -r requirements.txt
```

### ▶️ Run the Code

1. **Data Preparation**  
   Navigate to `data/cleaning/` and run the cleaning notebook to generate `2_cleaned_data.csv`.

2. **Model Training and Evaluation**  
   Launch Jupyter and open any notebook from `notebooks/` to explore the experiments 1, 2, 3, 4:
   - Experiment 1 (1_experiment_baseline_vs_other_models.ipynb) Train models and compare key statistic (Logistic Regression, Naïve Bayes, XGBoost, BERT)
   - Experiment 2 (2_experiment_data_preprocessing_pipelines.ipynb): Data Processing Pipelines Comparison
   - Experiment 3 (3_experiment_noise_bert): Evaluating BERT’s Robustness to Adversarial Noise
   - Experiment 4 (4_experiment_social_features.ipynb): Social Engineering Detection via Behavioral Feature Engineering


## 📚 References

- Champa, A. I., Zibran, M. F., & Rahayu, W. (2024a). *Why phishing emails escape detection: A closer look at the failure points.* 2024 12th International Symposium on Digital Forensics and Security (ISDFS), IEEE. [IEEE Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10527344)

- Champa, A. I., Zibran, M. F., & Rahayu, W. (2024b). *Curated datasets and feature analysis for phishing email detection with machine learning.* 2024 3rd IEEE International Conference on Computing and Machine Intelligence (ICMI). [Paper PDF](https://www2.cose.isu.edu/~minhazzibran/resources/MyPapers/Champa_ICMI2024_Published.pdf)


