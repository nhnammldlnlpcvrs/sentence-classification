# Sentence Classification

## Overview

Customer reviews contain valuable information about user experience. In this project, we aim to automatically classify customer reviews into **positive** or **negative** sentiment based solely on their textual content.

This task belongs to the field of **Natural Language Processing (NLP)**, specifically **text classification**. A successful model can be applied in real-world systems such as:

* Recommendation systems

* Customer support automation

* User experience analytics

---

## Objective

The goal is to build a machine learning model that predicts the sentiment of a given review:

* **Class 0 (Negative):** Reviews expressing dissatisfaction (typically 1–2 stars)

* **Class 1 (Positive):** Reviews expressing satisfaction (typically 3–4 stars)

---

## Dataset

The dataset consists of:

* **Train set:** Contains review text and corresponding labels

* **Test set:** Contains review text only (labels are hidden)

Each sample includes:

* `ID`: Unique identifier

* `Review`: Text data
* `Label`: Sentiment class (only in training set)

---

## Submission Format

Submissions must be in CSV format with the following structure:

```
ID,Label
1,1
2,0
3,1
...
```

Where:

* **ID:** Corresponds to the test sample ID

* **Label:** Predicted sentiment (0 or 1)

---

## Evaluation Metric

Submissions are evaluated using the **Macro F1-score**, which balances performance across both classes (positive and negative).

### Precision

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

### Recall

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

### F1-score

$$
F1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

### Macro F1-score

$$
F1_{\text{macro}} = \frac{F1_{class\ 0} + F1_{class\ 1}}{2}
$$
