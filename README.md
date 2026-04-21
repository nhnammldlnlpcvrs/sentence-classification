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
F1_{\text{macro}} = \frac{F1_{\text{class 0}} + F1_{\text{class 1}}}{2}
$$

---

## Techniques

This section summarizes all preprocessing, feature engineering, and modeling techniques used in the final notebook, along with mathematical justifications and concrete examples.

---

### 1. Unicode Normalization (NFKC)

**What it does:** Converts all Unicode characters to a canonical equivalent form.

```python
import unicodedata
text = unicodedata.normalize('NFKC', text)
```

**Why we use it:**

Reviews scraped from the web often contain visually identical but technically different characters:

| Before | After | Issue without fix |
|---|---|---|
| `ｈｅｌｌｏ` (full-width) | `hello` | Treated as different token |
| `café` (combining accent) | `cafe` | Vocabulary mismatch |
| `…` (Unicode ellipsis) | `...` | Not matched by regex |
| `½` (fraction) | `1/2` | Lost numerical meaning |

Without normalization, `hello` and `ｈｅｌｌｏ` become **two separate features** in TF-IDF, wasting vocabulary capacity and degrading generalization.

---

### 2. Contraction Expansion

**What it does:** Expands shortened forms into their full equivalents.

```
"don't"   →  "do not"
"isn't"   →  "is not"
"wouldn't" → "would not"
```

**Why we use it:**

TF-IDF tokenizes text into individual tokens. Without expansion:

$$
\text{"don't like"} \xrightarrow{\text{tokenize}} \{\ \texttt{"don"},\ \texttt{"'t"},\ \texttt{"like"}\ \}
$$

The word `not` is **completely lost**, making negation handling impossible. After expansion:

$$
\text{"do not like"} \xrightarrow{\text{tokenize}} \{\ \texttt{"do"},\ \texttt{"not"},\ \texttt{"like"}\ \}
$$

Now `not` appears explicitly and can be captured by the negation handler in the next step.

---

### 3. Number Replacement

**What it does:** Replaces all digit sequences with the token `num`.

```python
text = re.sub(r'\d+', 'num', text)

# "waited 20 minutes" → "waited num minutes"
# "2 stars out of 5"  → "num stars out of num"
# "over 10 years"     → "over num years"
```

**Why we use it:**

Specific numbers (`20`, `30`, `45`) create **distinct, sparse features** in TF-IDF despite carrying the same semantic meaning in context:

$$
\text{TF-IDF}(\texttt{"waited 20 minutes"}) \neq \text{TF-IDF}(\texttt{"waited 30 minutes"})
$$

Yet both reviews express the same sentiment — *long wait time*. By replacing all digits with `num`, we:

- Reduce vocabulary size (fewer features → less overfitting)
- Generalize across different number values
- Preserve context: `num stars` remains a meaningful sentiment phrase

---

### 4. Negation Handling *(Most Impactful Technique)*

**What it does:** Prepends `not_` to tokens following a negation word within a sliding window.

```python
NEGATION_WORDS = {'not', 'no', 'never', 'neither', 'nobody', 'nothing'}
WINDOW_SIZE = 3
```

**Example:**

```
Input:  "I do not like this product"
Output: "I do not not_like not_this not_product"

Input:  "The food was good but the service was terrible."
Output: "The food was good but the service was terrible."
         (dot resets window → no leakage across clauses)
```

**Why we use it:**

TF-IDF is a **bag-of-words** model — it treats each word independently without understanding context. This causes a critical failure on negated sentiment:

$$
\text{TF-IDF}(\texttt{"I like this"}) \approx \text{TF-IDF}(\texttt{"I do not like this"})
$$

Both vectors contain the feature `like`, which pushes the model toward **positive** — even for the negative review. After negation handling:

| Phrase | Feature | Correct signal |
|---|---|---|
| `"I like this"` | `like` | ✅ Positive |
| `"I do not like this"` | `not_like` | ✅ Negative |

The window size of 3 is empirically chosen — sentiment typically affects only the next few words after a negation:

```
"I do not  like    this    expensive product  ever"
            ↑       ↑        ↑
          [w=1]   [w=2]    [w=3]     outside window
```

Punctuation resets the window to prevent cross-sentence leakage:

```
"Terrible. Never buying again."
           ↑
          dot resets → "never" starts a new window
```

**Implementation note:** We use `re.findall` instead of `nltk.word_tokenize` to process 560k reviews efficiently:

$$
\text{word\_tokenize:} \approx 500 \text{ tokens/s} \quad \Rightarrow \quad \sim 30 \text{ hours}
$$

$$
\text{re.findall:} \approx 50{,}000 \text{ tokens/s} \quad \Rightarrow \quad \sim 15 \text{ minutes}
$$

---

### 5. TF-IDF Vectorization

**What it does:** Converts cleaned text into numerical feature vectors using Term Frequency–Inverse Document Frequency.

$$
\text{TF-IDF}(t, d) = \text{tf}(t, d) \times \log\left(\frac{N}{\text{df}(t) + 1}\right)
$$

Where:
- $\text{tf}(t, d)$: frequency of term $t$ in document $d$
- $N$: total number of documents
- $\text{df}(t)$: number of documents containing term $t$

We apply `sublinear_tf=True`, replacing raw frequency with:

$$
\text{tf}_{\text{sub}}(t, d) = 1 + \log(\text{tf}(t, d))
$$

This prevents a word appearing 100 times from being weighted 100× more than a word appearing once.

**Two vectorizers are combined:**

| Vectorizer | `ngram_range` | Purpose |
|---|---|---|
| Word-level | (1, 2) | Captures phrases: `"not good"`, `"highly recommend"` |
| Char-level | (3, 4) | Robust to typos: `"terribl"` ≈ `"terrible"` |

```python
# Combined feature matrix
X = hstack([word_tfidf, char_tfidf, meta_features])
# Shape: (560k, 50,012)
```

Using `dtype=np.float32` instead of the default `float64` reduces memory by **50%** with negligible accuracy loss.

---

### 6. Word Shape Meta-Features

**What it does:** Extracts shape-based statistics from the original (uncleaned) text to capture emphasis and style.

```python
# CAPS ratio: fraction of all-uppercase words
caps_ratio = sum(1 for w in text.split() if w.isupper() and len(w) > 1) / len(words)

# "AMAZING product!" → caps_ratio = 0.5  → strong emotion signal
# "ok product"       → caps_ratio = 0.0  → neutral
```

**Features extracted:**

| Feature | Description | Sentiment signal |
|---|---|---|
| `caps_ratio` | Fraction of ALL-CAPS words | High → strong emotion (± either) |
| `title_ratio` | Fraction of Title-Case words | Moderate → often positive |
| `exclaim_count` | Number of `!` marks | High → enthusiasm or frustration |
| `question_count` | Number of `?` marks | High → confusion or complaint |
| `negation_count` | Count of negation words | High → likely negative |
| `word_count` | Total words | Long reviews → often positive (loyalty) |
| `star_mentions` | Mentions of `"X stars"` | Direct rating signal |

**Why we use it:**

TF-IDF discards formatting information. A review like `"ABSOLUTE WORST EXPERIENCE EVER!!!"` carries strong sentiment through capitalization and punctuation — signals TF-IDF alone cannot capture.

---

### 7. Token Dropout Augmentation

**What it does:** During training only, randomly removes tokens from each review with probability $p$.

$$
x'_{\text{train}} = \text{Dropout}(x_{\text{train}},\ p=0.05)
$$

```
Original: "This product is absolutely fantastic and I totally love it"
Dropout:  "This product absolutely fantastic and totally love it"
          (dropped "is" and "I")
```

**Why we use it:**

Without augmentation, BERT may learn to rely on specific **co-occurrences** of words rather than their individual meanings:

```
Train: "absolutely fantastic" → positive
Test:  "truly fantastic"      → model has never seen this pair → weaker signal
```

Token dropout forces the model to learn each word's contribution **independently**, similar to Dropout in neural networks — preventing co-adaptation between features. We use $p = 0.05$ (5%) to avoid corrupting the review's meaning.

---

### 8. BERT Fine-tuning (DistilBERT)

**What it does:** Fine-tunes a pre-trained transformer model on the sentiment classification task.

$$
P(y \mid x) = \text{softmax}(W \cdot \text{BERT}(x) + b)
$$

We use **DistilBERT** — a distilled version of BERT that retains 97% of BERT's performance while being 40% smaller and 60% faster.

**Key hyperparameters and justifications:**

| Parameter | Value | Reason |
|---|---|---|
| `MAX_LEN` | 128 | Covers 95th percentile of review lengths |
| `LR` | `3e-5` | Small LR for fine-tuning (not training from scratch) |
| `WARMUP_RATIO` | 0.02 | Dataset is large (560k); long warmup would delay learning |
| `BATCH_SIZE` | 64 | Maximizes T4 GPU utilization |
| `EPOCHS` | 2 | Large dataset converges quickly; epoch 3+ risks overfitting |
| `N_FOLDS` | 1 | Time budget constraint; 560k samples sufficient for 1-fold |

**Why BERT outperforms TF-IDF:**

TF-IDF treats each word independently:

$$
\text{TF-IDF}(\texttt{"not good"}) = \text{vec}(\texttt{"not"}) + \text{vec}(\texttt{"good"})
$$

BERT encodes the **full context** of every token:

$$
h_{\texttt{"good"}} = \text{BERT}(\texttt{"not"}, \texttt{"good"}) \neq \text{BERT}(\texttt{"so"}, \texttt{"good"})
$$

The same word `"good"` receives a different representation depending on its context — allowing BERT to naturally handle negation, sarcasm, and complex sentence structure.

**Learning rate schedule:**

$$
\text{LR}(t) = \text{LR}_{\max} \times \min\left(\frac{t}{t_{\text{warmup}}},\ \frac{T - t}{T - t_{\text{warmup}}}\right)
$$

Linear warmup prevents catastrophic forgetting of pre-trained weights during the first few training steps.

---

### 9. Stratified K-Fold Cross-Validation

**What it does:** Splits the dataset into $k$ folds, ensuring each fold has the same class distribution as the full dataset.

$$
\text{CV Score} = \frac{1}{k} \sum_{i=1}^{k} F1_{\text{macro}}(\hat{y}^{(i)},\ y^{(i)})
$$

**Why we use Stratified over regular K-Fold:**

Suppose the dataset is 60% positive, 40% negative. Without stratification:

```
Fold 1: 80% positive, 20% negative  → F1 inflated
Fold 2: 40% positive, 60% negative  → F1 deflated
Fold 3: 60% positive, 40% negative  → F1 representative
```

Scores across folds are **not comparable**. With stratification, every fold maintains the 60/40 ratio, giving reliable and consistent evaluation.

---

### 10. Ensemble & Threshold Tuning

**What it does:** Combines predictions from multiple models and optimizes the decision boundary.

**Ensemble formula:**

$$
P_{\text{ensemble}}(y=1 \mid x) = \alpha \cdot P_{\text{BERT}}(y=1 \mid x) + (1-\alpha) \cdot P_{\text{ML}}(y=1 \mid x)
$$

Where $\alpha = 0.80$ (BERT weight) and $1 - \alpha = 0.20$ (TF-IDF + LR weight).

**Why ensemble works:** Each model captures different patterns — BERT learns contextual semantics, while TF-IDF + LR learns strong lexical signals (e.g., specific keywords). Their errors are partially independent, so averaging reduces variance.

**Threshold tuning:** Instead of always predicting positive when $P > 0.5$, we search for the optimal threshold $\tau^*$ on the Out-of-Fold predictions:

$$
\tau^* = \underset{\tau \in [0.3,\ 0.7]}{\arg\max}\ F1_{\text{macro}}\!\left(\mathbb{1}[P_{\text{OOF}} \geq \tau],\ y_{\text{train}}\right)
$$

This is valid because threshold tuning is done on **OOF predictions** (never seen during training), so it does not cause data leakage.

---

### Summary Table

| Technique | Stage | Key Benefit |
|---|---|---|
| Unicode Normalization (NFKC) | Preprocessing | Eliminates invisible character differences |
| Contraction Expansion | Preprocessing | Preserves `not` for negation handling |
| Number Replacement (`num`) | Preprocessing | Generalizes across specific digit values |
| Negation Handling (`not_X`) | Preprocessing | Distinguishes `like` from `not_like` |
| TF-IDF (word + char n-gram) | Features | Captures phrases and typo-robust substrings |
| Word Shape Meta-Features | Features | Captures capitalization and punctuation emphasis |
| Token Dropout | Augmentation | Reduces BERT overfitting on co-occurrences |
| DistilBERT Fine-tuning | Modeling | Context-aware representation of each token |
| Stratified K-Fold CV | Evaluation | Reliable F1 estimate with balanced class splits |
| Ensemble + Threshold Tuning | Post-processing | Combines model strengths; optimizes decision boundary |