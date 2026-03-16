# semantic-movie-recommender

**Emotion-aware movie retrieval with preference drift detection.**

Two self-contained ML modules that form the core of a movie recommendation
system designed to *improve* the user's emotional state — not just reflect it.

---


## Overview

This repository contains two core machine learning modules extracted from the
CineAI recommendation system.

1. `recommender.py` - semantic movie retrieval using FAISS embeddings and
   opposite-mood emotion logic.

2. `drift_detector.py` - preference drift detection using LightGBM with
   Stratified K-Fold cross-validation.

The repository demonstrates how I structure ML components, document design
decisions, and build interpretable recommendation pipelines.

---

## The Problem

Most recommendation systems answer: *"What matches how you feel right now?"*

This project answers: *"What would make you feel better?"*

A user detected as **sad** receives uplifting comedies.
An **angry** user receives calm, light-hearted films.
An **anxious** user receives easy, comforting animations.

This *opposite-mood* design is the founding hypothesis of the system
and is implemented entirely in `recommender.py`.

---

## Repository Contents

```
semantic-movie-recommender/
├── recommender.py       # Core ML retrieval — FAISS + opposite-mood logic
├── drift_detector.py    # Preference drift — LightGBM + Stratified K-Fold CV
└── README.md
```

These two files are the code sample submitted as part of a machine learning
engineering application. They are extracted from a larger full-stack system
(CineAI) and are presented here as standalone, well-documented modules.

---

### What is and is not runnable

| Component | Runnable? | Note |
|-----------|-----------|------|
| `drift_detector.py` | ✅ Fully runnable | No external data needed — works out of the box |
| `recommender.py` | ⚠️ Requires model files | Needs `movie_embeddings.pkl` and `movie_faiss.index` |

**Run `drift_detector.py` right now — no setup needed:**

```python
from drift_detector import DriftDetector
import random

detector = DriftDetector()
genres   = ["Action|Sci-Fi", "Comedy", "Drama|Romance", "Action", "Thriller"]

# Simulate 40 interactions
for _ in range(40):
    detector.add_event(random.choice(genres))

print(detector.get_report())
```

To use `recommender.py` you will need to generate the model files from
your movie dataset.

---

## How the two files connect

In the full CineAI system, `recommender.py` and `drift_detector.py`
work together in a feedback loop:

```
User watches a movie
        │
        ▼
recommender.py          ← retrieves movies using emotion + FAISS
        │
        ▼
drift_detector.py       ← logs the genre of each recommended movie
        │
        ▼
Every 10 interactions:
  DriftDetector.train() ← retrains LightGBM on updated history
        │
        ▼
  DriftDetector.detect_drift()
        │
   drift detected?
   YES → recommender adjusts weights, explores new genres
   NO  → recommender continues with current preference model
```

This means the system self-corrects over time. If a user who normally
watches action films starts watching romance movies, `drift_detector.py`
catches the shift and signals `recommender.py` to update its genre
priorities accordingly.

---

## `recommender.py`

### What it does

Retrieves personalised movie recommendations from a 62,000-title corpus
using a hybrid of semantic search and genre-centroid FAISS retrieval,
guided by opposite-mood emotion logic.

### Core concepts

**Opposite-mood mapping**

```python
EMOTION_GENRE_MAP = {
    "sad":     ["Comedy", "Animation", "Family", "Adventure"],
    "angry":   ["Comedy", "Drama", "Animation", "Family"],
    "anxious": ["Comedy", "Animation", "Family", "Music"],
    # ... 14 emotions total
}
```

Each detected emotion maps to counter-mood genres. A paired
`EMOTION_QUERY_MAP` provides a natural-language query string per emotion
that is encoded and searched semantically — so results reflect thematic
proximity, not just genre labels.

**Semantic FAISS search**

User queries and emotion strings are encoded using
`all-MiniLM-L6-v2` (384-dim sentence embeddings) and searched via
a pre-built FAISS flat index. Embeddings are L2-normalised at load
time so all inner-product searches are equivalent to cosine similarity.

**Genre-centroid retrieval**

The mean embedding of all movies matching a target genre is computed
and used as a FAISS query vector. This surfaces films *semantically
central* to the genre rather than simply filtering by tag.

**Five-level fallback chain**

```
1. Semantic search on free-text query
2. Opposite-mood emotion search
3. Genre-centroid search for explicit genres
4. Semantic search on "<emotion> movie" keyword
5. Return top dataset rows  ← absolute last resort
```

The system always returns results regardless of which upstream
signals are available.

### Key methods

| Method | Description |
|--------|-------------|
| `recommend(query, genres, emotion, top_n)` | Unified entry point |
| `recommend_by_emotion(emotion, top_n)` | Opposite-mood retrieval pipeline |
| `recommend_by_genre(genre, top_n)` | Genre-centroid FAISS search |
| `_semantic_search(query_text, top_k)` | Encode text → FAISS lookup |
| `_faiss_search(vector, top_k)` | Raw FAISS nearest-neighbour |

### Usage

```python
from recommender import MovieRecommender

rec = MovieRecommender(
    embeddings_pkl="models/movie_embeddings.pkl",
    faiss_index_path="models/movie_faiss.index",
)

# Emotion-driven recommendation
results = rec.recommend(emotion="sad", top_n=5)

# Free-text + emotion combined
results = rec.recommend(query="something funny", emotion="sad", top_n=5)

# Genre-based
results = rec.recommend(genres=["Comedy", "Animation"], top_n=5)

for r in results:
    print(f"{r['title']:40s}  score={r['score']:.4f}  genres={r['genres']}")
```

---

## `drift_detector.py`

### What it does

Detects when a user's movie genre preferences shift significantly over
time using a LightGBM binary classifier trained with Stratified K-Fold
cross-validation.

### Core concept

Drift detection is framed as a **binary classification problem**:

```
Full interaction history
         │
         ▼
First half  → label 0  (old behaviour)
Second half → label 1  (new behaviour)
         │
         ▼
Can a classifier reliably distinguish them?
  YES → mean AUC is high → preferences have drifted
  NO  → mean AUC ≈ 0.5  → preferences are stable
```

### Cross-validation pipeline

```
Interaction history (≥ 30 genre events)
         │
         ▼
One-hot encode → 15-dim genre vectors
         │
         ▼
StandardScaler normalisation
         │
         ▼
Stratified K-Fold CV (5 folds, shuffle=True, random_state=42)
  ┌──────────────────────────────────────────────┐
  │  For each fold:                              │
  │    Train LGBMClassifier on training split    │
  │    Predict probabilities on validation split │
  │    Compute ROC-AUC                           │
  │    Log: "Fold N AUC: X.XXXX"                │
  └──────────────────────────────────────────────┘
         │
         ▼
Mean CV AUC reported
         │
         ▼
Final model trained on full data
         │
         ▼
Score last 10 interactions:
  drift_score = mean(predict_proba[:, 1])
  drift_score > 0.65  →  drift detected
```

### Why Stratified K-Fold

Interaction histories are small (30–100 events) and class balance
between old/new behaviour is often uneven. Stratification preserves
the class ratio in every fold, giving reliable AUC estimates even
at small sample sizes.

### Why LightGBM

A simple cosine drift metric (distance between mean vectors) cannot
tell you *which* genres changed or whether the change is statistically
meaningful. LightGBM gives:

- **Probability scores** — continuous drift signal, not just a binary flag
- **Per-feature importances** — which genres are responsible for the drift
- **Cross-validated confidence** — AUC per fold, not just a single split
- **Fast retraining** — < 100ms on 100 samples with shallow trees

### LightGBM hyperparameters

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `n_estimators` | 100 | Sufficient for 15 sparse features |
| `learning_rate` | 0.05 | Conservative — avoids overfitting small histories |
| `max_depth` | 4 | Shallow trees suit sparse one-hot input |
| `num_leaves` | 15 | Consistent with max_depth=4 |
| `min_child_samples` | 5 | Prevents splits on tiny genre subsets |
| `subsample` | 0.8 | Row sampling for regularisation |
| `colsample_bytree` | 0.8 | Feature sampling per tree |

### Usage

```python
from drift_detector import DriftDetector

detector = DriftDetector(n_splits=5, drift_threshold=0.65)

# Log interactions as they happen
for genre_string in user_watch_history:
    detector.add_event(genre_string)   # auto-retrains every 10 events

# Check for drift
if detector.detect_drift():
    print("Preferences have shifted — adjusting recommendations.")

# Full report
report = detector.get_report()
print(f"Mean CV AUC : {report['mean_cv_auc']}")
print(f"Fold scores : {report['cv_fold_scores']}")
print(f"Top genres  : {report['top_drift_genres']}")
```

**Example output:**
```
[DriftDetector] Fold 1 AUC: 0.7812
[DriftDetector] Fold 2 AUC: 0.7634
[DriftDetector] Fold 3 AUC: 0.8021
[DriftDetector] Fold 4 AUC: 0.7455
[DriftDetector] Fold 5 AUC: 0.7908
[DriftDetector] Mean CV AUC: 0.7766
[DriftDetector] Drift score: 0.7124 | Threshold: 0.65

Mean CV AUC : 0.7766
Fold scores : [0.7812, 0.7634, 0.8021, 0.7455, 0.7908]
Top genres  : {'Sci-Fi': 42, 'Action': 31, 'Drama': 18, 'Comedy': 9}
```

---

## Dataset

The models in this repository were built on a custom scraped dataset of
**62,000+ movies** combining metadata from multiple public sources.

| Field | Description |
|-------|-------------|
| `title` | Movie title |
| `genres` | Pipe-separated genre labels (e.g. `Action|Sci-Fi|Thriller`) |
| `overview` | Plot summary text |
| `popularity` | Relative popularity score |

**Embedding generation:** Each movie was encoded into a 384-dimensional
vector by passing `title + genres + overview` through `all-MiniLM-L6-v2`.
Vectors were L2-normalised and saved as `movie_embeddings.pkl`. A FAISS
flat index was built from these vectors and saved as `movie_faiss.index`.

The raw dataset and model files are not included in this repository due to
file size. The embedding generation script is available on request.

---

## Installation

```bash
git clone https://github.com/priyankathotkar/semantic-movie-recommender.git
cd semantic-movie-recommender

pip install faiss-cpu sentence-transformers numpy pandas lightgbm scikit-learn
```


---

## Design Decisions Summary

| Decision | Why |
|----------|-----|
| Opposite-mood genre mapping | Counter the emotional state — improves user wellbeing rather than reinforcing negative moods |
| Semantic FAISS search over keyword filter | Captures thematic similarity beyond exact genre tags |
| Genre-centroid retrieval | Finds films semantically central to a genre, not just tagged with it |
| Stratified K-Fold CV for drift | Small histories + uneven class balance → stratification gives reliable AUC |
| LightGBM over cosine drift | Probability output, feature importances, and CV confidence — all needed for an interpretable drift signal |
| Five-level fallback chain | API must never return empty — every failure mode has a recovery path |

---

## Attribution

**All architecture, design decisions, and algorithmic approaches in this
repository were conceived and implemented solely by Priyanka Thotkar.**

Specifically, the following are original contributions:
- Opposite-mood emotion-to-genre mapping concept and implementation
- FAISS genre-centroid retrieval strategy
- Five-level fallback chain design
- LightGBM drift detection approach and Stratified K-Fold framing
- Overall system architecture

**AI tools used:** Claude (Anthropic) was used for debugging assistance
and docstring formatting during development — equivalent to how engineers
use GitHub Copilot. All design decisions and algorithmic approaches are
entirely my own.

**Open-source libraries:**

| Library | Purpose | License |
|---------|---------|---------|
| [FAISS](https://github.com/facebookresearch/faiss) | Vector similarity search | MIT |
| [sentence-transformers](https://www.sbert.net) | Sentence embeddings | Apache 2.0 |
| [LightGBM](https://lightgbm.readthedocs.io) | Gradient boosting classifier | MIT |
| [scikit-learn](https://scikit-learn.org) | Stratified K-Fold, AUC, StandardScaler | BSD |
| [numpy](https://numpy.org) | Numerical operations | BSD |
| [pandas](https://pandas.pydata.org) | Dataset handling | BSD |

---

## Limitations

Being upfront about what the system cannot do:

| Limitation | Details |
|------------|---------|
| **Cold start** | `drift_detector.py` needs ≥ 30 interactions before it can train. New users get no drift signal. |
| **Genre vocabulary is fixed** | The one-hot encoder covers 15 genres. Niche genres (e.g. "Anime", "K-Drama") are not captured. |
| **Flat FAISS index** | Search is O(N) — works well at 62,000 movies but would need IVF indexing above ~1M titles. |
| **Emotion accuracy** | The upstream facial emotion classifier is not perfect. A misclassified emotion produces a counter-mood recommendation for the wrong state. |
| **No personalisation memory** | Preferences reset on each session — there is no persistent user profile across sessions. |
| **Label split assumption** | Drift detection splits history at the midpoint. If a user's taste changed very recently, the signal may be weak until more post-drift interactions accumulate. |

---

## Future Improvements

- **Collaborative filtering** — incorporate user-user similarity signals
  alongside content embeddings
- **IVF FAISS index** — replace flat index with IVF for sub-linear search
  at larger scale (>1M movies)
- **Online drift adaptation** — adjust recommendations mid-session as drift
  is detected rather than waiting for a full retrain
- **Multi-modal embeddings** — combine movie poster image embeddings with
  text for richer retrieval
- **Reinforcement learning** — learn from explicit user feedback
  (thumbs up/down) to refine the opposite-mood mapping over time

---

*Part of the CineAI project — [Priyanka Thotkar](https://github.com/priyankathotkar)*
