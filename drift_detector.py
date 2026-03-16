import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


class DriftDetector:
    """
    Detects user behavior drift using LightGBM with Stratified K-Fold
    Cross-Validation. Tracks genre interaction history and determines
    if recent preferences differ significantly from past behavior.
    """

    def __init__(self, n_splits=5, drift_threshold=0.65):

        # interaction history as genre vectors
        self.history = []

        # cross-validation config
        self.n_splits = n_splits

        # drift threshold (mean probability of "new behavior")
        self.drift_threshold = drift_threshold

        # cross-validation AUC scores per fold
        self.cv_scores = []

        # mean AUC from last training run
        self.mean_auc = None

        # final model trained on full data after CV
        self.model = None

        # scaler for feature normalization
        self.scaler = StandardScaler()

        # training flag
        self.trained = False

        # supported genre features
        self.genres = [
            "Action", "Adventure", "Animation", "Comedy", "Crime",
            "Documentary", "Drama", "Fantasy", "Horror", "Mystery",
            "Romance", "Sci-Fi", "Thriller", "War", "Western"
        ]

        # LightGBM hyperparameters
        self.lgbm_params = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 4,
            "num_leaves": 15,
            "min_child_samples": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1
        }


    
    # Convert genre string → one-hot vector
   

    def genre_to_vector(self, genre_string):

        vector = np.zeros(len(self.genres), dtype=np.float32)

        if not genre_string:
            return vector

        genre_string = genre_string.lower()

        for i, g in enumerate(self.genres):
            if g.lower() in genre_string:
                vector[i] = 1.0

        return vector


   
    # Add user interaction event
  

    def add_event(self, genre):

        vector = self.genre_to_vector(genre)
        self.history.append(vector)

        # retrain automatically every 10 new events once baseline exists
        if len(self.history) >= 30 and len(self.history) % 10 == 0:
            self.train()


    # Cross-Validated LightGBM Training


    def train(self):
        """
        Trains LightGBM with Stratified K-Fold Cross-Validation.
        Labels: first half = old behavior (0), second half = new behavior (1).
        Returns mean AUC across folds.
        """

        if len(self.history) < 30:
            print("[DriftDetector] Not enough data to train (need >= 30 events).")
            return None

        X = np.array(self.history)

        # label: old=0, new=1 — split at midpoint
        y = np.zeros(len(X), dtype=int)
        y[len(X) // 2:] = 1

        # normalize features
        X_scaled = self.scaler.fit_transform(X)

        # ── Stratified K-Fold Cross-Validation ──
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        self.cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):

            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            fold_model = lgb.LGBMClassifier(**self.lgbm_params)
            fold_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
            )

            val_probs = fold_model.predict_proba(X_val)[:, 1]

            # AUC only valid if both classes present in val split
            if len(np.unique(y_val)) > 1:
                auc = roc_auc_score(y_val, val_probs)
                self.cv_scores.append(auc)
                print(f"[DriftDetector] Fold {fold + 1} AUC: {auc:.4f}")

        self.mean_auc = np.mean(self.cv_scores) if self.cv_scores else None
        if self.mean_auc:
            print(f"[DriftDetector] Mean CV AUC: {self.mean_auc:.4f}")
        else:
            print("[DriftDetector] CV AUC unavailable.")

        # ── Train final model on full data after CV ──
        self.model = lgb.LGBMClassifier(**self.lgbm_params)
        self.model.fit(X_scaled, y)

        self.trained = True

        return self.mean_auc


    # Detect drift on recent interactions

    def detect_drift(self):
        """
        Returns True if recent behavior significantly differs from past.
        Uses the LightGBM model trained via cross-validation.
        """

        if len(self.history) < 30:
            return False

        if not self.trained:
            self.train()

        if self.model is None:
            return False

        # score the last 10 interactions
        recent        = np.array(self.history[-10:])
        recent_scaled = self.scaler.transform(recent)

        probabilities = self.model.predict_proba(recent_scaled)[:, 1]
        drift_score   = float(np.mean(probabilities))

        print(f"[DriftDetector] Drift score: {drift_score:.4f} | Threshold: {self.drift_threshold}")

        return drift_score > self.drift_threshold


    # Get feature importances (which genres drive drift)
 

    def get_feature_importances(self):
        """
        Returns dict of genre -> importance score from the trained model.
        Useful for explaining which genres triggered drift.
        """

        if not self.trained or self.model is None:
            return {}

        importances = self.model.feature_importances_

        return dict(sorted(
            zip(self.genres, importances),
            key=lambda x: x[1],
            reverse=True
        ))


    # Full drift report
 

    def get_report(self):
        """
        Returns a summary dict of the current drift detection state.
        """

        return {
            "total_events":     len(self.history),
            "trained":          self.trained,
            "cv_fold_scores":   [round(s, 4) for s in self.cv_scores],
            "mean_cv_auc":      round(self.mean_auc, 4) if self.mean_auc else None,
            "drift_detected":   self.detect_drift() if self.trained else False,
            "drift_threshold":  self.drift_threshold,
            "top_drift_genres": self.get_feature_importances()
        }
