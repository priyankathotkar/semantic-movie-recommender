import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import Optional, List
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------
# OPPOSITE-MOOD MAPPING
# Mission: counter the user's emotion, not reinforce it.
# Sad user → uplifting/happy movies to cheer them up
# Angry user → calming/light movies to cool them down
# ---------------------------------------------------

EMOTION_GENRE_MAP = {
    # Feeling sad → recommend uplifting, feel-good, funny movies
    "sad":      ["Comedy", "Animation", "Family", "Adventure"],

    # Feeling happy → keep the energy going with fun/exciting movies
    "happy":    ["Action", "Adventure", "Sci-Fi", "Comedy"],

    # Feeling angry → calm them down with light, feel-good, inspiring movies
    "angry":    ["Comedy", "Drama", "Animation", "Family"],

    # Feeling fearful → comfort with light adventure or comedy
    "fear":     ["Comedy", "Family", "Animation", "Adventure"],

    # Feeling surprised → lean into the excitement with mystery/thriller
    "surprise": ["Mystery", "Thriller", "Sci-Fi", "Adventure"],

    # Feeling disgusted → lighten mood with comedy/animation
    "disgust":  ["Comedy", "Animation", "Family", "Romance"],

    # Neutral → well-rounded mix
    "neutral":  ["Drama", "Documentary", "Action", "Comedy"],

    # Feeling bored → excite them with action/adventure
    "bored":    ["Action", "Adventure", "Thriller", "Sci-Fi"],

    # Feeling excited → match their energy
    "excited":  ["Action", "Sci-Fi", "Adventure", "Animation"],

    # Feeling romantic → match the mood
    "romantic": ["Romance", "Drama", "Comedy"],

    # Feeling anxious → calm them down with light/easy films
    "anxious":  ["Comedy", "Animation", "Family", "Music"],

    # Feeling lonely → warm, human connection stories
    "lonely":   ["Drama", "Romance", "Comedy", "Animation"],

    # Feeling tired → easy, light viewing
    "tired":    ["Animation", "Comedy", "Family", "Documentary"],

    # Feeling stressed → laugh it off
    "stressed": ["Comedy", "Animation", "Music", "Family"],
}

# Natural language query per emotion — for semantic FAISS search
EMOTION_QUERY_MAP = {
    "sad":      "funny uplifting cheerful feel-good comedy movie to cheer you up",
    "happy":    "exciting fun thrilling adventure action movie",
    "angry":    "calming peaceful light-hearted funny relaxing movie",
    "fear":     "funny safe light-hearted family animated movie",
    "surprise": "mysterious suspenseful unexpected twist thriller movie",
    "disgust":  "fun silly lighthearted comedy animated movie",
    "neutral":  "interesting engaging well-crafted drama documentary movie",
    "bored":    "fast-paced exciting action adventure blockbuster movie",
    "excited":  "epic thrilling sci-fi action adventure movie",
    "romantic": "beautiful love story heartwarming romance drama",
    "anxious":  "calming funny easy relaxing animated family movie",
    "lonely":   "heartwarming friendship human connection emotional drama",
    "tired":    "light funny easy animated feel-good movie",
    "stressed": "hilarious funny comedy movie to make you laugh",
}


class MovieRecommender:

    def __init__(
        self,
        embeddings_pkl:  str = "models/movie_embeddings.pkl",
        faiss_index_path: str = "models/movie_faiss.index",
        embedding_model:  str = "all-MiniLM-L6-v2",
    ):
        self.embeddings_pkl   = Path(embeddings_pkl)
        self.faiss_index_path = Path(faiss_index_path)

        self.movies     = None
        self.embeddings = None
        self.index      = None
        self.titles     = []

        print("[Recommender] Loading sentence transformer...")
        self.encoder = SentenceTransformer(embedding_model)

        self._load()

    # ---------------------------------------------------
    # Load dataset + FAISS index
    # ---------------------------------------------------

    def _load(self):
        print("[Recommender] Loading dataset...")

        with open(self.embeddings_pkl, "rb") as f:
            data = pickle.load(f)

        self.movies     = data["movies"].reset_index(drop=True)
        self.embeddings = data["embeddings"].astype(np.float32)

        faiss.normalize_L2(self.embeddings)

        self.movies["genres"] = self.movies["genres"].fillna("")

        # Build genres_list — handles pipe, comma, or space separators
        self.movies["genres_list"] = self.movies["genres"].apply(
            lambda x: [
                g.strip().lower()
                for g in str(x).replace(",", "|").split("|")
                if g.strip()
            ]
        )

        self.titles = self.movies["title"].tolist()

        print("[Recommender] Loading FAISS index...")
        self.index = faiss.read_index(str(self.faiss_index_path))

        print(f"[Recommender] Ready — {len(self.movies)} movies loaded")

    # ---------------------------------------------------
    # Core FAISS search
    # ---------------------------------------------------

    def _faiss_search(self, vector: np.ndarray, top_k: int = 30):
        query = np.array(vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        distances, indices = self.index.search(query, top_k)
        return distances[0], indices[0]

    # ---------------------------------------------------
    # Semantic search via text query
    # ---------------------------------------------------

    def _semantic_search(self, query_text: str, top_k: int = 30):
        vector = self.encoder.encode([query_text])[0].astype(np.float32)
        return self._faiss_search(vector, top_k)

    # ---------------------------------------------------
    # Build result dict
    # ---------------------------------------------------

    def _build_result(self, idx: int, score: float) -> Optional[dict]:
        if idx < 0 or idx >= len(self.movies):
            return None
        movie = self.movies.iloc[idx]
        return {
            "title":  str(movie["title"]),
            "genres": str(movie["genres"]),
            "score":  round(float(score), 4),
        }

    # ---------------------------------------------------
    # Recommend by genre (centroid-based)
    # ---------------------------------------------------

    def recommend_by_genre(self, genre: str, top_n: int = 10) -> List[dict]:
        genre_lower = genre.strip().lower()

        # Exact match first
        mask      = self.movies["genres_list"].apply(lambda g: genre_lower in g)
        positions = self.movies[mask].index.tolist()

        # Partial string fallback
        if not positions:
            mask      = self.movies["genres"].str.lower().str.contains(genre_lower, na=False)
            positions = self.movies[mask].index.tolist()

        if not positions:
            print(f"[Recommender] No movies found for genre: '{genre}'")
            return []

        centroid             = self.embeddings[positions].mean(axis=0)
        distances, indices   = self._faiss_search(centroid, top_k=top_n * 3)

        results = []
        seen    = set()

        for dist, idx in zip(distances, indices):
            result = self._build_result(idx, 1 - dist)
            if result and result["title"] not in seen:
                seen.add(result["title"])
                results.append(result)
            if len(results) >= top_n:
                break

        return results

    # ---------------------------------------------------
    # Recommend by emotion (opposite-mood logic)
    # ---------------------------------------------------

    def recommend_by_emotion(self, emotion: str, top_n: int = 10) -> List[dict]:
        emotion = emotion.strip().lower()

        if emotion not in EMOTION_GENRE_MAP:
            print(f"[Recommender] Unknown emotion '{emotion}' → using neutral")
            emotion = "neutral"

        genres     = EMOTION_GENRE_MAP[emotion]
        query_text = EMOTION_QUERY_MAP.get(emotion, "good movie")

        print(f"[Recommender] Emotion: '{emotion}' → target genres: {genres}")
        print(f"[Recommender] Semantic query: '{query_text}'")

        seen    = set()
        results = []

        # ── 1. Semantic search (most powerful) ──────────────────────────────
        sem_distances, sem_indices = self._semantic_search(query_text, top_k=top_n * 2)

        for dist, idx in zip(sem_distances, sem_indices):
            result = self._build_result(idx, 1 - dist)
            if result and result["title"] not in seen:
                seen.add(result["title"])
                results.append(result)

        # ── 2. Genre centroid search as supplement ───────────────────────────
        per_genre = max(3, top_n // len(genres))

        for genre in genres:
            for m in self.recommend_by_genre(genre, per_genre):
                if m["title"] not in seen:
                    seen.add(m["title"])
                    results.append(m)
            if len(results) >= top_n * 2:
                break

        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_n]

    # ---------------------------------------------------
    # General recommend entry point
    # ---------------------------------------------------

    def recommend(
        self,
        query:   Optional[str]       = None,
        genres:  Optional[List[str]] = None,
        emotion: Optional[str]       = None,
        top_n:   int                 = 10,
    ) -> List[dict]:

        seen    = set()
        results = []

        # 1. Semantic search on user text query
        if query:
            print(f"[Recommender] Semantic query: '{query}'")
            distances, indices = self._semantic_search(query, top_k=top_n * 2)
            for dist, idx in zip(distances, indices):
                result = self._build_result(idx, 1 - dist)
                if result and result["title"] not in seen:
                    seen.add(result["title"])
                    results.append(result)

        # 2. Opposite-mood emotion recommendations
        if emotion:
            for m in self.recommend_by_emotion(emotion, top_n):
                if m["title"] not in seen:
                    seen.add(m["title"])
                    results.append(m)

        # 3. Genre-based
        if genres:
            for g in genres:
                for m in self.recommend_by_genre(g, top_n):
                    if m["title"] not in seen:
                        seen.add(m["title"])
                        results.append(m)

        # 4. Fallback: semantic search on emotion word alone
        if not results and emotion:
            print(f"[Recommender] Fallback semantic on emotion: '{emotion}'")
            distances, indices = self._semantic_search(emotion + " movie", top_k=top_n)
            for dist, idx in zip(distances, indices):
                result = self._build_result(idx, 1 - dist)
                if result and result["title"] not in seen:
                    seen.add(result["title"])
                    results.append(result)

        # 5. Last resort: top movies from dataset
        if not results:
            print("[Recommender] Last resort — returning top dataset movies")
            for idx in range(min(top_n, len(self.movies))):
                result = self._build_result(idx, 0.5)
                if result:
                    results.append(result)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_n]