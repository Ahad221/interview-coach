import json
import os


class MemoryAgent:
    """
    Tracks scores, weak areas and adaptive difficulty.

    persist=False (default): memory lives only in this visitor's session.
        Use this for the public demo so visitors don't share/overwrite data.
    persist=True: memory is saved to data/memory.json (good for local use).
    """

    def __init__(self, save_path="data/memory.json", persist=False):
        self.save_path = save_path
        self.persist = persist
        self.weak_areas = []
        self.history = []
        self.session_count = 0
        if self.persist:
            self._load()

    def _load(self):
        """Load memory from disk if it exists"""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, "r") as f:
                    data = json.load(f)
                self.weak_areas = data.get("weak_areas", [])
                self.history = data.get("history", [])
                self.session_count = data.get("session_count", 0)
            except (json.JSONDecodeError, OSError):
                pass  # corrupted file -> start fresh

    def _save(self):
        """Save memory to disk (only when persist=True)"""
        if not self.persist:
            return
        try:
            os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
            with open(self.save_path, "w") as f:
                json.dump({
                    "weak_areas": self.weak_areas,
                    "history": self.history,
                    "session_count": self.session_count
                }, f, indent=2)
        except OSError:
            pass  # read-only filesystem on some hosts; don't crash the app

    def update(self, evaluation: dict):
        """Update memory with new evaluation"""
        self.weak_areas.extend(evaluation.get("weaknesses", []))
        self.history.append({
            "score": evaluation.get("score", 0),
            "star_score": evaluation.get("star_score", {}),
            "soft_skills": evaluation.get("soft_skills", {}),
            "weaknesses": evaluation.get("weaknesses", [])
        })
        self.session_count += 1
        self._save()

    def get_weak_areas(self) -> list:
        """Return the 5 most recent unique weak areas"""
        seen = set()
        unique = []
        for w in reversed(self.weak_areas):
            if w not in seen:
                seen.add(w)
                unique.append(w)
        return unique[:5]

    def get_difficulty(self) -> str:
        """Adaptive difficulty based on avg score"""
        scores = [h["score"] for h in self.history if "score" in h]
        if not scores:
            return "easy"
        avg = sum(scores) / len(scores)
        if avg >= 7:
            return "hard"
        elif avg >= 4:
            return "medium"
        else:
            return "easy"

    def get_avg_score(self) -> float:
        scores = [h["score"] for h in self.history if "score" in h]
        return round(sum(scores) / len(scores), 1) if scores else 0.0

    def get_summary(self) -> str:
        if not self.history:
            return "No sessions yet."
        return (
            f"Sessions: {self.session_count} | "
            f"Avg Score: {self.get_avg_score()}/10 | "
            f"Difficulty: {self.get_difficulty().upper()} | "
            f"Weak areas: {', '.join(self.get_weak_areas()) or 'None yet'}"
        )

    def reset(self):
        """Clear all memory"""
        self.weak_areas = []
        self.history = []
        self.session_count = 0
        self._save()