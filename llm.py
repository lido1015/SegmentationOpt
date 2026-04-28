from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import openai
except ImportError:  # pragma: no cover
    openai = None

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:  # pragma: no cover
    SentenceTransformer = None
    util = None


@dataclass
class CoherenceScore:
    score: float
    prompt: str
    response: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class LlmEvaluator:
    """Interfaz para evaluar la coherencia de un segmento usando un LLM."""

    def __init__(
        self,
        model_name: str = "gpt-4",
        cache_path: Optional[str] = None,
        use_cache: bool = True,
        llm_temperature: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.use_cache = use_cache
        self.llm_temperature = llm_temperature
        self.cache_path = Path(cache_path or "llm_cache.json")
        self.cache: Dict[str, float] = self._load_cache()

        if openai is not None:
            self.client = openai
        else:
            self.client = None

        self.embedding_model = None
        if SentenceTransformer is not None:
            try:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self.embedding_model = None

    def _load_cache(self) -> Dict[str, float]:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, encoding="utf-8") as file:
                    raw = json.load(file)
                    return {str(k): float(v) for k, v in raw.items()}
            except Exception:
                return {}
        return {}

    def save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as file:
            json.dump(self.cache, file, indent=2, ensure_ascii=False)

    def _cache_key(self, text: str) -> str:
        normalized = " ".join(text.strip().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _build_prompt(self, text: str) -> str:
        return (
            "Evalúa la coherencia temática del siguiente fragmento de texto. "
            "Asigna una puntuación entre 1 (completamente incoherente) y 10 (altamente coherente). "
            "Responde exclusivamente con el número y, en la línea siguiente, precedida por 'EXPLICACIÓN:', una breve justificación. "
            f"Texto: {text}"
        )

    def _parse_response(self, response: str) -> float:
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", response)
        if not match:
            raise ValueError(f"No se pudo parsear la puntuación en la respuesta: {response}")
        score = float(match.group(1))
        return max(1.0, min(score, 10.0))

    def _call_llm(self, prompt: str) -> str:
        if self.client is None:
            raise RuntimeError(
                "OpenAI no está disponible. Instala la librería 'openai' y configura OPENAI_API_KEY."
            )
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("La variable de entorno OPENAI_API_KEY no está definida.")

        response = self.client.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.llm_temperature,
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()

    def score_segment(self, segment: Sequence[str]) -> CoherenceScore:
        text = " ".join(element.strip() for element in segment if isinstance(element, str))
        prompt = self._build_prompt(text)
        key = self._cache_key(text)

        if self.use_cache and key in self.cache:
            return CoherenceScore(score=self.cache[key], prompt=prompt, response="cached result")

        response = self._call_llm(prompt)
        score = self._parse_response(response)
        self.cache[key] = score
        return CoherenceScore(score=score, prompt=prompt, response=response)

    def score_segment_by_embedding(self, segment: Sequence[str]) -> CoherenceScore:
        if self.embedding_model is None or util is None:
            raise RuntimeError(
                "Sentence-Transformers no está disponible. Instala 'sentence-transformers' para usar la métrica de embeddings."
            )

        text = " ".join(element.strip() for element in segment if isinstance(element, str))
        sentences = [sentence.strip() for sentence in text.split(".") if sentence.strip()]
        if len(sentences) < 2:
            return CoherenceScore(score=1.0, prompt=text, response="segment too short")

        embeddings = self.embedding_model.encode(sentences, convert_to_tensor=True)
        similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)
        upper_triangular = similarity_matrix.triu(diagonal=1)
        score = float(upper_triangular.mean().item() * 9.0 + 1.0)
        score = max(1.0, min(score, 10.0))

        return CoherenceScore(score=score, prompt=text, response="embedding coherence")
