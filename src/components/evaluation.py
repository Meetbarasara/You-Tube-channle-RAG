"""
Evaluation module — RAGAS-based quality metrics.

Computes faithfulness and answer_relevancy for each RAG response,
using OpenAI models as the judge.

Usage:
    from src.components.evaluation import evaluate_response
    metrics = evaluate_response(question, answer, contexts)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def evaluate_response(
    question:  str,
    answer:    str,
    contexts:  list[str],
) -> dict:
    """
    Run RAGAS evaluation on a single question-answer-context tuple.

    Metrics computed:
      - faithfulness:       Is the answer grounded in the context? (0–1)
      - answer_relevancy:   Does the answer address the question?   (0–1)

    Returns:
        {
            "faithfulness":       float | None,
            "answer_relevancy":   float | None,
            "error":              str | None,   # set if evaluation failed
        }
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from src.components.config import settings

        # Build the single-row dataset RAGAS expects
        data = {
            "question":  [question],
            "answer":    [answer],
            "contexts":  [contexts],
        }
        dataset = Dataset.from_dict(data)

        # Use the same OpenAI credentials
        llm        = ChatOpenAI(model="gpt-4o-mini",       api_key=settings.openai_api_key)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.openai_api_key)

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
        )

        df = result.to_pandas()
        row = df.iloc[0]

        def _safe(val) -> Optional[float]:
            try:
                f = float(val)
                return round(f, 3) if f == f else None   # NaN check
            except Exception:
                return None

        return {
            "faithfulness":     _safe(row.get("faithfulness")),
            "answer_relevancy": _safe(row.get("answer_relevancy")),
            "error":            None,
        }

    except ImportError:
        logger.warning("RAGAS not installed. Run: pip install ragas datasets langchain-openai")
        return {
            "faithfulness":     None,
            "answer_relevancy": None,
            "error":            "ragas package not installed",
        }
    except Exception as exc:
        logger.error("RAGAS evaluation failed: %s", exc)
        return {
            "faithfulness":     None,
            "answer_relevancy": None,
            "error":            str(exc),
        }
