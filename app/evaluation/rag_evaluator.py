import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.core.rag import RAGService


class RAGEvaluator:
    """
    Evaluation framework for a Retrieval-Augmented Generation (RAG) pipeline.

    This class evaluates the performance of the RAG system by measuring:
        - Retrieval quality based on vector similarity scores.
        - Answer quality using keyword matching against expected keywords.

    The evaluation process consists of:
        1. Loading evaluation questions from a JSON file.
        2. Running each question through the RAG pipeline.
        3. Measuring retrieval metrics (similarity score, retrieved chunks).
        4. Measuring answer quality via keyword matches.
        5. Aggregating global evaluation metrics.

    Attributes:
        questions_file (Path): Path to the evaluation questions JSON file.
        rag (RAGService): RAG pipeline instance used for evaluation.
    """

    def __init__(self, questions_file: str) -> None:
        """
        Initialize the RAG evaluator.

        Args:
            questions_file (str): Path to the JSON file containing evaluation questions.
        """
        self.questions_file = Path(questions_file)
        self.rag = RAGService()

    def load_questions(self) -> List[Dict[str, Any]]:
        """
        Load evaluation questions from the JSON file.

        Returns:
            List[Dict[str, Any]]: List of evaluation question dictionaries.
        """
        with open(self.questions_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def keyword_match(self, answer: str, keywords: List[str]) -> Tuple[int, List[str]]:
        """
        Count how many expected keywords appear in the generated answer.

        Args:
            answer (str): Generated answer from the RAG system.
            keywords (List[str]): Expected keywords for the question.

        Returns:
            Tuple[int, List[str]]:
                - Number of matched keywords.
                - List of matched keywords.
        """
        answer_lower = answer.lower()

        matches = []
        for kw in keywords:
            if kw.lower() in answer_lower:
                matches.append(kw)

        return len(matches), matches

    def compute_global_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute aggregated evaluation metrics.

        Metrics include:
            - Average similarity score of retrieved chunks.
            - Keyword match rate across all questions.

        Args:
            results (List[Dict[str, Any]]): Evaluation results per question.

        Returns:
            Dict[str, Any]: Aggregated evaluation metrics.
        """
        if not results:
            return {"message": "There are no results."}

        total_score = sum([res["top_score"] for res in results])
        total_matches = sum([res["keyword_matches"] for res in results])
        total_expected = sum([len(res["expected_keywords"]) for res in results])

        avg_similarity = total_score / len(results) if total_score else 0

        keyword_match_rate = (
            total_matches / total_expected if total_expected != 0 else 0
        )

        metrics = {
            "avg_similarity_score": round(avg_similarity, 3),
            "keyword_match_rate": round(keyword_match_rate, 3),
            "total_questions": len(results),
        }

        return metrics

    def evaluate(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Run the full evaluation process.

        For each evaluation question:
            - Generate an answer using the RAG pipeline.
            - Measure retrieval quality.
            - Measure answer quality via keyword matching.

        Returns:
            Tuple[List[Dict[str, Any]], Dict[str, Any]]:
                - Detailed results for each question.
                - Aggregated evaluation metrics.
        """
        questions = self.load_questions()

        results = []

        print("\nStarting RAG evaluation\n")

        for q in questions:

            question = q["question"]
            expected_keywords = q["expected_keywords"]

            print(f"Question: {question}")

            response = self.rag.ask(question)

            answer = response["answer"]
            retrieved = response["sources"]

            # retrieval metrics
            top_score = retrieved[0]["score"] if retrieved else 0
            retrieved_chunks = len(retrieved)

            # answer keyword evaluation
            match_count, matches = self.keyword_match(answer, expected_keywords)

            result = {
                "question": question,
                "top_score": top_score,
                "retrieved_chunks": retrieved_chunks,
                "keyword_matches": match_count,
                "expected_keywords": expected_keywords,
                "matched_keywords": matches,
            }

            results.append(result)

            print(f"Top similarity score: {top_score:.3f}")
            print(f"Retrieved chunks: {retrieved_chunks}")
            print(f"Keyword matches: {match_count}/{len(expected_keywords)}")

        metrics = self.compute_global_metrics(results)

        print("\n--------------------------------------------")
        print("------------- EVALUATION SUMMARY -----------")
        print("--------------------------------------------")
        print(f'Total Questions: {metrics["total_questions"]}')
        print(f'Average similarity score: {metrics["avg_similarity_score"]}')
        print(f'Keyword Match Rate {metrics["keyword_match_rate"]}')

        return results, metrics


if __name__ == "__main__":

    evaluator = RAGEvaluator(questions_file="app/evaluation/evaluation_questions.json")

    results, metrics = evaluator.evaluate()

    with open("app/evaluation/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("app/evaluation/evaluation_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)
