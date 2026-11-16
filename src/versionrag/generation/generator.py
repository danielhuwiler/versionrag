"""Answer generation layer that produces natural-language answers from retrieval results."""

import logging

from versionrag.clients.llm import LLMClient
from versionrag.config import settings
from versionrag.generation.prompts import build_generation_prompt
from versionrag.models.schemas import Answer, ParsedQuery, RetrievalResult

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generates natural-language answers from parsed queries and retrieval results."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def generate(
        self, parsed: ParsedQuery, result: RetrievalResult
    ) -> Answer:
        """Generate a final answer by sending retrieval context to the LLM.

        Builds the prompt from the parsed query and retrieval result, calls the
        standard-tier LLM model, and packages the response into an Answer with
        source attribution.

        Args:
            parsed: The structured parsed query with the original question.
            result: The retrieval result containing chunks, changes, and/or versions.

        Returns:
            Answer with the generated text, sources, and metadata.
        """
        logger.debug(
            "Generating answer: query_type=%s, chunks=%d, changes=%d, versions=%d",
            parsed.query_type,
            len(result.chunks),
            len(result.changes),
            len(result.versions),
        )

        # Build the prompt pair from retrieval context
        system_prompt, user_prompt = build_generation_prompt(parsed, result)

        # Call the LLM with the standard model tier
        answer_text = await self._llm.chat(
            prompt=user_prompt,
            system_prompt=system_prompt,
            model=settings.llm_model_standard,
            purpose="evaluation.answer_generation",
        )

        logger.info(
            "Answer generated: model=%s, answer_len=%d, query_type=%s",
            settings.llm_model_standard,
            len(answer_text),
            parsed.query_type,
        )

        # Collect sources as plain dicts from both chunks and changes
        sources: list[dict] = [c.model_dump() for c in result.chunks] + [
            c.model_dump() for c in result.changes
        ]

        return Answer(
            question=parsed.question,
            answer=answer_text,
            sources=sources,
            retrieval_mode=result.retrieval_mode,
            query_type=parsed.query_type,
        )
