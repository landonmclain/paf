"""LongMemEval answer-judge helpers, ported VERBATIM from the official scorer.

Source of truth: github.com/xiaowu0162/LongMemEval `src/evaluation/evaluate_qa.py`.
Contains the binary yes/no correctness judge (get_lme_anscheck_prompt + the
label extraction) and the per-task type list used by the recall harness.

This module was carved out of the old `locomo_scoring.py` when LoCoMo was
retired (2026-08): the LoCoMo token-F1 answer scorer, the evidence-recall
heuristic (`tool_surfaced_evidence`), and the long-answer helpers lived
alongside it. They are gone. This file holds ONLY the LongMemEval
answer-correctness surface:

  - LME_TYPES            the six official LongMemEval question categories
  - get_lme_anscheck_prompt  VERBATIM per-task judge prompt
  - lme_is_abstention    does this question expect abstention ('_abs')?
  - lme_parse_label      extract the binary label from a judge response

The retrieval/recall metric that pairs with these lives in `run_lmem.py` as
ID-membership recall@k over surfaced `session=` tags (`_id_membership_recall`) --
it needs no token-overlap F1.

Do not paraphrase the judge prompt: the per-task clauses (off-by-one dates,
updated-answer-wins, rubric personalization, abstain-recognition) are what make
the scoring faithful to the official benchmark.
"""

LME_TYPES = [
    "single-session-user",
    "single-session-preference",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]


def get_lme_anscheck_prompt(
    task: str,
    question: str,
    answer: str,
    response: str,
    abstention: bool = False,
) -> str:
    """Official LongMemEval judge prompt, VERBATIM port of `get_anscheck_prompt`.

    Source of truth: github.com/xiaowu0162/LongMemEval `src/evaluation/evaluate_qa.py`.
    Yields a binary yes/no answer; 'yes' <=> correct (see `lme_parse_label`).
    Do not paraphrase: the per-task clauses (off-by-one dates, updated-answer-wins,
    rubric personalization, abstain-recognition) are what make the scoring faithful.
    """
    if not abstention:
        if task in ("single-session-user", "single-session-assistant", "multi-session"):
            template = (
                "I will give you a question, a correct answer, and a response "
                "from a model. Please answer yes if the response contains the correct answer. "
                "Otherwise, answer no. If the response is equivalent to the correct answer or "
                "contains all the intermediate steps to get the correct answer, you should also "
                "answer yes. If the response only contains a subset of the information required by "
                "the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel "
                "Response: {}\n\nIs the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "temporal-reasoning":
            template = (
                "I will give you a question, a correct answer, and a response "
                "from a model. Please answer yes if the response contains the correct answer. "
                "Otherwise, answer no. If the response is equivalent to the correct answer or "
                "contains all the intermediate steps to get the correct answer, you should also "
                "answer yes. If the response only contains a subset of the information required by "
                "the answer, answer no. In addition, do not penalize off-by-one errors for the "
                "number of days. If the question asks for the number of days/weeks/months, etc., "
                "and the model makes off-by-one errors (e.g., predicting 19 days when the answer "
                "is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect "
                "Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes "
                "or no only."
            )
            return template.format(question, answer, response)
        if task == "knowledge-update":
            template = (
                "I will give you a question, a correct answer, and a response "
                "from a model. Please answer yes if the response contains the correct answer. "
                "Otherwise, answer no. If the response contains some previous information along "
                "with an updated answer, the response should be considered as correct as long as "
                "the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect "
                "Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes "
                "or no only."
            )
            return template.format(question, answer, response)
        if task == "single-session-preference":
            template = (
                "I will give you a question, a rubric for desired personalized "
                "response, and a response from a model. Please answer yes if the response satisfies "
                "the desired response. Otherwise, answer no. The model does not need to reflect all "
                "the points in the rubric. The response is correct as long as it recalls and "
                "utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: "
                "{}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        raise NotImplementedError(task)
    template = (
        "I will give you an unanswerable question, an explanation, and a response from "
        "a model. Please answer yes if the model correctly identifies the question as "
        "unanswerable. The model could say that the information is incomplete, or some other "
        "information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: "
        "{}\n\nModel Response: {}\n\nDoes the model correctly identify the question as "
        "unanswerable? Answer yes or no only."
    )
    return template.format(question, answer, response)


def lme_is_abstention(question_id: str | object) -> bool:
    """LongMemEval abstention questions carry a trailing '_abs' in their id."""
    return "_abs" in str(question_id)


def lme_parse_label(text: str) -> bool:
    """Official label extraction: `label = 'yes' in eval_response.lower()`."""
    return "yes" in (text or "").lower()
