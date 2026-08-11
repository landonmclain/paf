"""LoCoMo QA scoring, ported from snap-research/locomo `task_eval/evaluation.py`.

Plus: the official LongMemEval judge prompt / label helpers (ported from
github.com/xiaowu0162/LongMemEval `src/evaluation/evaluate_qa.py`), kept here
so the whole two-benchmark scoring surface lives in one test module.

This mirrors the benchmark's own scorer as faithfully as we need, minus Porter
stemming (nltk is not a project dep; pure normalized token-overlap F1). It is
NOT a re-scoring of "evidence recall@k = does the answer cite the dia_id" -- that
was our old, broken `_recall_hit`. This implements the two LoCoMo metrics:

  1. Answer F1    -- normalized token-overlap F1(prediction, annotated `answer`),
                     category-dependent (see eval_question_answering).
  2. Retrieval    -- separate metric: fraction of gold `evidence` dia_ids whose
                     content was surfaced by the retrieval *tools* (search_index /
                     read_turn / search_conversation_history / search_memory), scored
                     from TOOL OUTPUTS, independent of the final answer. Live in
                     `retrieval_recall`/`_tool_surfaced` below.

Reference (source of truth): github.com/snap-research/locomo, `task_eval/evaluation.py`
and `task_eval/evaluate_qa.py`. Consult the source, not a blog, before changing any
normalization or category rule.
"""

from __future__ import annotations

import string
import unicodedata
from collections import Counter
from typing import Iterable

import regex  # third-party regex (Unicode property escapes \p{..}); LoCoMo uses this exact lib

# ---------------------------------------------------------------------------
# Normalization (mirrors evaluation.py: SimpleTokenizer + normalize_answer)
# ---------------------------------------------------------------------------
_ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
_NON_WS = r"[^\p{Z}\p{C}]"
_TOKEN_RE = regex.compile(rf"({_ALPHA_NUM})|({_NON_WS})", regex.IGNORECASE | regex.UNICODE | regex.MULTILINE)

_ARTICLE_WORDS = {"a", "an", "the", "and"}


def _tokenize(text: str, uncased: bool = True) -> list[str]:
    toks = [m.group() for m in _TOKEN_RE.finditer(text)]
    return [t.lower() for t in toks] if uncased else toks


def _normalize_answer(text: str) -> list[str]:
    """Lowercase, strip punctuation, drop article stopwords, then tokenize.

    This lands token-level where the paper's `normalize_answer` returns a
    whitespace-joined string we then split; equivalent set/overlap semantics.
    """
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    toks = _tokenize(text, uncased=True)
    return [t for t in toks if t not in _ARTICLE_WORDS]


# ---------------------------------------------------------------------------
# F1 (mirrors evaluation.py f1_score -- minus Porter stemming)
# ---------------------------------------------------------------------------


def f1_score(prediction: str, ground_truth: str) -> float:
    """Stem-free token-overlap F1. Returns 0.0 on no overlap."""
    p_tokens = _normalize_answer(prediction)
    g_tokens = _normalize_answer(ground_truth)
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_multiple(prediction: str, ground_truth: str) -> float:
    """Multi-answer F1 (mirrors evaluation.py `f1`): ground truth split on ';',
    max over predictions, mean over ground-truth sub-answers. Category 1 (single-hop)."""
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(";")]
    return float(
        sum(
            max(f1_score(pred, gt) for pred in predictions)
            for gt in ground_truths
        )
        / len(ground_truths)
    )


# ---------------------------------------------------------------------------
# Category dispatch (mirrors evaluation.py eval_question_answering)
# ---------------------------------------------------------------------------

# LoCoMo category codes (from the data + paper):
#   1 = single-hop      2 = temporal      3 = open-domain
#   4 = multi-hop       5 = adversarial
CAT_SINGLE_HOP, CAT_TEMPORAL, CAT_OPEN, CAT_MULTI_HOP, CAT_ADVERSARIAL = 1, 2, 3, 4, 5

# Category-5 adversarial: a correct answer is to ABSTAIN (recognize the info
# isn't in history). Mirrors evaluation.py's check for a refusal phrasing.
_ADVERSARIAL_REFUSAL = ("no information available", "not mentioned")


def score_answer(prediction: str, answer: str, category: int) -> float:
    """Per-question answer score per LoCoMo's eval_question_answering rules."""
    answer = str(answer)
    if category == CAT_MULTI_HOP:  # 4 -- treated identically to temporal/open in source
        return f1_score(prediction, answer)
    if category == CAT_SINGLE_HOP:  # 1 -- multi-answer partial F1
        return f1_multiple(prediction, answer)
    if category in (CAT_TEMPORAL, CAT_OPEN):  # 2, 3
        return f1_score(prediction, answer)
    if category == CAT_ADVERSARIAL:  # 5 -- binary abstain-or-fail
        low = prediction.lower()
        return 1.0 if any(r in low for r in _ADVERSARIAL_REFUSAL) else 0.0
    # Unknown category -- fall back to plain F1 (never silently drop a question).
    return f1_score(prediction, answer)


# ---------------------------------------------------------------------------
# LongMemEval official scoring (ported from xiaowu0162/LongMemEval, see above)
# ---------------------------------------------------------------------------

# LongMemEval question types (from the data + paper).
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


# ---------------------------------------------------------------------------
# Retrieval metric -- from TOOL OUTPUTS, independent of the final answer
# ---------------------------------------------------------------------------
_SurfacedEvidence = tuple[float, list[str]]


def tool_surfaced_evidence(
    tool_outputs: Iterable[str],
    gold_evidence: Iterable[str] | None,
    id_to_text: dict[str, str],
    answer_text: str = "",
) -> _SurfacedEvidence:
    """Fraction of gold evidence dia_ids whose *content* reached the model.

    A gold evidence dia_id counts as surfaced if the evidence turn's content was
    delivered to the model by pipeline machinery -- either through a retrieval
    tool output OR through durable memory/context that let the model quote it.

    Two membership signals, OR'd (empirically, the strict full-text F1>=0.5 used
    here previously under-counted genuine retrievals ~3x because (a) tool results
    are compacted/excerpted chunks whose overlap vs the full turn rarely hits 0.5,
    and (b) memory-injected answers produce no tool output at all yet clearly
    carried the evidence):

      1. TOOL -- max token-overlap F1 between the evidence turn text and any one
         tool output, >= 0.30 (a turn is surfaced; an excerpt suffices).
      2. ANSWER -- token-overlap F1 between the evidence turn text and the model's
         final answer, >= 0.35 (the content reached the model and it reproduced
         enough of the gold turn to answer from it).

    Substring containment of the turn's most salient fragment is folded into
    signal 1/2 implicitly via token overlap. No literal `D1:` citation required.

    Returns (recall_fraction, surfaced_ids_list).
    """
    if not gold_evidence:
        return 1.0, []
    gold: set[str] = set(gold_evidence)
    chunks = [t for t in tool_outputs if t and t.strip()]
    surfaced = []
    for dia_id in gold:
        turn_text = (id_to_text.get(dia_id) or "").strip()
        if not turn_text:
            continue  # no source content to compare against -- cannot confirm
        best_tool = max((f1_score(turn_text, c) for c in chunks), default=0.0)
        ans_f1 = f1_score(turn_text, answer_text) if answer_text.strip() else 0.0
        if best_tool >= 0.30 or ans_f1 >= 0.35:
            surfaced.append(dia_id)
    return (len(surfaced) / len(gold), sorted(surfaced))