"""Deterministic lexical and structural grading for the offline harness.

Milestone 1a validates the grader and the plumbing, not model quality.
"""
from __future__ import annotations

import re
import unicodedata

from pydantic import ValidationError

from market_intelligence.agents.schemas import (
    Comparison,
    NewsSummary,
    NewsTone,
    TopicReport,
)

PUNCT = ".,;:!?'\"()[]{}"
CUE = frozenset(
    "not never no none cannot can't isn't aren't wasn't weren't doesn't don't "
    "didn't won't wouldn't shouldn't couldn't nor neither without".split()
)
UP = frozenset({"grew", "rose", "increased", "gained", "up"})
DOWN = frozenset({"fell", "dropped", "declined", "decreased", "lost", "down"})
SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9}
CURRENCIES = frozenset("£$€")

TOKEN_RE = re.compile(
    r"^(?P<cur>[£$€])?(?P<num>\d[\d,]*(?:\.\d+)?)"
    r"(?P<unit>%|bn|billion|m|million|k|thousand)?$"
)

LIMITATIONS = (
    "This grader performs deterministic lexical and structural checks, not "
    "semantic truth judgement. Paraphrased correct facts can fail, and fluent "
    "wrong facts can pass. Negation handling is lexical, based on a closed cue "
    "list, and misses paraphrased negation and double negation. Direction "
    "handling is lexical, based on a closed verb list. Augmented unsupported "
    "content inside an otherwise matching fact is not detected as a precision "
    "failure. The key-fact metrics are named and reported as lexical matching, "
    "not factual accuracy. The harness is a regression and grading tool with "
    "documented blind spots; it does not determine whether an output is true."
)


def normalize(value: object) -> str:
    """Unicode NFKC, casefold, collapse whitespace, trim."""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(text.split()).strip()


def _tokens(text: str) -> list[str]:
    tokens = []
    for raw in normalize(text).split():
        token = raw.strip(PUNCT)
        if token:
            tokens.append(token)
    return tokens


def _scan(text: str):
    """Return (words, claims, has_cue, has_up, has_down)."""
    tokens = _tokens(text)
    words: list[str] = []
    claims: list[dict] = []
    has_cue = any(t in CUE for t in tokens)
    has_up = any(t in UP for t in tokens)
    has_down = any(t in DOWN for t in tokens)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        match = TOKEN_RE.match(token)
        if match is not None:
            unit = match.group("unit")
            if unit is None and i + 1 < len(tokens) and tokens[i + 1] in SCALE:
                unit = tokens[i + 1]
                i += 1
            claims.append(
                {
                    "cur": match.group("cur"),
                    "value": float(match.group("num").replace(",", "")),
                    "unit": unit,
                }
            )
            i += 1
            continue
        if token in CURRENCIES and i + 1 < len(tokens):
            nxt = TOKEN_RE.match(tokens[i + 1])
            if nxt is not None and nxt.group("cur") is None:
                unit = nxt.group("unit")
                consumed = 2
                if unit is None and i + 2 < len(tokens) and tokens[i + 2] in SCALE:
                    unit = tokens[i + 2]
                    consumed = 3
                claims.append(
                    {
                        "cur": token,
                        "value": float(nxt.group("num").replace(",", "")),
                        "unit": unit,
                    }
                )
                i += consumed
                continue
        if token in CUE or token in UP or token in DOWN:
            i += 1
            continue
        words.append(token)
        i += 1
    return words, claims, has_cue, has_up, has_down


def _scaled(claim: dict) -> float:
    unit = claim["unit"]
    return claim["value"] * SCALE[unit] if unit in SCALE else claim["value"]


def _assess(e: dict, g: dict) -> str:
    """Return 'equal', 'different' or 'unassessed' for one aligned claim pair."""
    eu, gu = e["unit"], g["unit"]
    ek = "scale" if eu in SCALE else ("percent" if eu == "%" else "none")
    gk = "scale" if gu in SCALE else ("percent" if gu == "%" else "none")
    if (e["cur"] is None) != (g["cur"] is None):
        return "unassessed"
    if e["cur"] is not None and e["cur"] != g["cur"]:
        return "unassessed"
    if ek != gk:
        return "unassessed"
    return "equal" if _scaled(e) == _scaled(g) else "different"


def _numeric_status(e_claims, g_claims, e_up, e_down, g_up, g_down) -> str:
    if len(e_claims) != len(g_claims):
        return "unassessed"
    statuses = [_assess(e, g) for e, g in zip(e_claims, g_claims)]
    if "unassessed" in statuses:
        return "unassessed"
    if (e_up and g_down) or (e_down and g_up):
        return "conflict"
    if "different" in statuses:
        return "conflict"
    return "equal"


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    index = 0
    for token in haystack:
        if index < len(needle) and token == needle[index]:
            index += 1
    return index == len(needle)


def _pair_key(e_words, g_words, g_text):
    return (len(g_words) - len(e_words), -len(e_words), normalize(g_text))


def _fact_parts(text: str):
    return _scan(text)


def match_key_facts(expected_facts, generated_facts):
    """Maximum-cardinality one-to-one lexical matching plus diagnostics."""
    expected = []
    for fact in expected_facts:
        words, claims, cue, up, down = _fact_parts(fact["text"])
        expected.append(
            {
                "id": fact["fact_id"],
                "text": fact["text"],
                "words": words,
                "claims": claims,
                "cue": cue,
                "up": up,
                "down": down,
            }
        )
    generated = []
    for text in generated_facts:
        words, claims, cue, up, down = _fact_parts(text)
        generated.append(
            {
                "text": text,
                "words": words,
                "claims": claims,
                "cue": cue,
                "up": up,
                "down": down,
            }
        )
    expected.sort(key=lambda item: item["id"])

    valid: dict[int, list[int]] = {}
    for i, e in enumerate(expected):
        candidates = []
        for j, g in enumerate(generated):
            if not _is_subsequence(e["words"], g["words"]):
                continue
            if e["cue"] != g["cue"]:
                continue
            if (
                _numeric_status(
                    e["claims"], g["claims"], e["up"], e["down"], g["up"], g["down"]
                )
                != "equal"
            ):
                continue
            candidates.append(j)
        candidates.sort(
            key=lambda j: (_pair_key(e["words"], generated[j]["words"], generated[j]["text"]), j)
        )
        valid[i] = candidates

    match_e: dict[int, int] = {}  # expected index -> generated index
    match_g: dict[int, int] = {}  # generated index -> expected index

    def augment(i: int, seen: set[int]) -> bool:
        for j in valid.get(i, []):
            if j in seen:
                continue
            seen.add(j)
            if j not in match_g or augment(match_g[j], seen):
                match_g[j] = i
                match_e[i] = j
                return True
        return False

    for i in range(len(expected)):
        augment(i, set())

    matches = sorted((i, j) for i, j in match_e.items())
    left_g = set(range(len(generated))) - set(match_g)

    negation = numeric = ag = ae = ug = ue = 0
    for i, e in enumerate(expected):
        if i in match_e:
            continue
        candidates = [
            j
            for j in sorted(
                left_g,
                key=lambda j: (_pair_key(e["words"], generated[j]["words"], generated[j]["text"]), j),
            )
            if _is_subsequence(e["words"], generated[j]["words"])
        ]
        if not candidates:
            ue += 1
            continue
        j = candidates[0]
        left_g.discard(j)
        g = generated[j]
        if e["cue"] != g["cue"]:
            negation += 1
            continue
        status = _numeric_status(
            e["claims"], g["claims"], e["up"], e["down"], g["up"], g["down"]
        )
        if status == "conflict":
            numeric += 1
        elif status == "unassessed":
            ae += 1
            ag += 1
        else:
            ue += 1
            ug += 1
    ug += len(left_g)

    diagnostics = {
        "key_fact_augmented_matches": sum(
            1
            for i, j in matches
            if len(generated[j]["words"]) > len(expected[i]["words"])
        ),
        "key_fact_negation_conflicts": negation,
        "key_fact_numeric_conflicts": numeric,
        "key_fact_unassessed_generated": ag,
        "key_fact_unassessed_expected": ae,
        "key_fact_unmatched_generated": ug,
        "key_fact_unmatched_expected": ue,
    }
    return matches, diagnostics, {"cg": negation + numeric, "ag": ag}


def fact_metrics(matched, expected_count, generated_count, diagnostics, sides):
    cg = sides["cg"]
    ag = sides["ag"]
    ce = diagnostics["key_fact_negation_conflicts"] + diagnostics["key_fact_numeric_conflicts"]
    ae = diagnostics["key_fact_unassessed_expected"]
    ug = diagnostics["key_fact_unmatched_generated"]
    ue = diagnostics["key_fact_unmatched_expected"]
    return {
        "precision": _ratio(matched, matched + ug + cg),
        "recall": _ratio(matched, matched + ue + ce),
        "coverage_generated": _ratio(matched + ug + cg, matched + ug + cg + ag),
        "coverage_expected": _ratio(matched + ue + ce, matched + ue + ce + ae),
    }


def _ratio(numerator: int, denominator: int):
    return None if denominator == 0 else numerator / denominator


def _set_f1(expected: set[str], generated: set[str]) -> float:
    if not expected and not generated:
        return 1.0
    common = len(expected & generated)
    if expected and generated and common == 0:
        return 0.0
    precision = common / len(generated) if generated else 0.0
    recall = common / len(expected) if expected else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _band(value: float, band: dict) -> bool:
    return band["min"] <= value <= band["max"]


def _sign_ok(overall: str, score: float) -> bool:
    if overall == "positive":
        return score > 0
    if overall == "negative":
        return score < 0
    return True


def _validate(model, data):
    try:
        model.model_validate(data)
        return True
    except ValidationError:
        return False


def _article_by_ref(articles, ref):
    for article in articles:
        if article["article_ref"] == ref:
            return article
    return None


def _doc_summary_for_article(document, article):
    for summary in document.get("summaries", []):
        if normalize(summary.get("source_title", "")) == normalize(article["title"]):
            return summary
    return None


def _grade_summaries(document, expected, articles):
    expected_summaries = expected.get("summaries", [])
    tone_correct = 0
    headline_ok = 0
    summary_ok = 0
    facts_expected = []
    facts_generated = []
    for item in expected_summaries:
        article = _article_by_ref(articles, item["article_ref"])
        summary = _doc_summary_for_article(document, article) if article else None
        if summary is None:
            continue
        if summary.get("tone_hint") == item["tone_hint"]:
            tone_correct += 1
        if all(
            normalize(phrase) in normalize(summary.get("headline", ""))
            for phrase in item.get("headline_must_include", [])
        ):
            headline_ok += 1
        if all(
            normalize(phrase) in normalize(summary.get("summary", ""))
            for phrase in item.get("summary_must_include", [])
        ):
            summary_ok += 1
        facts_expected.extend(item.get("key_facts", []))
        facts_generated.extend(summary.get("key_facts", []))
    matches, diagnostics, sides = match_key_facts(facts_expected, facts_generated)
    pairs = fact_metrics(len(matches), len(facts_expected), len(facts_generated), diagnostics, sides)
    return {
        "news.tone_hint_accuracy": _ratio(tone_correct, len(expected_summaries)),
        "news.headline_containment_rate": _ratio(headline_ok, len(expected_summaries)),
        "news.summary_containment_rate": _ratio(summary_ok, len(expected_summaries)),
        "news.key_fact_lexical_precision": pairs["precision"],
        "news.key_fact_lexical_recall": pairs["recall"],
        "diagnostics": diagnostics,
        "key_fact_coverage_generated": pairs["coverage_generated"],
        "key_fact_coverage_expected": pairs["coverage_expected"],
    }


def _grade_topics(document, expected, articles):
    expected_topics = expected.get("topics", [])
    doc_topics = document.get("topics", [])
    aliases = [{normalize(alias) for alias in t["topic_aliases"]} for t in expected_topics]
    edges = []
    for topic in doc_topics:
        name = normalize(topic.get("topic_name", ""))
        edges.append([j for j, alias_set in enumerate(aliases) if name in alias_set])
    match_g: dict[int, int] = {}
    match_doc: dict[int, int] = {}
    match_expected: dict[int, int] = {}

    def augment(i: int, seen: set[int]) -> bool:
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in match_g or augment(match_g[j], seen):
                match_g[j] = i
                match_doc[i] = j
                match_expected[j] = i
                return True
        return False

    for i in range(len(doc_topics)):
        augment(i, set())

    f1_values = []
    for j, expected_topic in enumerate(expected_topics):
        if j in match_expected:
            generated = doc_topics[match_expected[j]]
            f1_values.append(
                _set_f1(
                    {normalize(k) for k in expected_topic.get("keywords", [])},
                    {normalize(k) for k in generated.get("keywords", [])},
                )
            )
        else:
            f1_values.append(0.0)
    keyword_f1 = _ratio(sum(f1_values), len(f1_values))

    title_map = {normalize(a["title"]): a for a in articles}
    summary_by_id = {s.get("article_id"): s for s in document.get("summaries", [])}
    valid_citations = 0
    total_citations = 0
    for topic in doc_topics:
        for citation in topic.get("evidence", []):
            total_citations += 1
            summary = summary_by_id.get(citation.get("article_id"))
            if summary is None:
                continue
            article = title_map.get(normalize(summary.get("source_title", "")))
            if article is not None and normalize(citation.get("quote", "")) in normalize(
                article.get("excerpt", "")
            ):
                valid_citations += 1

    with_evidence = [
        (j, t) for j, t in enumerate(expected_topics) if t.get("evidence")
    ]
    covered = 0
    for j, expected_topic in with_evidence:
        if j not in match_expected:
            continue
        generated = doc_topics[match_expected[j]]
        accepted = False
        for citation in generated.get("evidence", []):
            summary = summary_by_id.get(citation.get("article_id"))
            if summary is None:
                continue
            article = title_map.get(normalize(summary.get("source_title", "")))
            if article is None:
                continue
            quote = normalize(citation.get("quote", ""))
            if quote not in normalize(article.get("excerpt", "")):
                continue
            for span in expected_topic.get("evidence", []):
                if span["article_ref"] == article["article_ref"] and quote in {
                    normalize(q) for q in span.get("quotes", [])
                }:
                    accepted = True
                    break
            if accepted:
                break
        if accepted:
            covered += 1

    return {
        "topics.topic_alias_accuracy": _ratio(len(match_expected), len(expected_topics)),
        "topics.keyword_f1": keyword_f1,
        "topics.evidence_coverage": _ratio(covered, len(with_evidence)),
        "topics.citation_precision": _ratio(valid_citations, total_citations),
    }


def _grade_tones(document, expected, articles):
    expected_tones = expected.get("tones", [])
    doc_tones = document.get("tones", [])
    title_map = {normalize(a["title"]): a for a in articles}
    overall_correct = 0
    band_correct = 0
    aspect_values = []
    for item in expected_tones:
        article = _article_by_ref(articles, item["article_ref"])
        wanted_id = None
        if article is not None:
            summary = _doc_summary_for_article(document, article)
            if summary is not None:
                wanted_id = summary.get("article_id")
        tone = next((t for t in doc_tones if t.get("article_id") == wanted_id), None)
        if tone is None:
            aspect_values.append(0.0)
            continue
        label_ok = tone.get("overall") == item["overall"]
        if label_ok:
            overall_correct += 1
        score_ok = (
            label_ok
            and _band(tone.get("score", -99.0), item["score"])
            and _sign_ok(tone.get("overall", ""), tone.get("score", 0.0))
        )
        if score_ok:
            band_correct += 1
        expected_aspects = {normalize(a["aspect"]): a for a in item.get("aspects", [])}
        doc_aspects = {normalize(a["aspect"]): a for a in tone.get("aspects", [])}
        correct = set()
        for name, spec in expected_aspects.items():
            generated = doc_aspects.get(name)
            if generated is None:
                continue
            if generated.get("tone") != spec["tone"]:
                continue
            if not _band(generated.get("score", -99.0), spec["score"]):
                continue
            correct.add(name)
        precision = _ratio(len(correct), len(doc_aspects)) if doc_aspects else (1.0 if not expected_aspects else 0.0)
        recall = _ratio(len(correct), len(expected_aspects)) if expected_aspects else (1.0 if not doc_aspects else 0.0)
        if precision is None or recall is None:
            aspect_values.append(0.0)
        elif precision + recall == 0:
            aspect_values.append(0.0)
        else:
            aspect_values.append(2 * precision * recall / (precision + recall))
    return {
        "tones.overall_accuracy": _ratio(overall_correct, len(expected_tones)),
        "tones.score_within_band_rate": _ratio(band_correct, len(expected_tones)),
        "tones.aspect_f1": _ratio(sum(aspect_values), len(aspect_values)),
    }


def _grade_comparison(document, expected, pages):
    expected_comp = expected.get("comparison", {})
    doc_comp = document.get("comparison", {})
    page_text = {(p["side"], p["source_id"]): p["text"] for p in pages}
    matches = 0
    total = 0
    violations = 0
    valid_citations = 0
    total_citations = 0
    for side_name in ("subject", "competitor"):
        spec = expected_comp.get(side_name)
        if spec is None:
            continue
        doc_side = doc_comp.get(side_name)
        if doc_side is None:
            total += 3
            continue
        for field_name in ("product_focus", "pricing_model", "positioning"):
            total += 1
            field_spec = spec["fields"][field_name]
            field = doc_side.get(field_name, {})
            value = field.get("value")
            evidence = field.get("evidence", [])
            if (value is None) != (len(evidence) == 0):
                violations += 1
            if "value" in field_spec:
                if field_spec["value"] is None:
                    if value is None:
                        matches += 1
                elif value is not None and normalize(value) == normalize(field_spec["value"]):
                    matches += 1
            elif field_spec.get("value_any"):
                if value is not None and normalize(value) in {
                    normalize(v) for v in field_spec["value_any"]
                }:
                    matches += 1
            else:
                if value is None:
                    matches += 1
            for citation in evidence:
                total_citations += 1
                text = page_text.get((side_name, citation.get("source_id")))
                if text is not None and normalize(citation.get("quote", "")) in normalize(text):
                    valid_citations += 1
    return {
        "comparison.field_value_match_rate": _ratio(matches, total),
        "comparison.evidence_valid_rate": _ratio(valid_citations, total_citations),
        "comparison.none_evidence_violations": violations,
    }


def grade_document(document: dict, expected: dict, articles: list, pages: list) -> dict:
    """Grade one enriched document against one expectation set."""
    invalid_sections = []
    summaries_valid = bool(document.get("summaries")) and all(
        _validate(NewsSummary, summary) for summary in document.get("summaries", [])
    )
    if not summaries_valid:
        invalid_sections.append("summaries")
        document = {**document, "summaries": []}
    if any(not _validate(TopicReport, topic) for topic in document.get("topics", [])):
        invalid_sections.append("topics")
        document = {**document, "topics": []}
    if any(not _validate(NewsTone, tone) for tone in document.get("tones", [])):
        invalid_sections.append("tones")
        document = {**document, "tones": []}
    if document.get("comparison") is not None and not _validate(
        Comparison, document["comparison"]
    ):
        invalid_sections.append("comparison")
        document = {**document, "comparison": {}}

    news = _grade_summaries(document, expected, articles)
    topics = _grade_topics(document, expected, articles)
    tones = _grade_tones(document, expected, articles)
    comparison = _grade_comparison(document, expected, pages)

    if "summaries" in invalid_sections:
        news["news.tone_hint_accuracy"] = 0.0
        news["news.headline_containment_rate"] = 0.0
        news["news.summary_containment_rate"] = 0.0
        news["news.key_fact_lexical_precision"] = 0.0
        news["news.key_fact_lexical_recall"] = 0.0
        news["key_fact_coverage_generated"] = None
        news["key_fact_coverage_expected"] = 1.0
        news["diagnostics"] = {
            "key_fact_augmented_matches": 0,
            "key_fact_negation_conflicts": 0,
            "key_fact_numeric_conflicts": 0,
            "key_fact_unassessed_generated": 0,
            "key_fact_unassessed_expected": 0,
            "key_fact_unmatched_generated": 0,
            "key_fact_unmatched_expected": len(expected.get("summaries", [])),
        }
    if "topics" in invalid_sections:
        topics["topics.topic_alias_accuracy"] = 0.0
        topics["topics.keyword_f1"] = 0.0
        topics["topics.evidence_coverage"] = (
            0.0 if any(t.get("evidence") for t in expected.get("topics", [])) else None
        )
        topics["topics.citation_precision"] = None
    if "tones" in invalid_sections:
        tones["tones.overall_accuracy"] = 0.0
        tones["tones.score_within_band_rate"] = 0.0
        tones["tones.aspect_f1"] = 0.0
    if "comparison" in invalid_sections:
        comparison["comparison.field_value_match_rate"] = 0.0
        comparison["comparison.evidence_valid_rate"] = None
        comparison["comparison.none_evidence_violations"] = 0

    diagnostics = dict(news["diagnostics"])
    diagnostics["key_fact_assessable_coverage_generated"] = news["key_fact_coverage_generated"]
    diagnostics["key_fact_assessable_coverage_expected"] = news["key_fact_coverage_expected"]

    return {
        "news.tone_hint_accuracy": news["news.tone_hint_accuracy"],
        "news.headline_containment_rate": news["news.headline_containment_rate"],
        "news.summary_containment_rate": news["news.summary_containment_rate"],
        "news.key_fact_lexical_precision": news["news.key_fact_lexical_precision"],
        "news.key_fact_lexical_recall": news["news.key_fact_lexical_recall"],
        "topics.topic_alias_accuracy": topics["topics.topic_alias_accuracy"],
        "topics.keyword_f1": topics["topics.keyword_f1"],
        "topics.evidence_coverage": topics["topics.evidence_coverage"],
        "topics.citation_precision": topics["topics.citation_precision"],
        "tones.overall_accuracy": tones["tones.overall_accuracy"],
        "tones.score_within_band_rate": tones["tones.score_within_band_rate"],
        "tones.aspect_f1": tones["tones.aspect_f1"],
        "comparison.field_value_match_rate": comparison["comparison.field_value_match_rate"],
        "comparison.evidence_valid_rate": comparison["comparison.evidence_valid_rate"],
        "comparison.none_evidence_violations": comparison["comparison.none_evidence_violations"],
        "pipeline.invalid_output_rate": 1.0 if invalid_sections else 0.0,
        "diagnostics": diagnostics,
        "invalid_output": bool(invalid_sections),
        "invalid_sections": invalid_sections,
    }
