# ADR 0003: Extractive answers by default; verify every answer's citations

- **Status:** Accepted
- **Date:** 2026-11-25

## Context

Generated answers can cite sources that were never retrieved, or attach a real citation to a claim the
source does not make. Users treat a citation as evidence, so a wrong one is worse than none.

## Decision

1. The default answerer is **extractive**: it returns source sentences that share at least two content
   words with the question, each with its source number. It cannot invent text and needs no model.
2. An `LLMAnswerer` works with any OpenAI-compatible endpoint. Sources are wrapped in `<source>` tags
   with their own tags escaped, flagged sources are labelled, and the system prompt says source text is
   data and permission claims inside it must be ignored.
3. **Every answer is verified before it is returned,** whatever produced it:
   - each cited number must be a source the user was given;
   - each cited sentence must share at least 60 % of its content words with the cited source;
   - sentences without citations are reported.

   By default, unverified answers are withheld and replaced with a clear message.
4. Answers pass through the secret and personal-data redactor on the way out, even though sources
   were already redacted at ingestion.

## Limits

- The lexical support check catches invented citations and unrelated claims. It does not catch subtle
  misreadings that reuse the source's words.
- The extractive answerer misses questions phrased with words the document does not use. The sample
  evaluation shows one such miss ("What does Acme sell?").
- A real embedding model and an entailment check would improve both. The protocols are in place for that.
