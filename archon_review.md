# Archon Review of Pydantic AI Implementation (July 24, 2024)

This document contains the review and suggestions provided by Archon regarding the Pydantic AI implementation in `src/agent.py`, `src/agent_tools.py`, and `src/agent_prompts.py`.

## Initial Request

Please review the following Python files (`src/agent.py`, `src/agent_tools.py`, `src/agent_prompts.py`) which implement an agent using Pydantic AI. Focus on potential improvements to the Pydantic AI usage, model validation, error handling, and overall agent design related to Pydantic AI. Provide specific suggestions for improvement.

_(File contents omitted for brevity)_

## Archon Response (Thread ID: 899eb172-2aeb-431f-8a00-75d39c33ae8a)

Here is a **focused expert review** of your `src/agent.py`, `src/agent_tools.py`, and `src/agent_prompts.py` with improvement recommendations emphasizing **Pydantic AI best practices, validation**, **error handling**, and overall **architecture** related to structured agent design.

---

# 1. Pydantic AI Output Workflow

### Issue

- **Manual parsing of LLM output:**  
  Your LLM is instructed to output JSON, but you use regex and normalization logic in `_parse_response` to repair/validate output, then call Pydantic validation. This is fragile (LLMs sometimes output nearly-correct but not-compliant structures).
- **Agent/LLM Invocation does not leverage `response_model`:**  
  [Pydantic AI recommends providing a `response_model` or equivalent to have the LLM return results that can be natively parsed as your Pydantic model(s)](https://ai.pydantic.dev/api/agent/#response-model). This eliminates much response-parsing and moves validation guarantees into the agent inference itself.

### Recommendations

- **Move all output validation to Pydantic**:  
  Leverage your `RankedArticle` model directly as:

  ```python
  response = await self.agent.run(user_message, response_model=List[RankedArticle])
  ```

  (adjust for your Agent's API—if it doesn't accept `response_model`, consider upgrading, or implement an output parser as shown in docs.)

- **Eliminate the `_normalize_article_data`/manual normalization:**  
  Instead, move normalization to Pydantic's `@model_validator(mode="before")` on `RankedArticle` (see [Pydantic v2 model validators](https://docs.pydantic.dev/latest/concepts/validators/#root-and-model-validators)).  
  This ensures any inbound dict (from LLM or API) "self-heals" fields before validation.

- **If necessary, retry on format errors**:  
  If parsing fails, you can instruct the agent to retry (see ModelRetry), passing in "the previous output was not JSON" as context.

---

# 2. Model Validation

## What You're Doing Well

- Use of Pydantic models to define contract for ranked articles and analysis.

## Improvements

- **Centralize and Move All Normalization to Model Validators**

  - In `RankedArticle`, add a `@model_validator(mode='before')` that:
    - Accepts synonyms (`subject`/`subjects`, etc.)
    - Repairs `authors` field (str→list, fill "Unknown", etc.)
    - Intelligently repairs URL variants
  - _Remove normalization from agent/tools code_, pass raw output to the model.

  Example:

  ```python
  class RankedArticle(BaseModel):
      ...
      @model_validator(mode="before")
      def fix_article_fields(cls, values):
          # repair synonyms, types, scores, etc.
          ...
          return values
  ```

- **Surface Validation Failures in Errors/Output**

  - Instead of silently skipping invalid articles, gather all `ValidationError`s and report them back (aggregated by field/title).

- **Strongly Type Returns**
  - Don't use `Dict[str, Any]` or `Any`—use concrete model types throughout except at interface boundaries.

---

# 3. Error Handling

## What You're Doing

- Wraps most agent/LLM interactions with try/catch
- Logs validation errors and skips bad articles

## Improvements

- **Fail Fast and Verbosely on Validation**

  - If multiple output articles are dropped for type/structure issues, print a summary of errors: which fields, what value, why dropped.

- **Retry on Recoverable Errors**

  - On failed JSON parse/response format, explicitly instruct the LLM in the next run to reformat output. Pydantic AI supports ModelRetry for this.

- **Consistent LLM Output Correction**

  - When the output isn't valid JSON for your model, retry with a clarifying system prompt.

- **Error Propagation**
  - When an article analysis or scrape fails, record that error in the result (e.g., as an "error" field).
  - This allows end-users (and yourself) to see what failed and why.

---

# 4. Agent & Tool Structure

## What You Do Well

- Clear file separation: core agent, tools, and prompts.

## Improvements

- **Remove Duplicated `_normalize_article_data`**

  - It's duplicated in `agent.py` and (commented) in `agent_tools.py`. This should be a model-local validator.

- **Separation of Pydantic Models**

  - Consider moving data models (RankedArticle, etc.) into their own module (e.g., `models.py` or `article_models.py`), imported by both agent and tools.

- **Tool Interfaces**

  - Annotate all tools with proper input/output Pydantic types, not Dict/Any.

- **Crawler Use in Batch Tools**
  - In your batch scraping code, don't create per-coroutine crawler instances; pass a crawler as a dependency and share (as you do in single scrape).

---

# 5. Prompt Structure & LLM Guidance

## What You Do Well

- The system prompt clearly asks for valid, strict JSON with schema.

## Improvements

- **If using Function/Tool Calls or response_model, you can drop much of the JSON guidance** since the returned output is automatically aligned.
- **Recommended:** include your _actual_ Pydantic schema for your article model in the prompt if you _aren't_ using response_model parsing.

- **Be explicit on "no markdown, no preamble, JSON only"** in both SYSTEM_PROMPT and analysis prompt (which you do—good!).

---

# 6. Observability and Logging

- Good use of logger in most places.
- _Minor_: For `agent_tools.py`, only set up handlers if not already attached, else logs may duplicate.

---

# 7. Testing, Dependency Injection, and Reuse

- Consider parameterizing the agent/LLM/crawler as dependencies in tool/APIs to facilitate testing and mocking.
- For large batches, consider reusing LLM/Agent objects instead of per-request instantiation.

---

# 8. Concrete Action Points Table

| Issue                                      | Suggestion                                                                             |
| ------------------------------------------ | -------------------------------------------------------------------------------------- |
| ~~Manual LLM output parsing/norm~~         | ~~Use Agent.run with response_model=List[RankedArticle] so output is model-validated~~ |
| ~~Normalization logic spread across code~~ | ~~Move all normalization to @model_validator(mode="before") on the Pydantic model~~    |
| Validation errors silently skipped         | Surface validation errors and aggregate summary, don't just log+skip                   |
| Inconsistent field names (subject etc.)    | Handle these in the @model_validator, not scattered in code                            |
| ~~Duplicated normalization logic~~         | ~~Remove duplicate functions, use model-only normalization~~                           |
| Scraping/Analysis error not surfaced       | Add 'error' (str) field when scraping or analysis fails, not just skipping             |
| Batch scraping instantiates per routine    | Pass/reuse shared crawler (async context manager, same as LLM agent)                   |
| Type hints with Dict/Any                   | Always use concrete Pydantic model types at function signatures                        |
| Too manual prompt-correction on error      | On format/validation errors, retry with LLM model, using ModelRetry or re-instruction  |
| ~~Models local to files~~                  | ~~Put all models (RankedArticle, ArticleAnalysis, ...) in one `models.py` module~~     |

---

# 9. Example: Model Validator Migration

Here's how you could migrate normalization into your model:

```python
class RankedArticle(BaseModel):
    title: str
    authors: List[str]
    subject: str
    score_reason: str
    relevance_score: int
    abstract_url: str = ""
    html_url: str = ""
    pdf_url: str = ""

    @model_validator(mode='before')
    @classmethod
    def normalize_fields(cls, values):
        # Authors synonym repair, type coercion, arxiv_id extraction, score clamp, etc.
        # Move the logic from `_normalize_article_data` here.
        return values
```

Then, everywhere validation/normalization is needed:

```python
validated = RankedArticle.model_validate(data)
```

No further normalization logic required _anywhere_ else.

---

# 10. Documentation/Reference

- [Pydantic AI: Agent API (`response_model` usage)](https://ai.pydantic.dev/api/agent/#response-model)
- [Pydantic Model Validators (V2+)](https://docs.pydantic.dev/latest/concepts/validators/#root-and-model-validators)
- [Weather agent example: model-level normalization & output contract](https://ai.pydantic.dev/examples/agents/weather/)

---

# Summary

## **Priority Actions:**

1. ~~Move ALL normalization to model validators.~~ (**DONE** - Validators added in `src/models.py`)
2. ~~Use `response_model` in Agent.run for direct model-validated output—no regex/manual parsing.~~ (**DONE** - Implemented in `src/agent.py`)
3. **Aggregate and report validation/analysis/scraping errors, not just skip them.** (**Partially Done** - Errors logged, but no aggregation yet)
4. **Eliminate duplicate code and weakly-typed returns; use strong Pydantic typing everywhere.** (**DONE** - Refactored `agent.py`, `agent_tools.py`, `state.py` to use models from `src/models.py`)
5. ~~Consider modularizing shared models into their own file.~~ (**DONE** - Created `src/models.py`)

Applying these will make your Pydantic AI agent more robust, maintainable, and production-grade per Pydantic AI best practices.

---

Would you like example code for migrating normalization to model validators, or a refactored version of your agent using these patterns? Or would you like the agent to be autonomously refined ("refine")?

---

## Implementation Summary (April 21, 2025)

Based on the Archon review (Thread ID: 899eb172...), the following changes were implemented:

1.  **Centralized Pydantic Models:** Created `src/models.py` and moved `RankedArticle`, `ArticleAnalysis`, and `UserContext` definitions there. Added `AgentStateModel` to this file.
2.  **Strong Typing:**
    - Refactored `src/agent.py` to import models from `src/models.py`, removed local definitions, and updated function signatures/internal logic to use these models (`UserContext`, `List[RankedArticle]`, `List[ArticleAnalysis]`).
    - Refactored `src/agent_tools.py` to update `analyze_article` signature and return type to use `UserContext` and `ArticleAnalysis`.
    - Refactored `src/state.py` to remove the local `AgentStateModel`, import the central one, and update methods to use the specific types (`Dict[str, RankedArticle]`, `Dict[str, UserContext]`). Implemented `model_dump_json` / `model_validate_json` for saving/loading.
3.  **Pydantic AI `response_model`:** Confirmed `arxiv_agent` in `src/agent.py` uses `response_model=List[RankedArticle]`.
4.  **Model Validators:** Added `@field_validator` and `@property` logic within `src/models.py` for basic normalization (authors, title, URLs, arxiv_id extraction).
5.  **Robust Ranking Output:** Addressed the issue where the LLM omitted optional fields (like `html_url`) during ranking:
    - Enhanced the prompt in `rank_articles` (`src/agent.py`) to explicitly instruct the LLM to copy all available fields.
    - Implemented a post-processing step in `rank_articles` to merge the LLM's ranked output with the original input data, ensuring missing fields present in the source are filled in.
6.  **Testing:** Ran `python -m src.agent` multiple times to test changes, including fixing relative import issues and verifying the file loading and post-processing logic worked as expected.

**Remaining Potential Improvements (from original review):**

- **Error Aggregation:** While errors are logged, there's no central aggregation or reporting of validation/scraping/analysis failures.
- **Model Normalization:** More complex normalization logic (e.g., handling synonyms like subject/subjects) mentioned by Archon is not yet implemented in `@model_validator` in `src/models.py`.
- **Error Propagation:** Specific error fields were not added to models to surface scraping/analysis failures directly in the output data.
