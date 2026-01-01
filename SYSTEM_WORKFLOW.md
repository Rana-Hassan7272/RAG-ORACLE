# RAG Failure Attribution System - Complete Workflow

## Overview
This document describes the complete end-to-end workflow of the RAG Failure Attribution System, from initialization to query processing and root cause analysis.

---

## System Architecture

The system consists of multiple components that work together to:
1. Process user queries through a RAG pipeline
2. Evaluate the quality of answers
3. Detect failures and identify root causes
4. Provide actionable fixes with confidence scores
5. Generate system health reports

---

## Phase 1: Initialization

### 1.1 Pipeline Setup
```python
pipeline = RAGPipeline(
    document_source="./documents",
    chunk_size=500,
    chunk_overlap=50,
    embedding_model="huggingface",
    top_k=5,
    temperature=0.7
)
```

**Components Initialized:**
- **DocumentLoader**: Loads documents from source
- **DocumentChunker**: Splits documents into chunks with overlap
- **EmbeddingStore**: Creates/persists vector embeddings
- **RAGRetriever**: Retrieves relevant chunks using similarity search
- **RAGGenerator**: Generates answers using LLM (OpenAI/Groq)
- **RAGEvaluator**: Evaluates answer quality (faithfulness, relevance, recall)
- **RootCauseOracle**: Analyzes failures and attributes root causes
- **CostOptimizer**: Tracks token usage and cost waste
- **QueryFeasibilityAnalyzer**: Detects typos and query issues
- **Tracer**: Logs all queries and results

### 1.2 Document Indexing
```
load_and_index_documents() OR load_existing_vector_store()
```
- Documents are chunked and embedded
- Vector store is created/persisted
- Ready for querying

---

## Phase 2: Query Processing Flow

### Step 1: User Query Input
```python
result = pipeline.query("What is the zone in trading?")
```

### Step 2: Retrieval
- **Input**: User question
- **Process**: 
  - Query is embedded using the same embedding model
  - Similarity search finds top-k most relevant chunks
  - Chunks are deduplicated (hash-based on first 200 chars)
- **Output**: List of retrieved document chunks with metadata

### Step 3: Answer Generation
- **Input**: Question + Retrieved chunks
- **Process**:
  - Chunks are formatted with metadata into context
  - System prompt + user prompt are constructed
  - LLM generates answer using context
- **Output**: Generated answer + full prompt used

### Step 4: Query Logging (Tracer)
- Creates unique `query_id` (timestamp-based)
- Logs: question, prompt, retrieved chunks, answer, config
- Persists to trace log file

### Step 5: Evaluation (RAGEvaluator)
**Metrics Calculated:**
- **Faithfulness**: How well answer is grounded in retrieved chunks
  - Extracts factual claims (numbers, dates, names)
  - Checks if claims appear in context
  - Score: 0.0 (hallucinated) to 1.0 (fully grounded)
- **Relevance**: How relevant retrieved chunks are to question
  - Semantic similarity between question and chunks
  - Score: 0.0 (irrelevant) to 1.0 (highly relevant)
- **Context Recall**: How much required information was retrieved
  - Extracts required concepts from question
  - Checks which concepts appear in retrieved chunks
  - Identifies missing concepts
  - Score: 0.0 (nothing retrieved) to 1.0 (all concepts retrieved)

**Output**: Dictionary with all evaluation scores and evidence

### Step 6: Early Success Check
**Condition**: If `faithfulness >= 0.7 AND recall >= 0.7 AND relevance >= 0.7`
- **Action**: Skip failure detection phases
- **Result**: Logged as `SUCCESS` in RootCauseOracle
- **Return**: Result with empty `root_causes`

### Step 7: Answer Intent Classification
**Purpose**: Prevents misclassifying correct abstentions as failures

**Intent Types:**
- **FullAnswer**: Complete answer with high faithfulness
- **PartialAnswer**: Incomplete but grounded answer
- **CorrectAbstention**: Model correctly says "I don't know" when context is insufficient
- **HallucinatedAnswer**: Answer contains unsupported claims
- **HallucinatedAbstention**: Model abstains but answer contains hallucinations

**Action**: If `CorrectAbstention` or `FullAnswer` → Skip failure detection, log to Oracle, return result

---

## Phase 3: Failure Detection & Analysis

### Step 8: Query Feasibility Analysis
**Checks:**
- **Typos**: Detects misspellings in question
- **Over-constrained**: Questions requiring exact formats (e.g., "list 5 items")
- **Ambiguous Grammar**: Declarative statements vs questions
- **Out-of-scope**: Questions about topics not in corpus

**Auto-correction**: If typos detected on first attempt:
- Automatically corrects question
- Re-runs query with corrected question (attempt=2)
- Returns result with correction log

### Step 9: Failure Detection
**Purpose**: Identifies if query resulted in failure

**Failure Types:**
- **Retrieval Failure**: Low recall, missing concepts
- **Generation Failure**: Low faithfulness, unsupported claims
- **Prompt Failure**: Good retrieval but poor answer quality

### Step 10: Exact Failure Point Detection
**Purpose**: Pinpoints which component failed

**Components Checked:**
- Retrieval component
- Generation component  
- Prompt design
- Embedding model

### Step 11: Corpus Concept Check
**Purpose**: Distinguishes Corpus Coverage vs Retrieval Configuration issues

**Process:**
- Takes missing concepts from evaluation
- Searches entire corpus (not just retrieved chunks) for these concepts
- Determines if concepts exist in corpus but weren't retrieved

**Output**: 
- `concepts_exist_in_corpus`: List of missing concepts that DO exist in corpus
- `concepts_missing_from_corpus`: List of missing concepts that DON'T exist in corpus

### Step 12: Conflict Resolution
**Purpose**: Resolves conflicts between different failure signals

**Logic:**
- If concepts exist in corpus → Retrieval Configuration issue
- If concepts missing from corpus → Corpus Coverage issue
- If correct abstention → Corpus Coverage (not Generation Control)
- If high recall but low faithfulness → Generation Control issue

**Output**: Final component attribution (Retrieval/Generation/Prompt/Corpus/Success)

### Step 13: Failure Surface Mapping
**Purpose**: Maps failure patterns to system configuration

**Analyzes:**
- Chunk size impact
- Top-k impact
- Overlap impact
- Query type vs failure correlation

### Step 14: Cost Optimization Analysis
**Purpose**: Identifies wasted tokens and cost

**Calculates:**
- Total tokens used (input + output)
- Unused chunks ratio (chunks retrieved but not used in answer)
- Wasted cost (tokens that didn't contribute to answer)
- Total cost (all tokens used)

**Output**: Cost metrics in USD

---

## Phase 4: Root Cause Attribution (RootCauseOracle)

### Step 15: Root Cause Analysis
**Input Signals:**
```python
signals = {
    "query_id": "query_20251230_004904_746345",
    "question": "Give 5 trading rules mentioned in the book.",
    "evaluation": {...},  # All evaluation metrics
    "query_feasibility": {...},  # Typo/grammar analysis
    "cost_optimization": {...},  # Cost metrics
    "retrieved_chunks": [...],  # Retrieved documents
    "answer": "...",  # Generated answer
    "config": {...},  # System configuration
    "corpus_concept_check": {...}  # Corpus vs retrieval distinction
}
```

### Step 16: Rule Application (10+ Rules)

**Rule 1: User Query (Typos)**
- **Trigger**: Severe typos detected (similarity < 0.7, not plural/singular differences)
- **Condition**: Only fires if recall < 0.6 or faithfulness < 0.7 (if retrieval succeeded, no error)
- **Fix**: "Correct spelling or normalize query before retrieval"
- **Confidence**: Based on recall, grounding, counterfactual delta
- **User Explanation**: Explains spelling errors prevent accurate retrieval

**Rule 1B: User Query (Ambiguous Grammar)**
- **Trigger**: Declarative statement instead of question
- **Fix**: "Rephrase as question"
- **Confidence**: Based on faithfulness, relevance, recall

**Rule: Metadata-Satisfied Answer**
- **Trigger**: Answer derived from metadata (page count, author) not content
- **Action**: Sets root_causes to empty, classifies as SUCCESS (Metadata Answer)
- **Purpose**: Prevents false Corpus Coverage flags for metadata queries

**Rule: Silent Hallucination**
- **Trigger**: Short answer (<20 tokens) with factual claims, low grounding
- **Skip Condition**: If answer contains uncertainty language ("does not explicitly state", "cannot be inferred", "not directly mentioned") → not hallucination
- **Fix**: "Hallucination Risk: Answer contains ungrounded claims"
- **Confidence**: High rank

**Rule 2: Corpus Coverage**
- **Trigger**: 
  - Query is answerable but concepts missing from corpus
  - OR: Correct abstention with missing concepts
- **Skip Condition**: If existence question + negation found in chunks/answer → Answerable negation (not missing corpus)
- **Fix**: "Expand corpus. Missing: [concepts]"
- **Confidence**: Based on recall, grounding, query concept overlap
- **Priority**: High - suppresses other rules when it fires
- **Is Unfixable**: true (requires data change, not tuning)
- **User Explanation**: Explains information doesn't exist in documents, no tuning can fix

**Rule 3: User Query (Over-constrained)**
- **Trigger**: Question requires exact format (e.g., "list 5 items")
- **Fix**: "Relax format constraints or use structured output"
- **Confidence**: Based on recall, grounding, counterfactual delta

**Rule 4: Retrieval Configuration**
- **Trigger**: 
  - Low recall BUT concepts exist in corpus
  - OR: High recall + high query concept overlap + abstention → underfetching
- **Sufficiency Gate**: Only fires if (answer_length < 30 tokens OR abstention_detected). If answer is coherent + grounded (grounding_overlap >= 0.6) → no error
- **Fix**: "Increase top_k from X to Y" (dynamic: Y = X + ceil(counterfactual_recall_gain * 10))
- **Confidence**: Based on recall, grounding, counterfactual delta
- **Distinction**: Only fires if concepts exist in corpus (not pure corpus gap)
- **User Explanation**: Explains retrieval needs more chunks, suggests specific top_k increase

**Rule 5: Retrieval Noise**
- **Trigger**: High recall but low relevance (irrelevant chunks retrieved)
- **Fix**: "Improve chunk filtering or reduce top_k"
- **Confidence**: Based on recall, grounding, counterfactual delta

**Rule 6: Embedding Model Mismatch**
- **Trigger**: 
  - `context_recall < 0.5`
  - Missing concepts exist in corpus
  - Retrieval experiments (top_k/overlap) already failed
  - Same failure appears ≥ 2 times historically
- **Fix**: "Switch embedding model or fine-tune"
- **Confidence**: Based on recall, grounding, counterfactual delta
- **Strict**: Only fires after repeated retrieval failures

**Rule 7: Prompt Design**
- **Trigger**: Good retrieval (relevance > 0.5) but poor answer quality
- **Fix**: "Improve prompt to better utilize retrieved context"
- **Confidence**: Based on recall, grounding, counterfactual delta

**Rule 8: Hallucination**
- **Trigger**: Low faithfulness + unsupported claims
- **Skip Conditions**: 
  - If correct abstention detected
  - If answer contains uncertainty language (hedging/cautious reasoning)
- **Fix**: "Lower temperature or enforce extractive answering"
- **Confidence**: Based on recall, grounding, counterfactual delta

**Rule 9: Cost Inefficiency**
- **Trigger**: High unused chunks ratio (>0.3) AND recall >= 0.6
- **Condition**: Never fires when recall < 0.6 (recall recovery takes priority over cost)
- **Fix**: "Reduce top_k from X"
- **Confidence**: Based on recall, grounding, counterfactual delta
- **User Explanation**: Explains system retrieving more chunks than necessary

**Rule 10: Systemic Drift**
- **Trigger**: Performance degradation over time
- **Fix**: "Review system configuration and corpus updates"
- **Confidence**: Based on historical trends

**Rule: Partial Answer (Acceptable)**
- **Trigger**: 
  - Answer is grounded (faithfulness >= 0.6, grounding_overlap >= 0.5)
  - Correct abstention detected
  - Missing concepts are from corpus (not retrieval issue)
- **Fix**: "No action required. Answer is grounded but incomplete due to corpus limits."
- **Purpose**: Distinguishes acceptable partial answers from failures
- **Priority**: Low - only fires if no major issues (Corpus Coverage, Retrieval Configuration, Hallucination, Generation Control)
- **User Explanation**: Explains answer is correct but incomplete, system correctly abstained

### Step 17: Confidence Calculation
**Formula**: `(recall * 0.4) + (grounding_overlap * 0.4) + (min(counterfactual_delta, 0.2) * 0.2)`

**Components:**
- **Recall**: How much required info was retrieved (0.0-1.0)
- **Grounding Overlap**: How well answer is supported by chunks (0.0-1.0)
- **Counterfactual Delta**: Improvement from suggested fix (0.0-0.2 max)

### Step 18: Root Cause Ranking & Filtering
**Process:**
1. All rules that fire are collected
2. Root causes are ranked by confidence
3. Only top 1-2 causes are kept (if confidence gap > 0.15)
4. If Corpus Coverage fires, other causes are suppressed

**Output**: 
```python
{
    "root_causes": [
        {
            "rank": 1,
            "type": "Corpus Coverage",
            "fix": "Expand corpus. Missing: trading, rules.",
            "user_explanation": "Your system failed because the required information does not exist in your documents. No retrieval or prompt tuning can fix this. You must add documents covering: trading, rules.",
            "is_unfixable": true,
            "evidence": {...},
            "confidence": 0.85
        }
    ],
    "primary_failure": "Corpus Coverage",
    "secondary_risk": null,
    "outcome": "SUCCESS_WITH_RISK"
}
```

### Step 19: Query History Logging
**Saved to**: `query_history.json`

**Record Format:**
```json
{
    "query_id": "query_20251230_004904_746345",
    "question": "Give 5 trading rules mentioned in the book.",
    "timestamp": "2025-12-30T00:49:04.746345",
    "root_cause": "Corpus Coverage",
    "fix": "Expand corpus. Missing: trading, rules.",
    "confidence": 0.85,
    "cost_waste": 0.000012,
    "total_cost": 0.000045,
    "outcome": "SUCCESS_WITH_RISK",
    "active_fix_id": null
}
```

### Step 20: Fix Tracking & Validation
**Purpose**: Track recommended fixes and validate their impact on system outcomes

**Fix ID Generation:**
- Each recommended fix generates a deterministic `fix_id`
- Format: `"<component>.<parameter>:<delta>"`
- Examples: `"retrieval.top_k:+5"`, `"generation.temperature:-0.2"`, `"retrieval.chunk_size:-100"`

**Fix History Logging:**
- Saved to: `fix_history.json`
- When a fix is recommended, logs:
  - `fix_id`: Deterministic identifier
  - `rule_type`: Root cause type
  - `fix`: Human-readable fix description
  - `config_before`: System configuration parameters affected by fix
  - `recommended_at`: Timestamp when fix was recommended
  - `applied_at`: Timestamp when fix was applied (null if not applied)

**Active Fix Detection:**
- Automatically detects when system configuration changes match a known fix pattern
- Queries are tagged with `active_fix_id` in query_history
- Can also accept explicit `applied_fix_id` via signals

**Fix Validation:**
- Method: `oracle.validate_fix(fix_id)` OR `oracle.validate_fix(before_query_ids=[...], after_query_ids=[...])`
- Compares metrics BEFORE vs AFTER fix activation
- Minimum sample size: 10 queries each (for fix_id method)
- Metrics compared:
  - `success_rate`
  - `failure_rate`
  - `average_cost`
  - `retrieval_recall` (from trace logs if available)
- Output format (query IDs method):
  - `fix_applied`: Fix description
  - `failure_rate_change`: Percentage change (e.g., "-18%")
  - `retrieval_recall_change`: Delta (e.g., "+0.22")
  - `cost_change_usd`: Delta (e.g., "+0.00004")
  - `verdict`: "Fix effective" | "Fix ineffective" | "No significant change"
- Verdicts (fix_id method):
  - `IMPROVED`: Significant improvement (>5% threshold)
  - `REGRESSED`: Significant degradation
  - `NO_SIGNIFICANT_CHANGE`: Within threshold
  - `FIX_NOT_APPLIED`: Fix not yet applied
  - `INSUFFICIENT_DATA`: Not enough queries for validation

**Fix Application:**
- Method: `oracle.apply_fix(fix_id)`
- Marks fix as applied with timestamp
- Used to track when user explicitly applies a fix

---

## Phase 5: Health Report Generation

### Step 21: System Health Report
**Method**: `pipeline.root_cause_oracle.get_report(last_n=None)`

**Process:**
1. Loads all query history from `query_history.json`
2. Calculates aggregate metrics:
   - Total queries, failed queries, success queries
   - Failure rate, success rate
   - Most common failure type and percentage
   - Most common risk type
   - Most expensive failure (by cost)
3. **Immediate Action** (last 10% of queries):
   - Minimum: 1 query if total < 10
   - Maximum: 100 queries if total > 1000
   - Aggregates fixes from recent queries
   - Filters out old format fixes
4. **Strategic Action** (all queries):
   - Aggregates fixes from entire history
   - Identifies long-term patterns
5. **Cost Metrics**:
   - `total_cost_waste_usd`: Sum of wasted costs
   - `total_cost_saved_usd`: Estimated savings from fixes
6. **System Verdict**:
   - Based on failure rate, most common failure type, and percentage
   - Examples: "Domain-Bounded System. Corpus coverage limits answerability." (failure rate ≥40%, Corpus Coverage ≥50%)
   - "Healthy but Corpus-Limited" (failure rate ≥20%, Corpus Coverage ≥40%)
   - "Retrieval-Constrained System" (Retrieval Configuration dominant)
   - More accurate than generic "System healthy" messages

**Output**:
```json
{
    "system_verdict": "Domain-Bounded System. Corpus coverage limits answerability.",
    "total_queries": 32,
    "failed_queries": 15,
    "success_queries": 5,
    "success_with_risk": 12,
    "failure_rate": 0.47,
    "most_common_failure": {
        "type": "Corpus Coverage",
        "count": 8,
        "percentage": 53.3
    },
    "most_common_risk": {
        "type": "Retrieval Configuration",
        "count": 4
    },
    "most_expensive_failure": {
        "type": "Corpus Coverage",
        "total_cost_usd": 0.0
    },
    "immediate_action": "Expand corpus. Missing: algorithms, frequency, hedge, funds.",
    "strategic_action": "Lower temperature from 0.7 or enforce extractive answering",
    "total_cost_waste_usd": 0.0,
    "total_cost_saved_usd": 0.000281,
    "average_confidence": 0.5
}
```

---

## Key Design Decisions

### 1. Corpus Coverage vs Retrieval Configuration
- **Corpus Coverage**: Concepts don't exist in corpus → Need to add documents
- **Retrieval Configuration**: Concepts exist but weren't retrieved → Need to adjust top_k/chunk_size
- **Detection**: Uses `corpus_concept_check` to search entire corpus, not just retrieved chunks

### 2. Correct Abstention Handling
- When model correctly abstains ("context doesn't contain enough information"):
  - If concepts missing from corpus → **Corpus Coverage**
  - If concepts exist in corpus but recall low → **Retrieval Configuration**
  - NOT classified as Generation Control failure

### 2B. Answerable Negation Detection
- Detects existence questions ("Does the book define/guarantee X?")
- Checks if corpus/answer contains negation ("no guarantee", "does not provide")
- Treats as answerable negation, NOT Corpus Coverage
- Prevents confusing "absence of claim" with "missing corpus"

### 3. Rule Priority
- **Corpus Coverage** has highest priority
- When it fires, other rules (Retrieval, Generation, Cost) are suppressed
- Prevents confusing users with multiple causes when root issue is missing documents

### 4. Confidence-Based Filtering
- Only top 1-2 root causes shown
- Causes with confidence gap > 0.15 are filtered out
- Prevents information overload

### 5. Immediate vs Strategic Actions
- **Immediate**: Last 10% of queries (recent trends)
- **Strategic**: All queries (long-term patterns)
- Helps prioritize urgent fixes vs systematic improvements

### 6. Cost Awareness
- Tracks token usage and cost for each query
- Identifies wasted costs (unused chunks)
- Estimates potential savings from fixes
- Helps prioritize high-cost failures
- **Priority Rule**: Cost optimization never competes with recall recovery (only fires when recall ≥ 0.6)

### 7. User-Friendly Explanations
- Every root cause includes `user_explanation` (plain language)
- `is_unfixable` flag distinguishes:
  - **Unfixable** (requires data/scope change): Corpus Coverage, Real-time queries
  - **Fixable** (tuning can help): Retrieval Configuration, Generation Control, etc.

### 8. Dynamic Recommendations
- Top_k suggestions based on counterfactual recall gain: `suggested_top_k = current_top_k + ceil(gain * 10)`
- Prevents conflicting recommendations (e.g., "Increase top_k" vs "Reduce top_k")

### 9. Answer Quality Gates
- **Retrieval Configuration**: Only fires if answer is insufficient (short OR abstaining). Good answers (coherent + grounded) are not penalized
- **Hallucination Detection**: Skips if answer contains uncertainty language (hedging/cautious reasoning is acceptable)
- **Partial Answer Classification**: Distinguishes acceptable incomplete answers from actual failures

---

## Output Format

### Per-Query Output
```python
{
    "query_id": "query_20251230_004904_746345",
    "question": "Give 5 trading rules mentioned in the book.",
    "root_causes": [
        {
            "rank": 1,
            "type": "Corpus Coverage",
            "fix": "Expand corpus. Missing: trading, rules.",
            "user_explanation": "Your system failed because the required information does not exist in your documents. No retrieval or prompt tuning can fix this. You must add documents covering: trading, rules.",
            "is_unfixable": true,
            "evidence": {
                "missing_concepts": ["trading", "rules"],
                "retrieval_delta": 0.0,
                "grounding_overlap": 0.2,
                "abstention_detected": true,
                "query_concept_overlap": 0.8,
                "confidence_formula": "recall(0.87)*0.4 + grounding(0.20)*0.4 + delta(0.00)*0.2"
            },
            "confidence": 0.428
        }
    ]
}
```

### System Health Report
- Comprehensive metrics across all queries
- Actionable recommendations (immediate + strategic)
- Cost analysis and savings estimates
- System health verdict

### Public Output Schema
**Purpose**: Provide stable, simplified output schema for external consumption

**Method**: `oracle.get_public_output(result)`

**Output Format:**
```python
{
    "query_id": "query_20251230_004904_746345",
    "outcome": "SUCCESS_WITH_RISK",
    "primary_failure": "Corpus Coverage",
    "recommended_fix": "Expand corpus. Missing: trading, rules.",
    "is_unfixable": true,
    "confidence": 0.85,
    "explanation": "Your system failed because the required information does not exist in your documents...",
    "diagnostic_maturity": "high-confidence"
}
```

**Fields:**
- `query_id`: Unique query identifier
- `outcome`: SUCCESS | SUCCESS_WITH_RISK | FAILURE | UNKNOWN
- `primary_failure`: Type of root cause (or null)
- `recommended_fix`: Human-readable fix recommendation (or null)
- `is_unfixable`: Boolean indicating if fix requires data/scope change
- `confidence`: Confidence score (0.0-1.0)
- `explanation`: User-friendly explanation
- `diagnostic_maturity`: "experimental" | "stable" | "high-confidence"
  - `high-confidence`: confidence >= 0.8
  - `stable`: confidence >= 0.5
  - `experimental`: confidence < 0.5

**Note**: Internal diagnostics (detailed evidence, secondary risks, etc.) remain unchanged and are separate from public schema.

---

## Workflow Summary

```
User Query
    ↓
Retrieval (Top-K chunks)
    ↓
Answer Generation (LLM)
    ↓
Evaluation (Faithfulness, Relevance, Recall)
    ↓
Early Success Check → [SUCCESS] → Log & Return
    ↓
Answer Intent Classification → [Correct Abstention] → Log & Return
    ↓
Query Feasibility Analysis → [Auto-correct typos if needed]
    ↓
Failure Detection
    ↓
Exact Failure Point Detection
    ↓
Corpus Concept Check (Corpus vs Retrieval distinction)
    ↓
Conflict Resolution
    ↓
Cost Optimization Analysis
    ↓
Root Cause Oracle (10+ Rules)
    ↓
Confidence Calculation & Ranking
    ↓
Query History Logging
    ↓
Return Result with Root Causes
    ↓
[Optional] Generate Health Report (all queries)
```

---

## Key Features

1. **Provable Root Causes**: Evidence-backed attribution with confidence scores
2. **Actionable Fixes**: Specific, implementable recommendations
3. **Cost Awareness**: Tracks waste and potential savings
4. **Historical Analysis**: Learns from past queries to improve recommendations
5. **Intelligent Distinction**: Separates Corpus Coverage from Retrieval Configuration
6. **Correct Abstention Handling**: Properly classifies when model correctly says "I don't know"
7. **Time-Aware Recommendations**: Immediate (recent) vs Strategic (long-term) actions
8. **System Health Monitoring**: Comprehensive health reports with verdict
9. **Fix Validation**: Tracks fix recommendations and validates their impact on system outcomes
10. **Public Schema**: Stable, simplified output schema for external consumption without breaking internal diagnostics

---

## Future Tool Integration

When packaged as a tool, the system would expose:

**Per-Query API:**
```python
oracle.analyze(signals) → {
    "root_causes": [...],
    "outcome": "SUCCESS|SUCCESS_WITH_RISK|FAILURE",
    "primary_failure": "...",
    "secondary_risk": "..."
}
```

**System Health API:**
```python
oracle.get_report(last_n=None) → {
    "system_verdict": "...",
    "immediate_action": "...",
    "strategic_action": "...",
    "total_cost_waste_usd": 0.0,
    "total_cost_saved_usd": 0.0,
    ...
}
```

**Public Output API:**
```python
oracle.get_public_output(result) → {
    "query_id": "...",
    "outcome": "SUCCESS|SUCCESS_WITH_RISK|FAILURE",
    "primary_failure": "...",
    "recommended_fix": "...",
    "is_unfixable": true|false,
    "confidence": 0.85,
    "explanation": "...",
    "diagnostic_maturity": "stable|high-confidence|experimental"
}
```

**Fix Validation API:**
```python
oracle.apply_fix(fix_id) → bool
oracle.validate_fix(fix_id) → {
    "fix_id": "...",
    "verdict": "IMPROVED|REGRESSED|NO_SIGNIFICANT_CHANGE|FIX_NOT_APPLIED|INSUFFICIENT_DATA",
    "before": {"success_rate": 0.6, "failure_rate": 0.4, "average_cost": 0.001, "sample_size": 50},
    "after": {"success_rate": 0.75, "failure_rate": 0.25, "average_cost": 0.001, "sample_size": 50}
}
oracle.validate_fix(before_query_ids=["id1", "id2"], after_query_ids=["id3", "id4"]) → {
    "fix_applied": "Increase top_k from 5 to 7",
    "failure_rate_change": "-18%",
    "retrieval_recall_change": "+0.22",
    "cost_change_usd": "+0.00004",
    "verdict": "Fix effective"
}
```

The system is designed to be a **drop-in forensic auditor** for any RAG pipeline, providing provable root causes and actionable fixes with confidence scores, fix validation, and stable public output schema.

