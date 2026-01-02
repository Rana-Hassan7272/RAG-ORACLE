# 🔮 RAG-ORACLE

[![PyPI](https://img.shields.io/pypi/v/rag-oracle.svg)](https://pypi.org/project/rag-oracle/)
[![Python](https://img.shields.io/pypi/pyversions/rag-oracle.svg)](https://pypi.org/project/rag-oracle/)
[![License](https://img.shields.io/pypi/l/rag-oracle.svg)](https://pypi.org/project/rag-oracle/)

**The First Production-Grade Root Cause Analysis System for RAG Failures**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RAG-ORACLE is the **world's first automated diagnostic system** that tells you **exactly why** your RAG pipeline failed and **how to fix it**—with production-grade accuracy and zero guesswork.

**Add 3 lines of code. Get instant root cause diagnosis.**

---

## 🎯 What Problem Does RAG-ORACLE Solve?

### The RAG Debugging Crisis

You built a RAG system. It works... sometimes. When it fails, you're left staring at:
- ❌ Low evaluation scores (faithfulness: 0.3, recall: 0.4)
- ❌ Generic metrics that don't tell you **why** it failed
- ❌ Hours of manual debugging to find the root cause
- ❌ Trial-and-error fixes that may or may not work

**Existing tools like LangSmith and RAGAS give you metrics. RAG-ORACLE gives you answers.**

### Why LangSmith & RAGAS Fall Short

| Feature | LangSmith | RAGAS | **RAG-ORACLE** |
|---------|-----------|-------|----------------|
| **Evaluation Metrics** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Root Cause Diagnosis** | ❌ No | ❌ No | ✅ **YES** |
| **Actionable Fixes** | ❌ Manual | ❌ Manual | ✅ **Automated** |
| **Corpus vs Retrieval Separation** | ❌ No | ❌ No | ✅ **YES** |
| **Cost Waste Detection** | ❌ No | ❌ No | ✅ **YES** |
| **Fix Validation** | ❌ No | ❌ No | ✅ **YES** |
| **Hallucination Detection** | ⚠️ Basic | ⚠️ Basic | ✅ **Advanced** |

**The Problem**: LangSmith and RAGAS tell you **"faithfulness is low"**. They don't tell you if it's because:
- Your corpus is missing information (unfixable without new data)
- Your retrieval config is wrong (fixable by increasing `top_k`)
- Your embedding model doesn't understand your domain (fixable by switching models)
- Your LLM is hallucinating (fixable by lowering temperature)

**RAG-ORACLE solves this.** It's the difference between a thermometer (tells you there's a fever) and a doctor (tells you it's strep throat and prescribes antibiotics).

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │  Query   │──▶│ Retrieval│──▶│Generation│──▶│  Answer  │    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAG-ORACLE DIAGNOSTIC ENGINE                  │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Phase 1: Signal Collection                               │   │
│  │  • Evaluation Metrics (Faithfulness, Recall, Relevance)  │   │
│  │  • Query Feasibility (Typos, Constraints)                │   │
│  │  • Corpus Concept Check (Missing vs Available)           │   │
│  │  • Cost Analysis (Token Usage, Waste)                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         │                                         │
│                         ▼                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Phase 2: Root Cause Analysis (10 Diagnostic Rules)       │   │
│  │  1. User Query Issues (Typos, Grammar)                   │   │
│  │  2. Corpus Coverage (Missing Information)                │   │
│  │  3. Retrieval Configuration (top_k, chunk_size)          │   │
│  │  4. Retrieval Noise (Irrelevant chunks)                  │   │
│  │  5. Embedding Mismatch (Domain-specific)                 │   │
│  │  6. Prompt Design (Task alignment)                       │   │
│  │  7. Hallucination Risk (Ungrounded claims)               │   │
│  │  8. Generation Control (Temperature)                     │   │
│  │  9. Cost Inefficiency (Wasted tokens)                    │   │
│  │ 10. Systemic Drift (Performance degradation)             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         │                                         │
│                         ▼                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Phase 3: Fix Recommendation & Validation                 │   │
│  │  • Actionable Fix (e.g., "Increase top_k from 3 to 5")  │   │
│  │  • Confidence Score (0.0 - 1.0)                          │   │
│  │  • Fix Validation (Before/After comparison)              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         │                                         │
│                         ▼                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Output: Diagnosis Report                                 │   │
│  │  • Primary Failure Type                                  │   │
│  │  • User-Friendly Explanation                             │   │
│  │  • Recommended Fix                                       │   │
│  │  • Is Unfixable? (Corpus vs System)                      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Real-World Example

### Input Query
```python
question = "What is the author's view on fear of failure?"
```

### RAG-ORACLE Output
```json
{
  "query_id": "abc123",
  "outcome": "FAILURE",
  "primary_failure": "Corpus Coverage",
  "recommended_fix": "Expand corpus. Missing: failure, fear, psychology.",
  "is_unfixable": true,
  "confidence": 0.82,
  "explanation": "Your system failed because the required information does not exist in your documents. No retrieval or prompt tuning can fix this. You must add documents covering: failure, fear, psychology.",
  "diagnostic_maturity": "high-confidence"
}
```

### What This Means
- ✅ **Clear Diagnosis**: The information doesn't exist in your corpus
- ✅ **Actionable**: Add documents about "failure", "fear", "psychology"
- ✅ **Honest**: Marked as `is_unfixable` because no system tweak will help
- ✅ **Confident**: 82% confidence in this diagnosis

---

## 🏥 System Health Report Example

After running 50 queries, get a comprehensive health report:

```json
{
  "system_verdict": "Retrieval-Constrained System. Adjust retrieval parameters.",
  "total_queries": 50,
  "failed_queries": 15,
  "success_queries": 30,
  "success_with_risk": 5,
  "failure_rate": 0.30,
  "most_common_failure": {
    "type": "Retrieval Configuration",
    "count": 8,
    "percentage": 53.3
  },
  "immediate_action": "Increase top_k from 3 to 5",
  "strategic_action": "Increase top_k from 3 to 5 or adjust chunk_size",
  "total_cost_waste_usd": 0.002340,
  "total_cost_saved_usd": 0.001200,
  "average_confidence": 0.76
}
```

**Insights**:
- 30% failure rate, mostly due to retrieval config
- Immediate fix: Increase `top_k` from 3 to 5
- Saved $0.0012 by optimizing successful queries
- High confidence (0.76) in diagnoses

---

## 🚀 Quickstart

### Installation

```bash
pip install rag-oracle
```

### Prerequisites

RAG-ORACLE works with any RAG system. You'll need:
- Python 3.8+
- Your existing RAG pipeline (OpenAI, LangChain, LlamaIndex, or custom)
- Optional: Embedding model for auto-evaluation (recommended for easiest setup)

### Setting Up Embeddings (Recommended)

For auto-evaluation, you need an embedding model. Here are common options:

```python
# Option 1: HuggingFace (Free, no API key needed)
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Option 2: OpenAI (Requires API key)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(openai_api_key="your-key")

# Option 3: Use your existing embeddings
# Any LangChain-compatible embedding model works
```

**Note**: If you don't provide embeddings, you must provide evaluation metrics manually in each `diagnose()` call (see Advanced Usage).

### Minimal Usage (3 Lines)

**For existing RAG systems - super simple integration:**

```python
from rag_pipeline import RAGOracle
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize (one time setup)
# Option 1: With embeddings for auto-evaluation (recommended)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
oracle = RAGOracle(embeddings=embeddings)

# Option 2: Without embeddings (you provide evaluation manually)
# oracle = RAGOracle()

# In your existing RAG code, add this:
result = oracle.diagnose(
    query="What is the capital of France?",
    answer=your_rag_answer,  # Your RAG system's generated answer
    chunks=your_retrieved_chunks,  # List of retrieved document chunks
    config={"top_k": 5, "temperature": 0.7}  # Optional: your system config
)

# Get instant diagnosis
if result["root_causes"]:
    primary = result["root_causes"][0]
    print(f"💡 Fix: {primary['fix']}")
    print(f"📝 {primary['user_explanation']}")
    print(f"⚠️ Unfixable: {primary['is_unfixable']}")
```

**That's it. No pipeline setup. No configuration. Just diagnose.**

**Note**: If you don't provide embeddings, you must provide `evaluation` in each `diagnose()` call (see Advanced Usage below).

### Full Pipeline Usage

If you want a complete RAG pipeline with built-in diagnostics:

```python
from rag_pipeline import RAGPipeline

# Initialize with your documents
pipeline = RAGPipeline(document_source="./documents")

# Query and get diagnosis
result = pipeline.query("What is the capital of France?")

# Get root cause analysis
print(result["root_causes"])
```

### Get System Health Report

```python
# Get aggregated insights across all queries
report = oracle.get_report()

print(f"Failure Rate: {report['failure_rate']*100:.1f}%")
print(f"Most Common Issue: {report['most_common_failure']['type']}")
print(f"Immediate Action: {report['immediate_action']}")
# Output:
# Failure Rate: 30.0%
# Most Common Issue: Retrieval Configuration
# Immediate Action: Increase top_k from 3 to 5

# Analyze only last 10 queries
recent_report = oracle.get_report(last_n=10)
```

### Public Output Example

```python
# Get clean, user-facing output (for APIs/production)
# Pass the result from diagnose() method
public_output = oracle.get_public_output(result)

print(public_output)
# {
#   "query_id": "xyz789",
#   "outcome": "SUCCESS_WITH_RISK",
#   "primary_failure": "Cost Inefficiency",
#   "recommended_fix": "Reduce top_k from 5",
#   "is_unfixable": false,
#   "confidence": 0.68,
#   "explanation": "Your system is retrieving more chunks than necessary...",
#   "diagnostic_maturity": "stable"
# }
```

### Fix Validation Example

```python
# After applying a fix, validate if it worked
validation = oracle.validate_fix(
    before_query_ids=["query_1", "query_2", "query_3"],  # Before fix
    after_query_ids=["query_4", "query_5", "query_6"]    # After fix
)

print(f"Fix Applied: {validation['fix_applied']}")
print(f"Failure Rate Change: {validation['failure_rate_change']}")
print(f"Verdict: {validation['verdict']}")
# Output:
# Fix Applied: Increase top_k from 3 to 5
# Failure Rate Change: -20%
# Verdict: Fix effective
```

---

## 📚 Basic Usage

### Complete Example

```python
from rag_pipeline import RAGPipeline
import json

# 1. Initialize pipeline
pipeline = RAGPipeline(
    document_source="./documents",
    chunk_size=500,
    chunk_overlap=50,
    embedding_model="huggingface",
    top_k=3,
    temperature=0.7
)

# 2. Query the system
question = "What are the main themes in the document?"
result = pipeline.query(question)

# 3. View the answer
print(f"Answer: {result['answer']}")

# 4. Check for failures
root_causes = result.get("root_causes", [])
if root_causes:
    primary = root_causes[0]
    print(f"\n⚠️ Issue Detected: {primary['type']}")
    print(f"Fix: {primary['fix']}")
    print(f"Explanation: {primary['user_explanation']}")
    print(f"Confidence: {primary['confidence']:.2f}")

# 5. Get system health report
report = pipeline.root_cause_oracle.get_report(last_n=10)
print(json.dumps(report, indent=2))
```

---

## 💼 Common Use Cases

### Use Case 1: Quick Diagnosis for Single Query

```python
from rag_pipeline import RAGOracle
from langchain_huggingface import HuggingFaceEmbeddings

oracle = RAGOracle(embeddings=HuggingFaceEmbeddings())

# After your RAG system generates an answer
result = oracle.diagnose(
    query="What is the capital of France?",
    answer=rag_answer,
    chunks=retrieved_chunks
)

# Check if there are issues
if result["root_causes"]:
    print(f"Issue: {result['root_causes'][0]['type']}")
    print(f"Fix: {result['root_causes'][0]['fix']}")
```

### Use Case 2: Batch Processing with Health Reports

```python
# Process multiple queries
queries = ["query1", "query2", "query3"]
results = []

for query in queries:
    answer, chunks = your_rag_system.process(query)
    result = oracle.diagnose(query=query, answer=answer, chunks=chunks)
    results.append(result)

# Get system-wide health report
health_report = oracle.get_report()
print(f"Overall Failure Rate: {health_report['failure_rate']*100:.1f}%")
print(f"Most Common Issue: {health_report['most_common_failure']['type']}")
```

### Use Case 3: Fix Validation Workflow

```python
# Step 1: Collect queries before fix
before_queries = []
for query in test_queries:
    result = oracle.diagnose(...)
    before_queries.append(result["query_id"])

# Step 2: Apply recommended fix (e.g., increase top_k from 3 to 5)
your_rag_system.update_config(top_k=5)

# Step 3: Collect queries after fix
after_queries = []
for query in test_queries:
    result = oracle.diagnose(...)
    after_queries.append(result["query_id"])

# Step 4: Validate if fix worked
validation = oracle.validate_fix(
    before_query_ids=before_queries,
    after_query_ids=after_queries
)

if validation["verdict"] == "Fix effective":
    print("✅ Fix validated! Keep the change.")
else:
    print("❌ Fix didn't help. Try a different approach.")
```

### Use Case 4: Production API Integration

```python
from flask import Flask, request, jsonify
from rag_pipeline import RAGOracle

app = Flask(__name__)
oracle = RAGOracle(embeddings=your_embeddings)

@app.route("/diagnose", methods=["POST"])
def diagnose_endpoint():
    data = request.json
    result = oracle.diagnose(
        query=data["query"],
        answer=data["answer"],
        chunks=data["chunks"]
    )
    
    # Return clean public output
    return jsonify(oracle.get_public_output(result))
```

## 🔧 Integration Guide

### Option 1: Simple Integration (Recommended)

**For 99% of users - just 3 lines of code:**

```python
from rag_pipeline import RAGOracle
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize once (with your embeddings for auto-evaluation)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
oracle = RAGOracle(embeddings=embeddings)

# In your existing RAG code, add this:
result = oracle.diagnose(
    query=user_query,
    answer=your_generated_answer,
    chunks=your_retrieved_chunks,  # Can be strings, dicts, or LangChain Documents
    config={"top_k": 5, "temperature": 0.7}  # optional
)

# Get actionable fix
if result["root_causes"]:
    fix = result["root_causes"][0]["fix"]
    explanation = result["root_causes"][0]["user_explanation"]
    is_unfixable = result["root_causes"][0]["is_unfixable"]
    print(f"💡 Fix: {fix}")
    print(f"📝 {explanation}")
    if is_unfixable:
        print("⚠️ This requires data changes, not system tuning.")
```

**Works with any RAG system** - OpenAI, LangChain, LlamaIndex, custom pipelines.

**Chunk Format Support**: The `chunks` parameter accepts:
- List of strings: `["chunk 1 text", "chunk 2 text"]`
- List of dicts: `[{"page_content": "text", "metadata": {...}}, ...]`
- List of LangChain Documents: `[Document(page_content="..."), ...]`

### Option 2: Advanced Integration (Manual Evaluation)

**For custom evaluation pipelines or when you already have evaluation metrics:**

```python
from rag_pipeline import RAGOracle

# Initialize without embeddings (you'll provide evaluation manually)
oracle = RAGOracle()

# Your existing RAG pipeline
question = "Your query"
chunks = your_retriever.retrieve(question)
answer = your_generator.generate(question, chunks)

# Your custom evaluation (must include faithfulness, relevance, context_recall)
evaluation = {
    "faithfulness": {
        "faithfulness": 0.65,  # Score 0.0-1.0
        "unsupported_claims": ["claim 1", "claim 2"]
    },
    "relevance": {
        "relevance": 0.72
    },
    "context_recall": {
        "context_recall": 0.58,
        "missing_concepts": ["concept1", "concept2"]
    }
}

# Diagnose with pre-computed evaluation
result = oracle.diagnose(
    query=question,
    answer=answer,
    chunks=chunks,
    evaluation=evaluation,  # Provide your evaluation
    config={"top_k": 3, "temperature": 0.7}
)

print(result["root_causes"])
```

### Option 3: Direct RootCauseOracle Usage

**For maximum control and custom signal processing:**

```python
from rag_pipeline import RootCauseOracle

# Initialize Oracle
oracle = RootCauseOracle(query_history_file="./query_history.json")

# Prepare signals dictionary
signals = {
    "query_id": "unique_query_id",
    "question": "Your query",
    "answer": "Generated answer",
    "retrieved_chunks": chunks,  # List of LangChain Documents
    "evaluation": {
        "faithfulness": {"faithfulness": 0.65, "unsupported_claims": []},
        "relevance": {"relevance": 0.72},
        "context_recall": {"context_recall": 0.58, "missing_concepts": []}
    },
    "config": {"top_k": 3, "temperature": 0.7},
    "query_feasibility": None,  # Optional
    "cost_optimization": None,  # Optional
    "corpus_concept_check": None  # Optional
}

# Analyze
diagnosis = oracle.analyze(signals)
print(diagnosis["root_causes"])
```

---

## 📖 API Reference

### `RAGPipeline`

Main entry point for the RAG system with built-in diagnostics.

#### Constructor
```python
RAGPipeline(
    document_source: str,           # Path to documents
    chunk_size: int = 500,          # Chunk size in characters
    chunk_overlap: int = 50,        # Overlap between chunks
    embedding_model: str = "huggingface",  # Embedding model
    top_k: int = 3,                 # Number of chunks to retrieve
    temperature: float = 0.7,       # Generation temperature
    enable_evaluation: bool = True  # Enable diagnostics
)
```

#### Methods

**`query(question: str) -> dict`**
- Query the RAG system and get diagnosis
- Returns: Full result with answer, evaluation, and root causes

**`get_config() -> dict`**
- Get current pipeline configuration

**`update_top_k(top_k: int)`**
- Update retrieval parameter

**`update_temperature(temperature: float)`**
- Update generation parameter

---

### `RAGOracle`

High-level wrapper for simplified integration (recommended for most users).

#### Constructor
```python
RAGOracle(
    query_history_file: Optional[str] = None,  # Defaults to "./query_history.json"
    embeddings: Optional[Any] = None,          # Embedding model for auto-evaluation
    generator: Optional[Any] = None            # Generator for LLM-as-a-Judge evaluation
)
```

#### Methods

**`diagnose(query: str, answer: str, chunks: list, config: dict = None, evaluation: dict = None, query_id: str = None) -> dict`**
- Diagnose RAG query execution and identify root causes
- `query`: User's question/query string
- `answer`: Generated answer from your RAG system
- `chunks`: Retrieved document chunks. Can be:
  - List of strings: `["chunk text 1", "chunk text 2"]`
  - List of dicts: `[{"page_content": "text", "metadata": {...}}, ...]`
  - List of LangChain Documents: `[Document(...), ...]`
- `config`: Optional dict with system config (e.g., `{"top_k": 5, "temperature": 0.7}`)
- `evaluation`: Optional pre-computed evaluation dict. If not provided and embeddings available, will auto-evaluate
- `query_id`: Optional query identifier (auto-generated if not provided)
- Returns: Diagnosis dict with `root_causes`, `primary_failure`, `outcome`, etc.

**`get_report(last_n: int = None) -> dict`**
- Get system health report
- `last_n`: Analyze last N queries (None = all)

**`validate_fix(fix_id: str = None, before_query_ids: list = None, after_query_ids: list = None) -> dict`**
- Validate if a fix actually worked
- Either provide `fix_id` OR `before_query_ids`/`after_query_ids`
- Returns: Before/after comparison with verdict

**`apply_fix(fix_id: str) -> bool`**
- Mark a recommended fix as "applied" in fix history
- Returns: True if fix was found and marked, False otherwise

**`get_public_output(result: dict) -> dict`**
- Get clean, user-facing output
- Returns: Simplified diagnosis for APIs/production

---

### `RootCauseOracle`

Core diagnostic engine for root cause analysis.

#### Constructor
```python
RootCauseOracle(
    query_history_file: str = "./query_history.json"
)
```

#### Methods

**`analyze(signals: dict) -> dict`**
- Analyze query execution signals
- Returns: Diagnosis with root causes and fixes

**`get_report(last_n: int = None) -> dict`**
- Get system health report
- `last_n`: Analyze last N queries (None = all)

**`validate_fix(fix_id: str = None, before_query_ids: list = None, after_query_ids: list = None) -> dict`**
- Validate if a fix actually worked
- Either provide `fix_id` OR `before_query_ids`/`after_query_ids`
- Returns: Before/after comparison with verdict

**`apply_fix(fix_id: str) -> bool`**
- Mark a recommended fix as "applied" in fix history
- Returns: True if fix was found and marked, False otherwise

**`get_public_output(result: dict) -> dict`**
- Get clean, user-facing output
- Returns: Simplified diagnosis

---

## ⚙️ Configuration Options

### Pipeline Configuration

```python
config = {
    # Chunking
    "chunk_size": 500,        # Characters per chunk
    "chunk_overlap": 50,      # Overlap between chunks
    
    # Retrieval
    "top_k": 3,               # Number of chunks to retrieve
    "embedding_model": "huggingface",  # or "openai"
    
    # Generation
    "temperature": 0.7,       # 0.0 (deterministic) to 1.0 (creative)
    "model_type": "openai",   # or "groq"
    "model_name": "gpt-4",    # Specific model
    
    # Diagnostics
    "enable_evaluation": True,
    "enable_tracing": True,
    "log_dir": "./logs"
}

pipeline = RAGPipeline(**config)
```

### Environment Variables

```bash
# API Keys
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key

# Optional
RAG_LOG_DIR=./logs
RAG_VECTOR_STORE=./vector_store
```

---

## 🎓 Core Concepts

### 1. Corpus Coverage vs Retrieval Failure

**The Critical Distinction**

| Scenario | Corpus Coverage | Retrieval Failure |
|----------|----------------|-------------------|
| **Problem** | Info doesn't exist in docs | Info exists but not retrieved |
| **Cause** | Missing documents | Wrong `top_k`, bad embeddings |
| **Fix** | Add documents (unfixable by system) | Increase `top_k` (fixable) |
| **Example** | "What's the CEO's salary?" (not in docs) | "What's the company mission?" (in docs, not retrieved) |

**How RAG-ORACLE Detects This**:
```python
# Checks if missing concepts exist in corpus
corpus_check = retriever.check_concepts_in_corpus(missing_concepts)

if concepts_missing_from_corpus:
    # Corpus Coverage - unfixable
    diagnosis = "Expand corpus. Missing: [concepts]"
    is_unfixable = True
else:
    # Retrieval Failure - fixable
    diagnosis = "Increase top_k from 3 to 5"
    is_unfixable = False
```

---

### 2. Correct Abstention

**What is it?**
When the RAG system correctly says "I don't know" because the information isn't in the corpus.

**Why it matters?**
- ✅ **Good**: System abstains when info is missing (honest)
- ❌ **Bad**: System hallucinates an answer (dangerous)

**How RAG-ORACLE Handles It**:
```python
# Detects abstention language
abstention_phrases = [
    "does not contain enough information",
    "cannot be answered from the provided context",
    "not found in the context"
]

if abstention_detected and info_missing_from_corpus:
    # This is GOOD behavior - don't flag as failure
    outcome = "SUCCESS_WITH_RISK"
    explanation = "System correctly abstained. Consider expanding corpus."
```

**Example**:
```
Q: "What is the CEO's favorite color?"
A: "The provided context does not contain information about the CEO's favorite color."

Diagnosis: ✅ Correct Abstention (not a failure!)
```

---

### 3. Fix Validation

**The Problem**: You applied a fix. Did it actually work?

**RAG-ORACLE's Solution**: Automated before/after comparison

```python
# Before fix
before_queries = ["query1", "query2", "query3"]
# Apply fix: top_k = 3 → 5
# After fix
after_queries = ["query4", "query5", "query6"]

# Validate
validation = oracle.validate_fix(before_query_ids=before_queries, after_query_ids=after_queries)

print(validation)
# {
#   "fix_applied": "Increase top_k from 3 to 5",
#   "failure_rate_change": "-20%",  # 20% reduction in failures
#   "retrieval_recall_change": "+0.15",  # Recall improved by 0.15
#   "verdict": "Fix effective"
# }
```

**Validation Criteria**:
- ✅ **Effective**: Failure rate drops >5% OR recall improves >0.05
- ❌ **Ineffective**: Failure rate increases OR recall drops
- ⚠️ **No Change**: Metrics stay within ±5%

---

### 4. Cost Awareness

**The Hidden Problem**: Your RAG system works, but wastes money.

**How RAG-ORACLE Detects Waste**:

```python
# Scenario: Retrieved 10 chunks, only used 2
cost_analysis = {
    "total_tokens": 5000,
    "wasted_tokens": 4000,  # 8 unused chunks
    "wasted_cost_usd": 0.0008,
    "unused_chunks_ratio": 0.8
}

# Diagnosis
if unused_ratio >= 0.4:
    diagnosis = "Cost Inefficiency"
    fix = "Reduce top_k from 10 to 3"
    explanation = "You're retrieving 10 chunks but only using 2. Reduce top_k to save costs."
```

**Cost Optimization Triggers**:
1. **Unused Chunks**: Retrieved 10, used 2
2. **High Cost Per Token**: Paying too much for little value
3. **Metadata Overfetch**: Retrieved 5 sources, used 1

**Real Impact**:
- Before: 10 chunks × 500 tokens = 5000 tokens
- After: 3 chunks × 500 tokens = 1500 tokens
- **Savings**: 70% token reduction

---

## 🔍 Root Cause Types

### 1. User Query Issues
- **Typos**: "What is the captial of France?" → "capital"
- **Ambiguous Grammar**: "The fear of failure" (statement, not question)
- **Over-Constrained**: "List exactly 5 items" (too rigid)

**Fix**: Query preprocessing, normalization

---

### 2. Corpus Coverage
- **Missing Information**: Required data not in documents
- **Unfixable**: System can't fix this—user must add documents

**Fix**: Expand corpus with relevant documents

---

### 3. Retrieval Configuration
- **Low Recall**: Info exists but not retrieved
- **Cause**: `top_k` too low, chunks too small

**Fix**: Increase `top_k`, adjust `chunk_size`

---

### 4. Retrieval Noise
- **High Recall, Low Relevance**: Retrieved too much junk
- **Cause**: `top_k` too high, poor filtering

**Fix**: Reduce `top_k`, improve chunk filtering

---

### 5. Embedding Mismatch
- **Domain-Specific Failure**: Generic embeddings don't understand domain jargon
- **Cause**: Using general-purpose embeddings for specialized domain

**Fix**: Switch to domain-specific embeddings or fine-tune

---

### 6. Prompt Design
- **Good Retrieval, Bad Answer**: Retrieved right info, generated wrong answer
- **Cause**: Prompt doesn't align with task

**Fix**: Rewrite system prompt

---

### 7. Hallucination Risk
- **Ungrounded Claims**: Answer contains facts not in retrieved text
- **Cause**: High temperature, creative generation

**Fix**: Lower temperature, enforce extractive answering

---

### 8. Generation Control
- **Unsupported Claims**: LLM making things up
- **Cause**: Temperature too high

**Fix**: Lower temperature from 0.7 to 0.3

---

### 9. Cost Inefficiency
- **Wasted Tokens**: Retrieving more than needed
- **Cause**: `top_k` too high for simple queries

**Fix**: Reduce `top_k`, compress context

---

### 10. Systemic Drift
- **Performance Degradation**: System worked before, failing now
- **Cause**: Config changes, corpus updates, model changes

**Fix**: Review recent changes, rollback if needed

---

## 🆚 RAG-ORACLE vs Alternatives

| Tool | What It Does | Integration | RAG-ORACLE Advantage |
|------|--------------|-------------|---------------------|
| **LangSmith** | Tracing & monitoring (what happened) | Complex setup | ✅ **3-line integration** + tells you WHY |
| **RAGAS** | Evaluation metrics (scores) | Requires pipeline | ✅ **Works with any RAG** + actionable fixes |
| **Manual Debugging** | Trial-and-error | N/A | ✅ **Instant diagnosis** instead of hours |

**The Key Difference**: 
- **LangSmith/RAGAS**: "Your faithfulness score is 0.3" ❓
- **RAG-ORACLE**: "Your faithfulness is low because corpus is missing info. Add documents about X, Y, Z." ✅

**Use Together**: LangSmith for tracing + RAGAS for metrics + **RAG-ORACLE for diagnosis = Complete RAG observability**

---

## 🔍 Troubleshooting

### Common Issues

**1. "Evaluation not provided and no embeddings available"**
```python
# Solution: Either provide embeddings during initialization
oracle = RAGOracle(embeddings=your_embeddings)

# OR provide evaluation in each diagnose() call
result = oracle.diagnose(..., evaluation=your_evaluation)
```

**2. "Unsupported chunk type"**
```python
# Solution: Ensure chunks are in one of these formats:
chunks = ["text1", "text2"]  # List of strings
chunks = [{"page_content": "text", "metadata": {}}]  # List of dicts
chunks = [Document(page_content="text")]  # List of LangChain Documents
```

**3. Import Error: "No module named 'rag_pipeline'"**
```bash
# Solution: Install the package
pip install rag-oracle

# Or install from source
pip install -e .
```

**4. Low confidence scores in diagnoses**
- This is normal for new systems with limited query history
- Confidence improves as the system learns from more queries
- Use `get_report()` to see system-wide patterns

### Getting Help

- Check the [examples/](examples/) directory for working code
- Review the API Reference section above
- Open an issue on [GitHub](https://github.com/Rana-Hassan7272/RAG-ORACLE/issues)

## 📦 Package Structure

```
rag-oracle/
├── rag_pipeline/
│   ├── root_cause_oracle.py    # Core diagnostic engine
│   ├── rag_oracle.py           # Simplified wrapper (recommended)
│   ├── pipeline.py             # Full RAG pipeline (optional)
│   ├── evaluator.py            # Evaluation metrics
│   ├── config.py               # Configuration defaults
│   └── ...                     # Other components
├── examples/
│   └── example_usage.py        # Usage examples
├── README.md                    # This file
├── LICENSE                      # MIT License
└── pyproject.toml              # Package configuration
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Built with:
- LangChain for RAG components
- OpenAI/Groq for LLMs
- HuggingFace for embeddings

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Rana-Hassan7272/RAG-ORACLE/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Rana-Hassan7272/RAG-ORACLE/discussions)
- **Repository**: [RAG-ORACLE on GitHub](https://github.com/Rana-Hassan7272/RAG-ORACLE)

---

## 🎯 Roadmap

- [ ] Support for more LLM providers (Anthropic, Cohere)
- [ ] Web UI for diagnostics dashboard
- [ ] Automated fix application
- [ ] Multi-language support
- [ ] Enterprise features (team collaboration, audit logs)

---

**Made with MUHAMMAD HASSAN SHAHBAZ by developers who are tired of debugging RAG systems manually.**

*RAG-ORACLE: Because your RAG system deserves a doctor, not just a thermometer.*
