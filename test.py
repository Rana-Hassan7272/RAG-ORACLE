from rag_pipeline import RAGOracle
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Initialize the diagnostic engine with embeddings for auto-evaluation
# Using HuggingFace embeddings (free, no API key needed)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
oracle = RAGOracle(embeddings=embeddings)

# 2. Run diagnosis
result = oracle.diagnose(
    query="What is the capital of France?",
    answer="Paris is the capital of France.",
    chunks=["France is a country in Europe. Paris is the capital city of France.", "Paris is the capital city of France and is located in the north-central part of the country."],
    config={"top_k": 2, "temperature": 0.7}  # Optional: your system config
)

# 3. Inspect results
print("=" * 50)
print("DIAGNOSIS RESULT")
print("=" * 50)
print(f"Query ID: {result.get('query_id')}")
print(f"Outcome: {result.get('outcome')}")
print(f"\nRoot Causes Found: {len(result.get('root_causes', []))}")

if result.get("root_causes"):
    primary = result["root_causes"][0]
    print(f"\nPrimary Issue: {primary['type']}")
    print(f"Suggested Fix: {primary['fix']}")
    print(f"Explanation: {primary['user_explanation']}")
    print(f"Confidence: {primary['confidence']:.2f}")
    print(f"Unfixable: {primary['is_unfixable']}")
else:
    print("\n✅ No issues detected - system is healthy!")
