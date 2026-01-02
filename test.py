from rag_pipeline import RAGOracle

# 1. Initialize the diagnostic engine
oracle = RAGOracle(embeddings=None)  # You *can* pass your embedding model if available

# 2. Run diagnosis
result = oracle.diagnose(
    query="What is the capital of France?",
    answer="Paris is the capital of France.",
    chunks=[ "France is a country in Europe...", "Paris is the capital city of France..." ],
)

# 3. Inspect results
print(result)
print("Suggested Fix:", result["root_causes"][0]["fix"])
print("Explanation:", result["root_causes"][0]["user_explanation"])
