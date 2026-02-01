import os

# ------------------------------------------------------------------------------
# Project Root
# ------------------------------------------------------------------------------
# Define the absolute path to the project's root directory.
# This ensures file paths remain correct regardless of where the script is run.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------------------
# Database Configuration
# ------------------------------------------------------------------------------
# Absolute path to the SQLite database
DB_NAME = os.path.join(PROJECT_ROOT, "fpl_data.db")

# Table definitions
SOURCE_TABLE = "historical_gameweek_data"
DESTINATION_TABLE = "historical_player_data"

# ------------------------------------------------------------------------------
# Model Configuration
# ------------------------------------------------------------------------------
# The specific quantized LLM from Hugging Face optimized for MLX (Apple Silicon)
# Option A: mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
# Option B: mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit (Current Selection)
MODEL_NAME = "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit"

# Sentence-transformer model for semantic embeddings
# 'all-MiniLM-L6-v2' is chosen for its balance of speed and accuracy
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ------------------------------------------------------------------------------
# Vector Store (FAISS)
# ------------------------------------------------------------------------------
# Paths for the FAISS index and associated artifacts
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "faiss_index.bin")
DOCUMENTS_PATH = os.path.join(PROJECT_ROOT, "documents.pkl")
METADATA_PATH = os.path.join(PROJECT_ROOT, "metadata.pkl")

# ------------------------------------------------------------------------------
# External API
# ------------------------------------------------------------------------------
FPL_API_URL = "https://fantasy.premierleague.com/api/"

# ------------------------------------------------------------------------------
# Retrieval Settings
# ------------------------------------------------------------------------------
# Number of documents to retrieve from FAISS context
TOP_K_RESULTS = 15
