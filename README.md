# ⚽ FPL RAG Agent (Local DeepSeek R1)

An intelligent Fantasy Premier League assistant that runs 100% locally on your machine. It combines **DeepSeek R1** (via MLX) with a **RAG (Retrieval-Augmented Generation)** system built on live FPL API data.

## 🚀 Features
* **Hybrid RAG:** Retrieves player stats *and* upcoming fixture difficulty to ground the LLM's answers.
* **Anti-Hallucination:** Hardcodes specific opponents into the context so the AI never guesses fixtures.
* **Ownership Awareness:** Connects to your specific Team ID and filters out players you already own from transfer suggestions.
* **Logic Reasoning:** Hides the raw "Chain of Thought" tokens for a clean user experience.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/fpl-rag-agent.git](https://github.com/yourusername/fpl-rag-agent.git)
    cd fpl-rag-agent
    ```

2.  **Set up a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

Run the all-in-one manager:
```bash
python main.py