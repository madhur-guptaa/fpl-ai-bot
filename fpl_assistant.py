import logging
import pickle
import sqlite3
from typing import Any, Dict, Generator, List, Optional

import faiss
import gradio as gr
from mlx_lm import load, stream_generate
from sentence_transformers import SentenceTransformer

import config
import prompts
import utils

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Singleton Services (Efficient Resource Management)
# ------------------------------------------------------------------------------

class ModelService:
    """
    Manages the Local LLM (Llama-3).
    Uses the Singleton pattern to ensure we only load the heavy model once.
    """
    _instance = None

    def __init__(self):
        self.model = None
        self.tokenizer = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        logger.info(f"🤖 Loading LLM: {config.MODEL_NAME}...")
        try:
            self.model, self.tokenizer = load(config.MODEL_NAME)
            logger.info("✅ LLM loaded successfully.")
        except Exception as e:
            logger.critical(f"❌ Failed to load LLM: {e}")
            raise

    def generate_stream(self, prompt: str, max_tokens: int = 1200) -> Generator[str, None, None]:
        """Streams text generation token-by-token."""
        for response in stream_generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens):
            yield response.text


class FAISSService:
    """
    Manages the Vector Database (FAISS).
    Handles semantic search retrieval.
    """
    _instance = None

    def __init__(self):
        self.index = None
        self.documents = None
        self.embedding_model = None
        self.metadatas = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_assets()
        return cls._instance

    def _load_assets(self):
        logger.info("🧠 Loading RAG Assets (FAISS + Embeddings)...")
        try:
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
            self.index = faiss.read_index(config.FAISS_INDEX_PATH)
            with open(config.DOCUMENTS_PATH, "rb") as f:
                self.documents = pickle.load(f)
            with open(config.METADATA_PATH, "rb") as f:
                self.metadatas = pickle.load(f)
            logger.info(f"✅ RAG System Ready ({len(self.documents)} documents loaded).")
        except Exception as e:
            logger.error(f"❌ Failed to load RAG assets. Did you run 'fpl_build_vector_db.py'? Error: {e}")
            raise

    def query_context(self, query: str, top_k: int = 15, doc_type: str = "all") -> List[str]:
        """Wrapper for the util function."""
        return utils.query_faiss(query, top_k, self.index, self.documents, self.embedding_model, self.metadatas,
                                 doc_type)


# ------------------------------------------------------------------------------
# Context Builders
# ------------------------------------------------------------------------------

class FPLService:
    @staticmethod
    def get_user_team_context(player_names: List[str], faiss_service: FAISSService) -> str:
        """Retrieves specific docs for the players currently in the user's team."""
        docs = utils.get_context_for_players(
            player_names, faiss_service.index, faiss_service.documents, faiss_service.embedding_model,
            faiss_service.metadatas
        )
        return "\n---\n".join(docs)

    @staticmethod
    def get_transfer_context(conn: sqlite3.Connection, current_gw: int, faiss_service: FAISSService,
                             user_player_ids: List[int]) -> str:
        """
        1. Identifies top transfer targets using the algorithm in utils.py.
           (Excludes players the user already owns)
        2. Retrieves their RAG documents (Form, Fixtures, etc.).
        """
        # Pass exclude_ids to prevent recommending owned players
        candidates = utils.find_transfer_targets(conn, current_gw, exclude_ids=user_player_ids)

        if not candidates:
            return "No transfer targets identified."

        # Clean names for FAISS (remove position tags like ' (MID)')
        candidate_names = [c.split(" (")[0] for c in candidates]

        docs = utils.get_context_for_players(
            candidate_names, faiss_service.index, faiss_service.documents, faiss_service.embedding_model,
            faiss_service.metadatas
        )
        return "\n---\n".join(docs)


# ------------------------------------------------------------------------------
# Prompt Construction
# ------------------------------------------------------------------------------

def format_history(history: List[Dict[str, str]]) -> str:
    """Converts the Gradio 'messages' format into a string for the prompt context."""
    if not history:
        return "No prior conversation."

    formatted = []
    for msg in history[-4:]:  # Only keep last 4 turns to save context window
        role = "User" if msg['role'] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)


def format_prompt(
        system_prompt: str,
        query: str,
        message_context: str,
        rank_info: str,
        user_team_list: str,
        user_team_context: str,
        transfer_context: str,
        chip_advice: str,
        risk_level: str,
        history_str: str
) -> List[Dict[str, str]]:
    """
    Assembles all data sources into a single structured prompt.
    """
    user_content = f"""
### 📊 USER CONTEXT
**Current Rank:** {rank_info}
**Current Team:** {user_team_list}
**Selected Risk Level:** {risk_level}

### 🧠 KNOWLEDGE BASE (RAG)
**Your Squad Analysis:**
{user_team_context}

**Transfer Market Insights (Recommended Buys):**
{transfer_context if transfer_context else "No transfer data requested."}

**Relevant General Info:**
{message_context}

### 💡 STRATEGIC ADVICE
{chip_advice if chip_advice else "No chip strategy relevant."}

### 📜 CONVERSATION HISTORY
{history_str}

---
### ❓ USER QUERY
"{query}"
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


# ------------------------------------------------------------------------------
# Main Chat Logic
# ------------------------------------------------------------------------------

def chat_logic(
        message: str,
        history: List[Dict[str, str]],  # Gradio 'messages' format
        team_id: Optional[int],
        risk_level: str,
        user_team: Dict[str, Any],
        user_team_list: str,
        user_context: str
) -> Generator[str, None, None]:
    # 1. Validation
    if not team_id:
        yield "⚠️ Please enter your Team ID above and click 'Load My Team' to start."
        return

    logger.info(f"📨 Query Received: '{message}' | Team: {team_id}")

    # 2. Service Initialization
    faiss_service = FAISSService.get_instance()
    model_service = ModelService.get_instance()
    current_gw = utils.get_current_gw()

    # 3. Intent Classification
    advice_keywords = [
        "transfer", "change", "sell", "buy", "captain", "wildcard", "chip", "replace",
        "optimize", "fix", "move", "suggestion", "recommend", "pick",
        "who", "whom", "best", "option", "trade", "bench", "start", "transfers"
    ]
    is_advice_query = any(k in message.lower() for k in advice_keywords)

    # 4. Intent Determination & Prompt Selection
    intent = "general"
    if any(k in message.lower() for k in ["fixture", "schedule", "run", "easy", "difficult"]):
        intent = "fixture_analysis"
        system_prompt = prompts.GENERAL_PROMPT
        logger.info("🎯 Intent: Fixture Analysis")
    elif is_advice_query:
        intent = "player_advice"
        system_prompt = f"{prompts.ADVICE_PROMPT}\n\n### EXAMPLES:{prompts.FEW_SHOT_ADVICE}"
        logger.info("🎯 Intent: Player Advice (Retrieving transfer targets...)")
    else:
        system_prompt = prompts.GENERAL_PROMPT
        logger.info("🎯 Intent: General Chat")

    # 5. Data Retrieval (Dynamic)
    transfer_context = ""
    chip_advice = ""

    if is_advice_query and current_gw:
        try:
            with sqlite3.connect(config.DB_NAME) as conn:
                # Extract IDs from the user_team state to exclude them from transfer targets
                current_ids = [p['player_id'] for p in user_team.get('player_data', []) if 'player_id' in p]

                transfer_context = FPLService.get_transfer_context(
                    conn, current_gw, faiss_service, user_player_ids=current_ids
                )
            chip_advice = utils.get_chip_strategy_advice(user_team.get("player_data", []), current_gw)
        except Exception as e:
            logger.warning(f"⚠️ Advice retrieval warning: {e}")

    # 6. Semantic Search (Static RAG)
    try:
        # Determine the doc_type based on intent
        if intent == "fixture_analysis":
            doc_type_filter = "fixture_run"
        else:
            doc_type_filter = "current_status"

        rag_hits = faiss_service.query_context(message, top_k=10, doc_type=doc_type_filter)
        message_context = "\n---\n".join(rag_hits)
    except Exception as e:
        message_context = ""
        logger.error(f"⚠️ FAISS Search failed: {e}")

    # 7. Prompt Assembly
    history_str = format_history(history)
    prompt_messages = format_prompt(
        system_prompt, message, message_context,
        user_team.get("rank_string", "N/A"),
        user_team_list, user_context,
        transfer_context, chip_advice,
        risk_level, history_str
    )

    # 8. Inference
    prompt_str = model_service.tokenizer.apply_chat_template(
        prompt_messages, add_generation_prompt=True, tokenize=False
    )
    prompt_str = utils.compress_prompt(prompt_str)

    logger.info("🤖 Generating response...")
    full_response = ""

    # Smart "Thinking" Logic for DeepSeek models
    is_thinking = True

    try:
        for token in model_service.generate_stream(prompt_str):
            full_response += token

            # --- LOGIC: Handle DeepSeek's <think> blocks ---
            if is_thinking:
                # 1. Did we find the closing tag?
                if "</think>" in full_response:
                    is_thinking = False
                    # Yield the answer immediately
                    answer = full_response.split("</think>")[-1].strip()
                    if answer: yield answer

                # 2. FAILSAFE: Did the model forget the tag but start the answer?
                # If we see standard Markdown headers or bold text, switch mode.
                elif "**" in full_response or "###" in full_response or "📉" in full_response:
                    is_thinking = False
                    # Try to strip the thought if possible, otherwise just yield everything
                    if "</think>" in full_response:
                        yield full_response.split("</think>")[-1].strip()
                    else:
                        yield full_response

                # 3. Buffer check: if text is huge and NO tag found, force yield
                elif len(full_response) > 300 and "<think>" not in full_response:
                    is_thinking = False
                    yield full_response

                # 4. Still thinking? Show loader.
                else:
                    yield "🧠 *Analyzing FPL stats & fixtures...*"

            else:
                # --- NORMAL STREAMING ---
                # Just stream the answer clean
                if "</think>" in full_response:
                    yield full_response.split("</think>")[-1].strip()
                else:
                    yield full_response

    except Exception as e:
        logger.error(f"❌ Generation Error: {e}")
        yield "⚠️ An error occurred during generation. Please check the logs."


# ------------------------------------------------------------------------------
# UI Callbacks
# ------------------------------------------------------------------------------

def set_team_info(team_id_str: str, risk_level: str):
    """
    Fetches the user's team data when they click the 'Set' button.
    Stores this in Gradio State so we don't re-fetch it every message.
    """
    if not team_id_str.strip().isdigit():
        raise gr.Error("Invalid Team ID. Please enter numbers only.")

    team_id = int(team_id_str)
    gr.Info(f"🔄 Fetching data for Team {team_id}...")

    try:
        current_gw = utils.get_current_gw()
        if not current_gw:
            raise gr.Error("Could not connect to FPL API (No current Gameweek).")

        # Fetch basic team info (Includes Next Match Logic from utils)
        user_team = utils.get_user_team(team_id, current_gw)
        if not user_team or not user_team.get("player_data"):
            raise gr.Error(f"Team {team_id} not found or has no players.")

        player_summaries = []
        player_names = []

        for p in user_team["player_data"]:
            name = p['name']
            player_names.append(name)

            # --- Rich Context Injection ---
            # Includes 'Next Match' to prevent hallucinations
            summary = (
                f"{name} | "
                f"Next: {p.get('next_match', 'N/A')} | "
                f"Form: {p.get('form', 0)} | "
                f"xG: {p.get('xG', 0)} | "
                f"xA: {p.get('xA', 0)}"
            )
            player_summaries.append(summary)

        user_team_list = "\n".join(player_summaries)  # Pass this rich list to the LLM

        # We need the FAISS service to get player context
        faiss_service = FAISSService.get_instance()
        user_context = FPLService.get_user_team_context(player_names, faiss_service)

        gr.Info(f"✅ Success! Loaded {len(player_names)} players for Team {team_id}.")
        return team_id, risk_level, user_team, user_team_list, user_context

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        raise gr.Error(f"Error loading team: {str(e)}")


# ------------------------------------------------------------------------------
# Gradio App Launcher
# ------------------------------------------------------------------------------

with gr.Blocks(theme=gr.themes.Soft(), title="FPL AI Agent") as demo:
    # Header
    gr.Markdown(
        """
        # ⚽ FPL RAG Assistant
        **Your Personal Fantasy Premier League Analyst** *Powered by Deepseek, FAISS, and Live FPL Data*
        """
    )

    # State Variables (Hidden memory)
    team_state = gr.State(None)
    risk_state = gr.State("Balanced")
    user_team_state = gr.State({})
    user_team_list_state = gr.State("")
    user_context_state = gr.State("")

    # Configuration Row
    with gr.Row(variant="panel"):
        with gr.Column(scale=1):
            team_input = gr.Textbox(
                label="FPL Team ID",
                placeholder="e.g. 5978138",
                info="Find this in your browser URL on the FPL Points page."
            )
        with gr.Column(scale=1):
            risk_input = gr.Radio(
                ["Conservative", "Balanced", "Aggressive"],
                label="Strategy / Risk Profile",
                value="Balanced",
                interactive=True
            )
        with gr.Column(scale=1):
            set_btn = gr.Button("🚀 Load My Team", variant="primary")

    # Chat Interface
    chatbot = gr.ChatInterface(
        fn=chat_logic,
        additional_inputs=[
            team_state,
            risk_state,
            user_team_state,
            user_team_list_state,
            user_context_state
        ],
        title="💬 Chat with Magujawischlichhetu the Astute",
        description="Ask questions about transfers, captaincy, and chips.",
        type="messages",  # Uses list of dicts for history
    )

    # Event Wiring
    set_btn.click(
        fn=set_team_info,
        inputs=[team_input, risk_input],
        outputs=[team_state, risk_state, user_team_state, user_team_list_state, user_context_state]
    )

if __name__ == "__main__":
    # Ensure assets are loaded before UI starts
    print("--- ⏳ Initializing System ---")
    try:
        # Pre-load to fail fast if config is wrong
        FAISSService.get_instance()
        ModelService.get_instance()
        print("--- ✅ System Ready. Launching UI ---")
        demo.launch()
    except Exception as e:
        print(f"--- ❌ Fatal Startup Error: {e} ---")
