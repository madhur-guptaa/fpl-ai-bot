import os
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

import config
import fpl_assistant
import fpl_build_vector_db
# Import your modules
import fpl_process_api

console = Console()


def print_header():
    console.clear()
    console.print(Panel.fit(
        "[bold green]⚽ FPL AI Agent Manager[/bold green]\n"
        "[italic]DeepSeek R1 • FAISS • Live FPL Data[/italic]",
        style="green"
    ))


def run_etl_refresh():
    console.print("\n[bold cyan]🔄 Step 1: Refreshing Live Data (ETL)...[/bold cyan]")
    try:
        # We call the main function of your ETL script
        fpl_process_api.main()
        console.print("[bold green]✅ Data Refresh Complete![/bold green]")
    except Exception as e:
        console.print(f"[bold red]❌ ETL Failed:[/bold red] {e}")
        sys.exit(1)


def run_knowledge_build():
    console.print("\n[bold cyan]🧠 Step 2: Rebuilding Knowledge Base (RAG)...[/bold cyan]")
    try:
        fpl_build_vector_db.main()
        console.print("[bold green]✅ Vector Database Rebuilt![/bold green]")
    except Exception as e:
        console.print(f"[bold red]❌ RAG Build Failed:[/bold red] {e}")
        sys.exit(1)


def launch_assistant():
    console.print("\n[bold cyan]🤖 Step 3: Launching AI Assistant UI...[/bold cyan]")
    try:
        # We need to ensure assets are loaded before UI launch
        print("Initializing services...")
        fpl_assistant.FAISSService.get_instance()
        fpl_assistant.ModelService.get_instance()
        fpl_assistant.demo.launch()
    except Exception as e:
        console.print(f"[bold red]❌ UI Launch Failed:[/bold red] {e}")


def main():
    while True:
        print_header()
        console.print("[1] [bold]Full Setup[/bold] (Refresh Data + Build DB + Launch UI)")
        console.print("[2] [bold]Refresh Data Only[/bold] (ETL + RAG Build)")
        console.print("[3] [bold]Launch UI Only[/bold] (Use existing data)")
        console.print("[4] Exit")

        choice = Prompt.ask("\nSelect an option", choices=["1", "2", "3", "4"], default="1")

        if choice == "1":
            run_etl_refresh()
            run_knowledge_build()
            launch_assistant()
        elif choice == "2":
            run_etl_refresh()
            run_knowledge_build()
            Prompt.ask("\nPress Enter to return to menu...")
        elif choice == "3":
            if not os.path.exists(config.FAISS_INDEX_PATH):
                console.print("[bold red]⚠️ Error:[/bold red] No Knowledge Base found. Please run Option 2 first.")
                time.sleep(2)
            else:
                launch_assistant()
        elif choice == "4":
            console.print("Goodbye! 👋")
            sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
