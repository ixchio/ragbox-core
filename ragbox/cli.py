import typer
import asyncio
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from ragbox import RAGBox
from ragbox.config.defaults import Settings

app = typer.Typer(
    name="ragbox",
    help="RAGBox: The RAG framework for people who don't want to think about RAG.",
    add_completion=False,
)

console = Console()


@app.command()
def init(
    document_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory containing local documents to index.",
    ),
    config_file: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to an optional JSON/YAML configuration file.",
    ),
):
    """
    Initialize a RAGBox instance in the specified directory.

    This command blocks until the initial index and knowledge graph are built.
    Afterward, it starts the background daemon to watch for changes.
    """
    console.print(
        Panel(
            f"Initializing RAGBox at [cyan]{document_dir}[/cyan]", border_style="green"
        )
    )

    # In a real implementation this might parse config_file into Settings
    settings = Settings()

    try:
        # Pre-flight cost estimation
        rg_pre = RAGBox(document_dir, config=settings)
        cost_estimate = rg_pre.estimate_cost(query=None)
        console.print(
            Panel(f"[bold yellow]{cost_estimate}[/bold yellow]", title="Cost Estimate")
        )

        # RAGBox __init__ calls _ensure_built which builds synchronously implicitly via asyncio
        with console.status(
            "[bold green]Building Vector Store and Knowledge Graph...", spinner="dots"
        ):
            pass  # It's already built by the init above
        console.print(
            "[bold green]✓ Indexing complete.[/bold green] RAGBox is ready for queries."
        )
        console.print(
            "The self-healing daemon is now active in the background for this session."
        )

        # Keep process alive to allow watchdog to run, or exit if init is purely
        # meant to pre-compute. Because RAGBox design spins up daemon threads,
        # we'll keep it alive here if we want continuous syncing.
        console.print("[yellow]Press Ctrl+C to stop the daemon.[/yellow]")
        try:
            while True:
                # Wait forever
                asyncio.run(asyncio.sleep(3600))
        except KeyboardInterrupt:
            console.print("Daemon stopped.")

    except Exception as e:
        console.print(f"[bold red]Error initializing RAGBox: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def query(
    question: str = typer.Argument(..., help="The question to ask the RAG system."),
    document_dir: Path = typer.Option(
        ".",
        "--dir",
        "-d",
        help="Directory where RAGBox is initialized.",
        exists=True,
        dir_okay=True,
    ),
):
    """
    Query an existing RAGBox instance.
    """
    console.print(f"[bold blue]Querying RAGBox in {document_dir}...[/bold blue]\n")
    try:
        rg = RAGBox(document_dir)

        # Estimation
        cost_estimate = rg.estimate_cost(query=question)
        console.print(f"[yellow]Estimation:[/yellow] {cost_estimate}")

        with console.status(
            "[bold green]Agentic Orchestrator thinking...", spinner="dots"
        ):
            answer = rg.query(question)

        console.print(Panel(Markdown(answer), title="Answer", border_style="blue"))

    except Exception as e:
        console.print(f"[bold red]Error querying RAGBox: {e}[/bold red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
