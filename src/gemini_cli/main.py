"""
Main entry point for the Gemini CLI application.
Targets Gemini 2.5 Pro Experimental. Includes ASCII Art welcome.
Passes console object to model.
"""

import os
import sys
import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from pathlib import Path
import yaml
import google.generativeai as genai
import logging
import time

from .models.gemini import GeminiModel, list_available_models
from .config import Config
from .utils import count_tokens
from .tools import AVAILABLE_TOOLS

# Setup console and config
console = Console() # Create console instance HERE
try:
    config = Config()
except Exception as e:
    console.print(f"[bold red]Error loading configuration:[/bold red] {e}")
    config = None

# Setup logging - MORE EXPLICIT CONFIGURATION
log_level = os.environ.get("LOG_LEVEL", "WARNING").upper() # <-- Default back to WARNING
log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'

# Get root logger and set level
root_logger = logging.getLogger()
root_logger.setLevel(log_level)

# Remove existing handlers to avoid duplicates if basicConfig was called elsewhere
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Add a stream handler to output to console
stream_handler = logging.StreamHandler(sys.stdout) 
stream_handler.setLevel(log_level)
formatter = logging.Formatter(log_format)
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)

log = logging.getLogger(__name__) # Get logger for this module
log.info(f"Logging initialized with level: {log_level}") # Confirm level

# --- Default Model ---
DEFAULT_MODEL = "gemini-2.5-pro-exp-03-25"
# --- ---

# --- ASCII Art Definition ---
GEMINI_CODE_ART = r"""

[medium_purple]
  ██████╗ ███████╗███╗   ███╗██╗███╗   ██╗██╗        ██████╗  ██████╗ ██████╗ ███████╗
 ██╔════╝ ██╔════╝████╗ ████║██║████╗  ██║██║       ██╔════╝ ██╔═══██╗██╔══██╗██╔════╝
 ██║ ███╗███████╗██╔████╔██║██║██╔██╗ ██║██║       ██║      ██║   ██║██║  ██║███████╗
 ██║  ██║██╔════╝██║╚██╔╝██║██║██║╚██╗██║██║       ██║      ██║   ██║██║  ██║██╔════╝
 ╚██████╔╝███████╗██║ ╚═╝ ██║██║██║ ╚████║██║       ╚██████╗ ╚██████╔╝██████╔╝███████╗
  ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝        ╚═════╝  ╚═════╝ ╚═════╝ ╚══════╝
[/medium_purple]
"""
# --- End ASCII Art ---


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
@click.option(
    '--model', '-m',
    help=f'Model ID to use (e.g., gemini-2.5-pro-exp-03-25). Default: {DEFAULT_MODEL}',
    default=None
)
@click.pass_context
def cli(ctx, model):
    """Interactive CLI for Gemini models with coding assistance tools."""
    if not config:
        console.print("[bold red]Configuration could not be loaded. Cannot proceed.[/bold red]")
        sys.exit(1)

    if ctx.invoked_subcommand is None:
        model_name_to_use = model or config.get_default_model() or DEFAULT_MODEL
        log.info(f"Attempting to start interactive session with model: {model_name_to_use}")
        # Pass the console object to start_interactive_session
        start_interactive_session(model_name_to_use, console)

# ... (setup, set_default_model, list_models functions remain the same) ...
@cli.command()
@click.argument('key', required=True)
def setup(key):
    if not config: console.print("[bold red]Config error.[/bold red]"); return
    try: config.set_api_key("google", key); console.print("[green]✓[/green] Google API key saved.")
    except Exception as e: console.print(f"[bold red]Error saving API key:[/bold red] {e}")

@cli.command()
@click.argument('model_name', required=True)
def set_default_model(model_name):
    if not config: console.print("[bold red]Config error.[/bold red]"); return
    try: config.set_default_model(model_name); console.print(f"[green]✓[/green] Default model set to [bold]{model_name}[/bold].")
    except Exception as e: console.print(f"[bold red]Error setting default model:[/bold red] {e}")

@cli.command()
@click.option('--detailed', '-d', is_flag=True, help='Show detailed capability information')
def list_models(detailed):
    """List available models with their capabilities."""
    
    if not config: console.print("[bold red]Config error.[/bold red]"); return
    api_key = config.get_api_key("google")
    if not api_key: console.print("[bold red]Error:[/bold red] API key not found. Run 'gemini setup'."); return
    console.print("[yellow]Fetching models...[/yellow]")
    try:
        models_list = list_available_models(api_key, detect_capabilities=True)
        if not models_list or (isinstance(models_list, list) and len(models_list) > 0 and isinstance(models_list[0], dict) and "error" in models_list[0]):
             console.print(f"[red]Error listing models:[/red] {models_list[0].get('error', 'Unknown error') if models_list else 'No models found or fetch error.'}"); return
        
        console.print("\n[bold cyan]Available Models (Access may vary):[/bold cyan]")
        
        # Define capability symbols legend
        if detailed:
            from .models.gemini import CAPABILITY_SYMBOLS
            legend = "\n[bold]Capability Legend:[/bold]\n"
            for cap, symbol in CAPABILITY_SYMBOLS.items():
                legend += f"  {symbol}: {cap.replace('_', ' ').title()}\n"
            console.print(legend)
        
        for model_data in models_list:
            # Basic model info
            model_info = f"- [bold green]{model_data['name']}[/bold green]"
            
            # Add capability symbols if available
            if "capability_symbols" in model_data and model_data["capability_symbols"]:
                symbols = " ".join([f"[{get_capability_color(cap)}]{cap}[/{get_capability_color(cap)}]" 
                                   for cap in model_data["capability_symbols"]])
                model_info += f" {symbols}"
            
            # Add display name
            model_info += f" (Display: {model_data.get('display_name', 'N/A')})"
            
            console.print(model_info)
            
            # Add detailed capabilities if requested
            if detailed and "capabilities" in model_data:
                capabilities_text = "  Capabilities: " + ", ".join([cap.replace("_", " ").title() for cap in model_data["capabilities"]])
                console.print(f"  [dim]{capabilities_text}[/dim]")
                if "description" in model_data and model_data["description"]:
                    desc_text = model_data["description"]
                    # Truncate long descriptions
                    if len(desc_text) > 100:
                        desc_text = desc_text[:97] + "..."
                    console.print(f"  [dim]Description: {desc_text}[/dim]")
                console.print("")
        
        console.print("\nUse 'gemini --model MODEL' or 'gemini set-default-model MODEL'.")
        console.print("Use 'gemini list-models --detailed' to see full capability information.")
    except Exception as e: console.print(f"[bold red]Error listing models:[/bold red] {e}"); log.error("List models failed", exc_info=True)


def get_capability_color(capability):
    """Returns an appropriate color based on capability type."""
    if "❌" in capability:
        return "red"
    elif "Code" in capability:
        return "green"
    elif "File" in capability:
        return "blue"
    elif "System" in capability:
        return "magenta"
    elif "Advanced" in capability or "🚀" in capability:
        return "yellow"
    else:
        return "cyan"


@cli.command()
@click.argument('model_name', required=True)
def test_capabilities(model_name):
    """Force test and update capabilities for a specific model."""
    if not config: console.print("[bold red]Config error.[/bold red]"); return
    api_key = config.get_api_key("google")
    if not api_key: console.print("[bold red]Error:[/bold red] API key not found. Run 'gemini setup'."); return
    
    console.print(f"[yellow]Testing capabilities for model: {model_name}...[/yellow]")
    
    try:
        # Initialize model with the specified name
        from .models.gemini import GeminiModel
        model = GeminiModel(api_key=api_key, console=console, model_name=model_name)
        
        # Force test capabilities
        with console.status(f"[yellow]Testing model capabilities...[/yellow]", spinner="dots"):
            capabilities = model.test_model_capabilities(force_test=True)
        
        # Display results
        console.print("\n[bold green]Capability test completed![/bold green]")
        console.print(f"[bold]Model:[/bold] {model_name}")
        console.print("[bold]Capabilities:[/bold]")
        
        from .models.gemini import CAPABILITY_SYMBOLS
        for capability in capabilities:
            symbol = CAPABILITY_SYMBOLS.get(capability, capability)
            color = get_capability_color(symbol)
            console.print(f"  [{color}]{symbol}[/{color}] ({capability.replace('_', ' ').title()})")
        
        console.print("\nCapabilities have been tested and cached for future use.")
        
    except Exception as e:
        console.print(f"[bold red]Error testing model capabilities:[/bold red] {e}")
        log.error(f"Failed to test capabilities for {model_name}", exc_info=True)


# --- MODIFIED start_interactive_session to accept and pass console ---
def start_interactive_session(model_name: str, console: Console):
    """Start an interactive chat session with the selected Gemini model."""
    if not config: console.print("[bold red]Config error.[/bold red]"); return

    # --- Display Welcome Art ---
    console.clear()
    console.print(GEMINI_CODE_ART)
    console.print(Panel("[b]Welcome to Gemini Code AI Assistant![/b]", border_style="blue", expand=False))
    time.sleep(0.1)
    # --- End Welcome Art ---

    api_key = config.get_api_key("google")
    if not api_key:
        console.print("\n[bold red]Error:[/bold red] Google API key not found.")
        console.print("Please run [bold]'gemini setup YOUR_API_KEY'[/bold] first.")
        return

    try:
        console.print(f"\nInitializing model [bold]{model_name}[/bold]...")
        # Pass the console object to GeminiModel constructor
        model = GeminiModel(api_key=api_key, console=console, model_name=model_name)
        console.print("[green]Model initialized successfully.[/green]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error initializing model '{model_name}':[/bold red] {e}")
        log.error(f"Failed to initialize model {model_name}", exc_info=True)
        console.print("Please check model name, API key permissions, network. Use 'gemini list-models'.")
        return

    # --- Session Start Message ---
    console.print("Type '/help' for commands, '/exit' or Ctrl+C to quit.")

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ")

            if user_input.lower() == '/exit': break
            elif user_input.lower() == '/help': show_help(); continue

            # Display initial "thinking" status - generate handles intermediate ones
            response_text = model.generate(user_input)

            if response_text is None and user_input.startswith('/'): console.print(f"[yellow]Unknown command:[/yellow] {user_input}"); continue
            elif response_text is None: console.print("[red]Received an empty response from the model.[/red]"); log.warning("generate() returned None unexpectedly."); continue

            console.print("[bold medium_purple]Gemini:[/bold medium_purple]")
            console.print(Markdown(response_text), highlight=True)

        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrupted. Exiting.[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[bold red]An error occurred during the session:[/bold red] {e}")
            log.error("Error during interactive loop", exc_info=True)


def show_help():
    """Show help information for interactive mode."""
    tool_list_formatted = ""
    if AVAILABLE_TOOLS:
        # Add indentation for the bullet points
        tool_list_formatted = "\n".join([f"  • [white]`{name}`[/white]" for name in sorted(AVAILABLE_TOOLS.keys())])
    else:
        tool_list_formatted = "  (No tools available)"
        
    # Use direct rich markup and ensure newlines are preserved
    help_text = f""" [bold]Help[/bold]

 [cyan]Interactive Commands:[/cyan]
  /exit
  /help

 [cyan]CLI Commands:[/cyan]
  gemini setup KEY
  gemini list-models
  gemini set-default-model NAME
  gemini --model NAME

 [cyan]Workflow Hint:[/cyan] Analyze -> Plan -> Execute -> Verify -> Summarize

 [cyan]Available Tools:[/cyan]
{tool_list_formatted}
"""
    # Print directly to Panel without Markdown wrapper
    console.print(Panel(help_text, title="Help", border_style="green", expand=False))


if __name__ == "__main__":
    cli()