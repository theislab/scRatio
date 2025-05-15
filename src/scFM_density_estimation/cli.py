"""Console script for scFM_density_estimation."""
import scFM_density_estimation

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for scFM_density_estimation."""
    console.print("Replace this message by putting your code into "
               "scFM_density_estimation.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
