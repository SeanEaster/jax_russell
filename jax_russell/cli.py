"""Console script for jax_russell."""

import click


@click.command()
def main():
    """Main entrypoint."""
    click.echo("jax_russell")
    click.echo("=" * len("jax_russell"))
    click.echo("Formulas to tell you when the price is right")


if __name__ == "__main__":
    main()  # pragma: no cover
