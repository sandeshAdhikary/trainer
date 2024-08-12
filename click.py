import click
from trainer.run_study import main

@click.group()
def cli():
    pass


@cli.command()
@click.argument('args', nargs=-1)
def run(args):
    main(args)


if __name__ == "__main__":
    cli()