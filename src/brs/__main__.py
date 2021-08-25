import logging
import os
import sys
 import click 

if os.environ.get("BRS_DEBUG"):
    LEVEL = logging.DEBUG
else:
    level_name = os.environ.get("BRS_LOG_LEVEL", "INFO")
    LEVEL = logging._nameToLevel.get(level_name, logging.INFO)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=LEVEL)


@click.command()
def main(args=None):
    """Console script for brs."""
    click.echo("Replace this message by putting your code into "
               "brs.__main__.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")

    return 0

if __name__ == "__main__":
    main(prog_name="Box Random Sets")
