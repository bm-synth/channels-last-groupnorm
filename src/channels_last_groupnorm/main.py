# Copyright (c) 2025 Synthesia Limited - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.

import logging

import typer

from datalib import InputPath

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(salutation: str = "Hello") -> None:
    logger.info(f"{salutation} World!")
    InputPath("README.md")


if __name__ == "__main__":
    typer.run(main)
