from __future__ import annotations

import logging
import time

from .config import load_config, paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
log = logging.getLogger("localscribe")


def main() -> None:
    config = load_config()
    directories = paths()
    log.info("LocalScribe bootstrap ready; watching will be added by PLA-138")
    log.info("Input directory: %s", directories["input"])
    log.debug("Configuration: %s", config)
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
