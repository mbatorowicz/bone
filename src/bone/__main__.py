"""python -m bone | python -m bone studio"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in {"studio", "ui", "view-ui"}:
        from bone.api.studio import main as studio_main

        studio_main()
    else:
        from bone.cli import main as cli_main

        cli_main()


if __name__ == "__main__":
    main()
