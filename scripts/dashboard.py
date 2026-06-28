from __future__ import annotations

import subprocess
import sys

from worldcup.utils.paths import project_root


def main() -> None:
    app_path = project_root() / "src" / "worldcup" / "dashboard" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        check=True,
    )


if __name__ == "__main__":
    main()
