"""VIO / GPS ATE Studio 進入點。

  python main.py                     # 開 GUI
  python main.py <log.bin>           # 開 GUI 並直接載入該 log
  python main.py --cli <log.bin> ... # 命令列(見 cli.py)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    if "--cli" in sys.argv:
        from cli import main as cli_main
        argv = [a for a in sys.argv[1:] if a != "--cli"]
        return cli_main(argv)
    try:
        from ui.main_window import run
    except ImportError as e:
        print(f"無法載入 GUI:{e}\n請先安裝相依套件:pip install -r requirements.txt", file=sys.stderr)
        return 1
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
