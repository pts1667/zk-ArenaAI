"""Writes sim/include/zksim/rawframe.hpp from the raw frame schema."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bots"))
from zkbot.policy import rawframe  # noqa: E402

(ROOT / "sim" / "include" / "zksim" / "rawframe.hpp").write_text(rawframe.cpp_header(), encoding="utf8", newline="\n")
