"""Filesystem locations of bundled assets."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
MENAGERIE = ASSETS / "third_party" / "menagerie"
PANDA_XML = MENAGERIE / "franka_emika_panda" / "panda.xml"
OBJECTS = ASSETS / "objects"
SCENES = ASSETS / "scenes"
