"""The app is installed as a package (the Docker image does `pip install .`), so every template and SQL
file must be listed in pyproject's package data. A missing one only shows up as a 500 on a server."""

from __future__ import annotations

import fnmatch
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_template_sql_and_static_file_is_packaged():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    patterns = config["tool"]["setuptools"]["package-data"]["broll"]
    package = ROOT / "src" / "broll"
    shipped = [p for pattern in ("db/*.sql", "search/*.txt", "web/templates/*.html", "web/templates/partials/*.html")
               for p in package.glob(pattern)]
    static = [p for p in (package / "web" / "static").glob("*") if p.is_file()] if (package / "web" / "static").exists() else []
    assert shipped, "no templates found - the test is looking in the wrong place"
    missing = [
        str(p.relative_to(package))
        for p in shipped + static
        if not any(fnmatch.fnmatch(str(p.relative_to(package)), pattern) for pattern in patterns)
    ]
    assert not missing, f"not in package-data, so missing from an installed copy: {missing}"
