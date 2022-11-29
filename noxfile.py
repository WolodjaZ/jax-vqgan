import os

import nox

os.environ.update({"PDM_IGNORE_SAVED_PYTHON": "1"})

SUPPORTED_PY_VERSIONS = ["3.8", "3.9", "3.10"]
nox.options.sessions = ["test", "test_extended", "lint", "coverage", "mypy", "docs", "lint_nb"]


@nox.session(python=SUPPORTED_PY_VERSIONS)
def test(session: nox.Session) -> None:
    """Pytesting."""
    session.run("pdm", "install", "-G", "test", external=True)
    session.run("pytest", "-m", "not training_long", "tests")


@nox.session(python=SUPPORTED_PY_VERSIONS)
def test_extended(session: nox.Session) -> None:
    """Pytesting. Extended version."""
    session.run("pdm", "install", "-G", "test", external=True)
    session.run("pytest", "tests")


@nox.session(python=SUPPORTED_PY_VERSIONS)
def coverage(session: nox.Session) -> None:
    """Coverage analysis."""
    session.run("pdm", "install", "-G", "test", external=True)
    session.run("coverage", "erase")
    session.run(
        "coverage",
        "run",
        "--source",
        "modules",
        "--append",
        "-m",
        "pytest",
        "-m",
        "not training_long",
        "tests",
    )
    session.run("coverage", "report", "--fail-under=1", "--ignore-errors")  # 90
    session.run("coverage", "erase")


@nox.session(python=SUPPORTED_PY_VERSIONS)
def mypy(session: nox.Session) -> None:
    """Type check."""
    session.run("pdm", "install", "-G", "type_check", external=True)
    """Run mypy."""
    session.run("mypy", "--ignore-missing-imports", "--check-untyped-defs", "modules")


@nox.session(python=SUPPORTED_PY_VERSIONS)
def lint(session: nox.Session) -> None:
    """Lint install."""
    session.run("pdm", "install", "-G", "lint", external=True)
    """Run mypy."""
    session.run(
        "flake8",
        "--count",
        "--select=F",
        "--show-source",
        "--statistics",
        "--max-line-length=100",
    )


@nox.session(python=SUPPORTED_PY_VERSIONS)
def lint_nb(session: nox.Session) -> None:
    """Lint notebooks."""
    session.run("pdm", "install", "-G", "lint", "-G", "notebook_lint", external=True)
    session.run("nbqa", "black", "notebooks")
    session.run("nbqa", "flake8", "--max-line-length=100", "notebooks")
    session.run("nbqa", "isort", "notebooks")
    session.run(
        "jupyter",
        "nbconvert",
        "--clear-output",
        "--inplace",
        "notebooks/visualize.ipynb",
        "notebooks/optimize.ipynb",
    )


@nox.session(python=SUPPORTED_PY_VERSIONS)
def docs(session: nox.Session) -> None:
    """Build documentation."""
    session.run("pdm", "install", "-G", "doc", external=True)
    session.run("mkdocs", "build", "--clean")
