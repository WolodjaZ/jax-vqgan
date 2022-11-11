from invoke import task

CURRENT_PYTHON_VERSION = "3.8"


@task
def help(c):
    """Print help message."""
    print("Available tasks:")
    print("  - help: Print help message.")
    print("  - venv: Create a virtual environment.")
    print("  - clean: Clean all unnecessary files.")
    print("  - test: Test project.")
    print("  - test_extended: Test project with extended version.")
    print("  - coverage: Coverage analysis.")
    print("  - style: Lint with flake8.")
    print("  - mypy: Typing analysis.")
    print("  - docs: Build documentation.")


@task
def style(c, python_version=CURRENT_PYTHON_VERSION):
    c.run(f"nox -s lint-{python_version}")


@task
def venv(c):
    c.run("python3 -m venv venv")
    cmd = """source venv/bin/activate &&
            python3 -m pip install --upgrade pip setuptools wheel &&
            python3 -m pip install -e requirements.txt &&
            python3 -m pip install nox==2022.8.7 pre-commit==2.20.0 &&
            pre-commit install && pre-commit autoupdate
            """
    c.run(cmd)


@task
def clean(c, python_version=CURRENT_PYTHON_VERSION):
    c.run('find . -type f -name "*.DS_Store" -ls -delete')
    c.run(r'find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf')
    c.run('find . | grep -E ".pytest_cache" | xargs rm -rf')
    c.run('find . | grep -E ".ipynb_checkpoints" | xargs rm -rf')
    c.run('find . | grep -E ".trash" | xargs rm -rf')
    c.run("rm -f .coverage")


@task
def test(c, python_version=CURRENT_PYTHON_VERSION):
    c.run(f"nox -s test-{python_version}")


@task
def test_extended(c, python_version=CURRENT_PYTHON_VERSION):
    c.run(f"nox -s test_extended-{python_version}")


@task
def coverage(c, python_version=CURRENT_PYTHON_VERSION):
    c.run(f"nox -s coverage-{python_version}")


@task
def mypy(c, python_version=CURRENT_PYTHON_VERSION):
    c.run(f"nox -s mypy-{python_version}")


@task
def docs(c, python_version=CURRENT_PYTHON_VERSION):
    c.run(f"nox -s docs-{python_version}")
