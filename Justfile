@default: lint

@init:
    poetry install

@lint:
    echo "Checking formatting..."
    poetry run ruff format --check .
    echo "\nLinting..."
    poetry run ruff check .
    echo "\nDependency analysis..."
    poetry run deptry .
    echo "\nType checking..."
    poetry run mypy .

@fix:
    poetry run ruff format .
    poetry run ruff check --fix .

@push: lint
    git push
