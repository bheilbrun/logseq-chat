@default: lint test

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

@test:
    poetry run pytest tests

@push: lint test
    git push
