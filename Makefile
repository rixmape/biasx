deploy:
	streamlit run app/app.py

test-all:
	pytest --cov=biasx --cov-report=term --cov-report=html -p no:warnings

test-unit:
	pytest tests/unit --cov=biasx --cov-report=term --cov-report=html -p no:warnings

test-integration:
	pytest tests/integration --cov=biasx --cov-report=term --cov-report=html -p no:warnings

test-system:
	pytest tests/system-level --cov=biasx --cov-report=term --cov-report=html -p no:warnings

clean:
	rm -rf .pytest_cache .coverage htmlcov __pycache__ .ipynb_checkpoints */__pycache__ dist build biasx.egg-info

build:
	rm -rf biasx.egg-info dist build && python -m build

upload:
	python -m twine upload dist/*

tree:
	tree -I ".git|tmp*|*tmp|.vscode|__pycache__|.venv|tests|outputs|logs"

docs-serve:
	mkdocs serve

docs-deploy:
	mkdocs gh-deploy
