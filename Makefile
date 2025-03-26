deploy:
	streamlit run app/app.py

test:
	pytest --cov=biasx --cov-report=term --cov-report=html -p no:warnings

clean:
	rm -rf .pytest_cache .coverage htmlcov __pycache__ .ipynb_checkpoints */__pycache__ dist build biasx.egg-info

build:
	rm -rf biasx.egg-info dist build && python -m build

upload:
	python -m twine upload dist/*
