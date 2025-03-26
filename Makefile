deploy:
	streamlit run app/app.py

test:
	pytest --cov=biasx --cov-report=term --cov-report=html -p no:warnings

clean:
	rm -rf .pytest_cache .coverage htmlcov __pycache__ .ipynb_checkpoints */__pycache__