install:
	pip install -r requirements.txt
	pip install -e .

train:
	python scripts/train.py

test:
	pytest -q

api:
	uvicorn api.main:app --reload
