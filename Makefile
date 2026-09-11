install:
	pip install -r requirements.txt

test:
	pytest -q

profile:
	python scripts/profile_dataset.py

train:
	python scripts/train.py

run:
	uvicorn app.main:app --reload
