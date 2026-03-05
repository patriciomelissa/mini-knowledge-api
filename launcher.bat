@echo off

echo Setup starting...

if not exist env-mini-knowledge-api (
    python -m venv env-mini-knowledge-api
)

call env-mini-knowledge-api\Scripts\activate

python -m pip install --upgrade pip

pip install -r requirements.txt
pip install -r requirements-dev.txt

pre-commit install

echo Setup completed!

pause