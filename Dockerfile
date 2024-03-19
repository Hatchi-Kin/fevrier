FROM python:3.10-slim-buster

WORKDIR /app

ADD . /app

RUN pip install --no-cache-dir -r requirements.txt

AWS_ACCESS_KEY_ID=AKIASANVIRYZNFV5H7VD
AWS_SECRET_ACCESS_KEY=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
REGION_NAME=eu-west-1


EXPOSE 80

CMD ["python", "api.py"]