FROM python:3.8

WORKDIR /env

COPY ./ ./
RUN pip install -e .

CMD ["python", "tracksegnet-main.py", "parms.csv"]
