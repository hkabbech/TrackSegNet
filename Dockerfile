FROM python:3.8

WORKDIR /env

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY ./ ./
# COPY data/ ./
# COPY src/ ./
# COPY tracksegnet.py ./

CMD ["python", "tracksegnet.py", "parms.csv"]
