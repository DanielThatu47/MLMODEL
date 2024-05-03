FROM python:3.8

ENV PYTHONUNBUFFERED 1
RUN mkdir /aimodel
WORKDIR /aimodel
COPY requirements.txt /aimodel/
RUN pip install --upgrade pip wheel
RUN pip install -r requirements.txt
COPY . /aimodel/

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"
