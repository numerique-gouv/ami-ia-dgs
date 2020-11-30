FROM tensorflow/tensorflow:2.2.1-py3

RUN mkdir -p /dgs_backend
WORKDIR /dgs_backend

COPY requirements.txt ./
RUN pip3 install -r requirements.txt

COPY src/ .

EXPOSE 5000

CMD ["python3", "main_server.py"]
