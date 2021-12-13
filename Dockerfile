FROM pytorch/pytorch:latest

RUN apt-get update

WORKDIR /home
ADD ./docker/requirements.txt .

RUN pip install -r requirements.txt


ADD . .

