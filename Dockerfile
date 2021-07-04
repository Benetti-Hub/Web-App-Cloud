FROM ubuntu:18-04

RUN apt-get update && apt-get install -y python3 python3-pip sudo

RUN useradd -m benetti

RUN chown -R benetti:benetti /home/benett/

COPY --chown==benetti . /home/benetti/app

USER benetti

RUN cv /home/benetti/app && pip3 install -r requirements.txt

WORKDIR /home/benetti/app