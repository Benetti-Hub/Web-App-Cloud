FROM ubuntu

RUN apt-get update && apt-get install -y python3 python3-pip sudo

RUN useradd -m benetti

RUN chown -R benetti:benetti /home/benetti/

COPY --chown==benetti . /home/benetti/app

USER benetti

RUN pip3 install --upgrade pip

RUN cd /home/benetti/app && pip3 install -r requirements.txt

WORKDIR /home/benetti/app

EXPOSE 8080

ENTRYPOINT python3 app.py