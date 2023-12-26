FROM wolframresearch/wolframengine:13.2

USER root
RUN apt-get update \
  && apt-get install -y build-essential python3-dev
USER wolframengine

COPY --chown=wolframengine . /backend

RUN mkdir /home/wolframengine/.WolframEngine/Licensing/
RUN (cd /backend; pip install -r requirements.txt)
RUN (cd /backend; pip install -r requirements2.txt)
RUN (cd /backend; python3 manage.py migrate)

CMD (cd /backend; python3 manage.py runserver 0.0.0.0:8000)
