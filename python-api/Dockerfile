
FROM python:3.9


WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./api.py /code/api.py


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["fastapi", "run", "/code/api.py"]