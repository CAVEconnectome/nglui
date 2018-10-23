FROM tiangolo/uwsgi-nginx-flask:python3.6

RUN mkdir -p /home/nginx/.cloudvolume/secrets && chown -R nginx /home/nginx && usermod -d /home/nginx -s /bin/bash nginx
COPY requirements.txt /app/.
RUN pip install numpy && \
    pip install -r requirements.txt
COPY . /app
ENV FLASK_APP=run.py
RUN python setup.py install
CMD ["flask","run","--port","9898","--host","0.0.0.0"]