ARG BASE_IMAGE_NAME=registry.daturum.ru/sberbank/agiki/data-models/base_images/base
ARG BASE_IMAGE_VERSION=latest
FROM ${BASE_IMAGE_NAME}:${BASE_IMAGE_VERSION}

COPY --chown=${APP_USER}:${APP_USER} . /app
RUN chmod +x -R /app/bin/*
WORKDIR /app

ENV MODEL_DATA=/app/data
ENV WEB_SERVER_PORT=9292
ENV WEB_SERVER_HOST=0.0.0.0


RUN cd /app/
RUN bundle install --path vendor/bundle
RUN pip3 install -r python/requirements.txt
RUN rm -rf ~/.cache/pip

CMD ["/app/bin/server"]
