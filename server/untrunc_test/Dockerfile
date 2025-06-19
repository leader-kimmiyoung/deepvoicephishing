# pull base image
FROM ubuntu:bionic AS build
LABEL stage=intermediate
ARG FF_VER=shared

# install packaged dependencies
RUN apt-get update && [ "$FF_VER" = 'shared' ] && \
	apt-get -y install --no-install-recommends libavformat-dev libavcodec-dev libavutil-dev g++ make git || \
	apt-get -y install --no-install-recommends yasm wget g++ make git ca-certificates xz-utils && \
	rm -rf /var/lib/apt/lists/*

# copy code
ADD . /untrunc-src
WORKDIR /untrunc-src

# build untrunc
#RUN /usr/bin/g++ -o untrunc *.cpp -lavformat -lavcodec -lavutil
RUN /usr/bin/make FF_VER=$FF_VER && strip untrunc


# deliver clean image
FROM ubuntu:bionic
ARG FF_VER=shared

# Install dependencies
RUN apt-get update && [ "$FF_VER" = 'shared' ] && \
	apt-get -y install --no-install-recommends libavformat57 libavcodec57 libavutil55 python3-setuptools python3-pip ffmpeg ca-certificates curl && \
	rm -rf /var/lib/apt/lists/* || true

# Copy untrunc binary
COPY --from=build /untrunc-src/untrunc /bin/untrunc

# Set up working directory
WORKDIR /app

# Copy application files
COPY ./requirement.txt /app/requirement.txt
COPY ./app.py /app/app.py
#COPY ./websocket_client.py /app/websocket_client.py
COPY ./ok.m4a /app/ok.m4a

# Install Python dependencies
RUN curl https://bootstrap.pypa.io/pip/3.6/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    pip3 install --no-cache-dir -r requirement.txt

# Create necessary directories
RUN mkdir -p /data/WebSocket

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_DATA_PATH=/data/WebSocket
ENV FLASK_FILE_PERIOD=10000
ENV FLASK_FILE_DUPLICATE=1000
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Expose port
EXPOSE 8301

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8301", "--timeout", "300", "app:app"]
