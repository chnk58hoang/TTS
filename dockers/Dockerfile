# Dockerfile
ARG DOCKER_BASE_IMAGE="nvidia/cuda:12.4.0-runtime-ubuntu22.04"
FROM $DOCKER_BASE_IMAGE

# host user
ARG USER=asr
# host user id
ARG UID=1011
# ARG UID=1028
# host group id
ARG GID=999
# password for sudo in docker
ARG PW=1
RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | chpasswd
USER ${UID}:${GID}
