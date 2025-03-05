FROM emmanuelgonzalez/sos:latest

WORKDIR /opt
COPY . /opt

USER root

# Set the entrypoint
ENTRYPOINT [ "/usr/bin/python3", "/opt/main.py" ]
# ENTRYPOINT [ "/opt/miniconda/envs/sos/bin/python", "/opt/main.py" ]