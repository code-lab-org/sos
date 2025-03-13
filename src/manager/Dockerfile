FROM emmanuelgonzalez/sos:latest

# Set working directory
WORKDIR /opt
COPY . /opt

# Set environment variables
USER root

# Set the entrypoint
ENTRYPOINT [ "python3", "/opt/main.py" ]