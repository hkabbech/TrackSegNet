# Installation and requirements

## Clone the repository

```bash
git clone https://github.com/hkabbech/TrackSegNet.git
cd TrackSegNet
```

## Either create and run a docker container

```bash
# Build a docker image (Rebuild the image after changing the parameters):
docker compose build
# Run the container:
docker compose run tracksegnet-env
```

## Or create a virtual environment and install the packages

Requirement: python3.8 or equivalent and the virtualenv library

```bash
# Create the environment:
python -m venv tracksegnet-env
# virtualenv -p /usr/bin/python3 tracksegnet-env
# Activate the environment:
source ./tracksegnet-env/bin/activate # for Windows PowerShell: .\tracksegnet-env\Scripts\Activate.ps1 (run as administrator)
# Install the required python libraries:
python -m pip install -r requirements.txt
```

Note for later, to deactivate the virtual environment, type `deactivate`.