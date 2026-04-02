import os

# VARIABLES
MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")

# CONSTANTS
ROOT_PATH = os.path.join(os.path.expanduser("~"), "Martyna", "dreamscope")
LOCAL_DATA_PATH = os.path.join(os.path.expanduser("~"), "Martyna", "dreamscope", "data")
