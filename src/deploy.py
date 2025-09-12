from modal import App

# Aggregate both Modal apps into a single deployable app.
# This file intentionally lives at src/deploy.py so you can:
#   modal deploy -m src.deploy

from .deployed_apps.extract_vector import app as extract_vector_app
from .deployed_apps.pythia_12b_modal_snapshot import app as pythia12b_snapshot_app

app = (
    App("activation-vector-project")
    .include(extract_vector_app)
    .include(pythia12b_snapshot_app)
)
