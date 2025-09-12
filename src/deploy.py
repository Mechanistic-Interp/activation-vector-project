from modal import App

# from .training_data.get_training_data import app as get_training_data_app
from .extract_vector import app as extract_vector_app

app = (
    App("activation-vector-project")
    # .include(get_training_data_app)
    .include(extract_vector_app)
)
