from flask import Flask
# from flask_migrate import Migrate
from .config import configure_app
from .utils import get_instance_folder_path
from .nglaunch import mod

__version__ = '0.0.11'


def create_app(test_config=None):
    # Define the Flask Object
    app = Flask(__name__,
                instance_path=get_instance_folder_path(),
                instance_relative_config=True,
                static_folder="../static")
    # load configuration (from test_config if passed)
    if test_config is None:
        app = configure_app(app)
    else:
        app.config.update(test_config)
    # register blueprints
    app.register_blueprint(mod)

    return app
