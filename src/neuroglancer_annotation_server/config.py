# Define the application directory
import os
import neuroglancer


class BaseConfig(object):
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    # Statement for enabling the development environment
    DEBUG = True

    NEUROGLANCER_URL = "https://neuroglancer-demo.appspot.com"
    NEUROGLANCER_PORT = 9000
    ANNOTATION_INFO_SERVICE_URL = 'https://www.dynamicannotationframework.com/info'
    ANNOTATION_ENGINE_URL = 'http://35.185.22.247'
    # Enable protection agains *Cross-site Request Forgery (CSRF)*
    CSRF_ENABLED = True

    # Use a secure, unique and absolutely secret key for
    # signing the data.
    CSRF_SESSION_KEY = "SECRETSESSION"

    # Secret key for signing cookies
    SECRET_KEY = b'SECRETKEY'


config = {
    "development": "neuroglancer_annotation_server.config.BaseConfig",
    "testing": "neuroglancer_annotation_server.config.BaseConfig",
    "default": "neuroglancer_annotation_server.config.BaseConfig"
}


def configure_app(app):
    config_name = os.getenv('FLASK_CONFIGURATION', 'default')
    # object-based default configuration
    app.config.from_object(config[config_name])
    if os.environ.get('NGANNOTATIONSERVER_SETTINGS', None) is not None:
        app.config.from_envvar('NGANNOTATIONSERVER_SETTINGS')
    else:
        # instance-folders configuration
        app.config.from_pyfile('config.cfg', silent=True)
    neuroglancer.set_server_bind_address('0.0.0.0', bind_port=app.config['NEUROGLANCER_PORT'])

    return app
