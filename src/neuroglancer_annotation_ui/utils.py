import neuroglancer

default_static_content_source='https://neuromancer-seung-import.appspot.com/'

def set_static_content_source(source=default_static_content_source):
    neuroglancer.set_static_content_source(url=source)

def stop_ngl_server():
    """
    Shuts down the neuroglancer tornado server
    """
    neuroglancer.server.stop()
