import requests

ngl_info_endpoint = "{neuroglancer_endpoint}/version.json"


def get_ngl_info(ngl_url):
    """
    Get the version of neuroglancer running at a given endpoint.
    """
    try:
        r = requests.get(ngl_info_endpoint.format(neuroglancer_endpoint=ngl_url))
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error getting neuroglancer version: {e}")
        return None
