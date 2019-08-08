import sys
import os
import sys
from pathlib import Path
base_dir = str(Path('..').resolve())

from neuroglancer_annotation_ui.base import AnnotationManager, stop_ngl_server
from neuroglancer_annotation_ui.state_list_browser_extension import StateListBrowserFactory
from annotationframeworkclient.infoservice import InfoServiceClient
from annotationframeworkclient.jsonservice import JSONService

import click

import neuroglancer
neuroglancer.set_static_content_source(url='https://nkem-multicut-dot-neuromancer-seung-import.appspot.com/')

dataset = 'pinky100'
infoclient = InfoServiceClient(dataset_name=dataset)
json_client = JSONService()

img_src = infoclient.image_source(format_for='neuroglancer')
seg_src = infoclient.pychunkedgraph_viewer_source(format_for='neuroglancer')

state_ids = [5753698579906560,
             5765899575361536,
             5642906408845312]

file_name = base_dir + '/chandelier_state_output.json'
StateListBrowser = StateListBrowserFactory(state_ids, json_client, file_name)

if __name__ == '__main__':
    manager = AnnotationManager()
    manager.add_extension('state_viewer',
                          StateListBrowser)
    click.echo(manager.viewer.url)
    click.echo('\tNeuroglancer server running. Press q to quit')

    while True:
        c = click.getchar()
        if c == 'q':
            break
        else:
            pass
    stop_ngl_server()