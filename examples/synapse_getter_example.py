import sys
from neuroglancer_annotation_ui import extension_mapping
from neuroglancer_annotation_ui.base import AnnotationManager, stop_ngl_server
from annotationframeworkclient.annotationengine import AnnotationClient
from annotationframeworkclient.infoservice import InfoServiceClient
from synapse_getter_extension import SynapseGetterFactory
from cell_type_getter import cell_type_extension_factory

import click
from time import sleep

from secret_config import config

info_url = 'https://www.dynamicannotationframework.com'
dataset = 'pinky100'

infoclient = InfoServiceClient(server_address=info_url,
                               dataset_name=dataset)
img_src = infoclient.image_source(format_for='neuroglancer')
#seg_src = infoclient.pychunkgraph_segmentation_source(format_for='neuroglancer')
seg_src = config['correct_seg_src']

cell_type_table = 'soma_valence'
cell_type_schema = 'cell_type_local'

if __name__ == '__main__':

    manager = AnnotationManager()
    manager.add_image_layer(layer_name='img',image_source=img_src)
    sleep(0.25)
    
    manager.add_segmentation_layer(layer_name='seg',segmentation_source=seg_src, watched=True)
    manager.viewer.set_view_options()
    
    SynapseGetterExtension = SynapseGetterFactory(config)
    manager.add_extension(extension_name='syns',
                          ExtensionClass=SynapseGetterExtension)

    CellTypeExtension = cell_type_extension_factory(cell_type_table,
                                                    cell_type_schema,
                                                    config)
    manager.add_extension(extension_name='ct',
                          ExtensionClass=CellTypeExtension)

    click.echo(manager.viewer.url)
    click.echo('\tNeuroglancer server running. Press q to quit')
    while True:
        c = click.getchar()
        if c == 'q':
            break
        else:
            pass
    
    stop_ngl_server()
