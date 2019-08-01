import sys
from neuroglancer_annotation_ui import extension_mapping
from neuroglancer_annotation_ui.base import AnnotationManager, stop_ngl_server
from annotationframeworkclient.annotationengine import AnnotationClient
from annotationframeworkclient.infoservice import InfoServiceClient
from neuroglancer_annotation_ui.synapse_getter import SynapseGetterFactory
from neuroglancer_annotation_ui.cell_type_getter import CellTypeGetterFactory

import click
from time import sleep

from secret_config import config

info_url = 'https://www.dynamicannotationframework.com'
dataset = 'pinky100'
cleft_src = config['cleft_src']
infoclient = InfoServiceClient(server_address=info_url,
                               dataset_name=dataset)
img_src = infoclient.image_source(format_for='neuroglancer')
seg_src = infoclient.pychunkgraph_viewer_source(format_for='neuroglancer')

cell_type_table = 'soma_valence'
synapse_table = 'pni_synapses_i3'

if __name__ == '__main__':

    manager = AnnotationManager()
    manager.add_image_layer(layer_name='img',image_source=img_src)
    manager.add_segmentation_layer(layer_name='seg',segmentation_source=seg_src, watched=True)
    manager.add_segmentation_layer(layer_name='clefts', segmentation_source=cleft_src)
    manager.viewer.set_view_options()
    
    with manager.viewer.txn() as s:
        s.voxel_size = [4,4,40]
 
    SynapseGetterExtension = SynapseGetterFactory(synapse_table,
                                                  config)
    manager.add_extension(extension_name='syns',
                          ExtensionClass=SynapseGetterExtension)

    CellTypeExtension = CellTypeGetterFactory(cell_type_table,
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
