import os
import sys
import click
from time import sleep

from neuroglancer_annotation_ui import AnnotationManager, stop_ngl_server, set_static_content_source
from annotationframeworkclient.annotationengine import AnnotationClient
from annotationframeworkclient.infoservice import InfoServiceClient
from neuroglancer_annotation_ui.synapse_getter import SynapseGetterFactory
from neuroglancer_annotation_ui.cell_type_getter import CellTypeGetterFactory


config_name = 'pinky100'
data_version = 58

# Here, I set key parameters via a .env, but 
from dotenv import load_dotenv
load_dotenv(dotenv_path='/Users/caseyschneider-mizell/.config/{}/.env'.format(config_name))
dataset_name = os.getenv('DATASET_NAME')
config = {
    'sqlalchemy_database_uri': os.getenv('MATERIALIZATION_DATABASE_URI'),
    'materialization_version': data_version,
    'annotation_endpoint': os.getenv('ANNOTATION_ENDPOINT'),
    'dataset_name': dataset_name,
}

cleft_src = os.getenv('CLEFT_SEGMENTATION')
infoclient = InfoServiceClient(server_address=os.getenv('INFO_URL'),
                               dataset_name=dataset_name)
img_src = infoclient.image_source(format_for='neuroglancer')
seg_src = infoclient.pychunkedgraph_viewer_source(format_for='neuroglancer')

cell_type_table = 'soma_valence'
synapse_table = os.getenv('AUTOMATED_SYNAPSE_TABLE')

# The 
set_static_content_source(os.getenv('DEFAULT_NEUROGLANCER_SOURCE'))

if __name__ == '__main__':

    manager = AnnotationManager()
    manager.add_image_layer(layer_name='img',image_source=img_src)
    manager.add_segmentation_layer(layer_name='seg', segmentation_source=seg_src, watched=True)
    # manager.add_segmentation_layer(layer_name='clefts', segmentation_source=cleft_src)
    manager.viewer.set_view_options()
    
    with manager.viewer.txn() as s:
        s.voxel_size = [4,4,40]

    SynapseGetterExtension = SynapseGetterFactory(synapse_table,
                                                  config)
    manager.add_extension(extension_name='syns',
                          ExtensionClass=SynapseGetterExtension)

    # CellTypeExtension = CellTypeGetterFactory(cell_type_table,
    #                                           config)
    # manager.add_extension(extension_name='ct',
    #                       ExtensionClass=CellTypeExtension)

    click.echo(manager.viewer.url)
    click.echo('\tNeuroglancer server running. Press q to quit')
    while True:
        c = click.getchar()
        if c == 'q':
            break
        else:
            pass
    
    stop_ngl_server()
