import sys
import json
from collections import defaultdict
from neuroglancer_annotation_ui.base import AnnotationManager, stop_ngl_server, set_static_content_source
from annotationframeworkclient.infoservice import InfoServiceClient
from annotationframeworkclient.jsonservice import JSONService
from neuroglancer_annotation_ui.sphere_dropper_extension import SphereDropperFactory
from neuroglancer_annotation_ui.annotation import sphere_annotation

import click
import tqdm

info_url = 'https://www.dynamicannotationframework.com'
dataset = 'basil'

infoclient = InfoServiceClient(server_address=info_url,
                               dataset_name=dataset)
img_src = infoclient.image_source(format_for='neuroglancer')
seg_src = infoclient.flat_segmentation_source(format_for='neuroglancer')

set_static_content_source('https://nkem-multicut-dot-neuromancer-seung-import.appspot.com')

@click.command()
@click.option('-r', '--radius', required=False, default=10000)
@click.option('-f', '--filename', required=False, default='temp_point_file.json')
@click.option('-p', '--previous_point_state_id', required=False, default=None)
@click.option('-w', '--working_file', required=False, default=None)
def launch_ngl(radius, previous_point_state_id, filename, working_file):
    manager = AnnotationManager()
    manager.add_image_layer(layer_name='img', image_source=img_src)
    manager.add_segmentation_layer(layer_name='seg', segmentation_source=seg_src)

    if working_file is not None and filename is None:
        filename = working_file
    SphereDropperExtension = SphereDropperFactory(radius=radius,
                                                  layer_names=('new', 'existing'),
                                                  annotation_layer_colors=('#2dfee7', '#e7298a'),
                                                  filename=filename)
    manager.add_extension(extension_name='sphere_dropper',
                          ExtensionClass=SphereDropperExtension)

    if previous_point_state_id is not None:
        annotation_dict = build_spheres_from_state(manager.extensions['sphere_dropper'],
                               previous_point_state_id,
                               layer_name='existing')
        manager.viewer.add_annotation_one_shot(annotation_dict)

    if working_file is not None:
        with open(working_file, 'r') as f:
            points = json.load(f)
        sphere_dropper = manager.extensions['sphere_dropper']
        annotation_dict = defaultdict(list)
        for ln, anno_pts in points.items():
            if ln in manager.viewer.layer_names:
                for pt in anno_pts:
                    annotation_dict[ln].append(sphere_dropper.make_sphere_annotation(pt))
        manager.viewer.add_annotation_one_shot(annotation_dict)

    click.echo(manager.viewer.url)
    click.echo('\tNeuroglancer server running. Press q to quit')
    while True:
        c = click.getchar()
        if c == 'q':
            break
        else:
            pass
    stop_ngl_server()

def build_spheres_from_state(sphere_dropper, state_id, layer_name):
    json_client = JSONService()
    state = json_client.get_state_json(state_id)
    annotation_dict = {layer_name:[]}
    for ln, lc in state['layers'].items():
        if lc['type'] == 'annotation':
            for anno in lc['annotations']:
                annotation_dict[layer_name].append(sphere_dropper.make_sphere_annotation(anno['point']))
    return annotation_dict

if __name__ == '__main__':
    launch_ngl()

