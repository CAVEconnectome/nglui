import requests
import neuroglancer
from flask import redirect, jsonify, Response, abort, Blueprint, current_app, render_template, url_for, request
import os
from .forms import build_extension_forms
from urllib.parse import urlparse
from neuroglancer_annotation_ui import get_extensions, get_extension_configs, extension_mapping, AnnotationManager
from annotationframeworkclient.infoservice import InfoServiceClient

mod = Blueprint('nglaunch', 'nglaunch', url_prefix='/annotationui')

__version__ = "0.0.23"

def setup_manager(info_client, anno_client=None, ngl_url = None):
    manager = AnnotationManager(annotation_client=anno_client)
    manager.add_layers(image_layers={'img':{'source': info_client.image_source(format_for='neuroglancer')}},
                       segmentation_layers={'seg':{'source':info_client.pychunkgraph_segmentation_source(format_for='neuroglancer')}},
                       resolution=[4,4,40])
    manager.watched_segmentation_layer = 'seg'
    if ngl_url is not None:
        manager.viewer.set_
    return manager


@mod.route('/version')
def version():
    return "Neuroglancer Annotation UI Server -- version {} \n {}".format(__version__, current_app.config)


@mod.route('/', methods=['GET', 'POST'])
def index():    
    info_url = current_app.config['INFOSERVICE_ENDPOINT']
    info_client = InfoServiceClient(server_address=info_url)
    datasets = info_client.get_datasets()
    extensions = get_extensions()
    extension_configs = get_extension_configs()

    form = build_extension_forms(datasets, extensions)

    if request.method == 'GET':
        return render_template('index.html',
                               form=form,
                               url=url_for('.index'),
                               info_url=info_url)

    if request.method == 'POST':
        if form.validate_on_submit():
            dataset = form.dataset.data
            neuroglancer.set_static_content_source(url=current_app.config['NEUROGLANCER_URL'])
            #ann_engine_url = current_app.config['ANNOTATION_ENGINE_URL']
            #client = AnnotationClient(endpoint=ann_engine_url, dataset_name=dataset)
            manager = setup_manager(InfoServiceClient(server_address=info_url, dataset_name=dataset),None)
            for f in form:
                if (f.label.text == 'extension') and (f.data is True):
                    ext_config = extension_configs[f.id]
                    ext_config['db_config'] = {'sqlalchemy_database_uri': current_app.config['MATERIALIZED_DB_URI'],
                                               'materialization_version': current_app.config['MATERIALIZED_DB_DATA_VERSION'],
                                               'dataset_name': dataset,
                                               'annotation_endpoint': current_app.config['ANNOTATION_ENGINE_URL']}
                    manager.add_extension(f.id, extension_mapping[f.id](**ext_config))
            url = manager.url
            o1 = urlparse(manager.url)
            o = urlparse(request.url)
            port_replace = current_app.config.get('NEUROGLANCER_FORWARD_PORT', o1.port)
            if port_replace is not None:
                port_string = ":{}".format(port_replace)
            else:
                port_string = ""

            new_url = o.scheme + "://" + o.netloc.split(':')[0] + port_string + o1.path
            return redirect(new_url)


@mod.route('/dataset/<dataset>')
def launch_dataset_viewer(dataset):
    info_url = current_app.config['ANNOTATION_INFO_SERVICE_URL']
    r = requests.get(info_url + "/api/dataset/{}".format(dataset))
    if r.status_code == 200:
        manager = setup_manager(r.json())
        return redirect(manager.url)
    else:
        abort(Response(r.text, status=404))


@mod.route('/viewers')
def get_viewers():
    if neuroglancer.server.global_server is not None:
        server_keys = {k: v.get_viewer_url()
                       for k, v in neuroglancer.server.global_server.viewers.items()}
        return jsonify(server_keys)
    else:
        return jsonify({})
