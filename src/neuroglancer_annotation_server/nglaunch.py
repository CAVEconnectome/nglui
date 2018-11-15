import requests
import neuroglancer
from flask import redirect, jsonify, Response, abort, Blueprint, current_app, render_template, url_for, request
import os
from .forms import NgDataSetExtensionForm
from urllib.parse import urlparse
from neuroglancer_annotation_ui import get_extensions, extension_mapping, AnnotationManager
from annotationframeworkclient.infoservice import InfoServiceClient

mod = Blueprint('nglaunch', 'nglaunch')


__version__ = "0.0.3"
def setup_manager(info_client, anno_client=None):
    manager = AnnotationManager(annotation_client=anno_client)
    manager.add_layers(image_layers={'img':{'source': info_client.image_source(format_for='neuroglancer')}},
                       segmentation_layers={'seg':{'source':info_client.pychunkgraph_segmentation_source(format_for='neuroglancer')}})
    manager.watched_segmentation_layer = 'seg'
    return manager


@mod.route('/', methods=['GET', 'POST'])
def index():    
    info_url = current_app.config['INFOSERVICE_ENDPOINT']
    info_client = InfoServiceClient(server_address=info_url)
    datasets = info_client.get_datasets()

    extensions = get_extensions()

    form = NgDataSetExtensionForm()
    form.dataset.choices = [(d, d) for d in datasets]

    for ext in extensions:
        form.extensions.append_entry()
    for ext, entry in zip(extensions, form.extensions.entries):
        entry.label.text = ext
        entry.id = ext

    if request.method == 'GET':
        return render_template('index.html',
                               form=form,
                               url=url_for('.index'),
                               info_url=info_url)

    if request.method == 'POST':
        if form.validate_on_submit():
            dataset = form.dataset.data
            # ann_engine_url = current_app.config['ANNOTATION_ENGINE_URL']
            #client = AnnotationClient(endpoint=ann_engine_url, dataset_name=dataset)
            manager = setup_manager(InfoServiceClient(server_address=info_url, dataset_name=dataset),
                                    None)
            for extension in form.extensions:
                if extension.data:
                    manager.add_extension(extension.id, extension_mapping[extension.id])
            url = manager.url
            o1 = urlparse(manager.url)
            o = urlparse(request.url)
            new_url = o.scheme + "://" + o.netloc.split(':')[0] + ":{}".format(o1.port) + o1.path
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
