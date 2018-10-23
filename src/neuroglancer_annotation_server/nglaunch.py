import neuroglancer
from flask import redirect, jsonify, Response, abort, Blueprint, current_app, render_template, url_for, request
import requests
from neuroglancer_annotation_ui.base import AnnotationManager
from annotationengine.annotationclient import AnnotationClient
import os
from .forms import NgDataSetExtensionForm
from neuroglancer_annotation_ui import get_extensions, extension_mapping
mod = Blueprint('nglaunch', 'nglaunch')


__version__ = "0.0.1"
def setup_manager(d, client=None):
    manager = AnnotationManager(annotation_client=client)
    manager.add_image_layer('img', d['image_source'])
    manager.add_segmentation_layer('seg',
                                   d['flat_segmentation_source'])
    return manager


@mod.route('/', methods=['GET', 'POST'])
def index():
    info_url = current_app.config['ANNOTATION_INFO_SERVICE_URL']
    r = requests.get(os.path.join(info_url, "api/datasets"))
    if r.status_code != 200:
        abort(Response(r.text, status=500))

    datasets = r.json()
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
            r = requests.get(info_url + "/api/dataset/{}".format(dataset))
            ann_engine_url = current_app.config['ANNOTATION_ENGINE_URL']
            #client = AnnotationClient(endpoint=ann_engine_url, dataset_name=dataset)
            manager = setup_manager(r.json())
            for extension in form.extensions:
                if extension.data:
                    manager.add_extension(extension.id, extension_mapping[extension.id])
            return redirect(manager.url)


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
