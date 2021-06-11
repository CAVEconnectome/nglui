from .. import nglite as neuroglancer
from . import annotation, utils
from numpy import issubdtype, integer, uint64, vstack
from collections import OrderedDict
import copy
import re

SEGMENTATION_LAYER_TYPES = ["segmentation", "segmentation_with_graph"]


class EasyViewer(neuroglancer.Viewer):
    """
    Extends the neuroglancer Viewer object to make simple operations simple.
    """

    def __init__(self):
        super(EasyViewer, self).__init__()

    def __repr__(self):
        return self.as_url()

    def _repr_html_(self):
        return '<a href="%s" target="_blank">Viewer</a>' % self.as_url()

    def load_url(self, url):
        """Load neuroglancer compatible url and updates viewer state

        Attributes:
        url (str): neuroglancer url

        """
        state = neuroglancer.parse_url(url)
        self.set_state(state)

    @staticmethod
    def _smart_add_segmentation_layer(s, layer_name, source, **kwargs):
        if re.search(r"^graphene://", source) is not None:
            s.layers[layer_name] = neuroglancer.ChunkedgraphSegmentationLayer(
                source=source, **kwargs
            )
        elif re.search(r"^precomputed://", source) is not None:
            s.layers[layer_name] = neuroglancer.SegmentationLayer(
                source=source, **kwargs
            )

    def add_layers(
        self,
        image_layers={},
        segmentation_layers={},
        annotation_layers={},
        resolution=None,
    ):
        with self.txn() as s:
            for ln, kws in image_layers.items():
                s.layers[ln] = neuroglancer.ImageLayer(**kws)
            for ln, kws in segmentation_layers.items():
                self._smart_add_segmentation_layer(s, **kws)
            for ln, kws in annotation_layers.items():
                s.layers[ln] = neuroglancer.AnnotationLayer(**kws)
            if resolution is not None:
                s.voxel_size = resolution
        pass

    def add_segmentation_layer(self, layer_name, source, **kwargs):
        """Add segmentation layer to viewer instance.

        Attributes:
            layer_name (str): name of layer to be displayed in neuroglancer ui.
            source (str): source of neuroglancer segment layer
        """
        with self.txn() as s:
            self._smart_add_segmentation_layer(
                s, layer_name=layer_name, source=source, **kwargs
            )

    def add_image_layer(self, layer_name, source, **kwargs):
        """Add segmentation layer to viewer instance.

        Attributes:
            layer_name (str): name of layer to be displayed in neuroglancer ui.
            source (str): source of neuroglancer image layer
        """
        with self.txn() as s:
            s.layers[layer_name] = neuroglancer.ImageLayer(source=source, **kwargs)

    def set_resolution(self, resolution):
        with self.txn() as s:
            s.voxel_size = resolution

    def add_contrast_shader(self, layer_name, black=0.0, white=1.0):
        shader_text = f"#uicontrol float black slider(min=0, max=1, default={black})\n#uicontrol float white slider(min=0, max=1, default={white})\nfloat rescale(float value) {{\n  return (value - black) / (white - black);\n}}\nvoid main() {{\n  float val = toNormalized(getDataValue());\n  if (val < black) {{\n    emitRGB(vec3(0,0,0));\n  }} else if (val > white) {{\n    emitRGB(vec3(1.0, 1.0, 1.0));\n  }} else {{\n    emitGrayscale(rescale(val));\n  }}\n}}\n"
        self._update_layer_shader(layer_name, shader_text)

    def _update_layer_shader(self, layer_name, shader_text):
        with self.txn() as s:
            s.layers[layer_name]._json_data["shader"] = shader_text

    def set_state_server(self, state_server):
        with self.txn() as s:
            s._json_data["jsonStateServer"] = state_server

    def add_annotation_layer(
        self,
        layer_name=None,
        color=None,
        linked_segmentation_layer=None,
        filter_by_segmentation=False,
        brackets_show_segmentation=True,
        selection_shows_segmentation=True,
        tags=None,
    ):
        """Add annotation layer to the viewer instance.

        Attributes:
            layer_name (str): name of layer to be created
        """
        if layer_name is None:
            layer_name = "annos"
        if layer_name in [l.name for l in self.state.layers]:
            return

        if linked_segmentation_layer is None:
            filter_by_segmentation = None

        with self.txn() as s:
            new_layer = neuroglancer.AnnotationLayer(
                linked_segmentation_layer=linked_segmentation_layer,
                filter_by_segmentation=filter_by_segmentation,
                brackets_show_segmentation=brackets_show_segmentation,
                selection_shows_segmentation=selection_shows_segmentation,
            )
            s.layers.append(name=layer_name, layer=new_layer)
            if color is not None:
                s.layers[layer_name].annotationColor = color
        if tags is not None:
            self.add_annotation_tags(layer_name=layer_name, tags=tags)

    def set_annotation_layer_color(self, layer_name, color):
        """Set the color for the annotation layer"""
        if layer_name in [l.name for l in self.state.layers]:
            with self.txn() as s:
                s.layers[layer_name].annotationColor = color
        else:
            pass

    def clear_annotation_layers(self, layer_names):
        with self.txn() as s:
            for ln in layer_names:
                s.layers[ln].annotations._data = []

    def set_annotation_one_shot(self, ln_anno_dict):
        """
        ln_anno_dict is a layer_name to annotation list dict.
        """
        with self.txn() as s:
            for ln, annos in ln_anno_dict.items():
                s.layers[ln].annotations._data = annos

    def add_annotations(self, layer_name, annotations):
        """Add annotations to a viewer instance, the type is specified.
           If layer name does not exist, add the layer

        Attributes:
            layer_name (str): name of layer to be displayed in neuroglancer ui.
            layer_type (str): can be: 'points, ellipse or line' only
        """
        with self.txn() as s:
            for anno in annotations:
                s.layers[layer_name].annotations.append(anno)

    def remove_annotations(self, layer_name, anno_ids):
        if isinstance(anno_ids, str):
            anno_ids = [anno_ids]
        try:
            with self.txn() as s:
                el = len(s.layers[layer_name].annotations)
                for anno in reversed(s.layers[layer_name].annotations):
                    el -= 1
                    if anno.id in anno_ids:
                        anno_ids.remove(anno.id)
                        s.layers[layer_name].annotations.pop(el)
                        if len(anno_ids) == 0:
                            break
        except:
            self.update_message("Could not remove annotation")

    def add_annotation_tags(self, layer_name, tags):
        """
        Add a list of tags to an annotation layer
        """
        if layer_name not in self.layer_names:
            raise ValueError("Layer is not an annotation layer")
        tag_list = [
            OrderedDict({"id": tag_id + 1, "label": label})
            for tag_id, label in enumerate(tags)
        ]
        with self.txn() as s:
            s.layers[layer_name]._json_data["annotationTags"] = tag_list

    def update_description(self, layer_id_dict, new_description):
        layer_id_dict = copy.deepcopy(layer_id_dict)
        with self.txn() as s:
            try:
                for layer_name, id_list in layer_id_dict.items():
                    for anno in s.layers[layer_name].annotations:
                        if anno.id in id_list:
                            if anno.description is None:
                                anno.description = new_description
                            else:
                                anno.description = "{}\n{}".format(
                                    anno.description, new_description
                                )
                            id_list.remove(anno.id)
                            if len(id_list) == 0:
                                break
            except Exception as e:
                print(e)
                self.update_message("Could not update descriptions!")

    @property
    def url(self):
        return self.get_viewer_url()

    def as_url(self, prefix=None, as_html=False, link_text="Neuroglancer link"):
        if prefix is None:
            prefix = utils.default_neuroglancer_base
        ngl_url = neuroglancer.to_url(self.state, prefix=prefix)
        if as_html:
            return '<a href="{}" target="_blank">{}</a>'.format(ngl_url, link_text)
        else:
            return ngl_url

    def update_message(self, message):
        with self.config_state.txn() as s:
            if message is not None:
                s.status_messages["status"] = message

    def set_selected_layer(self, layer_name, tool=None):
        if layer_name in self.layer_names:
            with self.txn() as s:
                s._json_data["selectedLayer"] = OrderedDict(
                    layer=layer_name, visible=True
                )
                if tool is not None:
                    s.layers[layer_name]._json_data["tool"] = tool

    def get_selected_layer(self):
        state_json = self.state.to_json()
        try:
            selected_layer = state_json["selectedLayer"]["layer"]
        except:
            selected_layer = None
        return selected_layer

    def get_annotation(self, layer_name, aid):
        if self.state.layers[layer_name].type == "annotation":
            for anno in self.state.layers[layer_name].annotations:
                if anno.id == aid:
                    return anno
            else:
                return None
        else:
            return None

    def get_selected_annotation_id(self):
        layer_name = self.get_selected_layer()
        try:
            aid_data = self.state.layers[layer_name]._json_data["selectedAnnotation"]
            if isinstance(aid_data, OrderedDict):
                aid = aid_data["id"]
            else:
                aid = aid
        except:
            aid = None
        return aid

    def select_annotation(self, layer_name, aid):
        if layer_name in self.layer_names:
            with self.txn() as s:
                s.layers[layer_name]._json_data["selectedAnnotation"] = aid
        self.set_selected_layer(layer_name)

    @property
    def layer_names(self):
        return [l.name for l in self.state.layers]

    def selected_objects(self, segmentation_layer):
        return list(self.state.layers[segmentation_layer].segments)

    def add_selected_objects(self, segmentation_layer, oids, colors=None):
        if issubdtype(type(oids), integer):
            oids = [oids]

        with self.txn() as s:
            for oid in oids:
                s.layers[segmentation_layer].segments.add(uint64(oid))

        if colors is not None:
            if isinstance(colors, dict):
                self.assign_colors(segmentation_layer, colors)
            elif len(colors) == len(oids):
                seg_colors = {str(oid): clr for oid, clr in zip(oids, colors)}
                self.assign_colors(segmentation_layer, seg_colors)

    def get_mouse_coordinates(self, s):
        pos = s.mouse_voxel_coordinates
        return pos

    def set_view_options(
        self,
        show_slices=None,
        layout=None,
        show_axis_lines=None,
        show_scale_bar=None,
        orthographic=None,
        position=None,
        zoom_image=None,
        zoom_3d=None,
    ):
        with self.txn() as s:
            if show_slices is not None:
                s.showSlices = show_slices
            if layout is not None:
                s.layout.type = layout
            if show_axis_lines is not None:
                s.show_axis_lines = show_axis_lines
            if show_scale_bar is not None:
                s.show_scale_bar = show_scale_bar
            if orthographic is not None:
                s.layout.orthographic_projection = orthographic
            if position is not None:
                s.position.voxelCoordinates = position
            if zoom_image is not None:
                s.navigation.zoomFactor = zoom_image
            if zoom_3d is not None:
                s.perspectiveZoom = zoom_3d

    def set_segmentation_view_options(
        self,
        layer_name,
        alpha_selected=None,
        alpha_3d=None,
        alpha_unselected=None,
    ):
        if self.state.layers[layer_name].type not in SEGMENTATION_LAYER_TYPES:
            return
        with self.txn() as s:
            l = s.layers[layer_name]
            if alpha_selected is not None:
                l.selectedAlpha = alpha_selected
            if alpha_3d is not None:
                l.objectAlpha = alpha_3d
            if alpha_unselected is not None:
                l.notSelectedAlpha = alpha_unselected

    def set_timestamp(
        self,
        layer_name,
        timestamp=None,
    ):
        """Set timestamp of a segmentation layer

        Parameters
        ----------
        layer_name : str
            Name of a segmentation layer
        timestamp : float, optional
            Timestamp in unix epoch time (e.g. `time.time.now()` in python), by default None
        """
        if self.state.layers[layer_name].type != "segmentation_with_graph":
            return
        with self.txn() as s:
            l = s.layers[layer_name]
            if timestamp is not None:
                l.timestamp = int(timestamp)
            else:
                l.timestamp = None

    def assign_colors(self, layer_name, seg_colors):
        """Assign colors to root ids in a segmentation layer

        Parameters
        ----------
        layer_name : str,
            Segmentation layer name
        seg_colors : dict
            dict with root ids as keys and colors as values.
        """
        with self.txn() as s:
            if seg_colors is not None:
                seg_colors = {
                    str(oid): k for oid, k in seg_colors.items() if k is not None
                }
                s.layers[layer_name]._json_data["segmentColors"] = seg_colors

    def set_multicut_points(
        self,
        layer_name,
        seg_id,
        points_red,
        points_blue,
        supervoxels_red=None,
        supervoxels_blue=None,
        focus=True,
    ):
        """Configures multicut points in the neuroglancer state. Note that points need to be in mesh units (e.g. nanometers), not voxels!

        Parameters
        ----------
        layer_name : str
            Segmentation layer name
        seg_id : np.uint64
            Segmentation id of the object in question
        points_red : np.array
            Nx3 array of locations in voxel space for side 1 of the cut.
        points_blue : np.array
            Mx3 array of locations in voxel space for side 2 of the cut.
        supervoxels_red : np.array or None, optional
            N-length array of supervoxel ids associated with locations in points_red or None. If None, supervoxels lookup occurs based on the mesh. By default None
        supervoxels_blue : np.array or None, optional
            M-length array of supervoxel ids associated with locations in points_blue or None. If None, supervoxels lookup occurs based on the mesh. By default None
        focus : bool, optional
            If True, makes the layer and graph tool focused. By default True
        """

        def _multicut_annotation(pt, oid, sv_id):
            if sv_id is None:
                sv_id = oid
            return annotation.point_annotation(
                pt, description=str(sv_id), linked_segmentation=[sv_id, oid]
            )

        if supervoxels_red is None:
            supervoxels_red = [None for x in points_red]
        if supervoxels_blue is None:
            supervoxels_blue = [None for x in points_blue]

        annos_red = neuroglancer.annotationHolder()
        for pt, sv_id in zip(points_red, supervoxels_red):
            annos_red.annotations.append(
                _multicut_annotation(pt * self.state.voxel_size, seg_id, sv_id)
            )

        annos_blue = neuroglancer.annotationHolder()
        for pt, sv_id in zip(points_blue, supervoxels_blue):
            annos_blue.annotations.append(
                _multicut_annotation(pt * self.state.voxel_size, seg_id, sv_id)
            )

        self.add_selected_objects(layer_name, [seg_id])

        with self.txn() as s:
            l = s.layers[layer_name]
            l.tab = "graph"
            l.graphOperationMarker.append(annos_red)
            l.graphOperationMarker.append(annos_blue)

        if focus:
            self.set_selected_layer(layer_name)
            ctr_pt = vstack([points_red, points_blue]).mean(axis=0)
            self.set_view_options(position=ctr_pt, zoom_3d=100)
