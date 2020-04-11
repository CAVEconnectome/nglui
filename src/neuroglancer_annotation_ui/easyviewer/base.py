import neuroglancer_annotation_ui.nglite as neuroglancer
from . import annotation, utils
from numpy import issubdtype, integer, uint64
from collections import OrderedDict
import copy
import re

neuroglancer.viewer_state.layer_types['segmentation_with_graph'] = utils.ChunkedgraphSegmentationLayer


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

    def set_source_url(self, ngl_url):
        self.ngl_url = neuroglancer.set_server_bind_address(ngl_url)

    def load_url(self, url):
        """Load neuroglancer compatible url and updates viewer state

        Attributes:
        url (str): neuroglancer url

        """
        state = neuroglancer.parse_url(url)
        self.set_state(state)

    @staticmethod
    def _smart_add_segmentation_layer(s, layer_name, source, **kwargs):
        if re.search('^graphene:\/\/', source) is not None:
            s.layers[layer_name] = utils.ChunkedgraphSegmentationLayer(
                source=source, **kwargs)
        elif re.search('^precomputed:\/\/', source) is not None:
            s.layers[layer_name] = neuroglancer.SegmentationLayer(
                source=source, **kwargs)

    def add_layers(self, image_layers={}, segmentation_layers={}, annotation_layers={}, resolution=None):
        with self.txn() as s:
            for ln, kws in image_layers.items():
                s.layers[ln] = neuroglancer.ImageLayer(**kws)
            for ln, kws in segmentation_layers.items():
                self._smart_add_segmentation_layer(s, **kws)
            for ln, kws in annotation_layers.items():
                s.layers[ln] = neuroglancer.AnnotationLayer(**kws)
            if resolution is not None:
                s.voxel_size = resolutionP
        pass

    def add_segmentation_layer(self, layer_name, source, **kwargs):
        """Add segmentation layer to viewer instance.

        Attributes:
            layer_name (str): name of layer to be displayed in neuroglancer ui.
            source (str): source of neuroglancer segment layer
        """
        with self.txn() as s:
            self._smart_add_segmentation_layer(s,
                                               layer_name=layer_name,
                                               source=source,
                                               **kwargs)

    def add_image_layer(self, layer_name, source, **kwargs):
        """Add segmentation layer to viewer instance.

        Attributes:
            layer_name (str): name of layer to be displayed in neuroglancer ui.
            source (str): source of neuroglancer image layer
        """
        with self.txn() as s:
            s.layers[layer_name] = neuroglancer.ImageLayer(
                source=source, **kwargs)

    def set_resolution(self, resolution):
        with self.txn() as s:
            s.voxel_size = resolution

    def add_annotation_layer(self, layer_name=None, color=None,
                             linked_segmentation_layer=None, filter_by_segmentation=True,
                             tags=None):
        """Add annotation layer to the viewer instance.

        Attributes:
            layer_name (str): name of layer to be created
        """
        if layer_name is None:
            layer_name = 'annos'
        if layer_name in [l.name for l in self.state.layers]:
            return

        if linked_segmentation_layer is None:
            filter_by_segmentation = None

        with self.txn() as s:
            new_layer = neuroglancer.AnnotationLayer(linked_segmentation_layer=linked_segmentation_layer,
                                                     filter_by_segmentation=filter_by_segmentation)
            s.layers.append(name=layer_name,
                            layer=new_layer)
            if color is not None:
                s.layers[layer_name].annotationColor = color
        if tags is not None:
            self.add_annotation_tags(layer_name=layer_name, tags=tags)

    def set_annotation_layer_color(self, layer_name, color):
        """Set the color for the annotation layer

        """
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
        '''
        ln_anno_dict is a layer_name to annotation list dict.
        '''
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

    def add_annotation(self, layer_name, annotation, color=None):
        raise DeprecationWarning(
            'This function is depreciated. Use ''add_annotation'' instead.')
        self.add_annotations(layer_name, annotation, color=color)

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
        except Exception as e:
            self.update_message('Could not remove annotation')

    def remove_annotation(self, layer_name, aids):
        raise DeprecationWarning(
            'This function is depreciated. Use ''remove_annotations'' instead.')
        self.remove_annotations(self, layer_name, aids)

    def add_annotation_tags(self, layer_name, tags):
        '''
        Add a list of tags to an annotation layer
        '''
        if layer_name not in self.layer_names:
            raise ValueError('Layer is not an annotation layer')
        tag_list = [OrderedDict({'id': tag_id+1, 'label': label})
                    for tag_id, label in enumerate(tags)]
        with self.txn() as s:
            s.layers[layer_name]._json_data['annotationTags'] = tag_list

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
                                    anno.description, new_description)
                            id_list.remove(anno.id)
                            if len(id_list) == 0:
                                break
            except Exception as e:
                print(e)
                self.update_message('Could not update descriptions!')

    @property
    def url(self):
        return self.get_viewer_url()

    def as_url(self, prefix=None, as_html=False, link_text='Neuroglancer link'):
        if prefix is None:
            prefix = utils.default_static_content_source
        ngl_url = neuroglancer.to_url(self.state, prefix=prefix)
        if as_html:
            return '<a href="{}" target="_blank">{}</a>'.format(ngl_url, link_text)
        else:
            return ngl_url

    def update_message(self, message):
        with self.config_state.txn() as s:
            if message is not None:
                s.status_messages['status'] = message

    def set_selected_layer(self, layer_name, tool=None):
        if layer_name in self.layer_names:
            with self.txn() as s:
                s._json_data['selectedLayer'] = OrderedDict(
                    layer=layer_name, visible=True)
                if tool is not None:
                    s.layers[layer_name]._json_data['tool'] = tool

    def get_selected_layer(self):
        state_json = self.state.to_json()
        try:
            selected_layer = state_json['selectedLayer']['layer']
        except:
            selected_layer = None
        return selected_layer

    def get_annotation(self, layer_name, aid):
        if self.state.layers[layer_name].type == 'annotation':
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
            aid_data = self.state.layers[layer_name]._json_data['selectedAnnotation']
            if isinstance(aid_data, OrderedDict):
                aid = aid_data['id']
            else:
                aid = aid
        except:
            aid = None
        return aid

    def select_annotation(self, layer_name, aid):
        if layer_name in self.layer_names:
            with self.txn() as s:
                s.layers[layer_name]._json_data['selectedAnnotation'] = aid
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

    def set_view_options(self,
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

    def set_segmentation_view_options(self,
                                      layer_name,
                                      alpha_selected=None,
                                      alpha_3d=None,
                                      alpha_unselected=None,
                                      ):
        if self.state.layers[layer_name].type is not 'segmentation':
            return
        with self.txn() as s:
            l = s.layers[layer_name]
            if alpha_selected is not None:
                l.selectedAlpha = alpha_selected
            if alpha_3d is not None:
                l.objectAlpha = alpha_3d
            if alpha_unselected is not None:
                l.notSelectedAlpha = alpha_unselected

    def assign_colors(self, layer_name, seg_colors):
        """ Assign colors to root ids in a segmentation layer

        Parameters
        ----------
        layer_name : str,
            Segmentation layer name
        seg_colors : dict
            dict with root ids as keys and colors as values.
        """
        with self.txn() as s:
            if seg_colors is not None:
                seg_colors = {str(oid): k for oid, k in seg_colors.items()}
                s.layers[layer_name]._json_data['segmentColors'] = seg_colors
