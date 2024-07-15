from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

from . import utils

SEGMENTATION_LAYER_TYPES = ["segmentation", "segmentation_with_graph"]


class EasyViewerBase(ABC):
    def __init__(self):
        self._default_viewer_url = None

    def __repr__(self):
        return self.as_url()

    def __repr_html__(self):
        return f'<a href="{self.as_url()}" target="_blank">Viewer</a>'

    @abstractmethod
    def load_url(self, url) -> None:
        pass

    def add_layers(
        self,
        image_layers: dict = {},
        segmentation_layers: dict = {},
        annotation_layers: dict = {},
        resolution: list = None,
    ) -> None:
        self.set_resolution(resolution)
        with self.txn() as s:
            for ln, kws in image_layers.items():
                s.layers[ln] = self._ImageLayer(**kws)
            for ln, kws in segmentation_layers.items():
                self._SegmentationLayer(s, **kws)
            for ln, kws in annotation_layers.items():
                s.layers[ln] = self._AnnotationLayer(**kws)

    @abstractmethod
    def _ImageLayer(self, source, **kwargs):
        pass

    @abstractmethod
    def _AnnotationLayer(self, **kwargs):
        pass

    @abstractmethod
    def _SegmentationLayer(self, source, **kwargs):
        pass

    def add_segmentation_layer(self, layer_name, source, **kwargs):
        """Add segmentation layer to viewer instance.

        Attributes:
            layer_name (str): name of layer to be displayed in neuroglancer ui.
            source (str): source of neuroglancer segment layer
        """
        with self.txn() as s:
            s.layers[layer_name] = self._SegmentationLayer(source=source, **kwargs)

    @abstractmethod
    def append_source_to_segmentation_layer(self, layer_name, source):
        pass

    @abstractmethod
    def add_skeleton_source(self, layer_name, source, shader_text=None):
        pass

    @abstractmethod
    def set_skeleton_shader(self, layer_name, shader_text=None):
        pass

    def add_image_layer(self, layer_name, source, contrast_range=None, **kwargs):
        """Add segmentation layer to viewer instance.

        Attributes:
            layer_name (str): name of layer to be displayed in neuroglancer ui.
            source (str): source of neuroglancer image layer
            contrast_range = (float, float): contrast range for image layer either in 0-1 (floats) or 0-255 (ints).
        """
        with self.txn() as s:
            s.layers[layer_name] = self._ImageLayer(source=source, **kwargs)
        if contrast_range is not None:
            self.add_contrast_shader(
                layer_name, black=contrast_range[0], white=contrast_range[1]
            )

    @abstractmethod
    def set_resolution(self, resolution) -> None:
        pass

    def add_contrast_shader(self, layer_name, black=0.0, white=1.0):
        shader_text = f"#uicontrol float black slider(min=0, max=1, default={black})\n#uicontrol float white slider(min=0, max=1, default={white})\nfloat rescale(float value) {{\n  return (value - black) / (white - black);\n}}\nvoid main() {{\n  float val = toNormalized(getDataValue());\n  if (val < black) {{\n    emitRGB(vec3(0,0,0));\n  }} else if (val > white) {{\n    emitRGB(vec3(1.0, 1.0, 1.0));\n  }} else {{\n    emitGrayscale(rescale(val));\n  }}\n}}\n"
        self.update_shader(layer_name, shader_text)

    def update_shader(self, layer_name, shader_text):
        with self.txn() as s:
            s.layers[layer_name]._json_data["shader"] = shader_text

    @abstractmethod
    def set_state_server(self, state_server) -> None:
        pass

    @abstractmethod
    def add_annotation_layer(
        self,
        layer_name=None,
        color=None,
        linked_segmentation_layer=None,
        filter_by_segmentation=False,
        brackets_show_segmentation=True,
        selection_shows_segmentation=True,
        tags=None,
    ) -> None:
        pass

    def set_annotation_layer_color(self, layer_name, color):
        """Set the color for the annotation layer"""
        if layer_name in [l.name for l in self.state.layers]:
            with self.txn() as s:
                s.layers[layer_name].annotationColor = utils.parse_color(color)
        else:
            pass

    def clear_annotation_layers(self, layer_names):
        with self.txn() as s:
            for ln in layer_names:
                s.layers[ln].annotations._data = []

    def add_annotations(
        self,
        layer_name: str,
        annotations: List,
    ):
        with self.txn() as s:
            s.layers[layer_name].annotations.extend(annotations)

    def add_multilayer_annotations(
        self,
        layer_anno_dict: Dict[str, List],
    ):
        """
        layer_anno_dict is a layer_name to annotation list dict.
        """
        with self.txn() as s:
            for ln, annos in layer_anno_dict.items():
                if annos is None:
                    continue
                s.layers[ln].annotations.extend(annos)

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

    @abstractmethod
    def add_annotation_tags(self, layer_name, tags):
        pass

    @property
    def url(self):
        return self.get_viewer_url()

    @abstractmethod
    def as_url(
        self,
        prefix: Optional[str] = None,
        as_html: Optional[bool] = False,
        link_text: Optional[str] = "Neuroglancer Link",
    ) -> str:
        pass

    @abstractmethod
    def set_selected_layer(self, layer_name):
        pass

    @abstractmethod
    def select_annotation(self, layer_name, anno_id):
        pass

    @property
    def layer_names(self):
        return [l.name for l in self.state.layers]

    def selected_objects(self, segmentation_layer):
        return list(self.state.layers[segmentation_layer].segments)

    @abstractmethod
    def add_selected_objects(
        self,
        segmentation_layer: str,
        oids: List[int],
        colors: Optional[Union[List, Dict]] = None,
    ) -> None:
        pass

    @abstractmethod
    def set_view_options(
        self,
        show_slices: Optional[bool] = None,
        layout: Optional[str] = None,
        show_axis_lines: Optional[bool] = None,
        show_scale_bar: Optional[bool] = None,
        orthographic: Optional[bool] = None,
        position: Optional[Tuple[float]] = None,
        zoom_image: Optional[float] = None,
        zoom_3d: Optional[float] = None,
        background_color: Optional[Tuple[float]] = None,
    ) -> None:
        pass

    @abstractmethod
    def set_segmentation_view_options(
        self,
        layer_name: str,
        alpha_selected: Optional[float] = None,
        alpha_3d: Optional[float] = None,
        alpha_unselected: Optional[float] = None,
        **kwargs,
    ):
        pass

    @abstractmethod
    def set_timestamp(
        self,
        layer_name,
        timestamp: Optional[int] = None,
    ):
        pass

    @abstractmethod
    def assign_colors(self, layer_name, seg_colors):
        """Assign colors to root ids in a segmentation layer

        Parameters
        ----------
        layer_name : str,
            Segmentation layer name
        seg_colors : dict
            dict with root ids as keys and colors as values.
        """

    @abstractmethod
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
        pass

    @staticmethod
    @abstractmethod
    def point_annotation(
        point,
        id=None,
        **kwargs,
    ):
        pass

    @staticmethod
    @abstractmethod
    def line_annotation(
        pointA,
        pointB,
        id=None,
        **kwargs,
    ):
        pass

    @staticmethod
    @abstractmethod
    def ellipsoid_annotation(
        center,
        radii,
        id=None,
        **kwargs,
    ):
        pass

    @classmethod
    def sphere_annotation(
        cls,
        center,
        radius,
        z_multiplier,
        **kwargs,
    ):
        return cls.ellipsoid_annotation(
            center,
            [radius, radius, radius * z_multiplier],
            **kwargs,
        )

    @staticmethod
    @abstractmethod
    def bounding_box_annotation(
        pointA,
        pointB,
        id=None,
        **kwargs,
    ):
        pass

    @staticmethod
    @abstractmethod
    def group_annotations(
        annotations,
        source=None,
        id=None,
        return_all=True,
        gather_linked_segmentations=True,
        share_linked_segmentations=False,
        children_visible=True,
    ):
        pass
