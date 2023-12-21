from .mappers import (
    SelectionMapper,
    AnnotationMapperBase,
    PointMapper,
    LineMapper,
    SphereMapper,
    BoundingBoxMapper,
)
from datetime import datetime
import numpy as np
import numbers

DEFAULT_IMAGE_LAYER = "img"
DEFAULT_SEG_LAYER = "seg"
DEFAULT_ANNO_LAYER = "anno"

DEFAULT_SEGMENTATION_VIEW_KWS = {
    "alpha_selected": 0.3,
    "alpha_3d": 1,
    "alpha_unselected": 0,
}


class LayerConfigBase(object):
    """Base class for configuring layers

    Parameters
    ----------
    name : str
        Layer name for reference and display
    type : str
        Layer type. Usually handled by the subclass
    source : str
        Datasource for the layer
    color : str
        Hex string (with starting hash).
    active : bool,
        If True, becomes a selected layer.
    """

    def __init__(
        self,
        name,
        type,
        source,
        color,
        active,
    ):
        self._config = dict(
            type=type,
            name=str(name),
            source=source,
            color=color,
            active=active,
        )

    @property
    def type(self):
        return self._config.get("type", None)

    @property
    def name(self):
        return self._config.get("name", None)

    @name.setter
    def name(self, n):
        self._config["name"] = n

    @property
    def source(self):
        return self._config.get("source", None)

    @source.setter
    def source(self, s):
        self._config["source"] = s

    @property
    def color(self):
        return self._config.get("color", None)

    @color.setter
    def color(self, c):
        self._config["color"] = c

    @property
    def active(self):
        return self._config.get("active", False)

    @active.setter
    def active(self, val):
        self._config["active"] = val

    def _render_layer(
        self, viewer, data, viewer_resolution=None, return_annos=False
    ):
        """Applies rendering rules"""
        annos = self._specific_rendering(
            viewer,
            data,
            viewer_resolution=viewer_resolution,
            return_annos=return_annos,
        )
        if self.active:
            viewer.set_selected_layer(self.name)
        if return_annos:
            return annos
        else:
            pass

    def _add_layer(
            self, viewer
    ):
        "Subclass implements layer addition rules"
        pass

    def _set_view_options(self, *args, **kwargs):
        "Subclass implements own rules"
        pass

    def _specific_rendering(
        self, viewer, data, viewer_resolution=None, return_annos=False,
    ):
        """Subclasses implement specific rendering rules"""
        pass


class ImageLayerConfig(LayerConfigBase):
    """Image layer configuration class.

    This provides the rules for setting up an image layer in neuroglancer.

    Parameters
    ----------
    source : str
        Cloudpath to an image source
    name : str, optional
        Name of the image layer. By default, 'img'.
    active : bool, optional
        If True, makes the layer active in neuroglancer. Default is False.
    contrast_controls : bool, optional
        If True, gives the layer a user-controllable brightness and contrast shader. Default is False.
    black : float, optional
        If contrast_controls is True, sets the default black level. Default is 0.0.
    white : float, optional
        If contrast_controls is True, sets the default white level. Default is 1.0.
    """

    def __init__(
        self,
        source,
        name=None,
        active=False,
        contrast_controls=False,
        black=0.0,
        white=1.0,
    ):
        if name is None:
            name = DEFAULT_IMAGE_LAYER
        super(ImageLayerConfig, self).__init__(
            name=name, type="image", source=source, color=None, active=active
        )
        self._contrast_controls = contrast_controls
        self._black = black
        self._white = white

    def _add_layer(self, viewer):
        viewer.add_image_layer(self.name, self.source)

    def _specific_rendering(
        self, viewer, data, viewer_resolution=None, return_annos=False,
    ):
        if self._contrast_controls:
            viewer.add_contrast_shader(self.name, black=self._black, white=self._white)


class SegmentationLayerConfig(LayerConfigBase):
    """Configuration class for segmentation layers

    Parameters
    ----------
    source : str
        Segmentation source
    name : str, optional,
        Layer name.
    selected_ids_column : str or list-like, optional.
        Column name (or list of column names) to use for selected ids.
    fixed_ids : list-like, optional.
        List of root ids to select directly.
    fixed_id_colors : list-like, optional.
        List of colors for fixed ids. Should be the same length as fixed_ids, although null entries can be padded with None values.
    color_column : str, optional.
        # at the begining.
        Column to use for color values for selected objects. Values should be RGB hex strings with a
    active : bool, optional.
        If True, makes the layer selected. Default is False.
    alpha_selected: float, optional
        Opacity of selected segmentations in the image layer. Optional, default is 0.3.
    alpha_3d: float, optional
        Opacity of meshes. Optional, default is 1.
    alpha_unselected: float, optional
        Opacity of unselected segments. Optional, default is 0.
    split_point_map: mappers.SplitPointMap or None, optional
        If not None, provides an object to map the dataframe input to multicut points. Default is None.
    view_kws : dict, optional.
        Keyword arguments for viewer.set_segmentation_view_options. Sets selected (and unselected) segmetation alpha values. Defaults to values in DEFAULT_SEGMENTATION_VIEW_KWS dict specified in this module.
    timestamp : float or datetime, optional.
        Timestamp at which to fix the chunkedgraph in either unix epoch or datetime format. Optional, default is None.
    """

    def __init__(
        self,
        source,
        name=None,
        selected_ids_column=None,
        fixed_ids=None,
        fixed_id_colors=None,
        color_column=None,
        active=False,
        alpha_selected=0.3,
        alpha_3d=1,
        alpha_unselected=0,
        split_point_map=None,
        view_kws=None,
        timestamp=None,
        data_resolution=None,
    ):
        if name is None:
            name = DEFAULT_SEG_LAYER

        super(SegmentationLayerConfig, self).__init__(
            name=name, type="segmentation", source=source, color=None, active=active
        )
        self._config["data_resolution"] = data_resolution

        if selected_ids_column is not None or fixed_ids is not None:
            self._selection_map = SelectionMapper(
                data_columns=selected_ids_column,
                fixed_ids=fixed_ids,
                fixed_id_colors=fixed_id_colors,
                color_column=color_column,
            )
        else:
            self._selection_map = None

        self._split_point_map = split_point_map

        if isinstance(timestamp, datetime):
            timestamp = timestamp.timestamp()
        self.timestamp = timestamp

        base_seg_kws = DEFAULT_SEGMENTATION_VIEW_KWS.copy()
        if view_kws is None:
            view_kws = {
                "alpha_selected": alpha_selected,
                "alpha_3d": alpha_3d,
                "alpha_unselected": alpha_unselected,
            }
        base_seg_kws.update(view_kws)
        self._view_kws = view_kws

    @property
    def data_resolution(self):
        return self._config.get("data_resolution", None)

    def add_selection_map(
        self,
        selected_ids_column=None,
        fixed_ids=None,
        fixed_id_colors=None,
        color_column=None,
    ):
        if self._selection_map is not None:
            if isinstance(selected_ids_column, str):
                selected_ids_column = [selected_ids_column]
            if selected_ids_column is not None:
                data_columns = self._selection_map.data_columns + selected_ids_column
            else:
                data_columns = self._selection_map.data_columns
            if fixed_ids is not None:
                new_fixed_ids = np.concatenate(
                    (self._selection_map.fixed_ids.tolist(), np.atleast_1d(fixed_ids))
                ).tolist()
                old_fixed_id_colors = self._selection_map.fixed_id_colors
                if len(old_fixed_id_colors) > len(
                    self._selection_map.fixed_ids.tolist()
                ):
                    old_fixed_id_colors = old_fixed_id_colors[
                        : len(len(self._selection_map.fixed_ids.tolist()))
                    ]
                else:
                    while len(old_fixed_id_colors) < len(
                        self._selection_map.fixed_ids.tolist()
                    ):
                        old_fixed_id_colors += [None]
                if fixed_id_colors is None:
                    fixed_id_colors = len(fixed_ids) * [None]
                elif isinstance(fixed_id_colors, str) or isinstance(
                    fixed_ids, numbers.Number
                ):
                    fixed_id_colors = [fixed_id_colors]
                new_fixed_id_colors = old_fixed_id_colors + fixed_id_colors
            else:
                new_fixed_ids = self._selection_map.fixed_ids
                new_fixed_id_colors = self._selection_map.fixed_id_colors
            if color_column is None:
                color_column = self._selection_map.color_column
        else:
            data_columns = selected_ids_column
            new_fixed_ids = fixed_ids
            new_fixed_id_colors = fixed_id_colors

        self._selection_map = SelectionMapper(
            data_columns=data_columns,
            fixed_ids=new_fixed_ids,
            fixed_id_colors=new_fixed_id_colors,
            color_column=color_column,
        )

    def _add_layer(self, viewer):
        viewer.add_segmentation_layer(self.name, self.source)

    def _specific_rendering(
        self, viewer, data, viewer_resolution=None, return_annos=False,
    ):
        if self._selection_map is not None:
            selected_ids = self._selection_map.selected_ids(data)
            colors = self._selection_map.seg_colors(data)
            viewer.add_selected_objects(self.name, selected_ids, colors)

        if self._split_point_map is not None:
            (
                seg_id,
                points_red,
                points_blue,
                sv_red,
                sv_blue,
            ) = self._split_point_map._render_data(
                data,
                data_resolution=self.data_resolution,
                viewer_resolution=viewer_resolution,
                viewer=viewer,
            )
            viewer.set_multicut_points(
                self.name,
                seg_id,
                points_red,
                points_blue,
                sv_red,
                sv_blue,
                self._split_point_map.focus,
            )

        viewer.set_timestamp(self.name, self.timestamp)
        viewer.set_segmentation_view_options(self.name, **self._view_kws)
        pass


class AnnotationLayerConfig(LayerConfigBase):
    """Configuration class for annotation layers

    Parameters
    ----------
    name : str, optional
        Layer name. By default, 'annos'
    color : str, optional
        Hex color code with an initial #. By default, None
    linked_segmentation_layer : str, optional
        Name of a linked segmentation layer for selected ids. By default, None
    mapping_rules : PointMapper, LineMapper, SphereMapper or list, optional
        One rule or a list of rules mapping data to annotations. By default, []
    array_data : bool, optional
        If True, allows simple mapping where one or more arrays are passed instead of a dataframe.
        Only allows basic annotation creation, no tags, linked segmentations, or other rich features.
    tags : list, optional
        List of tags for the layer.
    active : bool, optional
        If True, makes the layer selected. Default is True (unlike for image/segmentation layers).
    """

    def __init__(
        self,
        name=None,
        color=None,
        linked_segmentation_layer=None,
        mapping_rules=[],
        array_data=False,
        tags=None,
        active=True,
        filter_by_segmentation=False,
        brackets_show_segmentation=True,
        selection_shows_segmentation=True,
        filter_query=None,
        data_resolution=None,
    ):
        if name is None:
            name = DEFAULT_ANNO_LAYER

        super(AnnotationLayerConfig, self).__init__(
            name=name, type="annotation", color=color, source=None, active=active
        )
        self._config["linked_segmentation_layer"] = linked_segmentation_layer
        self._config["filter_by_segmentation"] = filter_by_segmentation
        self._config["selection_shows_segmentation"] = selection_shows_segmentation
        self._config["brackets_show_segmentation"] = brackets_show_segmentation
        self._config["filter_query"] = filter_query
        self._config["data_resolution"] = data_resolution

        if issubclass(type(mapping_rules), AnnotationMapperBase):
            mapping_rules = [mapping_rules]
        if array_data is True:
            if len(mapping_rules) > 1:
                raise ValueError("Only one mapping rule can be set using array data")
            for mr in mapping_rules:
                mr.array_data = array_data
        self._array_data = array_data
        self._annotation_map_rules = mapping_rules
        self._tags = tags

    @property
    def linked_segmentation_layer(self):
        return self._config.get("linked_segmentation_layer", None)

    @property
    def filter_by_segmentation(self):
        return self._config.get("filter_by_segmentation", None)

    @property
    def selection_shows_segmentation(self):
        return self._config.get("selection_shows_segmentation", None)

    @property
    def brackets_show_segmentation(self):
        return self._config.get("brackets_show_segmentation", None)

    @property
    def filter_query(self):
        return self._config.get("filter_query", None)

    @property
    def data_resolution(self):
        return self._config.get("data_resolution", None)

    def _add_layer(self, viewer):
        viewer.add_annotation_layer(
            self.name,
            color=self.color,
            linked_segmentation_layer=self.linked_segmentation_layer,
            filter_by_segmentation=self.filter_by_segmentation,
            selection_shows_segmentation=self.selection_shows_segmentation,
            brackets_show_segmentation=self.brackets_show_segmentation,
            tags=self._tags,
        )

    def _set_view_options(self, viewer, data, viewer_resolution=None):
        pos = None
        if data is not None:
            if self.filter_query is not None:
                data = data.query(self.filter_query)
            for rule in self._annotation_map_rules:
                pos = rule._get_position(
                        data,
                        data_resolution=self.data_resolution,
                        viewer_resolution=viewer_resolution,
                    )
                viewer.set_view_options(
                    position=pos
                )
                if pos is not None:
                    break
        return pos

    def _specific_rendering(
        self, viewer, data, viewer_resolution=None, return_annos=False,
    ):
        annos = []
        for rule in self._annotation_map_rules:
            rule.tag_map = self._tags
            if data is not None:
                if self.filter_query is not None:
                    data = data.query(self.filter_query)
                if len(data) > 0:
                    annos.extend(
                        rule._render_data(
                            data,
                            data_resolution=self.data_resolution,
                            viewer_resolution=viewer_resolution,
                            viewer=viewer,
                        )
                    )
        if return_annos:
            return annos
        else:
            viewer.add_annotations(self.name, annos)
            pass
