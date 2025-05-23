import json
import warnings
from typing import Literal, Optional, Self, Union

import neuroglancer
import numpy as np
from IPython.display import HTML
from neuroglancer import viewer, viewer_base

from ..site_utils import neuroglancer_url
from . import source_info
from .ngl_components import *


class UnservedViewer(viewer_base.UnsynchronizedViewerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._default_viewer_url = ""

    def get_server_url(self):
        return self._default_viewer_url


class ViewerState:
    def __init__(
        self,
        target_site: str = None,
        target_url: str = None,
        layers: Optional[list] = None,
        dimensions: Optional[Union[list, CoordSpace]] = None,
        *,
        position: Optional[Union[list, np.ndarray]] = None,
        scale_imagery: float = 1.0,
        scale_3d: float = 50000.0,
        show_slices: bool = False,
        selected_layer: Optional[str] = None,
        selected_layer_visible: bool = False,
        layout: Literal[
            "xy", "yz", "xz", "xy-3d", "xz-3d", "yz-3d", "4panel", "3d", "4panel-alt"
        ] = "xy-3d",
        base_state: Optional[dict] = None,
        interactive: bool = False,
        infer_coordinates: bool = True,
    ):
        """
        Parameters
        ----------
        target_site : str
            The target site for the viewer. If None, the default site will be used.
        target_url : str
            The target URL for the viewer. If None, the default URL will be used.
        layers : list of Layer
            The layers to add to the viewer.
        dimensions : list or CoordSpace
            The dimensions of the viewer. If None, the default dimensions will be used.
        position : list or np.ndarray
            The position of the viewer in 3D space. If None, the default position will be used.
        scale_imagery : float
            The scale factor for imagery layers. Default is 1.0.
        scale_3d : float
            The scale factor for 3D projections. Default is 50000.0.
        show_slices : bool
            Whether to show cross-sectional slices in the viewer. Default is False.
        selected_layer : str
            The name of the selected layer. If None, no layer is selected.
        selected_layer_visible : bool
            Whether the selected layer is visible. Default is False.
        layout : str
            The panel layout of the viewer. Default is "xy-3d".
        base_state : dict
            The base state of the viewer. If None, the default state will be used.
        interactive : bool
            Whether the viewer is interactive. Default is False.
        infer_coordinates : bool
            Whether to infer resolution and position from the source information using CloudVolume. Default is True.
        """

        self._target_site = target_site
        self._target_url = neuroglancer_url(target_url, target_site)
        self._layers = layers if layers else list()
        self._dimensions = dimensions
        self._position = position
        self._scale_imagery = scale_imagery
        self._scale_3d = scale_3d
        self._show_slices = show_slices
        self._selected_layer = selected_layer
        self._selected_layer_visible = selected_layer_visible
        self._layout = layout
        self._base_state = base_state
        self._interactive = interactive
        self._saved_state_url = None
        self._viewer = None
        self._source_info = None
        self._infer_coordinates = infer_coordinates

    def add_from_client(
        self,
        client: "caveclient.CAVEclient",
        imagery: Union[bool, str] = True,
        segmentation: Union[bool, str] = True,
        skeleton_source: bool = True,
        resolution: bool = True,
        target_url: bool = False,
    ) -> Self:
        """Configure the viewer with information from a CaveClient.
        Can set the target URL, viewer resolution, and add image and segmentation layers.

        Parameters
        ----------
        client : caveclient.CAVEclient
            The client to use for configuration.
        imagery : Union[bool, str], optional
            Whether to add an image layer, by default True.
            If a string is provided, it will be used as the name of the layer.
        segmentation : Union[bool, str], optional
            Whether to add a segmentation layer, by default True
            If a string is provided, it will be used as the name of the layer.
        skeleton_source : bool, optional
            Whether to try to add a skeleton source for the segmentation layer, if provided, by default True
        resolution : bool, optional
            Whether to infer viewer resolution from the client info, by default True
        target_url : bool, optional
            Whether to set the neuroglancer URL from the client, by default False

        Returns
        -------
        Self
            The updated viewer state object.
        """
        if target_url:
            self._target_url = client.info.viewer_site()
        if resolution:
            self.dimensions = client.info.viewer_resolution()
        if imagery:
            if isinstance(imagery, str):
                self.add_image_layer(
                    source=client.info.image_source(),
                    name=imagery,
                )
            else:
                self.add_image_layer(
                    source=client.info.image_source(),
                )
        if segmentation:
            seg_source = [client.info.segmentation_source()]
            skel_source_path = (
                client.info.get_datastack_info().get("skeleton_source"),
            )
            if skeleton_source and skel_source_path:
                seg_source.append(skel_source_path)

            if isinstance(segmentation, str):
                self.add_segmentation_layer(
                    source=seg_source,
                    name=segmentation,
                )
            else:
                self.add_segmentation_layer(
                    source=seg_source[0],
                )
        return self

    @property
    def viewer(self):
        if self._viewer is None:
            self._viewer = self.to_neuroglancer_state()
        return self._viewer

    def _reset_viewer(self):
        self._viewer = None

    @property
    def infer_coordinates(self):
        return self._infer_coordinates

    @infer_coordinates.setter
    def infer_coordinates(self, value: bool):
        if value != self._infer_coordinates:
            self._reset_viewer()
        self._infer_coordinates = value

    @property
    def layers(self):
        return self._layers

    @property
    def layer_names(self):
        return [layer.name for layer in self.layers]

    def add_layer(self, layers, selected: bool = False) -> Self:
        """Add a layer to the viewer.
        Parameters
        ----------
        layers : Layer or list of Layer
            The layer(s) to add to the viewer.
        """

        if isinstance(layers, list):
            self._layers.extend(layers)
            if selected is not False:
                for layer, vis in zip(layers, selected):
                    if vis:
                        self.set_selected_layer(layer.name)
        else:
            self._layers.append(layers)
            if selected:
                self.set_selected_layer(layers.name)
        self._reset_viewer()
        return self

    @property
    def source_info(self):
        if self._source_info is None:
            self._source_info = source_info.populate_info(self.layers)
        return self._source_info

    def _suggest_position_from_source(self, resolution=None) -> np.ndarray:
        """Suggest a position based on the source information.
        This uses the headers on each layer source to suggest a position based on the
        center of the first segmentation layer or, if no segmentation layer, the first image layer.
        If no position can be inferred, the position will be to None.

        NOTE: Requires cloudvolume to be installed and an internet connection.

        Parameters
        ----------
        resolution : list or np.ndarray, optional
            The resolution of the viewer. If not provided, the viewer's current resolution will be used.

        Returns
        -------
        Position : np.ndarray

        """
        try:
            position = source_info.suggest_position(self.source_info, resolution)
        except source_info.NoCloudvolumeError:
            warnings.warn(
                "Cloudvolume is not installed (Install with `pip install cloud-volume`). Cannot suggest position."
            )
            position = None
        return position

    def _suggest_resolution_from_source(self) -> np.ndarray:
        """Suggest a resolution based on the source information.
        This uses the headers on each layer source to suggest a resolution based on the
        first image layer. If no resolution can be inferred, the resolution will be None.

        NOTE: Requires cloudvolume to be installed and an internet connection.

        Returns
        -------
        Resolution : np.ndarray

        """
        try:
            resolution = source_info.suggest_resolution(self.source_info)
        except source_info.NoCloudvolumeError:
            warnings.warn(
                "Cloudvolume is not installed (Install with `pip install cloud-volume`). Cannot suggest resolution."
            )
            resolution = None
        return resolution

    @property
    def dimensions(self):
        return self._dimensions

    @dimensions.setter
    def dimensions(self, value):
        if isinstance(value, CoordSpace):
            self._dimensions = value
        else:
            self._dimensions = CoordSpace(value)
        self._reset_viewer()

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value
        self._reset_viewer()

    @property
    def scale_imagery(self):
        return self._scale_imagery

    @scale_imagery.setter
    def scale_imagery(self, value):
        self._scale_imagery = value
        self._reset_viewer()

    @property
    def scale_3d(self):
        return self._scale_3d

    @scale_3d.setter
    def scale_3d(self, value):
        self._scale_3d = value
        self._viewer = None

    @property
    def show_slices(self):
        return self._show_slices

    @show_slices.setter
    def show_slices(self, value):
        self._show_slices = value
        self._viewer = None

    @property
    def selected_layer(self):
        return {
            "layer": self._selected_layer,
            "visible": self._selected_layer_visible,
        }

    def set_selected_layer(
        self, selected_layer: Union[str, ImageLayer, SegmentationLayer, AnnotationLayer]
    ) -> Self:
        if isinstance(selected_layer, str):
            self._selected_layer = selected_layer
        else:
            self._selected_layer = selected_layer.name
        self._reset_viewer()
        return self

    @property
    def selected_layer_visible(self):
        return self._selected_layer_visible

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, value):
        if value not in [
            "xy",
            "yz",
            "xz",
            "xy-3d",
            "xz-3d",
            "yz-3d",
            "4panel",
            "3d",
            "4panel-alt",
        ]:
            raise ValueError(
                f"Invalid layout: {value}. Must be one of 'xy', 'yz', 'xz', 'xy-3d', 'xz-3d', 'yz-3d', '4panel', '3d', or '4panel-alt'."
            )
        self._layout = value
        self._reset_viewer()

    @property
    def base_state(self):
        return self._base_state

    @base_state.setter
    def base_state(self, value):
        self._base_state = value
        self._reset_viewer()

    @property
    def interactive(self):
        return self._interactive

    @interactive.setter
    def interactive(self, value):
        if value != self._interactive:
            self._viewer = None
        self._interactive = value

    def set_viewer_properties(
        self,
        position: Optional[Union[list, np.ndarray]] = None,
        dimensions: Optional[Union[list, np.ndarray]] = None,
        scale_imagery: Optional[float] = None,
        scale_3d: Optional[float] = None,
        show_slices: Optional[bool] = None,
        selected_layer: Optional[Union[str, ImageLayer]] = None,
        selected_layer_visible: Optional[bool] = None,
        layout: Optional[
            Literal[
                "xy",
                "yz",
                "xz",
                "xy-3d",
                "xz-3d",
                "yz-3d",
                "4panel",
                "3d",
                "4panel-alt",
            ]
        ] = None,
        base_state: Optional[dict] = None,
        interactive: Optional[bool] = None,
        infer_coordinates: bool = False,
    ) -> Self:
        """
        Set various properties of the viewer state.

        This function allows you to configure the viewer's position, dimensions,
        scale, layout, and other properties. Any property not explicitly set will
        remain unchanged.

        Parameters
        ----------
        position : list or np.ndarray, optional
            The position of the viewer in 3D space.
        dimensions : list or np.ndarray, optional
            The dimensions of the viewer's coordinate space.
            Can be either a list of 3 values treated as nanometer resolution or
            a CoordSpace object with more detailed options.
        scale_imagery : float, optional
            The scale factor for imagery layers.
        scale_3d : float, optional
            The scale factor for 3D projections.
        show_slices : bool, optional
            Whether to show cross-sectional slices in the viewer.
        selected_layer : str or ImageLayer, optional
            The name of the selected layer or the layer object itself.
        selected_layer_visible : bool, optional
            Whether the selected layer is visible.
        layout : {"xy", "yz", "xz", "xy-3d", "xz-3d", "yz-3d", "4panel", "3d", "4panel-alt"}, optional
            The panel layout of the viewer.
        base_state : dict, optional
            The base state of the viewer.
        interactive : bool, optional
            Whether the viewer is interactive.
        infer_coordinates : bool, optional
            Whether to infer resolution and position from the source information.
            If True, the viewer will attempt to extract this information from the
            info provided by the source if not provided explicitly.


        Returns
        -------
        self
            The updated viewer state object.
        """

        if position is not None:
            self.position = position
        if dimensions is not None:
            self.dimensions = dimensions
        if scale_imagery is not None:
            self.scale_imagery = scale_imagery
        if scale_3d is not None:
            self.scale_3d = scale_3d
        if show_slices is not None:
            self.show_slices = show_slices
        if selected_layer is not None:
            if isinstance(selected_layer, str):
                self.selected_layer = selected_layer
            else:
                self.selected_layer = selected_layer.name
        if selected_layer_visible is not None:
            self.selected_layer_visible = selected_layer_visible
        if layout is not None:
            self.layout = layout
        if base_state is not None:
            self.base_state = base_state
        if interactive is not None:
            self.interactive = interactive
        if infer_coordinates is not None:
            self.infer_coordinates = infer_coordinates
        return self

    def add_image_layer(
        self,
        source: str,
        name: str = "imagery",
        resolution: Optional[Union[list, np.ndarray]] = None,
        **kwargs,
    ) -> Self:
        if resolution is None:
            resolution = self.dimensions
        img_layer = ImageLayer(
            name=name,
            source=source,
            resolution=resolution,
            **kwargs,
        )
        self.add_layer(img_layer)
        return self

    def add_segmentation_layer(
        self,
        source: str,
        name: str = "segmentation",
        resolution: Optional[Union[list, np.ndarray]] = None,
        **kwargs,
    ) -> Self:
        if resolution is None:
            resolution = self.dimensions
        seg_layer = SegmentationLayer(
            name=name,
            source=source,
            resolution=resolution,
            **kwargs,
        )
        self.add_layer(seg_layer)
        return self

    def add_annotation_layer(
        self,
        name: str = "annotation",
        source: Optional[str] = None,
        resolution: Optional[Union[list, np.ndarray]] = None,
        tags: Optional[list] = None,
        linked_segmentation: Union[str, bool] = True,
        shader: str = None,
        **kwargs,
    ) -> Self:
        """Add an annotation layer, either local or with precomputed annotation source

        Parameters
        ----------
        name : str, optional
            Layer name, by default "annotation"
        source : Optional[str], optional
            Cloud annotation source path, if using. By default None
        resolution : Optional[Union[list, np.ndarray]], optional
            Layer resolution, only used by local annotation layers.
            By default None, where it is inferred from the viewer.
        tags : Optional[list], optional
            Ordered list of tags, by default None.
        linked_segmentation : Union[str, bool], optional
            If True, will link to the first segmentation layer found in the viewer.
            If False, will not link to any segmentation layer.
            If a string is provided, it will be used as the name of the segmentation layer to link to.


        Returns
        -------
        Self
            The viewer state object with the added annotation layer.
        """
        if linked_segmentation:
            if linked_segmentation is True:
                for layer in self.layers:
                    if isinstance(layer, SegmentationLayer):
                        linked_segmentation = layer.name
                        break
                else:
                    linked_segmentation = None
        if source:
            anno_layer = AnnotationLayer(
                name=name,
                source=source,
                linked_segmentation=linked_segmentation,
                shader=shader,
                **kwargs,
            )
        else:
            if resolution is None:
                resolution = self.dimensions
            anno_layer = LocalAnnotationLayer(
                name=name,
                resolution=resolution,
                tags=tags,
                linked_segmentation=linked_segmentation,
                shader=shader,
                **kwargs,
            )
        self.add_layer(anno_layer)
        return self

    def get_layer(self, name: str):
        """Get a layer by name.

        Parameters
        ----------
        name : str
            The name of the layer to get.

        Returns
        -------
        Layer
            The layer with the specified name.
        """
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise ValueError(f"Layer {name} not found in viewer.")

    @contextmanager
    def with_datamap(self, datamap: dict):
        """Context manager to apply datamaps."""
        if not isinstance(datamap, dict):
            datamap = {None: datamap}
        viewer_copy = copy.deepcopy(self)
        viewer_copy._apply_datamaps(datamap)
        yield viewer_copy

    def _apply_datamaps(self, datamap: dict):
        for layer in self.layers:
            layer._apply_datamaps(datamap)

    def add_points(
        self,
        data: Union[list, np.ndarray, pd.DataFrame],
        name: str = "annotation",
        point_column: Optional[str] = None,
        segment_column: Optional[str] = None,
        description_column: Optional[str] = None,
        tag_column: Optional[str] = None,
        tag_bools: Optional[list] = None,
        data_resolution: Optional[list] = None,
        tags: Optional[list] = None,
        linked_segmentation: Union[str, bool] = True,
        shader: Optional[str] = None,
    ) -> Self:
        """Add points to an existing annotation layer or create a new one.
        Parameters
        ----------
        data : Union[list, np.ndarray, pd.DataFrame]
            The data to add to the annotation layer.
        name : str, optional
            The name of the annotation layer, by default "annotation".
        point_column : Optional[str], optional
            The name of the column containing point coordinates, by default None.
            None is needed if the data is an array, but is required if the data is a DataFrame.
        segment_column : Optional[str], optional
            The name of the column containing linked segment IDs, by default None.
        description_column : Optional[str], optional
            The name of the column containing descriptions, by default None.
            If None, no descriptions will be added.
        tag_column : Optional[str], optional
            The name of a column containing tags, by default None.
        tag_bools : Optional[list], optional
            A list of column names indicating tags as booleans, by default None.
        data_resolution : Optional[list], optional
            The resolution of the data, by default None.
            If None, the viewer's current resolution will be used.
        tags : Optional[list], optional
            A list of tags to add to the annotation layer, by default None.
            If None, tags will be inferred from the data, if any tag columns are provided.
        linked_segmentation : str or bool, optional
            The name of the segmentation layer to link to, by default None.
            If True, will link to the first segmentation layer found in the viewer.
            If False, will not link to any segmentation layer.
        shader : Optional[str], optional
            The shader to use for the annotation layer, by default None.
            If None, the default shader will be used.

        Returns
        -------
        Self
            The viewer state object with the added points.
        """

        if name in self.layer_names:
            layer = self.get_layer(name)
            if not isinstance(layer, LocalAnnotationLayer):
                raise ValueError(
                    f"Layer {name} already exists but is not a LocalAnnotationLayer."
                )
        else:
            layer = LocalAnnotationLayer(
                name=name,
                resolution=data_resolution,
                tags=tags,
                linked_segmentation=linked_segmentation,
                shader=shader,
            ).add_points(
                data,
                point_column=point_column,
                segment_column=segment_column,
                description_column=description_column,
                tag_column=tag_column,
                tag_bools=tag_bools,
                data_resolution=data_resolution,
            )
            self.add_layer(layer)

    def to_neuroglancer_state(self):
        if self.dimensions is None:
            if self.infer_coordinates:
                self._dimensions = self._suggest_resolution_from_source()
            else:
                self._dimensions = CoordSpace()
        if not isinstance(self.dimensions, CoordSpace):
            self._dimensions = CoordSpace(self.dimensions)

        for layer in self.layers:
            if isinstance(layer, LocalAnnotationLayer):
                if layer.resolution is None:
                    layer.resolution = self.dimensions.resolution

        if self.interactive:
            self._viewer = viewer.Viewer()
        else:
            self._viewer = UnservedViewer()
        if self.base_state:
            self._viewer.set_state(self.base_state)

        with self._viewer.txn() as s:
            if self.position:
                s.position = self.position
            elif self.infer_coordinates:
                s.position = self._suggest_position_from_source(
                    resolution=self.dimensions.resolution
                )
            if self.layout:
                s.layout = self.layout
            s.dimensions = self.dimensions.to_neuroglancer()
            s.cross_section_scale = self.scale_imagery
            s.projection_scale = self.scale_3d
            s.show_slices = self.show_slices
            for layer in self.layers:
                layer.to_neuroglancer(s)

        return self._viewer

    def map(self, datamap: dict):
        """Apply a datamap to the viewer state to freeze the state of the viewer.

        Parameters
        ----------
        datamap : dict
            A dictionary mapping layer names to their corresponding datamaps.

        Returns
        -------
        Self
            A new ViewerState object with the datamap applied.
        """
        with self.with_datamap(datamap) as viewer_copy:
            return viewer_copy

    def to_dict(self) -> dict:
        """Return a dictionary representation of the viewer state.

        Returns
        -------
        dict
            A dictionary representation of the viewer state.
        """
        return self.viewer.state.to_json()

    def to_json_string(self, indent: int = 2) -> str:
        """Return a JSON string representation of the viewer state.

        Parameters
        ----------
        indent : int
            The number of spaces to use for indentation in the JSON string.
            Default is 2.

        Returns
        -------
        str
            A JSON string representation of the viewer state.
        """

        return json.dumps(self.to_dict(), indent=indent)

    def to_url(
        self,
        target_url: str = None,
        datamap: Optional[dict] = None,
    ):
        """Return a URL representation of the viewer state.

        Parameters
        ----------
        target_url : str
            The base URL to use for the Neuroglancer state. If not provided,
            the default server URL will be used.
        html : bool
            If True, return an HTML link instead of a plain URL.
        link_text : str
            The text to display for the HTML link. Only used if `html` is True.

        Returns
        -------
        str
            A URL representation of the viewer state.
        """
        if target_url is None:
            if self.interactive:
                return self.viewer.get_viewer_url()
            else:
                target_url = self._target_url

        url = neuroglancer.to_url(
            self.viewer.state,
            target_url,
        )
        return url

    def to_link(
        self,
        datamap: Optional[dict] = None,
        target_url: str = None,
        link_text: str = "Neuroglancer Link",
    ):
        """Return an HTML link representation of the viewer state.

        Parameters
        ----------
        target_url : str
            The base URL to use for the Neuroglancer state. If not provided,
            the default server URL will be used.
        link_text : str
            The text to display for the HTML link.

        Returns
        -------
        str
            An HTML link representation of the viewer state.
        """
        url = self.to_url(target_url)
        return HTML(f'<a href="{url}" target="_blank">{link_text}</a>')
