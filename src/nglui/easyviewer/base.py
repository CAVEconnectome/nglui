import json
from typing import Self

import neuroglancer
import numpy as np
from neuroglancer import viewer, viewer_base

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
            "xy", "yz", "xz", "xy-3d", "xz-3d", "yz-3d", "4panel", "3d", "4panel-alt"  # noqa: F722
        ] = "xy-3d",
        base_state: Optional[dict] = None,
        interactive: bool = False,
    ):
        self._target_site = target_site
        self._target_url = target_url
        self._layers = layers if layers else list
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

    @property
    def viewer(self):
        if self._viewer is None:
            self._viewer = self.to_neuroglancer_state()
        return self._viewer

    @property
    def layers(self):
        return self._layers

    @property
    def add_layer(self, layers):
        if isinstance(layers, list):
            self._layers.extend(layers)
        else:
            self._layers.append(layers)
        self._viewer = None

    @property
    def dimensions(self):
        if self._dimensions is None:
            return CoordSpace()
        return self._dimensions

    @dimensions.setter
    def dimensions(self, value):
        if isinstance(value, CoordSpace):
            self._dimensions = value
        else:
            self._dimensions = CoordSpace(value)
        self._viewer = None

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value
        self._viewer = None

    @property
    def scale_imagery(self):
        return self._scale_imagery

    @scale_imagery.setter
    def scale_imagery(self, value):
        self._scale_imagery = value
        self._viewer = None

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

    @selected_layer.setter
    def selected_layer(self, value):
        self._selected_layer = value
        self._viewer = None

    @property
    def selected_layer_visible(self):
        return self._selected_layer_visible

    @selected_layer_visible.setter
    def selected_layer_visible(self, value):
        self._selected_layer_visible = value
        self._viewer = None

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
        self._viewer = None

    @property
    def base_state(self):
        return self._base_state

    @base_state.setter
    def base_state(self, value):
        self._base_state = value
        self._viewer = None

    @property
    def interactive(self):
        return self._interactive

    @interactive.setter
    def interactive(self, value):
        if value != self._interactive:
            self._viewer = None
        self._interactive = value

    def image_layer(
        self,
        name: str,
        source: Optional[str] = None,
        resolution: Optional[Union[list, np.ndarray]] = None,
        client: Optional["caveclient.CAVEclient"] = None,
        shader: str = None,
        **kwargs,
    ) -> Self:
        if resolution is None:
            resolution = self.dimensions
        if client is not None:
            if not source:
                source = client.info.image_source()
            if not resolution:
                resolution = client.info.viewer_resolution()
        img_layer = ImageLayer(
            name=name,
            source=source,
            resolution=resolution,
            shader=shader,
            **kwargs,
        )
        self.add_layer(img_layer)
        return self

    def segmentation_layer(
        self,
        name: str,
        source: Optional[str] = None,
        resolution: Optional[Union[list, np.ndarray]] = None,
        client: Optional["caveclient.CAVEclient"] = None,
        shader: str = None,
        **kwargs,
    ) -> Self:
        if resolution is None:
            resolution = self.dimensions
        if client is not None:
            if not source:
                source = client.info.segmentation_source()
            if not resolution:
                resolution = client.info.viewer_resolution()
        seg_layer = SegmentationLayer(
            name=name,
            source=source,
            resolution=resolution,
            shader=shader,
            **kwargs,
        )
        self.add_layer(seg_layer)
        return self

    def to_neuroglancer_state(self):
        if self.dimensions is None:
            self._dimensions = CoordSpace()
        elif not isinstance(self.dimensions, CoordSpace):
            self._dimensions = CoordSpace(self.dimensions)

        if self.interactive:
            self._viewer = viewer.Viewer()
        else:
            self._viewer = UnservedViewer()
        if self.base_state:
            self._viewer.set_state(self.base_state)

        with self._viewer.txn() as s:
            if self.position:
                s.position = self.position
            if self.layeout:
                s.layout = self.layout
            s.dimensions = self.dimensions.to_neuroglancer()
            s.cross_section_scale = self.scale_imagery
            s.projection_scale = self.scale_3d
            s.show_slices = self.show_slices
            for layer in self.layers:
                layer.to_neuroglancer(s)

        return self._viewer

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
        html: bool = False,
        link_text: str = "Neuroglancer Link",
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
            target_url = self._target_url

        url = neuroglancer.to_url(
            self.viewer.state,
            target_url,
        )
        if html:
            return f'<a href="{url}" target="_blank">{link_text}</a>'
        else:
            return url
