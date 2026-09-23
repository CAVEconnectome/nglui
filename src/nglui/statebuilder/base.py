from __future__ import annotations

import copy
import json
import warnings
import webbrowser
from contextlib import contextmanager
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Union,
    overload,
)

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import attrs
import neuroglancer
import numpy as np
import pyperclip
from IPython.display import HTML
from neuroglancer import viewer, viewer_base

from . import source_info
from .capabilities import Capabilities, capabilities_for_url, parse_capabilities
from .layouts import Layout, Panel, StackLayout, layout_to_json, validate_layout
from .ngl_components import (
    AnnotationLayer,
    CoordSpace,
    DataMap,
    ImageLayer,
    RawLayer,
    SegmentationLayer,
    Source,
)
from .site_utils import MAX_URL_LENGTH, neuroglancer_url
from .tools import Tool, ToolPalette, add_palettes, bind_tools
from .utils import (
    NamedList,
    deep_merge,
    parse_color,
    strip_layers,
    strip_numpy_types,
    strip_state_properties,
)
from .viewer_config import Camera, Orientation, SidePanel

if TYPE_CHECKING:
    import caveclient
    import pandas as pd

_DEFAULT_LAYOUT = "xy-3d"
_DEFAULT_CROSS_SECTION_SCALE = 1.0
_DEFAULT_PROJECTION_SCALE = 50000.0
_DEFAULT_SHOW_SLICES = False


# Top-level presentation options, emitted under the same attribute name on the
# neuroglancer ViewerState. Each maps to its parser.
def _require(kind: type):
    """A parser that rejects values of the wrong type instead of coercing them.

    ``bool("false")`` is True, so coercion would silently invert a mistyped flag.
    """

    def parse(name: str, value):
        if kind is bool and isinstance(value, np.bool_):
            value = bool(value)
        if not isinstance(value, kind) or (
            kind is not bool and isinstance(value, bool)
        ):
            raise TypeError(
                f"{name} must be a {kind.__name__}, got {type(value).__name__} {value!r}."
            )
        return value

    return parse


def _color(name: str, value):
    return parse_color(value)


T = TypeVar("T")


class _DisplayOption(Generic[T]):
    """A top-level presentation option of a ViewerState, e.g. ``vs.title``.

    Declared in the class body, rather than attached in a loop, so editors and type
    checkers see each option. Setting None unsets it.
    """

    def __init__(self, parse: Callable[[str, Any], T], doc: str):
        self.parse = parse
        self.__doc__ = doc

    def __set_name__(self, owner, name: str) -> None:
        self.name = name

    @overload
    def __get__(self, obj: None, objtype: Any = None) -> _DisplayOption[T]: ...

    @overload
    def __get__(self, obj: object, objtype: Any = None) -> Optional[T]: ...

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj._display.get(self.name)

    def __set__(self, obj, value: Any) -> None:
        if value is None:
            obj._display.pop(self.name, None)
            obj._reset_viewer()
        else:
            obj._set_display(**{self.name: value})


# nglui panel name -> attribute of the neuroglancer ViewerState holding its location
_PANELS = {
    "layer_list_panel": "layer_list_panel",
    "statistics_panel": "statistics",
    "help_panel": "help_panel",
    "selected_layer_panel": "selected_layer",
}


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy and pandas types.

    Converts numpy scalars, arrays, and pandas Series to native Python types
    for JSON serialization.
    """

    def default(self, obj):
        """Override default method to handle numpy types.

        Parameters
        ----------
        obj : any
            Object to serialize

        Returns
        -------
        any
            JSON-serializable representation of the object
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        try:
            import pandas as pd

            if isinstance(obj, pd.Series):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


class UnservedViewer(viewer_base.UnsynchronizedViewerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._default_viewer_url = ""

    def get_server_url(self):
        return self._default_viewer_url


class ViewerState:
    title = _DisplayOption[str](_require(str), "Title shown in the browser tab.")
    show_axis_lines = _DisplayOption[bool](
        _require(bool), "Whether the red/green/blue axis lines are drawn."
    )
    show_scale_bar = _DisplayOption[bool](
        _require(bool), "Whether the scale bar is drawn."
    )
    show_default_annotations = _DisplayOption[bool](
        _require(bool),
        "Whether default annotations, such as the cross-section outline in 3d, are drawn.",
    )
    cross_section_background_color = _DisplayOption[str](
        _color, "Background color of the 2d views, as a hex string."
    )
    projection_background_color = _DisplayOption[str](
        _color, "Background color of the 3d view, as a hex string."
    )
    hide_cross_section_background_3d = _DisplayOption[bool](
        _require(bool),
        "Whether the cross-section plane's background is hidden in the 3d view.",
    )
    wire_frame = _DisplayOption[bool](
        _require(bool), "Whether meshes are drawn as wire frames."
    )

    def __init__(
        self,
        target_site: str = None,
        target_url: str = None,
        layers: Optional[list] = None,
        dimensions: Optional[Union[list, CoordSpace]] = None,
        *,
        position: Optional[Union[list, np.ndarray]] = None,
        scale_imagery: Optional[float] = None,
        scale_3d: Optional[float] = None,
        show_slices: Optional[bool] = None,
        selected_layer: Optional[str] = None,
        selected_layer_visible: bool = True,
        layout: Optional[Layout] = None,
        title: Optional[str] = None,
        show_axis_lines: Optional[bool] = None,
        show_scale_bar: Optional[bool] = None,
        show_default_annotations: Optional[bool] = None,
        cross_section_background_color: Optional[Union[str, tuple]] = None,
        projection_background_color: Optional[Union[str, tuple]] = None,
        hide_cross_section_background_3d: Optional[bool] = None,
        wire_frame: Optional[bool] = None,
        camera: Optional[Camera] = None,
        extra: Optional[dict] = None,
        base_state: Optional[dict] = None,
        interactive: bool = False,
        infer_coordinates: bool = True,
        client: Optional["caveclient.CAVEclient"] = None,
        capabilities: Optional[Union[str, "Capabilities"]] = None,
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
            The name of the selected layer, whose settings panel is shown. If None,
            no layer is selected.
        selected_layer_visible : bool
            Whether the selected layer's panel is open. Default is True.
        layout : str, Panel, or StackLayout
            The panel layout of the viewer: a preset (default "xy-3d"), or rows and
            columns of `Panel`s showing different layers. See `set_layout`.
        title : str, optional
            Title shown in the browser tab.
        show_axis_lines : bool, optional
            Whether to draw the red/green/blue axis lines. Neuroglancer's default is True.
        show_scale_bar : bool, optional
            Whether to draw the scale bar. Neuroglancer's default is True.
        show_default_annotations : bool, optional
            Whether to draw the default annotations, such as the outline of the
            cross-section plane in the 3d view. Neuroglancer's default is True.
        cross_section_background_color : str or tuple, optional
            Background color of the 2d views, as a name, hex string, or RGB (0-1 or 0-255).
        projection_background_color : str or tuple, optional
            Background color of the 3d view, as a name, hex string, or RGB (0-1 or 0-255).
        hide_cross_section_background_3d : bool, optional
            Whether to hide the background of the cross-section plane where it
            appears in the 3d view.
        wire_frame : bool, optional
            Whether to render meshes as wire frames.
        camera : Camera, optional
            Zoom, orientation, and depth for the 2d and 3d views. Its scales take
            precedence over `scale_imagery` and `scale_3d`. See `set_camera`.
        extra : dict, optional
            Raw Neuroglancer state JSON (camelCase keys) for options nglui does not
            wrap, deep-merged over the built state last so it wins over anything
            nglui sets. A value of None removes that key.
            For example, ``{"gpuMemoryLimit": 2_000_000_000}``.
        base_state : dict
            The base state of the viewer. If None, the default state will be used.
            When given, its values for `scale_imagery`, `scale_3d`, `show_slices`,
            and `layout` are kept unless those are set explicitly.
        interactive : bool
            Whether the viewer is interactive. Default is False.
        infer_coordinates : bool
            Whether to infer resolution and position from the source information using CloudVolume. Default is True.
        client : caveclient.CAVEclient, optional
            A CAVE client to use for configuration. If provided, it will be used by default in functions that can accept it.
        capabilities : str or Capabilities, optional
            What the target Neuroglancer deployment supports, which determines how
            annotation tags are encoded. Either a `Capabilities` or an alias such as
            ``"main"`` or ``"legacy"``. If None, nglui probes the target site and
            falls back to the conservative default when it cannot tell.
        """

        self._target_site = target_site
        self._target_url = neuroglancer_url(target_url, target_site)
        self._capabilities = parse_capabilities(capabilities)
        self._layers = NamedList(layers) if layers else NamedList()
        self._dimensions = dimensions
        self._position = position
        # View settings the user set explicitly, which override a base state. The
        # rest fall back to nglui's defaults only when there is no base state.
        # Camera fields are None until set, so they track this themselves.
        self._explicit_view = {
            name
            for name, value in [("show_slices", show_slices), ("layout", layout)]
            if value is not None
        }
        self._camera = Camera(
            cross_section_scale=scale_imagery, projection_scale=scale_3d
        )
        if camera is not None:
            for shorthand, value, field_name in [
                ("scale_imagery", scale_imagery, "cross_section_scale"),
                ("scale_3d", scale_3d, "projection_scale"),
            ]:
                from_camera = getattr(camera, field_name)
                if (
                    value is not None
                    and from_camera is not None
                    and float(value) != from_camera
                ):
                    raise ValueError(
                        f"{shorthand}={value} conflicts with camera.{field_name}="
                        f"{from_camera}; they set the same zoom, so give only one."
                    )
            self._camera = self._camera.merge(camera)
        self._show_slices = _DEFAULT_SHOW_SLICES
        self._layout = _DEFAULT_LAYOUT
        self._display = {}
        self._set_display(
            title=title,
            show_axis_lines=show_axis_lines,
            show_scale_bar=show_scale_bar,
            show_default_annotations=show_default_annotations,
            cross_section_background_color=cross_section_background_color,
            projection_background_color=projection_background_color,
            hide_cross_section_background_3d=hide_cross_section_background_3d,
            wire_frame=wire_frame,
        )
        self._panels: dict[str, SidePanel] = {}
        self._tools: list[Tool] = []
        self._palettes: dict[str, ToolPalette] = {}
        self._extra = strip_numpy_types(dict(extra)) if extra else {}
        if show_slices is not None:
            self._show_slices = show_slices
        if layout is not None:
            validate_layout(layout)
            self._layout = layout
        self._selected_layer = selected_layer
        self._selected_layer_visible = selected_layer_visible
        self._base_state = base_state
        self._interactive = interactive
        self._saved_state_url = None
        self._viewer = None
        self._source_info = None
        self._infer_coordinates = infer_coordinates
        self._client = client

    def add_layers_from_client(
        self,
        client: Optional["caveclient.CAVEclient"] = None,
        imagery: Union[bool, str] = True,
        segmentation: Union[bool, str] = True,
        skeleton_source: bool = True,
        resolution: bool = True,
        target_url: bool = False,
        selected_alpha: Optional[float] = None,
        alpha_3d: Optional[float] = None,
        mesh_silhouette: Optional[float] = None,
        imagery_kws: Optional[dict] = None,
        segmentation_kws: Optional[dict] = None,
    ) -> Self:
        """Configure the viewer with information from a CaveClient.
        Can set the target URL, viewer resolution, and add image and segmentation layers.

        Parameters
        ----------
        client : caveclient.CAVEclient, optional
            The client to use for configuration. If None, will use the client set in the viewer state if allowed.
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
        selected_alpha : Optional[float], optional
            The alpha value for the segmentation layer, by default None.
        alpha_3d : Optional[float], optional
            The alpha value for 3D meshes, by default None.
        mesh_silhouette : Optional[float], optional
            The mesh silhouette value, by default None.
        imagery_kws : Optional[dict], optional
            Additional keyword arguments to pass to the image layer constructor.
        segmentation_kws : Optional[dict], optional
            Additional keyword arguments to pass to the segmentation layer constructor.P

        Returns
        -------
        Self
            The updated viewer state object.
        """
        if client is None:
            client = self._client
        if client is None:
            raise ValueError(
                "No client provided and no client set in the viewer state. "
                "Please provide a CAVEclient instance."
            )

        if target_url:
            self._target_url = client.info.viewer_site()
        if resolution:
            self.dimensions = client.info.viewer_resolution()
        if imagery:
            if isinstance(imagery, str):
                self.add_image_layer(
                    source=client.info.image_source(),
                    name=imagery,
                    **(imagery_kws or {}),
                )
            else:
                self.add_image_layer(
                    source=client.info.image_source(),
                    **(imagery_kws or {}),
                )
        if segmentation:
            seg_source = [client.info.segmentation_source()]
            skel_source_path = client.info.get_datastack_info().get("skeleton_source")

            if skeleton_source and skel_source_path:
                seg_source.append(skel_source_path)

            if isinstance(segmentation, str):
                self.add_segmentation_layer(
                    source=seg_source,
                    name=segmentation,
                    selected_alpha=selected_alpha,
                    alpha_3d=alpha_3d,
                    mesh_silhouette=mesh_silhouette,
                    **(segmentation_kws or {}),
                )
            else:
                self.add_segmentation_layer(
                    source=seg_source,
                    selected_alpha=selected_alpha,
                    alpha_3d=alpha_3d,
                    mesh_silhouette=mesh_silhouette,
                    **(segmentation_kws or {}),
                )
        return self

    @property
    def viewer(self):
        """Get the Neuroglancer viewer object."""
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
        position = source_info.suggest_position(self.source_info, resolution)
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
        resolution = source_info.suggest_resolution(self.source_info)
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
        """Zoom of the 2d views; the camera's `cross_section_scale`."""
        if self._camera.cross_section_scale is None:
            return _DEFAULT_CROSS_SECTION_SCALE
        return self._camera.cross_section_scale

    @scale_imagery.setter
    def scale_imagery(self, value):
        self.set_camera(cross_section_scale=value)

    @property
    def scale_3d(self):
        """Zoom of the 3d view; the camera's `projection_scale`."""
        if self._camera.projection_scale is None:
            return _DEFAULT_PROJECTION_SCALE
        return self._camera.projection_scale

    @scale_3d.setter
    def scale_3d(self, value):
        self.set_camera(projection_scale=value)

    @property
    def camera(self) -> Camera:
        """The camera settings that have been set explicitly."""
        return self._camera

    @camera.setter
    def camera(self, value: Camera):
        self._camera = value if value is not None else Camera()
        self._reset_viewer()

    def set_camera(
        self,
        camera: Optional[Camera] = None,
        *,
        cross_section_scale: Optional[float] = None,
        cross_section_orientation: Optional[Orientation] = None,
        cross_section_depth: Optional[float] = None,
        projection_scale: Optional[float] = None,
        projection_orientation: Optional[Orientation] = None,
        projection_depth: Optional[float] = None,
    ) -> Self:
        """Set the zoom, orientation, and depth of the 2d and 3d views.

        Only the values given are changed; the rest keep their current settings.

        Parameters
        ----------
        camera : Camera, optional
            A camera whose set fields are applied, for reusing one view across states.
        cross_section_scale : float, optional
            Zoom of the 2d views. Same as `scale_imagery`.
        cross_section_orientation : str or sequence of float, optional
            Orientation of the 2d views, as a plane name (``"xy"``, ``"xz"``,
            ``"yz"``) or an ``[x, y, z, w]`` quaternion.
        cross_section_depth : float, optional
            Depth of field of the 2d views.
        projection_scale : float, optional
            Zoom of the 3d view. Same as `scale_3d`.
        projection_orientation : str or sequence of float, optional
            Orientation of the 3d view, as a plane name or quaternion.
        projection_depth : float, optional
            Depth of the 3d view's clipping volume.

        Returns
        -------
        ViewerState
            The viewer state, for chaining.

        Examples
        --------
        >>> vs.set_camera(projection_orientation="xz", projection_scale=5000)
        """
        if camera is not None:
            self._camera = self._camera.merge(camera)
        self._camera = self._camera.merge(
            Camera(
                cross_section_scale=cross_section_scale,
                cross_section_orientation=cross_section_orientation,
                cross_section_depth=cross_section_depth,
                projection_scale=projection_scale,
                projection_orientation=projection_orientation,
                projection_depth=projection_depth,
            )
        )
        self._reset_viewer()
        return self

    def _all_layer_names(self, s) -> list:
        """Names of every layer the built state will have: the base state's and ours."""
        names = [layer.name for layer in s.layers]
        return names + [layer.name for layer in self.layers if layer.name not in names]

    def _set_display(self, **options) -> None:
        """Set presentation options, ignoring any passed as None."""
        for name, value in options.items():
            if value is not None:
                self._display[name] = getattr(type(self), name).parse(name, value)
        self._reset_viewer()

    def set_panels(
        self,
        *,
        layer_list_panel: Optional[Union[bool, SidePanel, dict]] = None,
        statistics_panel: Optional[Union[bool, SidePanel, dict]] = None,
        help_panel: Optional[Union[bool, SidePanel, dict]] = None,
        selected_layer_panel: Optional[Union[bool, SidePanel, dict]] = None,
    ) -> Self:
        """Open, close, or place Neuroglancer's side panels.

        Each panel takes a `SidePanel`, a dict of its fields, or a bool as shorthand
        for ``SidePanel(visible=...)``. Only the panels and fields given are changed.

        Parameters
        ----------
        layer_list_panel : bool, SidePanel, or dict, optional
            The panel listing all layers.
        statistics_panel : bool, SidePanel, or dict, optional
            The chunk-download statistics panel.
        help_panel : bool, SidePanel, or dict, optional
            The keyboard and mouse bindings help panel.
        selected_layer_panel : bool, SidePanel, or dict, optional
            The panel for the selected layer's settings. Which layer it shows is set
            with `set_selected_layer`; ``visible`` here takes precedence over it.

        Returns
        -------
        ViewerState
            The viewer state, for chaining.

        Examples
        --------
        >>> vs.set_panels(
        ...     layer_list_panel=True,
        ...     selected_layer_panel=SidePanel(side="right", size=500),
        ... )
        """
        given = dict(
            layer_list_panel=layer_list_panel,
            statistics_panel=statistics_panel,
            help_panel=help_panel,
            selected_layer_panel=selected_layer_panel,
        )
        for name, value in given.items():
            if value is None:
                continue
            panel = SidePanel.coerce(value)
            if name in self._panels:
                panel = self._panels[name].merge(panel)
            self._panels[name] = panel
        self._reset_viewer()
        return self

    @property
    def panels(self) -> dict[str, SidePanel]:
        """Side panel settings, keyed by panel name."""
        return dict(self._panels)

    def add_tools(
        self, *tools: Tool, palette: Optional[Union[str, ToolPalette]] = None
    ) -> Self:
        """Bind tools to keys, and optionally show them in a tool palette.

        Keys are allocated across the whole viewer when the state is built, since
        Neuroglancer has one key namespace for every layer: a tool's explicit `key`
        must be free, and tools with ``key=None`` get the next free letter.

        Parameters
        ----------
        *tools : Tool
            Tools from `nglui.statebuilder.tools`. Layer tools need `layer` set, as
            a layer name or object. Use ``key=False`` for a palette-only tool.
        palette : str or ToolPalette, optional
            A palette to also show these tools in, by name or as a `ToolPalette`
            for placement. Tools are added to an existing palette of that name.

        Returns
        -------
        ViewerState
            The viewer state, for chaining.

        Examples
        --------
        >>> from nglui.statebuilder import tools
        >>> vs.add_tools(
        ...     tools.SelectSegments(layer="seg", key="S"),
        ...     tools.MeshSilhouette(layer="seg"),
        ...     palette=tools.ToolPalette("Review", side="right"),
        ... )
        """
        for tool in tools:
            if not isinstance(tool, Tool):
                raise TypeError(
                    f"Expected a Tool from nglui.statebuilder.tools, got {tool!r}."
                )
        self._tools.extend(tools)
        if palette is not None:
            if isinstance(palette, str):
                palette = ToolPalette(palette)
            self.add_tool_palette(palette)
            self._palettes[palette.name] = self._palettes[palette.name].with_tools(
                *tools
            )
        self._reset_viewer()
        return self

    def add_tool_palette(self, palette: ToolPalette) -> Self:
        """Add a tool palette, or update the placement of one with the same name.

        Tools listed in `palette` are shown in it but not bound to keys; use
        `add_tools` to bind them as well.

        Parameters
        ----------
        palette : ToolPalette
            The palette to add.

        Returns
        -------
        ViewerState
            The viewer state, for chaining.
        """
        existing = self._palettes.get(palette.name)
        if existing is not None:
            palette = attrs.evolve(palette, tools=(*existing.tools, *palette.tools))
        self._palettes[palette.name] = palette
        self._reset_viewer()
        return self

    @property
    def tools(self) -> list[Tool]:
        """Tools added with `add_tools`."""
        return list(self._tools)

    @property
    def tool_palettes(self) -> dict[str, ToolPalette]:
        """Tool palettes, keyed by name."""
        return dict(self._palettes)

    @property
    def extra(self) -> dict:
        """Raw Neuroglancer state JSON merged over the built state last.

        Read-only; assign a new dict, or use ``set_viewer_properties(extra=...)`` to
        merge into it, so the cached state is rebuilt.
        """
        return MappingProxyType(copy.deepcopy(self._extra))

    @extra.setter
    def extra(self, value: Optional[dict]):
        self._extra = strip_numpy_types(dict(value)) if value else {}
        self._reset_viewer()

    @property
    def show_slices(self):
        return self._show_slices

    @show_slices.setter
    def show_slices(self, value):
        self._show_slices = value
        self._explicit_view.add("show_slices")
        self._reset_viewer()

    @property
    def selected_layer(self):
        return {
            "layer": self._selected_layer,
            "visible": self._selected_layer_visible,
        }

    @selected_layer.setter
    def selected_layer(
        self, value: Union[str, ImageLayer, SegmentationLayer, AnnotationLayer]
    ):
        self.set_selected_layer(value)

    def set_selected_layer(
        self,
        selected_layer: Union[str, ImageLayer, SegmentationLayer, AnnotationLayer],
        visible: bool = True,
    ) -> Self:
        """Select a layer, showing its settings in the selected-layer panel.

        Parameters
        ----------
        selected_layer : str or Layer
            The layer, by name or object.
        visible : bool, optional
            Whether the panel is open. Default is True. Place the panel with
            ``set_panels(selected_layer_panel=SidePanel(...))``.

        Returns
        -------
        ViewerState
            The viewer state, for chaining.
        """
        if isinstance(selected_layer, str):
            self._selected_layer = selected_layer
        else:
            self._selected_layer = selected_layer.name
        self._selected_layer_visible = visible
        self._reset_viewer()
        return self

    @property
    def selected_layer_visible(self):
        return self._selected_layer_visible

    @selected_layer_visible.setter
    def selected_layer_visible(self, value: bool):
        self._selected_layer_visible = value
        self._reset_viewer()

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, value):
        validate_layout(value)
        self._layout = value
        self._explicit_view.add("layout")
        self._reset_viewer()

    def set_layout(self, layout: Layout) -> Self:
        """Set the panel layout: a preset, or rows and columns of `Panel`s.

        Parameters
        ----------
        layout : str, Panel, or StackLayout
            A preset such as "xy-3d" or "4panel", which shows every layer in every
            view; or a `Panel` or `row`/`column` of panels, each showing its own
            layers with its own view type and optionally its own camera.

        Returns
        -------
        ViewerState
            The viewer state, for chaining.

        Examples
        --------
        >>> from nglui.statebuilder import Panel, row
        >>> vs.set_layout(row(Panel([img, seg], layout="xy"), Panel([seg], layout="3d")))
        """
        self.layout = layout
        return self

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
        layout: Optional[Layout] = None,
        title: Optional[str] = None,
        show_axis_lines: Optional[bool] = None,
        show_scale_bar: Optional[bool] = None,
        show_default_annotations: Optional[bool] = None,
        cross_section_background_color: Optional[Union[str, tuple]] = None,
        projection_background_color: Optional[Union[str, tuple]] = None,
        hide_cross_section_background_3d: Optional[bool] = None,
        wire_frame: Optional[bool] = None,
        camera: Optional[Camera] = None,
        extra: Optional[dict] = None,
        base_state: Optional[dict] = None,
        interactive: Optional[bool] = None,
        infer_coordinates: Optional[bool] = None,
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
        layout : str, Panel, or StackLayout, optional
            The panel layout of the viewer; see `set_layout`.
        title, show_axis_lines, show_scale_bar, show_default_annotations : optional
            Presentation options; see the `ViewerState` constructor.
        cross_section_background_color, projection_background_color : optional
            Background colors; see the `ViewerState` constructor.
        hide_cross_section_background_3d, wire_frame : bool, optional
            Presentation options; see the `ViewerState` constructor.
        camera : Camera, optional
            Camera settings to apply; see `set_camera`.
        extra : dict, optional
            Raw Neuroglancer state JSON, deep-merged into any existing `extra`.
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
        self._set_display(
            title=title,
            show_axis_lines=show_axis_lines,
            show_scale_bar=show_scale_bar,
            show_default_annotations=show_default_annotations,
            cross_section_background_color=cross_section_background_color,
            projection_background_color=projection_background_color,
            hide_cross_section_background_3d=hide_cross_section_background_3d,
            wire_frame=wire_frame,
        )
        if camera is not None:
            self.set_camera(camera)
        if extra is not None:
            # Keep None values: they delete keys from the built state at build time
            self.extra = deep_merge(self._extra, extra, remove_none=False)
        if base_state is not None:
            self.base_state = base_state
        if interactive is not None:
            self.interactive = interactive
        if infer_coordinates is not None:
            self.infer_coordinates = infer_coordinates
        return self

    def add_raw_layer(
        self,
        data: dict,
        name: Optional[str] = None,
        visible: Optional[bool] = None,
        archived: Optional[bool] = None,
        source_remap: Optional[dict] = None,
    ) -> Self:
        """Add a layer based on a raw state dictionary. Useful when copying existing layers, but offers limited customization.

        Parameters
        ----------
        data : dict
            The raw state dictionary for the layer.
        name : str, optional
            The new name of the layer, if you want to change it.
        visible : bool, optional
            Whether the layer is visible, if you want to change it.
        archived : bool, optional
            Whether the layer is archived, if you want to change it.
        source_remap: dict, optional
            A dictionary mapping source cloudpaths to new cloudpaths, for example to update the layer's data source without changing anything else.

        Returns
        -------
        Self
            The updated viewer state object.
        """
        layer = RawLayer(
            json_data=data, name=name, visible=visible, archived=archived
        ).remap_sources(source_remap)
        self.add_layer(layer)
        return self

    def add_image_layer(
        self,
        source: Union[str, list, Source],
        name: str = "imagery",
        resolution: Optional[Union[list, np.ndarray]] = None,
        **kwargs,
    ) -> Self:
        """Add an image layer to the viewer.

        Parameters
        ----------
        source : str
            The source path for the image layer.
        name : str, optional
            The name of the image layer, by default "imagery".
        resolution : Optional[Union[list, np.ndarray]], optional
            The resolution of the image layer. If None, the viewer's current resolution will be used or it will be inferred from the source (if cloud-volume is installed).
        **kwargs : dict, optional
            Additional keyword arguments to pass to the image layer constructor.
        """
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
        source: Union[str, list, Source],
        name: str = "segmentation",
        resolution: Optional[Union[list, np.ndarray]] = None,
        segments: Optional[list] = None,
        **kwargs,
    ) -> Self:
        """Add a segmentation layer to the viewer.

        Parameters
        ----------
        source : str
            The source path for the segmentation layer.
        name : str, optional
            The name of the segmentation layer, by default "segmentation".
        resolution : Optional[Union[list, np.ndarray]], optional
            The resolution of the segmentation layer. If None, the viewer's current resolution will be used or it will be inferred from the source (if cloud-volume is installed).
        segments: list, optional
            A list of segments to set as selected in the segmentation layer.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the segmentation layer constructor.
        """
        if resolution is None:
            resolution = self.dimensions
        seg_layer = SegmentationLayer(
            name=name,
            source=source,
            resolution=resolution,
            segments=segments,
            **kwargs,
        )
        self.add_layer(seg_layer)
        return self

    def add_segments(
        self,
        segments: Union[list, np.ndarray, "pd.DataFrame", DataMap],
        visible: Optional[Union[list, np.ndarray]] = None,
        segment_colors: Optional[Union[str, list, dict]] = None,
        name: Optional[str] = None,
    ) -> Self:
        """Add segments directly to an existing segmentation layer.
        By default, it will use the first segmentation layer found in the viewer, otherwise you must specify a layer name.

        Parameters
        ----------
        segments : list or dict or VisibleSegments
            The segments to add. If a dict, the keys are the segment IDs and the values are the boolean visibility.
        visible: list, optional
            The visibility of the segments, assumed to be True if not provided.
            Should be the same length as segments, and segments should be a list of the same length.
        segment_colors : Union[str, list, dict], optional
            The color(s) to assign to the segments. If a string or list, all segments will be assigned the same color.
            A list is assumed to be a color tuple.
            If a dict, the keys are segment IDs and the values are colors.
        name: Optional[str]
            The name of the segmentation layer to add segments to.
            If None, it will use the first segmentation layer found in the viewer.

        Returns
        -------
        The viewer state object with the added segments.
        """
        if name is None:
            for l in self.layers:
                if isinstance(l, SegmentationLayer):
                    name = l.name
                    break
            else:
                raise ValueError("No segmentation layer found in the viewer.")
        self.layers[name].add_segments(
            segments,
            visible=visible,
        )
        if segment_colors is not None:
            if not isinstance(segment_colors, dict):
                segment_colors = {s: segment_colors for s in segments}
            self.layers[name].add_segment_colors(segment_colors)
        return self

    def add_segment_properties(
        self,
        data: Union[pd.DataFrame, DataMap],
        id_column: str = "pt_root_id",
        client: Optional["caveclient.CAVEclient"] = None,
        label_column: Optional[Union[str, list]] = None,
        description_column: Optional[str] = None,
        string_columns: Optional[list] = None,
        number_columns: Optional[list] = None,
        tag_value_columns: Optional[list] = None,
        tag_bool_columns: Optional[list] = None,
        tag_descriptions: Optional[list] = None,
        allow_disambiguation: bool = True,
        label_separator: str = "_",
        label_format_map: Optional[str] = None,
        prepend_col_name: bool = False,
        random_columns: Optional[int] = None,
        random_column_prefix: str = None,
        dry_run: bool = False,
        name: Optional[str] = None,
    ) -> Self:
        """Upload segment properties and add to the layer.
        If you already have a segment properties cloud path, use `add_source` to add it to the layer.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the segment properties.
        id_column : str, optional
            The column name for the segment IDs. Default is 'pt_root_id'.
        client : CAVEclient
            The CAVEclient object needed to upload .
        label_column : str or list, optional
            The column name(s) for the segment labels. Default is None.
        description_column : str, optional
            The column name for the segment descriptions. Default is None.
        string_columns : list, optional
            The column names for string properties. Default is None.
        number_columns : list, optional
            The column names for number properties. Default is None.
        tag_value_columns : list, optional
            The column names for tag value properties. Default is None.
        tag_bool_columns : list, optional
            The column names for tag boolean properties. Default is None.
        tag_descriptions : list, optional
            The descriptions for the tags. Default is None.
        allow_disambiguation : bool, optional
            Whether to allow disambiguation of segment IDs. Default is True.
        label_separator : str, optional
            The separator for label formatting. Default is "_".
        label_format_map : str, optional
            The format map for labels. Default is None.
        prepend_col_name: bool, optional
            Whether to prepend the column name to the label. Default is False.
        random_columns: int, optional
            Number of random columns to add. Default is None.
        random_column_prefix: str, optional
            Name prefix of the random columns. Default is None.
        dry_run: bool, optional
            If dry run is true, build but do not actually upload and instead use a placeholder source text. Default is False.
        name: str, optional
            Layer name. Must be specified if there is more than one segmentation layer in the state.

        Returns
        -------
        Self
        """

        if name is None:
            for l in self.layers:
                if isinstance(l, SegmentationLayer):
                    name = l.name
                    break
            else:
                raise ValueError("No segmentation layer found in the viewer.")

        if client is None:
            client = self._client
        if client is None:
            raise ValueError("Client must be specified.")

        self.layers[name].add_segment_properties(
            data,
            id_column=id_column,
            client=client,
            label_column=label_column,
            description_column=description_column,
            string_columns=string_columns,
            number_columns=number_columns,
            tag_value_columns=tag_value_columns,
            tag_bool_columns=tag_bool_columns,
            tag_descriptions=tag_descriptions,
            allow_disambiguation=allow_disambiguation,
            label_separator=label_separator,
            label_format_map=label_format_map,
            prepend_col_name=prepend_col_name,
            random_columns=random_columns,
            random_column_prefix=random_column_prefix,
            dry_run=dry_run,
        )

        return self

    def add_segments_from_data(
        self,
        data: Union[pd.DataFrame, DataMap],
        segment_column: str,
        visible_column: str = None,
        color_column: str = None,
        name: Optional[str] = None,
    ) -> Self:
        """Add segments from a DataFrame or DataMap to an existing segmentation layer.
        By default, it will use the first segmentation layer found in the viewer, otherwise you must specify a layer name.

        Parameters
        ----------
        data : Union[pd.DataFrame, DataMap]
            The data containing segment information or a datamap to be filled in later
        segment_column : str
            The name of the column containing segment IDs in the DataFrame.
        visible_column : str, optional
            The name of the column containing visibility information in the DataFrame.
            If not provided, all segments will be considered visible.
        color_column : str, optional
            The name of the column containing color information for the segments.
        name: Optional[str]
            The name of the segmentation layer to add segments to.
            If None, it will use the first segmentation layer found in the viewer.
        """

        if name is None:
            for l in self.layers:
                if isinstance(l, SegmentationLayer):
                    name = l.name
                    break
            else:
                raise ValueError("No segmentation layer found in the viewer.")
        self.layers[name].add_segments_from_data(
            data,
            segment_column=segment_column,
            visible_column=visible_column,
            color_column=color_column,
        )
        return self

    def add_annotation_layer(
        self,
        name: str = "annotation",
        source: Optional[Union[str, list, Source]] = None,
        resolution: Optional[Union[list, np.ndarray]] = None,
        tags: Optional[list] = None,
        linked_segmentation: Union[str, bool, dict] = True,
        shader: str = None,
        capabilities: Optional[Union[str, Capabilities]] = None,
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
        capabilities : str or Capabilities, optional
            Override how this layer's tags are encoded, regardless of what the viewer
            state resolved. See `ViewerState` for details.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the annotation layer constructor.

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
        if resolution is None:
            resolution = self.dimensions
        anno_layer = AnnotationLayer(
            name=name,
            source=source,
            resolution=resolution,
            tags=tags,
            linked_segmentation=linked_segmentation,
            shader=shader,
            capabilities=capabilities,
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
        if datamap is None:
            yield self
        else:
            if not isinstance(datamap, dict):
                datamap = {None: datamap}
            viewer_copy = copy.deepcopy(self)
            viewer_copy.map(datamap, inplace=True)
            yield viewer_copy

    def _apply_datamaps(self, datamap: dict):
        for layer in self.layers:
            layer._apply_datamaps(datamap)

    def add_annotation_source(
        self,
        source: Union[str, list, Source],
        name: str = "annotation",
        linked_segmentation: Union[str, bool] = True,
        shader: Optional[str] = None,
    ) -> Self:
        """Add a precomputed annotation source to the viewer.

        Parameters
        ----------
        source : str
            The source path for the annotation layer.
        name : str, optional
            The name of the annotation layer, by default "annotation".
        shader : Optional[str], optional
            The shader to use for the annotation layer, by default None.

        Returns
        -------
        Self
            The viewer state object with the added annotation layer.
        """
        self.add_annotation_layer(
            source=source,
            name=name,
            linked_segmentation=linked_segmentation,
            shader=shader,
        )
        return self

    def _get_or_create_annotation_layer(
        self, name: str, layer_kwargs: dict, **create_kwargs
    ) -> AnnotationLayer:
        """The annotation layer called `name`, creating it if it doesn't exist.

        `create_kwargs` and `layer_kwargs` are only used to create the layer; the
        dataframe helpers raise rather than silently drop `layer_kwargs` given for
        a layer that already exists.
        """
        if name not in self.layer_names:
            layer = AnnotationLayer(name=name, **create_kwargs, **layer_kwargs)
            self.add_layer(layer)
            return layer
        layer = self.get_layer(name)
        if not isinstance(layer, AnnotationLayer):
            raise ValueError(
                f"Layer '{name}' already exists but is not an AnnotationLayer."
            )
        if layer_kwargs:
            raise ValueError(
                f"Layer '{name}' already exists, so layer options "
                f"{sorted(layer_kwargs)} cannot be applied. Pass them when the "
                "layer is first created, or set them on the layer object."
            )
        return layer

    def add_points(
        self,
        data: Optional[Union[list, np.ndarray, pd.DataFrame, DataMap]] = None,
        name: str = "annotation",
        point_column: Optional[Union[str, list]] = None,
        segment_column: Optional[str] = None,
        description_column: Optional[str] = None,
        tag_column: Optional[str] = None,
        tag_bools: Optional[list] = None,
        data_resolution: Optional[list] = None,
        tags: Optional[list] = None,
        linked_segmentation: Union[str, bool] = True,
        shader: Optional[str] = None,
        color: Optional[str] = None,
        swap_visible_segments_on_move: Union[bool, Literal["auto"]] = "auto",
        **layer_kwargs: Any,
    ) -> Self:
        """Add points to an existing annotation layer or create a new one.
        Parameters
        ----------
        data : Union[list, np.ndarray, pd.DataFrame]
            The data to add to the annotation layer.
        name : str, optional
            The name of the annotation layer, by default "annotation".
        point_column : Optional[Union[str, list]] = None,
            The name of the column containing point coordinates, by default None.
            If you have a split point representation in the format {prefix}_x, {prefix}_y, {prefix}_z,
            you can specify this with "prefix" only and the suffixes will be ignored.
            If you provide a list, they will be used explicitly as the x,y, and z columns.
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
        swap_visible_segments_on_move: bool or "auto", optional
            If True, will swap the visibility of segments when moving points.
            If "auto" (default), will swap only when segment_column is provided.
        **layer_kwargs
            Further options for the annotation layer when this call creates it, such
            as `active_tool`, `shader_controls`, `filter_by_segmentation`, `tools`, or
            `extra`. See `AnnotationLayer`. Passing any when the layer already exists
            raises, rather than silently ignoring them.
        Returns
        -------
        Self
            The viewer state object with the added points.
        """
        if swap_visible_segments_on_move == "auto":
            swap_visible_segments_on_move = segment_column is not None

        layer = self._get_or_create_annotation_layer(
            name,
            layer_kwargs,
            resolution=data_resolution,
            tags=tags,
            linked_segmentation=linked_segmentation,
            color=color,
            shader=shader,
            swap_visible_segments_on_move=swap_visible_segments_on_move,
        )
        layer.add_points(
            data,
            point_column=point_column,
            segment_column=segment_column,
            description_column=description_column,
            tag_column=tag_column,
            tag_bools=tag_bools,
            data_resolution=data_resolution,
        )
        return self

    def add_lines(
        self,
        data: Optional[
            Union[list, tuple[np.ndarray, np.ndarray], pd.DataFrame, DataMap]
        ] = None,
        name: str = "annotation",
        point_a_column: Optional[Union[str, list]] = None,
        point_b_column: Optional[Union[str, list]] = None,
        segment_column: Optional[str] = None,
        description_column: Optional[str] = None,
        tag_column: Optional[str] = None,
        tag_bools: Optional[list] = None,
        data_resolution: Optional[list] = None,
        tags: Optional[list] = None,
        linked_segmentation: Union[str, bool] = True,
        shader: Optional[str] = None,
        color: Optional[str] = None,
        swap_visible_segments_on_move: Union[bool, Literal["auto"]] = "auto",
        **layer_kwargs: Any,
    ) -> Self:
        """Add lines to an existing annotation layer or create a new one.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the line annotations or a tuple of (start_points, end_points) as Nx3 arrays.
        name : str, optional
            The name of the annotation layer, by default "annotation".
        point_a_column : str
            The column name for the start point coordinates.
            If you have a split point representation in the format {prefix}_x, {prefix}_y, {prefix}_z,
            you can specify this with "prefix" only and the suffixes will be ignored.
            If you provide a list, they will be used explicitly as the x,y, and z columns.
        point_b_column : str
            The column name for the end point coordinates, formatted similarly as `point_a_column`.
        segment_column : Optional[str], optional
            The name of the column containing linked segment IDs, by default None.
        description_column : Optional[str], optional
            The name of the column containing descriptions, by default None.
        tag_column : Optional[str], optional
            The name of a column containing tags, by default None.
        tag_bools : Optional[list], optional
            A list of column names indicating tags as booleans, by default None.
        data_resolution : Optional[list], optional
            The resolution of the data, by default None.
        tags : Optional[list], optional
            A list of tags to add to the annotation layer, by default None.
        linked_segmentation : str or bool, optional
            The name of the segmentation layer to link to, by default None.
            If True, will link to the first segmentation layer found in the viewer.
            If False, will not link to any segmentation layer.
        shader : Optional[str], optional
            The shader to use for the annotation layer, by default None.
        color : Optional[str], optional
            The color to use for the lines, by default None.
        swap_visible_segments_on_move : bool or "auto", optional
            If True, will swap the visibility of segments when moving lines.
            If "auto" (default), will swap only when segment_column is provided.
        **layer_kwargs
            Further options for the annotation layer when this call creates it, such
            as `active_tool`, `shader_controls`, `filter_by_segmentation`, `tools`, or
            `extra`. See `AnnotationLayer`. Passing any when the layer already exists
            raises, rather than silently ignoring them.
        Returns
        -------
        Self
            The viewer state object with the added lines.
        """
        if swap_visible_segments_on_move == "auto":
            swap_visible_segments_on_move = segment_column is not None

        layer = self._get_or_create_annotation_layer(
            name,
            layer_kwargs,
            resolution=data_resolution,
            tags=tags,
            linked_segmentation=linked_segmentation,
            color=color,
            shader=shader,
            swap_visible_segments_on_move=swap_visible_segments_on_move,
        )
        layer.add_lines(
            data,
            point_a_column=point_a_column,
            point_b_column=point_b_column,
            segment_column=segment_column,
            description_column=description_column,
            tag_column=tag_column,
            tag_bools=tag_bools,
            data_resolution=data_resolution,
        )
        return self

    def add_ellipsoids(
        self,
        data: Optional[pd.DataFrame] = None,
        name: str = "annotation",
        center_column: str = None,
        radii_column: str = None,
        segment_column: Optional[str] = None,
        description_column: Optional[str] = None,
        tag_column: Optional[str] = None,
        tag_bools: Optional[list] = None,
        data_resolution: Optional[list] = None,
        tags: Optional[list] = None,
        linked_segmentation: Union[str, bool] = True,
        shader: Optional[str] = None,
        color: Optional[str] = None,
        swap_visible_segments_on_move: Union[bool, Literal["auto"]] = "auto",
        **layer_kwargs: Any,
    ) -> Self:
        """Add ellipsoid annotations to an existing annotation layer or create a new one.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the ellipsoid annotations.
        name : str, optional
            The name of the annotation layer, by default "annotation".
        center_column : str
            The column name for the ellipsoid center coordinates.
        radii_column : str
            The column name for the ellipsoid radii.
        segment_column : Optional[str], optional
            The name of the column containing linked segment IDs, by default None.
        description_column : Optional[str], optional
            The name of the column containing descriptions, by default None.
        tag_column : Optional[str], optional
            The name of a column containing tags, by default None.
        tag_bools : Optional[list], optional
            A list of column names indicating tags as booleans, by default None.
        data_resolution : Optional[list], optional
            The resolution of the data, by default None.
        tags : Optional[list], optional
            A list of tags to add to the annotation layer, by default None.
        linked_segmentation : str or bool, optional
            The name of the segmentation layer to link to, by default None.
            If True, will link to the first segmentation layer found in the viewer.
            If False, will not link to any segmentation layer.
        shader : Optional[str], optional
            The shader to use for the annotation layer, by default None.
        color : Optional[str], optional
            The color to use for the ellipsoids, by default None.
        swap_visible_segments_on_move : bool or "auto", optional
            If True, will swap the visibility of segments when moving ellipsoids.
            If "auto" (default), will swap only when segment_column is provided.
        **layer_kwargs
            Further options for the annotation layer when this call creates it, such
            as `active_tool`, `shader_controls`, `filter_by_segmentation`, `tools`, or
            `extra`. See `AnnotationLayer`. Passing any when the layer already exists
            raises, rather than silently ignoring them.
        Returns
        -------
        Self
            The viewer state object with the added ellipsoids.
        """
        if swap_visible_segments_on_move == "auto":
            swap_visible_segments_on_move = segment_column is not None

        layer = self._get_or_create_annotation_layer(
            name,
            layer_kwargs,
            resolution=data_resolution,
            tags=tags,
            linked_segmentation=linked_segmentation,
            color=color,
            shader=shader,
            swap_visible_segments_on_move=swap_visible_segments_on_move,
        )
        layer.add_ellipsoids(
            data,
            center_column=center_column,
            radii_column=radii_column,
            segment_column=segment_column,
            description_column=description_column,
            tag_column=tag_column,
            tag_bools=tag_bools,
            data_resolution=data_resolution,
        )
        return self

    def add_boxes(
        self,
        data: Optional["pd.DataFrame"] = None,
        name: str = "annotation",
        point_a_column: str = None,
        point_b_column: str = None,
        segment_column: Optional[str] = None,
        description_column: Optional[str] = None,
        tag_column: Optional[str] = None,
        tag_bools: Optional[list] = None,
        data_resolution: Optional[list] = None,
        tags: Optional[list] = None,
        linked_segmentation: Union[str, bool] = True,
        shader: Optional[str] = None,
        color: Optional[str] = None,
        swap_visible_segments_on_move: Union[bool, Literal["auto"]] = "auto",
        **layer_kwargs: Any,
    ) -> Self:
        """Add bounding box annotations to an existing annotation layer or create a new one.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the bounding box annotations.
        name : str, optional
            The name of the annotation layer, by default "annotation".
        point_a_column : str
            The column name for the start point coordinates of the box.
        point_b_column : str
            The column name for the end point coordinates of the box.
        segment_column : Optional[str], optional
            The name of the column containing linked segment IDs, by default None.
        description_column : Optional[str], optional
            The name of the column containing descriptions, by default None.
        tag_column : Optional[str], optional
            The name of a column containing tags, by default None.
        tag_bools : Optional[list], optional
            A list of column names indicating tags as booleans, by default None.
        data_resolution : Optional[list], optional
            The resolution of the data, by default None.
        tags : Optional[list], optional
            A list of tags to add to the annotation layer, by default None.
        linked_segmentation : str or bool, optional
            The name of the segmentation layer to link to, by default None.
            If True, will link to the first segmentation layer found in the viewer.
            If False, will not link to any segmentation layer.
        shader : Optional[str], optional
            The shader to use for the annotation layer, by default None.
        color : Optional[str], optional
            The color to use for the bounding boxes, by default None.
        swap_visible_segments_on_move : bool or "auto", optional
            If True, will swap the visibility of segments when moving boxes.
            If "auto" (default), will swap only when segment_column is provided.

        **layer_kwargs
            Further options for the annotation layer when this call creates it, such
            as `active_tool`, `shader_controls`, `filter_by_segmentation`, `tools`, or
            `extra`. See `AnnotationLayer`. Passing any when the layer already exists
            raises, rather than silently ignoring them.
        Returns
        -------
        Self
            The viewer state object with the added bounding boxes.
        """
        if swap_visible_segments_on_move == "auto":
            swap_visible_segments_on_move = segment_column is not None

        layer = self._get_or_create_annotation_layer(
            name,
            layer_kwargs,
            resolution=data_resolution,
            tags=tags,
            linked_segmentation=linked_segmentation,
            color=color,
            shader=shader,
            swap_visible_segments_on_move=swap_visible_segments_on_move,
        )
        layer.add_boxes(
            data,
            point_a_column=point_a_column,
            point_b_column=point_b_column,
            segment_column=segment_column,
            description_column=description_column,
            tag_column=tag_column,
            tag_bools=tag_bools,
            data_resolution=data_resolution,
        )
        return self

    def add_polylines(
        self,
        data: Optional[Union[list, np.ndarray, pd.DataFrame, DataMap]] = None,
        name: str = "annotation",
        points_column: Optional[Union[str, list]] = None,
        segment_column: Optional[str] = None,
        description_column: Optional[str] = None,
        tag_column: Optional[str] = None,
        tag_bools: Optional[list] = None,
        data_resolution: Optional[list] = None,
        tags: Optional[list] = None,
        linked_segmentation: Union[str, bool] = True,
        shader: Optional[str] = None,
        color: Optional[str] = None,
        swap_visible_segments_on_move: Union[bool, Literal["auto"]] = "auto",
        **layer_kwargs: Any,
    ) -> Self:
        """Add points to an existing annotation layer or create a new one.
        Parameters
        ----------
        data : Union[list, np.ndarray, pd.DataFrame]
            The data to add to the annotation layer.
        name : str, optional
            The name of the annotation layer, by default "annotation".
        points_column : Optional[Union[str, list]] = None,
            The name of the column containing lists of point coordinates, by default None.
            None is needed if the data is an array, but is required if the data is a DataFrame.
            Because the data is a list of 3d points, this cannot be a split prefix.
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
        swap_visible_segments_on_move: bool or "auto", optional
            If True, will swap the visibility of segments when moving points.
            If "auto" (default), will swap only when segment_column is provided.
        **layer_kwargs
            Further options for the annotation layer when this call creates it, such
            as `active_tool`, `shader_controls`, `filter_by_segmentation`, `tools`, or
            `extra`. See `AnnotationLayer`. Passing any when the layer already exists
            raises, rather than silently ignoring them.
        Returns
        -------
        Self
            The viewer state object with the added points.
        """
        if swap_visible_segments_on_move == "auto":
            swap_visible_segments_on_move = segment_column is not None

        layer = self._get_or_create_annotation_layer(
            name,
            layer_kwargs,
            resolution=data_resolution,
            tags=tags,
            linked_segmentation=linked_segmentation,
            color=color,
            shader=shader,
            swap_visible_segments_on_move=swap_visible_segments_on_move,
        )
        layer.add_polylines(
            data,
            points_column=points_column,
            segment_column=segment_column,
            description_column=description_column,
            tag_column=tag_column,
            tag_bools=tag_bools,
            data_resolution=data_resolution,
        )
        return self

    def _has_local_tags(self) -> bool:
        """Whether any local annotation layer actually carries tags.

        The capability probe is only worth a network round trip when the answer can
        change the output, which is exactly this case. Cloud annotation layers cannot
        carry tags at all.
        """
        return any(
            isinstance(layer, AnnotationLayer)
            and layer.source is None
            and layer.tags
            and layer.capabilities is None
            for layer in self.layers
        )

    def _resolve_capabilities(self):
        """Determine the capabilities to encode this state against.

        An explicit setting wins. Otherwise the target deployment is probed, but only
        when a local annotation layer has tags -- an offline user building a state with
        no tags should never wait on the network.
        """
        if self._capabilities is not None:
            return self._capabilities
        if not self._has_local_tags():
            return None
        return capabilities_for_url(self._target_url)

    def _state_for_target(self, target_url):
        """Build the neuroglancer state as the given target deployment would need it.

        `to_url` and friends let a caller name a different target after the viewer has
        already been built and cached. Since the target determines how annotation tags
        are encoded, reusing that cached state would quietly ship the wrong encoding,
        so rebuild when the target changes and the difference can matter.
        """
        unaffected = (
            target_url is None
            or target_url == self._target_url
            or self._capabilities is not None
            or not self._has_local_tags()
        )
        if unaffected:
            return self.viewer.state
        previous_url = self._target_url
        try:
            self._target_url = target_url
            self._reset_viewer()
            return self.viewer.state
        finally:
            self._target_url = previous_url
            self._reset_viewer()

    @property
    def capabilities(self):
        """Capabilities of the target deployment, if explicitly set."""
        return self._capabilities

    @capabilities.setter
    def capabilities(self, value):
        self._capabilities = parse_capabilities(value)
        self._reset_viewer()

    def to_neuroglancer_state(self):
        if self.dimensions is None:
            if self.infer_coordinates and source_info.HAS_CLOUDVOLUME:
                self._dimensions = self._suggest_resolution_from_source()
            else:
                warnings.warn(
                    "No dimensions provided or inferred. Using a null CoordSpace which may cause unintended effects in Neuroglancer."
                )
                self._dimensions = CoordSpace()
        if not isinstance(self.dimensions, CoordSpace):
            self._dimensions = CoordSpace(self.dimensions)

        for layer in self.layers:
            if isinstance(layer, AnnotationLayer):
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
            elif self.infer_coordinates and s.position is None:
                s.position = self._suggest_position_from_source(
                    resolution=self.dimensions.resolution
                )
            if s.dimensions.rank == 0:
                s.dimensions = self.dimensions.to_neuroglancer()
            view_settings = [
                (
                    "layout",
                    "layout",
                    layout_to_json(self.layout, self._all_layer_names(s)),
                ),
                ("show_slices", "show_slices", self.show_slices),
            ]
            for name, attr, value in view_settings:
                if not self.base_state or name in self._explicit_view:
                    setattr(s, attr, value)
            camera = self._camera
            if not self.base_state:
                camera = Camera(
                    cross_section_scale=_DEFAULT_CROSS_SECTION_SCALE,
                    projection_scale=_DEFAULT_PROJECTION_SCALE,
                ).merge(camera)
            camera.apply_to_neuroglancer(s)
            for name, value in self._display.items():
                setattr(s, name, value)
            if self._selected_layer is not None:
                if self._selected_layer not in self._all_layer_names(s):
                    raise ValueError(
                        f"The selected layer {self._selected_layer!r} is not in the "
                        f"viewer. Layers: {self._all_layer_names(s)}"
                    )
                s.selected_layer.layer = self._selected_layer
                s.selected_layer.visible = self._selected_layer_visible
            elif (
                "selected_layer_panel" in self._panels
                and self._panels["selected_layer_panel"].visible
            ):
                raise ValueError(
                    "The selected-layer panel is set to open, but no layer is "
                    "selected; choose one with set_selected_layer()."
                )
            for name, panel in self._panels.items():
                panel.apply_to_neuroglancer(getattr(s, _PANELS[name]))
            capabilities = self._resolve_capabilities()
            for layer in self.layers:
                layer.apply_to_neuroglancer(s, capabilities=capabilities)
            # After every layer exists, so keys already bound by tags, raw layers,
            # or a base state are known before any are handed out.
            requests = [(tool, None) for tool in self._tools] + [
                (tool, layer.name) for layer in self.layers for tool in layer.tools
            ]
            # Tag tools nglui generated may move to free keys; bindings from raw
            # layers or a base state were chosen by the user and stay put.
            tag_layers = tuple(
                layer.name
                for layer in self.layers
                if isinstance(layer, AnnotationLayer) and layer.tags
            )
            # Keys the viewer's extra JSON binds are merged in after this, so the
            # allocator must treat them as taken
            reserved = [
                key
                for key, value in (self._extra.get("toolBindings") or {}).items()
                if value is not None
            ]
            if requests or tag_layers:
                bind_tools(s, requests, tag_layers=tag_layers, reserved=reserved)
            add_palettes(s, self._palettes)

        if self._extra:
            self._viewer.set_state(
                deep_merge(self._viewer.state.to_json(), self._extra)
            )
        return self._viewer

    def map(self, datamap: dict, inplace: bool = False) -> Self:
        """Apply a datamap to the viewer state to freeze the state of the viewer.
        Must be used if any layers use a datamap.
        By default, this will return a new viewer state with the datamap applied without changing the current object.

        Parameters
        ----------
        datamap : dict
            A dictionary mapping layer names to their corresponding datamaps.
        inplace : bool, optional
            If True, apply the datamap in place and return the modified viewer state.
            If False, return a new viewer state with the datamap applied.

        Returns
        -------
        Self
            A new ViewerState object with the datamap applied.
        """
        self._reset_viewer()
        if inplace:
            self._apply_datamaps(datamap)
            return self
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

    def to_json_string(self, indent: int = 2, compact: bool = False) -> str:
        """Return a JSON string representation of the viewer state.

        Parameters
        ----------
        indent : int
            The number of spaces to use for indentation in the JSON string.
            Default is 2.
        compact: bool
            If True, the JSON string will be compact with no extra whitespace or newlines.
            Default is False.

        Returns
        -------
        str
            A JSON string representation of the viewer state.
        """

        if compact:
            return json.dumps(self.to_dict(), separators=(",", ":"), cls=NumpyEncoder)
        return json.dumps(self.to_dict(), indent=indent, cls=NumpyEncoder)

    def to_url(
        self,
        target_url: str = None,
        target_site: str = None,
        shorten: Union[bool, Literal["if_long"]] = False,
        client: Optional["caveclient.CAVEclient"] = None,
    ):
        """Return a URL representation of the viewer state.

        Parameters
        ----------
        target_url : str
            The base URL to use for the Neuroglancer state. If not provided,
            the default server URL will be used.
        target_site : str, optional
            The target site for the URL, based on the keys in site_utils.NEUROGLANCER . If not provided, the default server URL will be used.
        shorten: Union[bool, Literal["if_long"]], optional
            If True, the URL will be shortened using the CAVE link shortener service.
            If "if_long", the URL will only be shortened if it exceeds a certain length.
        client : Optional[caveclient.CAVEclient], optional
            The CAVE client to use for shortening the URL. If not provided, the URL will not be shortened.

        Returns
        -------
        str
            A URL representation of the viewer state.
        """
        if target_url is None:
            if target_site is None:
                if self.interactive:
                    return self.viewer.get_viewer_url()
                else:
                    target_url = self._target_url
            else:
                target_url = neuroglancer_url(target_site=target_site)

        url = neuroglancer.to_url(
            self._state_for_target(target_url),
            prefix=target_url,
        )
        if shorten == "if_long":
            if len(url) > MAX_URL_LENGTH:
                shorten = True
            else:
                shorten = False
        if shorten:
            url = self.to_link_shortener(client, target_url=target_url)
        return url

    def to_link(
        self,
        target_url: str = None,
        target_site: str = None,
        link_text: str = "Neuroglancer Link",
        shorten: Union[bool, Literal["if_long"]] = False,
        client: Optional["caveclient.CAVEclient"] = None,
    ):
        """Return an HTML link representation of the viewer state.

        Parameters
        ----------
        target_url : str
            The base URL to use for the Neuroglancer state. If not provided,
            the default server URL will be used.
        target_site : str, optional
            The target site for the URL, based on the keys in site_utils.NEUROGLANCER_SITES.
            If not provided, the default server URL will be used.
        link_text : str
            The text to display for the HTML link.
        shorten: Union[bool, Literal["if_long"]], optional
            If True, the URL will be shortened using the CAVE link shortener service.
            If "if_long", the URL will only be shortened if it exceeds a certain length.
        client : Optional[caveclient.CAVEclient], optional
            The CAVE client to use for shortening the URL. If not provided, the URL will not be shortened.

        Returns
        -------
        HTML.HTML
            An HTML link representation of the viewer state.
        """
        url = self.to_url(
            target_url=target_url,
            target_site=target_site,
            shorten=shorten,
            client=client,
        )
        return HTML(f'<a href="{url}" target="_blank">{link_text}</a>')

    def to_link_shortener(
        self,
        client: "caveclient.CAVEclient" = None,
        target_url: str = None,
        target_site: str = None,
    ) -> str:
        """Shorten the URL using the CAVE link shortener service.

        Parameters
        ----------
        client : CAVEclient
            The CAVE client to use for shortening the URL.
        target_url : str, optional
            The base URL to use for the Neuroglancer state. If not provided,
            the default server URL will be used.
        target_site : str, optional
            The target site for the URL, based on the keys in site_utils.NEUROGLANCER_SITES.
            If not provided, the default server URL will be used.

        Returns
        -------
        str
            A shortened URL representation of the viewer state.
        """
        if client is None:
            client = self._client
        if client is None:
            raise ValueError("A CAVEclient instance is required to shorten the URL.")

        # Resolve the target before building the state: the target determines how
        # annotation tags are encoded, so uploading first would pin the wrong one.
        if target_url is None:
            if target_site is None:
                if self.interactive:
                    return self.viewer.get_viewer_url()
                else:
                    target_url = self._target_url
            else:
                target_url = neuroglancer_url(target_site=target_site)
        state_id = client.state.upload_state_json(
            self._state_for_target(target_url).to_json()
        )
        return client.state.build_neuroglancer_url(state_id, target_url)

    def to_clipboard(
        self,
        target_url: str = None,
        target_site: str = None,
        shorten: Union[bool, Literal["if_long"]] = False,
        client: Optional["caveclient.CAVEclient"] = None,
    ) -> str:
        """Copy the viewer state URL to the system clipboard.

        Parameters
        ----------
        target_url : str
            The base URL to use for the Neuroglancer state. If not provided,
            the default server URL will be used.
        target_site : str, optional
            The target site for the URL, based on the keys in site_utils.NEUROGLANCER_SITES.
            If not provided, the default server URL will be used.
        shorten: Union[bool, Literal["if_long"]], optional
            If True, the URL will be shortened using the CAVE link shortener service.
            If "if_long", the URL will only be shortened if it exceeds a certain length.
        client : Optional[caveclient.CAVEclient], optional
            The CAVE client to use for shortening the URL. If not provided, the URL will not be shortened.

        Returns
        -------
        str
            The URL representation of the viewer state that has also been copied to the
            clipboard.
        """
        url = self.to_url(
            target_url=target_url,
            target_site=target_site,
            shorten=shorten,
            client=client,
        )
        pyperclip.copy(url)
        return url

    def to_browser(
        self,
        target_url: str = None,
        target_site: str = None,
        shorten: Union[bool, Literal["if_long"]] = False,
        client: Optional["caveclient.CAVEclient"] = None,
        new: int = 2,
        autoraise: bool = True,
        browser: Optional[str] = None,
    ):
        """Open the viewer state URL in a web browser.

        Parameters
        ----------
        target_url : str
            The base URL to use for the Neuroglancer state. If not provided,
            the default server URL will be used.
        target_site : str, optional
            The target site for the URL, based on the keys in site_utils.NEUROGLANCER_SITES.
            If not provided, the default server URL will be used.
        shorten: Union[bool, Literal["if_long"]], optional
            If True, the URL will be shortened using the CAVE link shortener service.
            If "if_long", the URL will only be shortened if it exceeds a certain length.
        client : Optional[caveclient.CAVEclient], optional
            The CAVE client to use for shortening the URL. If not provided, the URL will not be shortened.
        new : int, optional
            If new is 0, the url is opened in the same browser window if possible. If
            new is 1, a new browser window is opened if possible. If new is 2, a new
            browser page (“tab”) is opened if possible. Note that not all browsers
            support all values of new, and some may ignore this parameter.
        autoraise : bool, optional
            If True, the browser window will be raised to the front when opened. Note
            that under many window managers this will occur regardless of the setting
            of this variable.
        browser : Optional[str], optional
            The name of the browser to use. If None, the system default browser will be
            used. Note that the browser name needs to be registered on your system,
            see [webbrowser.register][] for more details.

        See Also
        --------
        [webbrowser.open][] : Opens a URL in the web browser.

        Returns
        -------
        str
            The URL representation of the viewer state that has also been opened in the
            browser.
        """
        url = self.to_url(
            target_url=target_url,
            target_site=target_site,
            shorten=shorten,
            client=client,
        )
        if browser is None:
            webbrowser.open(url, new=new, autoraise=autoraise)
        else:
            browser_controller = webbrowser.get(browser)
            browser_controller.open(url, new=new, autoraise=autoraise)
        return url
