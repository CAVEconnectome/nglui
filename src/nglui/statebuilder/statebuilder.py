from __future__ import annotations

from warnings import warn

from IPython.display import HTML

from nglui.easyviewer import EasyViewer
from nglui.easyviewer.ev_base.nglite.json_utils import encode_json

from ..easyviewer.ev_base.utils import (
    default_seunglab_neuroglancer_base,
    neuroglancer_url,
)
from .utils import check_target_site

DEFAULT_TARGET_SITE = "seunglab"
DEFAULT_URL = default_seunglab_neuroglancer_base

DEFAULT_VIEW_KWS = {
    "layout": "xy-3d",
    "zoom_image": 2,
    "show_slices": False,
    "zoom_3d": 2000,
}


class StateBuilder:
    """A class for schematic mapping data frames into neuroglancer states.

    Parameters
    ----------
    layers : list of nglui.statebuilder.layers.LayerConfigBase, optional
        List of layers to add. Defaults to [].
    base_state : dict, optional
        JSON state to add to. Defaults to None.
    url_prefix : str, optional
        http(s) path to Neuroglancer deployment to use. Defaults to None,
        which will use https://neuromancer-seung-import.appspot.com
    state_server : str, optional
        State server to post links to. Defaults to None.
    resolution : list of float, optional
        3-element vector controlling the viewer resolution. Defaults to None.
        If None and a client is set, uses the client viewer resolution.
    view_kws : dict, optional
        Dictionary controlling view parameters. Defaults to {}.

        Keys are:
            show_slices : bool, optional
                Sets if slices are shown in the 3d view. Defaults to False.
            layout : str, optional
                `xy-3d`/`xz-3d`/`yz-3d` (sections plus 3d pane),
                `xy`/`yz`/`xz`/`3d` (only one pane), or `4panel` (all panes).
                Default is `xy-3d`.
            show_axis_lines : bool, optional
                Determines if the axis lines are shown in the middle of each view.
            show_scale_bar : bool, optional
                Toggles showing the scale bar.
            orthographic : bool, optional
                Toggles orthographic view in the 3d pane.
            position : list of float, optional (length-3)
                Determines the centered location.
            zoom_image : float, optional
                Zoom level for the imagery in units of nm per voxel. Defaults to 8.
            zoom_3d : float, optional
                Zoom level for the 3d pane. Defaults to 2000. Smaller numbers are
                more zoomed in.
            background_color : str or list of float, optional
                Sets the background color of the 3d view. Arguments can be rgb
                values, hex colors, or named web colors. Defaults to black.
    client : caveclient.CAVEclient, optional
        A caveclient to get defaults from. Defaults to None.
    target_site : str, optional
        Target Neuroglancer category: either "seunglab" or "mainline"/"cave-explorer"/"spelunker". Defaults to None.
        Will be looked up automatically based on ngl_url, if used.
    """

    def __init__(
        self,
        layers=[],
        base_state=None,
        url_prefix=None,
        state_server=None,
        resolution=None,
        view_kws={},
        client=None,
        target_site=None,
    ):
        if client is not None:
            if state_server is None:
                state_server = client.state.state_service_endpoint
            if url_prefix is None:
                url_prefix = client.info.viewer_site()
            if resolution is None:
                resolution = client.info.viewer_resolution().tolist()
            if target_site is None:
                target_site = check_target_site(url_prefix, client)
        self._client = client
        url_prefix = neuroglancer_url(url_prefix, target_site)

        self._base_state = base_state
        self._layers = layers
        self._resolution = resolution
        self._url_prefix = url_prefix
        self._state_server = state_server
        self._target_site = target_site

        base_kws = DEFAULT_VIEW_KWS.copy()
        base_kws.update(view_kws)
        self._view_kws = base_kws

    def _reset_state(self, base_state=None, target_site=None):
        """
        Resets the neuroglancer state status to a default viewer.
        """
        if base_state is None:
            base_state = self._base_state
        self._temp_viewer = EasyViewer(target_site=target_site)
        self._temp_viewer.set_state(base_state)

    def initialize_state(self, base_state=None, target_site=None):
        """Generate a new Neuroglancer state with layers as needed for the schema.

        Parameters
        ----------
        base_state : str, optional
            Optional initial state to build on, described by its JSON. By default None.
        """
        self._reset_state(base_state, target_site=target_site)

        if self._state_server is not None:
            self._temp_viewer.set_state_server(self._state_server)

        if self._resolution is not None:
            self._temp_viewer.set_resolution(self._resolution)

        self._temp_viewer.set_view_options(**self._view_kws)
        for l in self._layers:
            l._add_layer(self._temp_viewer)

    def handle_positions(self, data):
        for l in self._layers[::-1]:
            pos = l._set_view_options(
                self._temp_viewer, data, viewer_resolution=self._resolution
            )
            if pos is not None:
                break

    def render_state(
        self,
        data=None,
        base_state=None,
        return_as="url",
        url_prefix=None,
        link_text="Neuroglancer Link",
        target_site=None,
        client=None,
    ):
        """Build a Neuroglancer state out of a DataFrame.

        Parameters
        ----------
        data : pandas.DataFrame, optional
            DataFrame to use as a point source. By default None, for which
            it will return only the base_state and any fixed values.
        base_state : dict, optional
            Initial state to build on, expressed as Neuroglancer JSON. By default None
        return_as : ['url', 'viewer', 'html', 'json', 'dict', 'short'], optional
            Choice of output types. Note that if a viewer is returned, the state is not reset.
                url : Returns the raw url describing the state
                viewer : Returns an EasyViewer object holding the state information
                html : Returns an HTML link to the url, useful for notebooks.
                json : Returns a JSON string describing the state.
                dict : Returns a dict version of the JSON state.
                short : Posts the state to the state server and returns a short link. Requires a client.
            By default 'url'
        url_prefix : str, optional
            Neuroglancer URL prefix to use. By default None, for which it will open with the
            class default.
        link_text : str, optional
            Text to use for the link when returning as html, by default 'Neuroglancer Link'
        target_site : str, optional
            Target Neuroglancer category: either "seunglab" or one of "mainline"/"cave-explorer"/"spelunker". Defaults to None.
            Will be looked up automatically based on ngl_url, if used and a client is set.
        client : caveclient.CAVEclient, optional
            A caveclient to get defaults from. Defaults to None, which falls back on the statebuilder.

        Returns
        -------
        string or neuroglancer.Viewer
            A link to or viewer for a Neuroglancer state with layers, annotations, and selected objects determined by the data.
        """
        if client is None:
            client = self._client
        if base_state is None:
            base_state = self._base_state
        if target_site is None:
            target_site = self._target_site
        if url_prefix is None:
            url_prefix = self._url_prefix

        if target_site is None and url_prefix is not None:
            if client is not None:
                target_site = check_target_site(url_prefix, client)
            else:
                warn(
                    f"Cannot check Neuroglancer target site without a client set in the statebuilder. Defaulting to '{DEFAULT_TARGET_SITE}'"
                )
        elif target_site is None and url_prefix is None:
            target_site = DEFAULT_TARGET_SITE
            url_prefix = DEFAULT_URL
            warn(
                'Deprecation warning: No target site or url prefix set, using default "seunglab" site. This will switch to "spelunker" in the future.'
            )

        self.initialize_state(base_state=base_state, target_site=target_site)
        self.handle_positions(data)

        self._render_layers(
            data,
            client=client,
        )

        if url_prefix is None:
            url_prefix = self._url_prefix

        if return_as == "viewer":
            return self.viewer
        elif return_as == "url":
            url = self._temp_viewer.as_url(prefix=url_prefix)
            return url
        elif return_as == "html":
            out = self._temp_viewer.as_url(
                prefix=url_prefix, as_html=True, link_text=link_text
            )
            out = HTML(out)
            return out
        elif return_as == "dict":
            out = self._temp_viewer.state.to_json()
            return out
        elif return_as == "json":
            out = self._temp_viewer.state.to_json()
            return encode_json(out)
        elif return_as == "short":
            if client is None:
                raise ValueError("Cannot generate short link without a client")
            state_id = client.state.upload_state_json(self._temp_viewer.state.to_json())
            short_link = client.state.build_neuroglancer_url(
                state_id,
                ngl_url=url_prefix,
                target_site=target_site.replace("spelunker", "mainline"),
            )
            return short_link
        else:
            raise ValueError("No appropriate return type selected")

    def _render_layers(
        self,
        data,
        client=None,
    ):
        if client is None:
            client = self._client
        # Inactivate all layers except last.
        found_active = False
        for l in self._layers[::-1]:
            if l.active and not found_active:
                found_active = True
            else:
                l.active = False
        anno_dict = {}
        for layer in self._layers:
            anno_dict[layer.name] = layer._render_layer(
                self._temp_viewer,
                data,
                viewer_resolution=self._resolution,
                return_annos=True,
                client=client,
            )
        self._temp_viewer.add_multilayer_annotations(anno_dict)

    @property
    def viewer(self):
        return self._temp_viewer


class ChainedStateBuilder:
    """Builds a collection of states that sequentially add annotations based on a sequence of dataframes.

    Parameters
    ----------
    statebuilders : list
        List of DataStateBuilders, in same order as dataframes will be passed
    """

    def __init__(self, statebuilders):
        self._statebuilders = statebuilders
        if len(self._statebuilders) == 0:
            raise ValueError("Must have at least one statebuilder")

    def render_state(
        self,
        data_list=None,
        base_state=None,
        return_as="url",
        url_prefix=None,
        link_text="Neuroglancer Link",
        target_site=None,
        client=None,
    ):
        """Generate a single neuroglancer state by addatively applying an ordered collection of
        dataframes to an collection of StateBuilder renders.
        Parameters
            data_list : Collection of DataFrame. The order must match the order of StateBuilders
                        contained in the class on creation.
            base_state : JSON neuroglancer state (optional, default is None).
                         Used as a base state for adding everything else to.
            return_as: ['url', 'viewer', 'html', 'json', 'short']. optional, default='url'.
                       Sets how the state is returned. Note that if a viewer is returned,
                       the state is not reset to default.
            url_prefix: string, optional (default is https://neuromancer-seung-import.appspot.com).
                        Overrides the default neuroglancer url for url generation.
            link_text: string, optional (default is 'Neuroglancer Link').
                        Text to use for the link when returning as html.
            target_site: string, optional (default is None).
                         Target Neuroglancer category: either "seunglab" or "mainline"/"cave-explorer"/"spelunker".
                         Will be looked up automatically based on url_prefix, if used.
            client: caveclient.CAVEclient, optional (default is None).
                    A caveclient to get defaults from. Defaults to None, which falls back on the last statebuilder.
        """
        if client is None:
            client = self._statebuilders[-1]._client
        if target_site is None:
            target_site = self._statebuilders[-1]._target_site
        if url_prefix is None:
            url_prefix = self._statebuilders[-1]._url_prefix

        if target_site is None and url_prefix is not None:
            if client is not None:
                target_site = check_target_site(url_prefix, client)
            else:
                warn(
                    f"Cannot check Neuroglancer target site without a client. Defaulting to '{DEFAULT_TARGET_SITE}'"
                )
        elif target_site is None and url_prefix is None:
            target_site = DEFAULT_TARGET_SITE
            url_prefix = DEFAULT_URL
            warn(
                'Deprecation warning: No target site or url prefix set, using default "seunglab" site. This will switch to "spelunker" in the future.'
            )

        if data_list is None:
            data_list = len(self._statebuilders) * [None]

        if len(data_list) != len(self._statebuilders):
            raise ValueError("Must have as many dataframes as statebuilders")

        temp_state = base_state
        for builder, data in zip(self._statebuilders[:-1], data_list[:-1]):
            temp_state = builder.render_state(
                data=data,
                base_state=temp_state,
                return_as="dict",
                url_prefix=url_prefix,
                target_site=target_site,
            )
        last_builder = self._statebuilders[-1]
        return last_builder.render_state(
            data=data_list[-1],
            base_state=temp_state,
            return_as=return_as,
            url_prefix=url_prefix,
            link_text=link_text,
            target_site=target_site,
            client=client,
        )
