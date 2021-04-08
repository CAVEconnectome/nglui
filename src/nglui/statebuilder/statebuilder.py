from nglui import EasyViewer
from nglui.easyviewer.utils import default_neuroglancer_base
import pandas as pd
import numpy as np
from collections.abc import Collection
from IPython.display import HTML
from .utils import bucket_of_values

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
    """

    def __init__(
        self,
        layers=[],
        base_state=None,
        url_prefix=None,
        state_server=None,
        resolution=[4, 4, 40],
        view_kws={},
        client=None,
    ):
        if client is not None:
            if state_server is None:
                state_server = client.state.state_service_endpoint
            if url_prefix is None:
                url_prefix = client.info.viewer_site()
        if url_prefix is None:
            url_prefix = default_neuroglancer_base

        self._base_state = base_state
        self._layers = layers
        self._resolution = resolution
        self._url_prefix = url_prefix
        self._state_server = state_server

        base_kws = DEFAULT_VIEW_KWS.copy()
        base_kws.update(view_kws)
        self._view_kws = base_kws

    def _reset_state(self, base_state=None):
        """
        Resets the neuroglancer state status to a default viewer.
        """
        if base_state is None:
            base_state = self._base_state
        self._temp_viewer = EasyViewer()
        self._temp_viewer.set_state(base_state)

    def initialize_state(self, base_state=None):
        """Generate a new Neuroglancer state with layers as needed for the schema.

        Parameters
        ----------
        base_state : str, optional
            Optional initial state to build on, described by its JSON. By default None.
        """
        self._reset_state(base_state)

        if self._state_server is not None:
            self._temp_viewer.set_state_server(self._state_server)

        if self._resolution is not None:
            self._temp_viewer.set_resolution(self._resolution)
        self._temp_viewer.set_view_options(**self._view_kws)

    def render_state(
        self,
        data=None,
        base_state=None,
        return_as="url",
        url_prefix=None,
        link_text="Neuroglancer Link",
    ):
        """Build a Neuroglancer state out of a DataFrame.

        Parameters
        ----------
        data : pandas.DataFrame, optional
            DataFrame to use as a point source. By default None, for which
            it will return only the base_state and any fixed values.
        base_state : str, optional
            Initial state to build on, expressed as Neuroglancer JSON. By default None
        return_as : ['url', 'viewer', 'html', 'json', 'dict', 'shared'], optional
            Choice of output types. Note that if a viewer is returned, the state is not reset.
                url : Returns the raw url describing the state
                viewer : Returns an EasyViewer object holding the state information
                html : Returns an HTML link to the url, useful for notebooks.
                json : Returns a JSON string describing the state.
                dict : Returns a dict version of the JSON state.
            By default 'url'
        url_prefix : str, optional
            Neuroglancer URL prefix to use. By default None, for which it will open with the
            class default.
        link_text : str, optional
            Text to use for the link when returning as html, by default 'Neuroglancer Link'
        Returns
        -------
        string or neuroglancer.Viewer
            A link to or viewer for a Neuroglancer state with layers, annotations, and selected objects determined by the data.
        """
        if base_state is None:
            base_state = self._base_state
        self.initialize_state(base_state=base_state)

        if url_prefix is None:
            url_prefix = self._url_prefix

        self._render_layers(data)

        if return_as == "viewer":
            return self.viewer
        elif return_as == "url":
            url = self._temp_viewer.as_url(prefix=url_prefix)
            self.initialize_state()
            return url
        elif return_as == "html":
            out = self._temp_viewer.as_url(
                prefix=url_prefix, as_html=True, link_text=link_text
            )
            out = HTML(out)
            self.initialize_state()
            return out
        elif return_as == "dict":
            out = self._temp_viewer.state.to_json()
            self.initialize_state()
            return out
        elif return_as == "json":
            from json import dumps

            out = self._temp_viewer.state.to_json()
            self.initialize_state()
            return dumps(out)
        else:
            raise ValueError("No appropriate return type selected")

    def _render_layers(self, data):
        for layer in self._layers:
            layer._render_layer(self._temp_viewer, data)

    @property
    def viewer(self):
        return self._temp_viewer


class ChainedStateBuilder:
    def __init__(self, statebuilders):
        """Builds a collection of states that sequentially add annotations based on a sequence of dataframes.

        Parameters
        ----------
        statebuilders : list
            List of DataStateBuilders, in same order as dataframes will be passed
        """
        self._statebuilders = statebuilders
        if len(self._statebuilders) == 0:
            raise ValueError("Must have at least one statebuilder")

    def render_state(
        self,
        data_list=None,
        base_state=None,
        return_as="url",
        url_prefix=default_neuroglancer_base,
        link_text="Neuroglancer Link",
    ):
        """Generate a single neuroglancer state by addatively applying an ordered collection of
        dataframes to an collection of StateBuilder renders.
        Parameters
            data_list : Collection of DataFrame. The order must match the order of StateBuilders
                        contained in the class on creation.
            base_state : JSON neuroglancer state (optional, default is None).
                         Used as a base state for adding everything else to.
            return_as: ['url', 'viewer', 'html', 'json']. optional, default='url'.
                       Sets how the state is returned. Note that if a viewer is returned,
                       the state is not reset to default.
            url_prefix: string, optional (default is https://neuromancer-seung-import.appspot.com).
                        Overrides the default neuroglancer url for url generation.
        """
        if data_list is None:
            data_list = len(self._statebuilders) * [None]

        if len(data_list) != len(self._statebuilders):
            raise ValueError("Must have as many dataframes as statebuilders")

        temp_state = base_state
        for builder, data in zip(self._statebuilders[:-1], data_list[:-1]):
            temp_state = builder.render_state(
                data=data, base_state=temp_state, return_as="dict"
            )
        last_builder = self._statebuilders[-1]
        return last_builder.render_state(
            data=data_list[-1],
            base_state=temp_state,
            return_as=return_as,
            url_prefix=url_prefix,
            link_text=link_text,
        )
