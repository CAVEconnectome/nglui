from neuroglancer_annotation_ui import EasyViewer, annotation
import pandas as pd
import numpy as np
from .utils import bucket_of_values

class StateBuilder():
    def __init__(self, base_state=None,
                 image_sources={}, seg_sources={},
                 selected_ids={}, annotation_layers={},
                 resolution=[4,4,40], fixed_selection={},
                 url_prefix=None):
        """
        Class for turning data frames into neuroglancer states 
        """
        self._base_state = base_state
        self._image_sources = image_sources
        self._seg_sources = seg_sources
        self._resolution = resolution
        self._selected_ids = selected_ids
        self.fixed_selection = fixed_selection
        self._annotation_layers = annotation_layers
        self._url_prefix = url_prefix
        self._data_columns = self._compute_data_columns()
        self.initialize_state()

    def _compute_data_columns(self):
        data_columns = []
        for ln, kws in self._annotation_layers.items():
            if 'points' in kws:
                for col in kws['points']:
                    data_columns.append(col)
            if 'lines' in kws:
                for colpair in kws['lines']:
                    data_columns.extend(colpair)
            if 'spheres' in kws:
                for colpair in kws['spheres']:
                    data_columns.extend(colpair)

        for ln, idcols in self._selected_ids.items():
            data_columns.extend(idcols)

        return data_columns

    
    def _reset_state(self, base_state=None):
        """
        Resets the neuroglancer state status to a default viewer.
        """
        if base_state is None:
            base_state = self._base_state
        self._temp_viewer = EasyViewer()
        self._temp_viewer.set_state(base_state)

    def _validate_dataframe(self, data):
        '''
        Makes sure the dataframe has data expected from the render rules
        '''
        # Check that each layer has a column
        if not np.all(np.isin(self.data_columns, data.columns)):
            missing_cols = [c for c in self.data_columns if c not in data.columns]
            raise ValueError('Dataframe does not have all needed columns. Missing {}'.format(missing_cols))


    def _add_layers(self):
        for ln, src in self._image_sources.items():
            if ln not in self._temp_viewer.layer_names:
                self._temp_viewer.add_image_layer(ln, src)
        for ln, src in self._seg_sources.items():
            if ln not in self._temp_viewer.layer_names:
                self._temp_viewer.add_segmentation_layer(ln, src)
        for ln, kws in self._annotation_layers.items():
            if ln not in self._temp_viewer.layer_names:
                self._temp_viewer.add_annotation_layer(ln, color=kws.get('color', None))
            # TODO tags for tag-mapping!

    def _render_data(self, data):
        self._add_selected_ids(data)
        self._add_annotations(data)

    def _add_selected_ids(self, data):
        for ln, oids in self._fixed_selection.items():
            self._temp_viewer.add_selected_objects(ln, oids)
            
        for ln, cols in self._selected_ids.items():
            for col in cols:
                oids = bucket_of_values(col, data)
                self._temp_viewer.add_selected_objects(ln, oids)

    def _add_annotations(self, data):
        if len(data)==0:
            return

        for ln, kws in self._annotation_layers.items():
            annos = []
            for pt_column in kws.get('points', []):
                pts = bucket_of_values(pt_column, data, item_is_array=True)
                annos.extend([annotation.point_annotation(pt) for pt in pts])

            for pt_column_pair in kws.get('lines', []):
                pt_col_a, pt_col_b = pt_column_pair
                pts_a = bucket_of_values(pt_col_a, data, item_is_array=True)
                pts_b = bucket_of_values(pt_col_b, data, item_is_array=True)
                annos.extend([annotation.line_annotation(ptA, ptB) for ptA, ptB in zip(pts_a, pts_b)])

            for pt_column_pair in kws.get('spheres', []):
                z_factor = self._resolution[0]/self._resolution[2]
                pt_col, radius_col = pt_column_pair
                pts = bucket_of_values(pt_col, data, item_is_array=True)
                rads = bucket_of_values(radius_col, data, item_is_array=False)
                annos.extend([annotation.sphere_annotation(pt, radius, z_factor) for pt, radius in zip(pts, rads)])

            self._temp_viewer.add_annotations(ln, annos)

    @property
    def data_columns(self):
        return self._data_columns

    def initialize_state(self, base_state=None):
        self._reset_state(base_state)
        self._temp_viewer.set_resolution(self._resolution)
        self._add_layers()
        self._temp_viewer.set_view_options()

    def render_state(self, data=None, base_state=None, return_as='url', url_prefix=None):
        """
        Use the render rules to make a neuroglancer state out of a dataframe.
        Parameters
            data : DataFrame. Source of data for the rendering rules. Optional, default is None.
                   If no data given, only the base state, image layers, and segmentation layers are generated.
            base_state : JSON neuroglancer state (optional, default is None).
                         Used as a base state for adding to.
            return_as: ['url', 'viewer', 'html', 'json']. optional, default='url'.
                       Sets how the state is returned. Note that if a viewer is returned,
                       the state is not reset to default.
            url_prefix: string, optional (default=None). Overrides the default neuroglancer url for url generation.
        """
        if base_state is not None:
            self.initialize_state(base_state=base_state)

        if url_prefix is None:
            url_prefix = self._url_prefix

        if data is not None:
            self._validate_dataframe(data)
            self._render_data(data)

        if return_as == 'viewer':
            return self.viewer
        elif return_as == 'url':
            out = self._temp_viewer.as_url(prefix=url_prefix)
            self.initialize_state()
            return out
        elif return_as == 'html':
            out = self._temp_viewer.as_url(prefix=url_prefix, as_html=True)
            self.initialize_state()
            return out
        elif return_as == 'json':
            out = self._temp_viewer.state.to_json()
            self.initialize_state()
            return out
        else:
            raise ValueError('No appropriate return type selected')

    @property
    def viewer(self):
        return self._temp_viewer

class FilteredDataStateBuilder(StateBuilder):
    def __init__(self, *args, **kwargs):
        super(FilteredDataStateBuilder, self).__init__( *args, **kwargs)

    def render_state(self, indices=None, data=None,
                     base_state=None, return_as='url', url_prefix=None):
        """
        Use the render rules to make a neuroglancer state out of a dataframe and a set of indices
        Parameters
            indices : Collection of index values to filter the data dataframe. Optional, default=None.
                      Optional, default is None. If None, all data is used.
            data : DataFrame. Source of data for the rendering rules. Optional, default is None.
                   If no data given, only the base state, image layers, and segmentation layers are generated.
            base_state : JSON neuroglancer state (optional, default is None).
                         Used as a base state for adding to.
            return_as: ['url', 'viewer', 'html', 'json']. optional, default='url'.
                       Sets how the state is returned. Note that if a viewer is returned,
                       the state is not reset to default.
            url_prefix: string, optional (default=None). Overrides the default neuroglancer url for url generation.
        """

        if data is not None:
            if indices is None:
                data_render = data
            else:
                data_render = data.loc[indices]
        else:
            data_render = None
        return super(FilteredDataStateBuilder, self).render_state(
                     data=data_render, base_state=base_state,
                     return_as=return_as, url_prefix=url_prefix)
