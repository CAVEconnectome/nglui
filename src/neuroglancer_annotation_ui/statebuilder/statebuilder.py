from neuroglancer_annotation_ui import EasyViewer, annotation, set_static_content_source
from neuroglancer_annotation_ui.utils import default_static_content_source
import pandas as pd
import numpy as np
from IPython.display import HTML
from collections import defaultdict
from .utils import bucket_of_values, make_basic_dataframe, sources_from_infoclient

def build_state_direct(dataset_name=None, selected_ids=[], point_annotations={},
                      line_annotations={}, sphere_annotations={},
                      default_anno_layer_name='annos',
                      return_as='url', render_kws={},
                      state_kws={}):
    """Build a Neuroglancer state from data directly, using the statebuilder as an intermediate layer.

    Parameters
    ----------
    selected_ids : list, optional
        List of object ids to make selected, by default []
    point_annotations : Nx3 list or numpy array, or a dict, optional
        Either an array of points or a dict with layernames as keys as array of point locations as values.
        By default, an empty dict.
    line_annotations : list or tuple, optional
        Either a pair of Mx3 arrays of points or a dict with layernames as keys as pairs of Mx3 arrays as values.
        Points at the same index along the 1st axis will be at the ends of each line annotation.
        By default, an empty dict.
    sphere_annotations : list or tuple, optional
        Either a collection whose first element is an Lx3 array and whose second elemnt is a length L array-like, or a dict
        with layernames as keys and values being that type of collection. This will generate spheres for each row with a center at each
        point in the first element radius determined by the second.
        By default, an empty dict.
    return_as : ['url', 'viewer', 'html', 'json'], optional
        Choice of output types. Note that if a viewer is returned, the state is not reset.
        By default 'url'
    render_kwargs : dict, optional
        Keyword arguments to pass to the render_state function.
    state_kwargs : dict, optional
        Keyword arguments to pass to the StateBuilder initialization.

    Returns
    -------
    string or neuroglancer.Viewer
        A Neuroglancer state with layers, annotations, and selected objects determined by the data.        
    """
    if point_annotations is not None:
        if not isinstance(point_annotations, dict):
            point_annotations = {default_anno_layer_name: point_annotations}
    
    if line_annotations is not None:
        if not isinstance(line_annotations, dict):
            line_annotations = {default_anno_layer_name: line_annotations}
    
    if sphere_annotations is not None:
        if not isinstance(sphere_annotations, dict):
            sphere_annotations = {default_anno_layer_name: sphere_annotations}
    
    df, pals, lals, sals = make_basic_dataframe(point_annotations, line_annotations, sphere_annotations)
    sb = StateBuilder(dataset_name=dataset_name, point_annotations=pals, line_annotations=lals, sphere_annotations=sals, **state_kws)
    return sb.render_state(data=df, return_as=return_as, **render_kws)


class StateBuilder():
    def __init__(self, base_state=None,
                 dataset_name=None, segmentation_type='graphene', image_layer_name='img', seg_layer_name='seg',
                 image_sources={}, seg_sources={},
                 selected_ids={}, point_annotations={},
                 line_annotations={}, sphere_annotations={},
                 resolution=[4,4,40], fixed_selection={},
                 url_prefix=None):
        """A class for schematic mapping data frames into neuroglancer states.
        Parameters
        ----------
        base_state : str, optional
            Neuroglancer json state. This is set before all
            layers are added from other arguments. Optional,
            default is None.
        dataset_name : str or None, optional
            Dataset name to populate image and segmentation layers from the InfoService. Default is None.
        segmentation_type : 'graphene' or 'flat', optional
            If dataset_name is used, specifies whether to take the flat or graph segmentation. Default is 'graphene'.
        image_layer_name : str, optional
            If dataset_name is used, defines the image layer name. Default is 'img'. Will not override explicit setting from image_sources.
        seg_layer_name : str, optional
            If dataset_name is used, defines the segmentation layer name. Default is 'seg'. Will not override explicit setting from seg_sources.
        image_sources : dict, optional
            Dict where keys are layer names and values are
            neuroglancer image sources, by default {}
        seg_sources : dict, optional
            Dict where keys are layer names and values are
            neuroglancer segmentation sources, by default {}
        selected_ids : dict, optional
            Dict where keys are segmentation layer names and
            values are an iterable of dataframe column names.
            Object root ids from these columns are added to the
            selected ids list in the segmentation layer. By default {}
        point_annotations : dict, optional
            Dict where the keys define annotation layer names and the values
            are a collection of column names to use as a source data for points
            in that annotation layer.
            By default {}
        line_annotations : dict, optional
            Dict where the keys define annotation layer names and the values
            are a collection of column name 2-tuples to use as a source for line annotatons. 
            By default {}
        sphere_annotations : dict, optional
            Dict where keys are annotation layer names and values are a collection of
            (center, radius) 2-tuples of column names for getting the source data.
            By default {}
        resolution : list, optional
            Numpy array for voxel resolution, by default [4,4,40]
        fixed_selection : dict, optional
            Dict where keys are segmentation layers and values are
            a collection of object ids to make selected for all dataframes.
            By default {}
        url_prefix : str, optional
            Default neuroglancer prefix to use, by default None.
        """
        if dataset_name is not None:
            info_img, info_seg = sources_from_infoclient(dataset_name, segmentation_type=segmentation_type,
                                                         image_layer_name=image_layer_name, seg_layer_name=seg_layer_name)
            for ln, src in info_img.items():  # Do not override manual choices
                if ln not in image_sources:
                    image_sources[ln] = src
            for ln, src in info_seg.items():
                if ln not in seg_sources:
                    seg_sources[ln] = src

        self._base_state = base_state
        self._image_sources = image_sources
        self._seg_sources = seg_sources
        self._resolution = resolution
        self._selected_ids = selected_ids
        self._fixed_selection = fixed_selection

        annotation_layers = defaultdict(dict)
        for ln, col_names in point_annotations.items():
            annotation_layers[ln]['points'] = col_names
        for ln, col_names in line_annotations.items():
            annotation_layers[ln]['lines'] = col_names
        for ln, col_names in sphere_annotations.items():
            annotation_layers[ln]['spheres'] = col_names
        self._annotation_layers = annotation_layers

        self._url_prefix = url_prefix
        self._data_columns = self._compute_data_columns()
        self.initialize_state()

    def _compute_data_columns(self):
        data_columns = []
        for _, kws in self._annotation_layers.items():
            if 'points' in kws:
                for col in kws['points']:
                    data_columns.append(col)
            if 'lines' in kws:
                for colpair in kws['lines']:
                    data_columns.extend(colpair)
            if 'spheres' in kws:
                for colpair in kws['spheres']:
                    data_columns.extend(colpair)

        for _, idcols in self._selected_ids.items():
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
        """Generate a new Neuroglancer state with layers as needed for the schema.
        
        Parameters
        ----------
        base_state : str, optional
            Optional initial state to build on, described by its JSON. By default None.
        """
        self._reset_state(base_state)
        self._temp_viewer.set_resolution(self._resolution)
        self._add_layers()
        self._temp_viewer.set_view_options()

    def render_state(self, data=None, base_state=None, return_as='url', static_content_source=default_static_content_source,
                     url_prefix=None, link_text='Neuroglancer Link'):
        """Build a Neuroglancer state out of a DataFrame.
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            DataFrame to use as a point source. By default None, for which
            it will return only the base_state and any fixed values.
        base_state : str, optional
            Initial state to build on, expressed as Neuroglancer JSON. By default None
        return_as : ['url', 'viewer', 'html', 'json'], optional
            Choice of output types. Note that if a viewer is returned, the state is not reset.
            By default 'url'
        static_content_source : str, optional
            Sets the neuroglancer web site source to use for a viewer, if a viewer is returned.
            Optional, default is None. If none is set, uses the default value from neuroglancer_annotation_ui.utils.
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
        if base_state is not None:
            self.initialize_state(base_state=base_state)
    
        if url_prefix is None:
            url_prefix = self._url_prefix

        if data is not None:
            self._validate_dataframe(data)
            self._render_data(data)

        if return_as == 'viewer':
            if static_content_source is not None:
                set_static_content_source(static_content_source)
            return self.viewer
        elif return_as == 'url':
            out = self._temp_viewer.as_url(prefix=url_prefix)
            self.initialize_state()
            return out
        elif return_as == 'html':
            out = HTML(self._temp_viewer.as_url(prefix=url_prefix, as_html=True, link_text=link_text))
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

    def render_state(self, indices=None, *args, **kwargs):
        """Make a Neuroglancer state with a dataframe and selection of indices. This would generally be
        better done by slicing the dataframe itself, but can be useful for interoperability with automation.
        Arguments after indices are passed to DataStateBuilder.render_state.
        
        Parameters
        ----------
        indices : Collection of ints, optional
            Indices to choose from the dataframe via iloc.  By default None, which plots the whole dataframe.

        Returns
        -------
        string or neuroglancer.Viewer
            A Neuroglancer state with layers, annotations, and selected objects determined by the data.
        """

        if data is not None:
            if indices is None:
                data_render = data
            else:
                data_render = data.loc[indices]
        else:
            data_render = None
        return super(FilteredDataStateBuilder, self).render_state(*args, **kwargs)


class ChainedStateBuilder():
    def __init__(self, statebuilders):
        """Builds a collection of states that sequentially add annotations based on a sequence of dataframes.
        
        Parameters
        ----------
        statebuilders : list
            List of DataStateBuilders, in same order as dataframes will be passed
        """
        self._statebuilders = statebuilders
        if len(self._statebuilders) == 0:
            raise ValueError('Must have at least one statebuilder')

    def render_state(self, data_list=None, base_state=None, return_as='url', url_prefix=None):
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
            url_prefix: string, optional (default=None). Overrides the default neuroglancer url for url generation.
        """
        if data_list is None:
            data_list = len(self._statebuilders) * [None]

        if len(data_list) != len(self._statebuilders):
            raise ValueError('Must have as many dataframes as statebuilders')

        temp_state = base_state
        for builder, data in zip(self._statebuilders[:-1], data_list[:-1]):
            temp_state = builder.render_state(data=data,
                                              base_state=temp_state,
                                              return_as='json')
        last_builder = self._statebuilders[-1]
        return last_builder.render_state(data = data_list[-1],
                                         base_state=temp_state,
                                         return_as=return_as,
                                         url_prefix=url_prefix)
