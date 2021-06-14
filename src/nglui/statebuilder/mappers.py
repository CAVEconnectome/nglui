import numpy as np
from nglui import annotation
import pandas as pd
from collections.abc import Collection


def _multipoint_transform(row, pt_columns, squeeze_cols):
    """Reshape dataframe to accomodate multiple points in a single row"""
    pts = {pcol: np.atleast_2d(row[pcol]) for pcol in pt_columns}
    row_dat = {}
    for col in row.index:
        if col in pt_columns:
            if col in squeeze_cols:
                # Revert the atleast_2d if it's a 1-d array, namely sphere radius
                row_dat[col] = pts[col].squeeze().tolist()
            else:
                row_dat[col] = pts[col].tolist()
        else:
            row_dat[col] = row[col]
    return pd.DataFrame(row_dat)


class SelectionMapper(object):
    """Class for configuring object selections based on root id

    Parameters
    ----------
    data_columns : str or list, optional
        Name (or list of names) of the data columns to get ids from. Default is None.
    fixed_ids : list, optional
        List of ids to select irrespective of data.
    """

    def __init__(
        self, data_columns=None, fixed_ids=None, fixed_id_colors=None, color_column=None
    ):
        if isinstance(data_columns, str):
            data_columns = [data_columns]
        self._config = dict(
            data_columns=data_columns,
            fixed_ids=fixed_ids,
            fixed_id_colors=fixed_id_colors,
            color_column=color_column,
        )

    @property
    def data_columns(self):
        if self._config.get("data_columns", None) is None:
            return []
        else:
            return self._config.get("data_columns")

    @property
    def fixed_ids(self):
        if self._config.get("fixed_ids", None) is None:
            return np.array([], dtype=np.uint64)
        else:
            return np.array(self._config.get("fixed_ids", []), dtype=np.uint64)

    @property
    def fixed_id_colors(self):
        if self._config.get("fixed_id_colors", None) is None:
            return []
        else:
            return list(self._config.get("fixed_id_colors", None))

    @property
    def color_column(self):
        return self._config.get("color_column", None)

    def selected_ids(self, data):
        """Uses the rules to generate a list of ids from a dataframe."""
        selected_ids = []
        if data is not None:
            for col in self.data_columns:
                selected_ids.append(data[col].values.astype(np.uint64))
        selected_ids.append(self.fixed_ids)
        return np.concatenate(selected_ids)

    def seg_colors(self, data):
        colors = {}
        if len(self.fixed_id_colors) == len(self.fixed_ids):
            for ii, oid in enumerate(self.fixed_ids):
                colors[oid] = self.fixed_id_colors[ii]

        if self.color_column is not None:
            clist = data[self.color_column].to_list()
            for col in self.data_columns:
                for ii, oid in enumerate(data[col]):
                    colors[oid] = clist[ii]

        return colors


class AnnotationMapperBase(object):
    def __init__(
        self,
        type,
        data_columns,
        description_column,
        linked_segmentation_column,
        tag_column,
        group_column,
        set_position,
        gather_linked_segmentations,
        share_linked_segmentations,
        multipoint,
    ):

        self._config = dict(
            type=type,
            data_columns=data_columns,
            array_data=False,
            description_column=description_column,
            linked_segmentation_column=linked_segmentation_column,
            tag_column=tag_column,
            group_column=group_column,
            set_position=set_position,
            gather_linked_segmentations=gather_linked_segmentations,
            share_linked_segmentations=share_linked_segmentations,
            multipoint=multipoint,
        )
        self._tag_map = None

    @property
    def type(self):
        return self._config.get("type", None)

    @property
    def data_columns(self):
        return self._config.get("data_columns", None)

    @property
    def description_column(self):
        return self._config.get("description_column", None)

    @property
    def linked_segmentation_column(self):
        return self._config.get("linked_segmentation_column", None)

    @property
    def tag_column(self):
        return self._config.get("tag_column", None)

    @property
    def group_column(self):
        return self._config.get("group_column", None)

    @property
    def gather_linked_segmentations(self):
        return self._config.get("gather_linked_segmentations", True)

    @property
    def share_linked_segmentations(self):
        return self._config.get("share_linked_segmentations", False)

    @property
    def set_position(self):
        return self._config.get("set_position", False)

    @property
    def multipoint(self):
        return self._config.get("multipoint", False)

    def multipoint_reshape(self, data, pt_columns, squeeze_cols=[]):
        if data is None or len(data) == 0:
            return data
        else:
            return pd.concat(
                data.apply(
                    lambda x: _multipoint_transform(
                        x, pt_columns=pt_columns, squeeze_cols=squeeze_cols
                    ),
                    axis=1,
                ).tolist(),
                ignore_index=True,
            )

    @property
    def tag_map(self):
        if self._tag_map is None:
            return {}
        else:
            return self._tag_map

    @tag_map.setter
    def tag_map(self, tag_list):
        if tag_list is None:
            self._tag_map = {}
        else:
            self._tag_map = {tag: ii + 1 for ii, tag in enumerate(tag_list)}

    @property
    def array_data(self):
        return self._config.get("array_data", False)

    @array_data.setter
    def array_data(self, new_array_data):
        self._config["array_data"] = new_array_data
        if new_array_data:
            self._config["data_columns"] = self._default_array_data_columns()
            self._config["description_column"] = None
            self._config["linked_segmentation_column"] = None
            self._config["tag_column"] = None

    def _default_array_data_columns(self):
        return []

    def _assign_tags(self, data):
        if self.tag_column is not None:
            anno_tags = []
            for row in data[self.tag_column]:
                if isinstance(row, Collection) and not isinstance(row, str):
                    add_annos = [self.tag_map.get(r, None) for r in row]
                else:
                    add_annos = [self.tag_map.get(row, None)]
                anno_tags.append(add_annos)
        else:
            anno_tags = [[None] for x in range(len(data))]
        return anno_tags

    def _render_data(self, data):
        # Set per subclass
        return None

    def _linked_segmentations(self, data):
        if self.linked_segmentation_column is not None:
            seg_array = np.vstack(data[self.linked_segmentation_column].values)
            linked_segs = [row[~pd.isnull(row)].astype(int) for row in seg_array]
        else:
            linked_segs = [None for x in range(len(data))]
        return linked_segs

    def _descriptions(self, data):
        if self.description_column is not None:
            descriptions = data[self.description_column].values
        else:
            descriptions = [None for x in range(len(data))]
        return descriptions

    def _add_groups(self, groups, annos):
        vals, inverse = np.unique(groups, return_inverse=True)
        inv_inds = np.flatnonzero(~pd.isnull(vals))
        group_annos = []

        for ii in inv_inds:
            anno_to_group = [annos[jj] for jj in np.flatnonzero(inverse == ii)]
            group_annos.append(
                annotation.group_annotations(
                    anno_to_group,
                    return_all=False,
                    gather_linked_segmentations=self.gather_linked_segmentations,
                    share_linked_segmentations=self.share_linked_segmentations,
                )
            )
        annos.extend(group_annos)
        return annos

    def _get_position(self, data):
        if len(data) > 0 and self.set_position is True:
            return list(data[self.data_columns[0]].iloc[0])
        else:
            return None


class PointMapper(AnnotationMapperBase):
    """Sets rules to map dataframes to point annotations

    Parameters
    ----------
    point_column : str, optional
        Column name with 3d position data
    array_data : bool, optional
        If True, the expected data is a Nx3 numpy array. This will only work for point positions,
        to use any other features you must build a dataframe.
    decription_column : str, optional
        Column name with string data for annotation descriptions
    linked_segmentation_column : str, optional
        Column name for root ids to link to annotations
    tag_column : str, optional
        Column name for categorical tag data. Tags must match those set in the
        annotation layer.
    group_column : str, optional
        Column name for grouping data. Data in this row should be numeric with possible NaNs.
        Rows with the same non-NaN value will be collected into a grouped annotation.
    set_position : bool, optional
        If set to True, moves the position to center on the first point in the
        data.
    multipoint: bool, optional
        If True, permits multiple points in a given row, sharing data in other columns. Default is False.
    """

    def __init__(
        self,
        point_column=None,
        description_column=None,
        linked_segmentation_column=None,
        tag_column=None,
        group_column=None,
        gather_linked_segmentations=True,
        share_linked_segmentations=False,
        set_position=False,
        multipoint=False,
    ):
        super(PointMapper, self).__init__(
            type="point",
            data_columns=[point_column],
            description_column=description_column,
            linked_segmentation_column=linked_segmentation_column,
            tag_column=tag_column,
            group_column=group_column,
            gather_linked_segmentations=gather_linked_segmentations,
            share_linked_segmentations=share_linked_segmentations,
            set_position=set_position,
            multipoint=multipoint,
        )

    def _default_array_data_columns(self):
        return ["pt"]

    def _render_data(self, data):
        if self.array_data:
            data = pd.DataFrame(data={"pt": np.array(data).tolist()})

        if self.multipoint:
            data = self.multipoint_reshape(data, pt_columns=self.data_columns)

        col = self.data_columns[0]
        relinds = ~pd.isnull(data[col])

        pts = np.vstack(data[col][relinds])
        descriptions = self._descriptions(data[relinds])

        linked_segs = self._linked_segmentations(data[relinds])
        tags = self._assign_tags(data)
        annos = [
            annotation.point_annotation(
                pt, description=d, linked_segmentation=ls, tag_ids=t
            )
            for pt, d, ls, t in zip(pts, descriptions, linked_segs, tags)
        ]

        if self.group_column is not None:
            groups = data[self.group_column].values[relinds]
            annos = self._add_groups(groups, annos)

        return annos


class LineMapper(AnnotationMapperBase):
    """Sets rules to map dataframes to line annotations

    Parameters
    ----------
    point_column_a : str
        Column name with 3d position data for the first point of the line.
        Must be set if array_data is False (the default)
    point_column_b : str
        Column name with 3d position data for the first point of the line.
        Must be set if array_data is False (the default)
    decription_column : str, optional
        Column name with string data for annotation descriptions
    linked_segmentation_column : str, optional
        Column name for root ids to link to annotations
    tag_column : str, optional
        Column name for categorical tag data. Tags must match those set in the annotation layer.
    group_column : str, optional
        Column name for grouping data. Data in this row should be numeric with possible NaNs.
        Rows with the same non-NaN value will be collected into a grouped annotation.
    set_position : bool, optional
        If set to True, moves the position to center on the first point in the data (using point_column_a).
    multipoint: bool, optional
        If True, permits multiple points in a given row, sharing data in other columns.
        Each point row must have the same number of points. Default is False.
    """

    def __init__(
        self,
        point_column_a=None,
        point_column_b=None,
        description_column=None,
        linked_segmentation_column=None,
        tag_column=None,
        group_column=None,
        gather_linked_segmentations=True,
        share_linked_segmentations=False,
        set_position=False,
        multipoint=False,
    ):
        super(LineMapper, self).__init__(
            type="line",
            data_columns=[point_column_a, point_column_b],
            description_column=description_column,
            linked_segmentation_column=linked_segmentation_column,
            tag_column=tag_column,
            group_column=group_column,
            gather_linked_segmentations=gather_linked_segmentations,
            share_linked_segmentations=share_linked_segmentations,
            set_position=set_position,
            multipoint=multipoint,
        )

    def _default_array_data_columns(self):
        return ["pt_a", "pt_b"]

    def _render_data(self, data):
        if self.array_data:
            data = pd.DataFrame(
                data={"pt_a": data[0].tolist(), "pt_b": data[1].tolist()}
            )

        if self.multipoint:
            data = self.multipoint_reshape(data, pt_columns=self.data_columns)

        colA, colB = self.data_columns
        relinds = np.logical_and(~pd.isnull(data[colA]), ~pd.isnull(data[colB]))

        ptAs = np.vstack(data[colA][relinds])
        ptBs = np.vstack(data[colB][relinds])
        descriptions = self._descriptions(data[relinds])
        linked_segs = self._linked_segmentations(data[relinds])
        tags = self._assign_tags(data)
        annos = [
            annotation.line_annotation(
                ptA, ptB, description=d, linked_segmentation=ls, tag_ids=t
            )
            for ptA, ptB, d, ls, t in zip(ptAs, ptBs, descriptions, linked_segs, tags)
        ]

        if self.group_column is not None:
            groups = data[self.group_column].values[relinds]
            annos = self._add_groups(groups, annos)

        return annos


class SphereMapper(AnnotationMapperBase):
    """Sets rules to map dataframes to sphere annotations

    Parameters
    ----------
    center_column : str
        Column name with 3d position data for the center of the sphere
    radius_column : str
        Column name with a radius for the sphere (in nm)
    decription_column : str, optional
        Column name with string data for annotation descriptions
    linked_segmentation_column : str, optional
        Column name for root ids to link to annotations
    tag_column : str, optional
        Column name for categorical tag data. Tags must match those set in the annotation layer.
    group_column : str, optional
        Column name for grouping data. Data in this row should be numeric with possible NaNs.
        Rows with the same non-NaN value will be collected into a grouped annotation.
    set_position : bool, optional
        If set to True, moves the position to center on the first point in the data.
    multipoint: bool, optional
        If True, permits multiple points in a given row, sharing data in other columns.
        Each point row must have the same number of points. Default is False.
    """

    def __init__(
        self,
        center_column=None,
        radius_column=None,
        description_column=None,
        linked_segmentation_column=None,
        tag_column=None,
        group_column=None,
        gather_linked_segmentations=True,
        share_linked_segmentations=False,
        z_multiplier=0.1,
        set_position=False,
        multipoint=False,
    ):
        super(SphereMapper, self).__init__(
            type="sphere",
            data_columns=[center_column, radius_column],
            description_column=description_column,
            linked_segmentation_column=linked_segmentation_column,
            tag_column=tag_column,
            group_column=group_column,
            gather_linked_segmentations=gather_linked_segmentations,
            share_linked_segmentations=share_linked_segmentations,
            set_position=set_position,
            multipoint=multipoint,
        )
        self._z_multiplier = z_multiplier

    def _default_array_data_columns(self):
        return ["ctr_pt", "rad"]

    def _render_data(self, data):
        if self.array_data:
            data = pd.DataFrame(data={"ctr_pt": data[0].tolist(), "rad": data[1]})

        col_ctr, col_rad = self.data_columns
        if self.multipoint:
            data = self.multipoint_reshape(
                data, pt_columns=self.data_columns, squeeze_cols=[col_rad]
            )

        relinds = np.logical_and(~pd.isnull(data[col_ctr]), ~pd.isnull(data[col_rad]))

        pts = np.vstack(data[col_ctr][relinds])
        rs = data[col_rad][relinds].values
        descriptions = self._descriptions(data[relinds])
        linked_segs = self._linked_segmentations(data[relinds])
        tags = self._assign_tags(data)
        annos = [
            annotation.sphere_annotation(
                pt,
                r,
                description=d,
                linked_segmentation=ls,
                tag_ids=t,
                z_multiplier=self._z_multiplier,
            )
            for pt, r, d, ls, t in zip(pts, rs, descriptions, linked_segs, tags)
        ]

        if self.group_column is not None:
            groups = data[self.group_column].values[relinds]
            annos = self._add_groups(groups, annos)

        return annos


class BoundingBoxMapper(AnnotationMapperBase):
    def __init__(
        self,
        point_column_a=None,
        point_column_b=None,
        description_column=None,
        linked_segmentation_column=None,
        tag_column=None,
        group_column=None,
        gather_linked_segmentations=True,
        share_linked_segmentations=False,
        set_position=False,
        multipoint=False,
    ):
        super(BoundingBoxMapper, self).__init__(
            type="axis_aligned_bounding_box",
            data_columns=[point_column_a, point_column_b],
            description_column=description_column,
            linked_segmentation_column=linked_segmentation_column,
            tag_column=tag_column,
            group_column=group_column,
            gather_linked_segmentations=gather_linked_segmentations,
            share_linked_segmentations=share_linked_segmentations,
            set_position=set_position,
            multipoint=multipoint,
        )

    def _default_array_data_columns(self):
        return ["pt_a", "pt_b"]

    def _render_data(self, data):
        if self.array_data:
            data = pd.DataFrame(
                data={"pt_a": data[0].tolist(), "pt_b": data[1].tolist()}
            )

        if self.multipoint:
            data = self.multipoint_reshape(data, pt_columns=self.data_columns)

        colA, colB = self.data_columns
        relinds = np.logical_and(~pd.isnull(data[colA]), ~pd.isnull(data[colB]))

        ptAs = np.vstack(data[colA][relinds])
        ptBs = np.vstack(data[colB][relinds])
        descriptions = self._descriptions(data[relinds])
        linked_segs = self._linked_segmentations(data[relinds])
        tags = self._assign_tags(data)
        annos = [
            annotation.bounding_box_annotation(
                ptA, ptB, description=d, linked_segmentation=ls, tag_ids=t
            )
            for ptA, ptB, d, ls, t in zip(ptAs, ptBs, descriptions, linked_segs, tags)
        ]

        if self.group_column is not None:
            groups = data[self.group_column].values[relinds]
            annos = self._add_groups(groups, annos)

        return annos


class SplitPointMapper(object):
    """Mapper to create split points in a segmentation layer.

    Parameters
    ----------
    id_column : str
        Column name for segment ids. The id column must contain the same id in all rows.
    point_column : str
        Name of the column containing points in space.
    team_column : str
        Name of the column describing the team for the points. The contents of the column should have two values, by default "red" and "blue".
    team_names : list, optional
        List of two values for the team names used in the team column. The first is mapped to red points, the second blue. Default is ["red", "blue"].
    supervoxel_column : str or None, optional
        Name of a column providing supervoxel ids. If None (default), the supervoxel must be looked up on the server.
    focus : bool, optional
        If True, sets the focus on the split tool and sets the position to the center of split points. Default is True.

    Returns
    -------
    SplitPointMapper instance to pass to a segmentation layer.
    """

    def __init__(
        self,
        id_column,
        point_column,
        team_column,
        team_names=["red", "blue"],
        supervoxel_column=None,
        focus=True,
    ):
        self.id_column = id_column
        self.point_column = point_column
        self.team_column = team_column
        self.team_names = team_names
        self.supervoxel_column = supervoxel_column
        self.focus = focus

    def _render_data(self, df):
        if len(df) == 0:
            return None, np.atleast_2d([]), np.atleast_2d([]), []

        team_pts = []
        team_svs = []
        for team_name in self.team_names:
            team_df = df.query(f"{self.team_column} == @team_name")
            if len(team_df) > 0:
                team_pts.append(np.vstack(team_df[self.point_column].values))
                if self.supervoxel_column:
                    team_svs.append(team_df[self.supervoxel_column].values)
                else:
                    team_svs.append(None)
            else:
                team_pts.append(np.atleast_2d([]))

        seg_id = np.unique(df[self.id_column])
        if len(seg_id) > 1:
            raise ValueError("Multiple seg ids provided for SplitPointMapper")
        else:
            seg_id = seg_id[0]

        return seg_id, team_pts[0], team_pts[1], team_svs[0], team_svs[1]
