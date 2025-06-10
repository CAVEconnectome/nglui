from typing import TYPE_CHECKING, Literal, Optional, Union

from caveclient import CAVEclient

from ..site_utils import neuroglancer_url
from .base import *
from .ngl_components import *
from .shaders import DEFAULT_SHADER_MAP

if TYPE_CHECKING:
    from nglui.segmentprops import SegmentProperties


DEFAULT_POSTSYN_COLOR = "turquoise"
DEFAULT_PRESYN_COLOR = "tomato"

MAX_URL_LENGTH = 1_750_000


def sort_dataframe_by_root_id(
    df, root_id_column, ascending=False, num_column="n_times", drop=False
) -> pd.DataFrame:
    """Sort a dataframe so that rows belonging to the same root id are together, ordered by how many times the root id appears.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to sort
    root_id_column : str
        Column name to use for sorting root ids
    ascending : bool, optional
        Whether to sort ascending (lowest count to highest) or not, by default False
    num_column : str, optional
        Temporary column name to use for count information, by default 'n_times'
    drop : bool, optional
        If True, drop the additional column when returning.

    Returns
    -------
    pd.DataFrame
    """
    cols = df.columns[df.columns != root_id_column]
    if len(cols) == 0:
        df = df.copy()
        df[num_column] = df[root_id_column]
        use_column = num_column
    else:
        use_column = cols[0]

    df[num_column] = df.groupby(root_id_column).transform("count")[use_column]
    if drop:
        return df.sort_values(
            by=[num_column, root_id_column], ascending=ascending
        ).drop(columns=[num_column])
    else:
        return df.sort_values(by=[num_column, root_id_column], ascending=ascending)


def add_random_column(
    df: pd.DataFrame,
    col_prefix: str = "sample_",
    n_cols: int = 1,
) -> pd.DataFrame:
    """Add a column of uniformly distributed random numbers to a dataframe.
    This can be useful to use to subsample ids in a segment property efficiently.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that will be passed into a column
    n_cols : str, optional
        Name of the numerical column to use, by default 'sample_'
    num_column : int, optional
        Number of distinct random columns, by default 1

    Returns
    -------
    pd.DataFrame
        Copy of the input dataframe with the random columns added
    """
    df = df.copy()
    for ii in range(n_cols):
        df[f"{col_prefix}{ii}"] = np.random.rand(len(df))
    return df


def make_point_state(
    client: CAVEclient,
    point_column: str = "pt_position",
    segment_column: str = "pt_root_id",
    description_column: Optional[str] = None,
    tag_column: Optional[str] = None,
    data_resolution: Optional[list] = None,
    tags: Optional[list] = None,
    layer_name: str = "Points",
    shader: Optional[Union[bool, str]] = True,
) -> ViewerState:
    """Generate a state builder that puts points on a single column with a linked segmentaton id

    Parameters
    ----------
    client : CAVEclient
        CAVEclient configured for the datastack desired
    point_column : str, optional
        column in dataframe to pull points from. Defaults to "pre_pt_position".
    linked_seg_column : str, optional
        column to link to segmentation, None for no column. Defaults to "pt_root_id".
    group_column : str, or list, optional
        column(s) to group annotations by, None for no grouping (default=None)
    tag_column : str, optional
        column to use for tags, None for no tags (default=None)
    description_column : str, optional
        column to use for descriptions, None for no descriptions (default=None)
    contrast : list, optional
        Two elements specifying the black level and white level as
        floats between 0 and 1, by default None. If None, no contrast
        is set.
    view_kws : dict, optional
        dictionary of view keywords to configure neuroglancer view

    Returns
    -------
    ViewerState:
        A viewerstate to make points with linked segmentations
    """
    if shader is True:
        shader = DEFAULT_SHADER_MAP.get("points")

    return (
        ViewerState(infer_coordinates=True)
        .add_layers_from_client(
            client,
        )
        .add_points(
            data=DataMap(),
            name=layer_name,
            point_column=point_column,
            segment_column=segment_column,
            description_column=description_column,
            tag_column=tag_column,
            data_resolution=data_resolution,
            tags=tags,
            shader=shader,
        )
    )


def make_line_state(
    client: CAVEclient,
    point_a_column="pre_pt_position",
    point_b_column="post_pt_position",
    segment_column="pt_root_id",
    description_column=None,
    tag_column=None,
    data_resolution=None,
    layer_name="lines",
    shader=None,
):
    if shader is True:
        shader = DEFAULT_SHADER_MAP.get("lines")
    return (
        ViewerState(infer_coordinates=True)
        .add_layers_from_client(
            client,
        )
        .add_lines(
            data=DataMap(),
            name=layer_name,
            point_a_column=point_a_column,
            point_b_column=point_b_column,
            segment_column=segment_column,
            description_column=description_column,
            tag_column=tag_column,
            data_resolution=data_resolution,
            shader=shader,
        )
    )
