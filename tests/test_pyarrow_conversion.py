"""Tests for PyArrow DataFrame conversion functionality."""

import numpy as np
import pandas as pd
import pytest

try:
    import pyarrow as pa

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

from nglui.segmentprops import SegmentProperties
from nglui.statebuilder import AnnotationLayer, ViewerState
from nglui.utils import convert_arrow_to_numpy

# Skip all tests if PyArrow is not available
pytestmark = pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not installed")


@pytest.fixture
def pyarrow_df():
    """Create a DataFrame with PyArrow-backed columns of various types."""
    return pd.DataFrame(
        {
            "id": pd.array([1, 2, 3, 4, 5], dtype=pd.ArrowDtype(pa.int64())),
            "name": pd.array(
                ["Alice", "Bob", "Charlie", "David", "Eve"],
                dtype=pd.ArrowDtype(pa.string()),
            ),
            "value": pd.array(
                [10.5, 20.3, 30.1, 40.9, 50.2], dtype=pd.ArrowDtype(pa.float64())
            ),
            "flag": pd.array(
                [True, False, True, False, True], dtype=pd.ArrowDtype(pa.bool_())
            ),
        }
    )


@pytest.fixture
def pyarrow_string_dtype_df():
    """Create a DataFrame with pd.StringDtype(storage='pyarrow') columns."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": pd.array(
                ["Alice", "Bob", "Charlie", "David", "Eve"],
                dtype=pd.StringDtype(storage="pyarrow"),
            ),
            "category": pd.array(
                ["A", "B", "A", "C", "B"], dtype=pd.StringDtype(storage="pyarrow")
            ),
        }
    )


@pytest.fixture
def mixed_dtype_df():
    """Create a DataFrame with mixed PyArrow and numpy dtypes."""
    return pd.DataFrame(
        {
            "id": pd.array([1, 2, 3, 4, 5], dtype=pd.ArrowDtype(pa.int64())),
            "numpy_col": np.array([10, 20, 30, 40, 50]),
            "string_col": pd.array(
                ["A", "B", "C", "D", "E"], dtype=pd.ArrowDtype(pa.string())
            ),
            "regular_float": [1.1, 2.2, 3.3, 4.4, 5.5],
        }
    )


@pytest.fixture
def pyarrow_segment_props_df():
    """Create a PyArrow DataFrame suitable for SegmentProperties."""
    return pd.DataFrame(
        {
            "seg_id": pd.array(np.arange(0, 100), dtype=pd.ArrowDtype(pa.int64())),
            "cell_type": pd.array(
                30 * ["ct_a"] + 30 * ["ct_b"] + 40 * ["ct_c"],
                dtype=pd.ArrowDtype(pa.string()),
            ),
            "category": pd.array(
                50 * ["cat_1"] + 40 * ["cat_2"] + 10 * [None],
                dtype=pd.ArrowDtype(pa.string()),
            ),
            "number_int": pd.array(
                np.arange(300, 400), dtype=pd.ArrowDtype(pa.int32())
            ),
            "number_float": pd.array(
                np.arange(300, 400) + 0.1, dtype=pd.ArrowDtype(pa.float32())
            ),
            "tag_a": pd.array(
                90 * [False] + 10 * [True], dtype=pd.ArrowDtype(pa.bool_())
            ),
            "tag_b": pd.array(
                95 * [True] + 5 * [False], dtype=pd.ArrowDtype(pa.bool_())
            ),
        }
    )


@pytest.fixture
def pyarrow_annotation_df():
    """Create a PyArrow DataFrame suitable for annotation layers."""
    return pd.DataFrame(
        {
            "x": pd.array([100, 200, 300, 400], dtype=pd.ArrowDtype(pa.int64())),
            "y": pd.array([150, 250, 350, 450], dtype=pd.ArrowDtype(pa.int64())),
            "z": pd.array([10, 20, 30, 40], dtype=pd.ArrowDtype(pa.int64())),
            "segment_id": pd.array(
                [12345, 67890, 11111, 22222], dtype=pd.ArrowDtype(pa.int64())
            ),
            "description": pd.array(
                ["Point A", "Point B", "Point C", "Point D"],
                dtype=pd.ArrowDtype(pa.string()),
            ),
        }
    )


class TestConvertArrowToNumpy:
    """Test the convert_arrow_to_numpy utility function."""

    def test_convert_all_columns(self, pyarrow_df):
        """Test converting all PyArrow columns."""
        result = convert_arrow_to_numpy(pyarrow_df)

        # Check that columns are no longer PyArrow backed
        assert not isinstance(result["id"].dtype, pd.ArrowDtype)
        assert not isinstance(result["name"].dtype, pd.ArrowDtype)
        assert not isinstance(result["value"].dtype, pd.ArrowDtype)
        assert not isinstance(result["flag"].dtype, pd.ArrowDtype)

        # Check that values are preserved
        assert result["id"].tolist() == [1, 2, 3, 4, 5]
        assert result["name"].tolist() == ["Alice", "Bob", "Charlie", "David", "Eve"]
        assert result["flag"].tolist() == [True, False, True, False, True]

        # Original DataFrame should be unchanged
        assert isinstance(pyarrow_df["id"].dtype, pd.ArrowDtype)

    def test_convert_specific_columns(self, mixed_dtype_df):
        """Test converting only specific columns."""
        result = convert_arrow_to_numpy(mixed_dtype_df, columns=["id", "string_col"])

        # Specified columns should be converted
        assert not isinstance(result["id"].dtype, pd.ArrowDtype)
        assert not isinstance(result["string_col"].dtype, pd.ArrowDtype)

        # Unspecified columns should remain unchanged
        assert result["numpy_col"].dtype == mixed_dtype_df["numpy_col"].dtype
        assert result["regular_float"].dtype == mixed_dtype_df["regular_float"].dtype

    def test_convert_single_column_string(self, pyarrow_df):
        """Test converting a single column specified as string."""
        result = convert_arrow_to_numpy(pyarrow_df, columns="name")

        # Only 'name' should be converted
        assert not isinstance(result["name"].dtype, pd.ArrowDtype)
        assert isinstance(result["id"].dtype, pd.ArrowDtype)
        assert isinstance(result["value"].dtype, pd.ArrowDtype)

    def test_convert_with_none_in_list(self, pyarrow_df):
        """Test that None values in columns list are filtered out."""
        result = convert_arrow_to_numpy(pyarrow_df, columns=["id", None, "name", None])

        # Only non-None columns should be converted
        assert not isinstance(result["id"].dtype, pd.ArrowDtype)
        assert not isinstance(result["name"].dtype, pd.ArrowDtype)
        assert isinstance(result["value"].dtype, pd.ArrowDtype)
        assert isinstance(result["flag"].dtype, pd.ArrowDtype)

    def test_convert_nested_list(self, pyarrow_annotation_df):
        """Test converting with nested list (for point columns)."""
        result = convert_arrow_to_numpy(
            pyarrow_annotation_df, columns=[["x", "y", "z"], "segment_id"]
        )

        # All specified columns should be converted
        assert not isinstance(result["x"].dtype, pd.ArrowDtype)
        assert not isinstance(result["y"].dtype, pd.ArrowDtype)
        assert not isinstance(result["z"].dtype, pd.ArrowDtype)
        assert not isinstance(result["segment_id"].dtype, pd.ArrowDtype)

        # Unspecified column should remain PyArrow
        assert isinstance(result["description"].dtype, pd.ArrowDtype)

    def test_convert_nonexistent_columns(self, pyarrow_df):
        """Test that nonexistent columns are safely ignored."""
        result = convert_arrow_to_numpy(pyarrow_df, columns=["id", "nonexistent_col"])

        # Should convert only existing columns without error
        assert not isinstance(result["id"].dtype, pd.ArrowDtype)
        assert isinstance(result["name"].dtype, pd.ArrowDtype)

    def test_convert_empty_columns_list(self, pyarrow_df):
        """Test with empty columns list returns original DataFrame."""
        result = convert_arrow_to_numpy(pyarrow_df, columns=[])

        # Should return original (with copy)
        assert isinstance(result["id"].dtype, pd.ArrowDtype)
        pd.testing.assert_frame_equal(result, pyarrow_df)

    def test_convert_string_dtype_pyarrow(self, pyarrow_string_dtype_df):
        """Test conversion of pd.StringDtype(storage='pyarrow')."""
        result = convert_arrow_to_numpy(pyarrow_string_dtype_df)

        # StringDtype with pyarrow storage should be converted to object
        assert result["name"].dtype == object
        assert result["category"].dtype == object
        assert result["name"].tolist() == ["Alice", "Bob", "Charlie", "David", "Eve"]

    def test_no_pyarrow_columns_returns_unchanged(self):
        """Test that DataFrame with no PyArrow columns is returned unchanged."""
        regular_df = pd.DataFrame(
            {"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.1, 2.2, 3.3]}
        )
        result = convert_arrow_to_numpy(regular_df)

        pd.testing.assert_frame_equal(result, regular_df)

    def test_convert_timestamp_columns(self):
        """Test conversion of PyArrow timestamp columns."""
        df_with_timestamps = pd.DataFrame(
            {
                "datetime": pd.array(
                    pd.date_range("2024-01-01", periods=3),
                    dtype=pd.ArrowDtype(pa.timestamp("ns")),
                ),
                "datetime_tz": pd.array(
                    pd.date_range("2024-01-01", periods=3, tz="UTC"),
                    dtype=pd.ArrowDtype(pa.timestamp("ns", tz="UTC")),
                ),
            }
        )

        result = convert_arrow_to_numpy(df_with_timestamps)

        # Timestamps should be converted to datetime64[ns]
        assert result["datetime"].dtype == np.dtype("datetime64[ns]")
        assert result["datetime_tz"].dtype == np.dtype("datetime64[ns]")

    def test_convert_various_int_types(self):
        """Test conversion of various PyArrow integer types."""
        df = pd.DataFrame(
            {
                "int8": pd.array([1, 2, 3], dtype=pd.ArrowDtype(pa.int8())),
                "int16": pd.array([1, 2, 3], dtype=pd.ArrowDtype(pa.int16())),
                "int32": pd.array([1, 2, 3], dtype=pd.ArrowDtype(pa.int32())),
                "int64": pd.array([1, 2, 3], dtype=pd.ArrowDtype(pa.int64())),
                "uint8": pd.array([1, 2, 3], dtype=pd.ArrowDtype(pa.uint8())),
                "uint32": pd.array([1, 2, 3], dtype=pd.ArrowDtype(pa.uint32())),
            }
        )

        result = convert_arrow_to_numpy(df)

        # Check that all are converted to pandas nullable integer types
        assert result["int8"].dtype == pd.Int8Dtype()
        assert result["int16"].dtype == pd.Int16Dtype()
        assert result["int32"].dtype == pd.Int32Dtype()
        assert result["int64"].dtype == pd.Int64Dtype()
        assert result["uint8"].dtype == pd.UInt8Dtype()
        assert result["uint32"].dtype == pd.UInt32Dtype()

    def test_convert_float_types(self):
        """Test conversion of PyArrow float types."""
        df = pd.DataFrame(
            {
                "float32": pd.array([1.1, 2.2, 3.3], dtype=pd.ArrowDtype(pa.float32())),
                "float64": pd.array([1.1, 2.2, 3.3], dtype=pd.ArrowDtype(pa.float64())),
            }
        )

        result = convert_arrow_to_numpy(df)

        # Check that floats are converted to pandas nullable float types
        assert result["float32"].dtype == pd.Float32Dtype()
        assert result["float64"].dtype == pd.Float64Dtype()


class TestSegmentPropertiesWithPyArrow:
    """Test SegmentProperties.from_dataframe with PyArrow DataFrames."""

    def test_segment_props_from_pyarrow_df(self, pyarrow_segment_props_df):
        """Test creating SegmentProperties from PyArrow DataFrame."""
        props = SegmentProperties.from_dataframe(
            pyarrow_segment_props_df,
            id_col="seg_id",
            label_col="seg_id",
            number_cols=["number_int", "number_float"],
            tag_value_cols="cell_type",
            tag_bool_cols=["tag_a", "tag_b"],
        )

        assert len(props) == 100
        assert len(props.property_description()) == 4

        p_dict = props.to_dict()
        assert p_dict["inline"]["properties"][2]["data_type"] == "int32"
        assert "ct_c" in p_dict["inline"]["properties"][1]["tags"]

    def test_segment_props_with_string_dtype_pyarrow(self):
        """Test SegmentProperties with pd.StringDtype(storage='pyarrow')."""
        df = pd.DataFrame(
            {
                "seg_id": np.arange(0, 50),
                "cell_type": pd.array(
                    25 * ["type_a"] + 25 * ["type_b"],
                    dtype=pd.StringDtype(storage="pyarrow"),
                ),
                "value": np.arange(100, 150),
            }
        )

        props = SegmentProperties.from_dataframe(
            df,
            id_col="seg_id",
            tag_value_cols="cell_type",
            number_cols="value",
        )

        assert len(props) == 50
        p_dict = props.to_dict()
        assert "type_a" in p_dict["inline"]["properties"][0]["tags"]
        assert "type_b" in p_dict["inline"]["properties"][0]["tags"]

    def test_segment_props_multicolumn_label_pyarrow(self, pyarrow_segment_props_df):
        """Test multicolumn label with PyArrow DataFrame."""
        props = SegmentProperties.from_dataframe(
            pyarrow_segment_props_df,
            id_col="seg_id",
            label_col=["seg_id", "cell_type", "category"],
        )

        assert len(props) == 100
        p_dict = props.to_dict()
        # Check that label was properly constructed from multiple columns
        assert "_" in p_dict["inline"]["properties"][0]["values"][1]


class TestAnnotationLayersWithPyArrow:
    """Test annotation layers with PyArrow DataFrames."""

    def test_add_points_with_pyarrow_df(self, pyarrow_annotation_df):
        """Test adding points from PyArrow DataFrame."""
        layer = AnnotationLayer(name="test_points")

        layer.add_points(
            data=pyarrow_annotation_df,
            point_column=["x", "y", "z"],
            segment_column="segment_id",
            description_column="description",
        )

        # Check that points were added
        assert len(layer.annotations) == 4

        # Check that values are correct
        first_point = layer.annotations[0]
        assert first_point.point == [100, 150, 10]
        assert first_point.segments == [[12345]]  # Segments are wrapped in a list
        assert first_point.description == "Point A"

    def test_add_points_with_string_dtype_pyarrow(self):
        """Test adding points with pd.StringDtype(storage='pyarrow')."""
        df = pd.DataFrame(
            {
                "x": [100, 200, 300],
                "y": [150, 250, 350],
                "z": [10, 20, 30],
                "description": pd.array(
                    ["Desc A", "Desc B", "Desc C"],
                    dtype=pd.StringDtype(storage="pyarrow"),
                ),
            }
        )

        layer = AnnotationLayer(name="test_points")
        layer.add_points(
            data=df,
            point_column=["x", "y", "z"],
            description_column="description",
        )

        assert len(layer.annotations) == 3
        assert layer.annotations[0].description == "Desc A"

    def test_add_lines_with_pyarrow_df(self):
        """Test adding lines from PyArrow DataFrame."""
        df = pd.DataFrame(
            {
                "x1": pd.array([100, 200], dtype=pd.ArrowDtype(pa.int64())),
                "y1": pd.array([150, 250], dtype=pd.ArrowDtype(pa.int64())),
                "z1": pd.array([10, 20], dtype=pd.ArrowDtype(pa.int64())),
                "x2": pd.array([110, 210], dtype=pd.ArrowDtype(pa.int64())),
                "y2": pd.array([160, 260], dtype=pd.ArrowDtype(pa.int64())),
                "z2": pd.array([15, 25], dtype=pd.ArrowDtype(pa.int64())),
            }
        )

        layer = AnnotationLayer(name="test_lines")
        layer.add_lines(
            data=df,
            point_a_column=["x1", "y1", "z1"],
            point_b_column=["x2", "y2", "z2"],
        )

        assert len(layer.annotations) == 2
        first_line = layer.annotations[0]
        assert first_line.pointA == [100, 150, 10]
        assert first_line.pointB == [110, 160, 15]

    def test_viewer_state_with_pyarrow_df(self, pyarrow_annotation_df):
        """Test full ViewerState workflow with PyArrow DataFrame."""
        vs = ViewerState()
        layer = AnnotationLayer(name="pyarrow_points", resolution=[1, 1, 1])
        layer.add_points(
            data=pyarrow_annotation_df,
            point_column=["x", "y", "z"],
            segment_column="segment_id",
        )
        vs.add_layer(layer)

        # Check that state can be generated
        state = vs.to_dict()
        assert "layers" in state
        assert len(state["layers"]) == 1

        # Check annotations in state
        annotations = state["layers"][0]["annotations"]
        assert len(annotations) == 4

    def test_add_ellipsoids_with_pyarrow_df(self):
        """Test adding ellipsoids from PyArrow DataFrame."""
        df = pd.DataFrame(
            {
                "cx": pd.array([100, 200], dtype=pd.ArrowDtype(pa.int64())),
                "cy": pd.array([150, 250], dtype=pd.ArrowDtype(pa.int64())),
                "cz": pd.array([10, 20], dtype=pd.ArrowDtype(pa.int64())),
                "rx": pd.array([5, 10], dtype=pd.ArrowDtype(pa.int64())),
                "ry": pd.array([5, 10], dtype=pd.ArrowDtype(pa.int64())),
                "rz": pd.array([5, 10], dtype=pd.ArrowDtype(pa.int64())),
            }
        )

        layer = AnnotationLayer(name="test_ellipsoids")
        layer.add_ellipsoids(
            data=df,
            center_column=["cx", "cy", "cz"],
            radii_column=["rx", "ry", "rz"],
        )

        assert len(layer.annotations) == 2
        first_ellipsoid = layer.annotations[0]
        assert first_ellipsoid.center == [100, 150, 10]
        assert first_ellipsoid.radii == [5, 5, 5]

    def test_add_polylines_with_pyarrow_df(self):
        """Test adding polylines from PyArrow DataFrame."""
        df = pd.DataFrame(
            {
                "path": [
                    [[100, 150, 10], [200, 250, 20]],
                    [[300, 350, 30], [400, 450, 40], [500, 550, 50]],
                ],
                "segment_id": pd.array([12345, 67890], dtype=pd.ArrowDtype(pa.int64())),
                "description": pd.array(
                    ["Path A", "Path B"],
                    dtype=pd.ArrowDtype(pa.string()),
                ),
            }
        )

        layer = AnnotationLayer(name="test_polylines")
        layer.add_polylines(
            data=df,
            points_column="path",
            segment_column="segment_id",
            description_column="description",
        )

        assert len(layer.annotations) == 2
        first_polyline = layer.annotations[0]
        assert len(first_polyline.points) == 2
        assert first_polyline.description == "Path A"

    def test_add_boxes_with_pyarrow_df(self):
        """Test adding bounding boxes from PyArrow DataFrame."""
        df = pd.DataFrame(
            {
                "x1": pd.array([100, 200], dtype=pd.ArrowDtype(pa.int64())),
                "y1": pd.array([150, 250], dtype=pd.ArrowDtype(pa.int64())),
                "z1": pd.array([10, 20], dtype=pd.ArrowDtype(pa.int64())),
                "x2": pd.array([120, 220], dtype=pd.ArrowDtype(pa.int64())),
                "y2": pd.array([170, 270], dtype=pd.ArrowDtype(pa.int64())),
                "z2": pd.array([30, 40], dtype=pd.ArrowDtype(pa.int64())),
            }
        )

        layer = AnnotationLayer(name="test_boxes")
        layer.add_boxes(
            data=df,
            point_a_column=["x1", "y1", "z1"],
            point_b_column=["x2", "y2", "z2"],
        )

        assert len(layer.annotations) == 2
        first_box = layer.annotations[0]
        assert first_box.pointA == [100, 150, 10]
        assert first_box.pointB == [120, 170, 30]
