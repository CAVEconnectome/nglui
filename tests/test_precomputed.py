"""Tests for nglui.precomputed annotation writing."""

import json
import os
import struct
import tempfile

import numpy as np
import pandas as pd
import pytest
from neuroglancer.coordinate_space import CoordinateSpace
from neuroglancer.viewer_state import AnnotationPropertySpec

from nglui.precomputed import (
    BoundingBoxAnnotationWriter,
    EllipsoidAnnotationWriter,
    LineAnnotationWriter,
    PointAnnotationWriter,
)
from nglui.precomputed._encoding import (
    build_dtype,
    encode_by_id_entries,
    encode_fixed_blocks,
    encode_multiple_annotations,
    sort_properties,
)
from nglui.precomputed._sharding import ShardSpec, choose_output_spec
from nglui.precomputed._spatial import (
    SpatialLevel,
    _IsotropicHierarchy,
    _SpatialHierarchy,
    _UniformHierarchy,
    auto_chunk_size,
    build_spatial_levels,
    compressed_morton_code,
    compute_chunk_assignments,
    compute_multiscale_assignments,
)
from nglui.precomputed._writer import _PrecomputedAnnotationWriter


@pytest.fixture
def coordinate_space_3d():
    return CoordinateSpace(names=["x", "y", "z"], scales=[1, 1, 1], units="nm")


@pytest.fixture
def sample_points():
    rng = np.random.default_rng(42)
    return rng.uniform(0, 1000, size=(100, 3)).astype(np.float32)


@pytest.fixture
def sample_df(sample_points):
    df = pd.DataFrame(
        {
            "pt_x": sample_points[:, 0],
            "pt_y": sample_points[:, 1],
            "pt_z": sample_points[:, 2],
            "score": np.random.default_rng(42)
            .uniform(0, 1, size=100)
            .astype(np.float32),
            "segment_id": np.random.default_rng(42)
            .integers(1, 50, size=100)
            .astype(np.uint64),
        }
    )
    df.index = np.arange(100, dtype=np.uint64)
    return df


@pytest.fixture
def sample_df_array_col(sample_points):
    df = pd.DataFrame(
        {
            "pt_position": list(sample_points),
            "score": np.random.default_rng(42)
            .uniform(0, 1, size=100)
            .astype(np.float32),
            "segment_id": np.random.default_rng(42)
            .integers(1, 50, size=100)
            .astype(np.uint64),
        }
    )
    df.index = np.arange(100, dtype=np.uint64)
    return df


# ── Encoding Tests ──────────────────────────────────────────────────


class TestEncoding:
    def test_build_dtype_point(self):
        dtype = build_dtype("point", 3, [])
        assert "geometry" in dtype.names
        assert dtype["geometry"].shape == (3,)
        assert dtype["geometry"].base == np.dtype("<f4")

    def test_build_dtype_line(self):
        dtype = build_dtype("line", 3, [])
        assert dtype["geometry"].shape == (6,)

    def test_build_dtype_with_properties(self):
        props = [
            AnnotationPropertySpec(id="score", type="float32"),
            AnnotationPropertySpec(id="label", type="uint8"),
        ]
        sorted_props = sort_properties(props)
        dtype = build_dtype("point", 3, sorted_props)
        assert "score" in dtype.names
        assert "label" in dtype.names

    def test_sort_properties(self):
        props = [
            AnnotationPropertySpec(id="a", type="uint8"),
            AnnotationPropertySpec(id="b", type="float32"),
            AnnotationPropertySpec(id="c", type="uint16"),
        ]
        sorted_p = sort_properties(props)
        assert sorted_p[0].id == "b"  # 4-byte first
        assert sorted_p[1].id == "c"  # 2-byte
        assert sorted_p[2].id == "a"  # 1-byte

    def test_encode_fixed_blocks(self):
        props = [AnnotationPropertySpec(id="score", type="float32")]
        sorted_props = sort_properties(props)
        dtype = build_dtype("point", 3, sorted_props)

        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        prop_data = {"score": np.array([0.5, 0.9], dtype=np.float32)}

        blocks = encode_fixed_blocks(coords, prop_data, dtype)
        assert len(blocks) == 2
        assert blocks[0]["geometry"].tolist() == [1.0, 2.0, 3.0]
        assert blocks[0]["score"] == pytest.approx(0.5)

    def test_encode_by_id_no_relationships(self):
        dtype = build_dtype("point", 3, [])
        coords = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        blocks = encode_fixed_blocks(coords, {}, dtype)
        entries = encode_by_id_entries(blocks, None)
        assert len(entries) == 1
        assert len(entries[0]) == dtype.itemsize

    def test_encode_by_id_with_relationships(self):
        dtype = build_dtype("point", 3, [])
        coords = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        blocks = encode_fixed_blocks(coords, {}, dtype)

        # One relationship, one annotation, with 2 segment IDs
        relationships = [[[100, 200]]]
        entries = encode_by_id_entries(blocks, relationships)
        assert len(entries) == 1

        entry = entries[0]
        # fixed block + uint32(count=2) + 2*uint64(ids)
        expected_size = dtype.itemsize + 4 + 2 * 8
        assert len(entry) == expected_size

        # Parse relationship suffix
        offset = dtype.itemsize
        count = struct.unpack_from("<I", entry, offset)[0]
        assert count == 2
        offset += 4
        id1 = struct.unpack_from("<Q", entry, offset)[0]
        id2 = struct.unpack_from("<Q", entry, offset + 8)[0]
        assert id1 == 100
        assert id2 == 200

    def test_encode_multiple_annotations(self):
        dtype = build_dtype("point", 3, [])
        coords = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        blocks = encode_fixed_blocks(coords, {}, dtype)
        ids = np.array([10, 20], dtype=np.uint64)

        data = encode_multiple_annotations(blocks, ids)
        # uint64(count) + 2*fixed_block + 2*uint64(ids)
        expected_size = 8 + 2 * dtype.itemsize + 2 * 8
        assert len(data) == expected_size

        count = struct.unpack_from("<Q", data, 0)[0]
        assert count == 2


# ── Spatial Tests ───────────────────────────────────────────────────


class TestSpatial:
    def test_chunk_assignment_single_chunk(self):
        coords = np.array([[5, 5, 5], [15, 15, 15]], dtype=np.float32)
        lower = np.array([0, 0, 0], dtype=np.float32)
        chunk_size = np.array([100, 100, 100], dtype=np.float64)
        num_chunks = np.array([1, 1, 1])

        assignments = compute_chunk_assignments(coords, lower, chunk_size, num_chunks)
        assert (0, 0, 0) in assignments
        assert len(assignments[(0, 0, 0)]) == 2

    def test_chunk_assignment_multiple_chunks(self):
        coords = np.array([[5, 5, 5], [150, 150, 150]], dtype=np.float32)
        lower = np.array([0, 0, 0], dtype=np.float32)
        chunk_size = np.array([100, 100, 100], dtype=np.float64)
        num_chunks = np.array([2, 2, 2])

        assignments = compute_chunk_assignments(coords, lower, chunk_size, num_chunks)
        assert (0, 0, 0) in assignments
        assert (1, 1, 1) in assignments
        assert len(assignments[(0, 0, 0)]) == 1
        assert len(assignments[(1, 1, 1)]) == 1

    def test_compressed_morton_code_single(self):
        code = compressed_morton_code([0, 0, 0], [2, 2, 2])
        assert code == 0

        code = compressed_morton_code([1, 0, 0], [2, 2, 2])
        assert code == 1

    def test_compressed_morton_code_batch(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.uint32)
        codes = compressed_morton_code(pts, [2, 2, 2])
        assert len(codes) == 3
        assert codes[0] == 0

    def test_auto_chunk_size_small(self):
        lower = np.array([0, 0, 0])
        upper = np.array([1000, 1000, 1000])
        cs = auto_chunk_size(lower, upper, 10, target_per_chunk=500)
        # 10 annotations < 500 target → single chunk
        np.testing.assert_array_equal(cs, [1000, 1000, 1000])

    def test_auto_chunk_size_large(self):
        lower = np.array([0, 0, 0])
        upper = np.array([1000, 1000, 1000])
        cs = auto_chunk_size(lower, upper, 10000, target_per_chunk=500)
        # Should subdivide
        assert np.all(cs < 1000)
        assert np.all(cs > 0)


# ── Sharding Tests ──────────────────────────────────────────────────


class TestSharding:
    def test_choose_output_spec_single(self):
        assert choose_output_spec(1, 100) is None

    def test_choose_output_spec_returns_spec(self):
        spec = choose_output_spec(1000, 100000)
        assert spec is not None
        assert isinstance(spec, ShardSpec)
        assert spec.type == "neuroglancer_uint64_sharded_v1"

    def test_shard_spec_to_json(self):
        spec = choose_output_spec(1000, 100000)
        j = spec.to_json()
        assert j["@type"] == "neuroglancer_uint64_sharded_v1"
        assert "hash" in j
        assert "shard_bits" in j


# ── Low-Level Writer Tests ──────────────────────────────────────────


class TestPrecomputedAnnotationWriter:
    def test_write_points_unsharded(self, coordinate_space_3d, sample_points):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = _PrecomputedAnnotationWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
            )
            writer.set_coordinates(sample_points)
            writer.write(tmpdir)

            # Verify info file
            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            assert info["@type"] == "neuroglancer_annotations_v1"
            assert info["annotation_type"] == "point"
            assert len(info["lower_bound"]) == 3
            assert len(info["upper_bound"]) == 3

            # Verify by_id files exist (sharded format)
            by_id_dir = os.path.join(tmpdir, "by_id")
            assert os.path.isdir(by_id_dir)
            by_id_files = os.listdir(by_id_dir)
            # Should have at least one file (either individual files or shard files)  
            assert len(by_id_files) >= 1
            # If sharded, should have .shard files; if unsharded, should have numbered files
            has_shard = any(f.endswith('.shard') for f in by_id_files)
            has_individual = any(f.isdigit() for f in by_id_files)
            assert has_shard or has_individual

            # Verify spatial0 directory has files
            spatial_dir = os.path.join(tmpdir, "spatial0")
            assert os.path.isdir(spatial_dir)
            assert len(os.listdir(spatial_dir)) > 0

    def test_write_points_with_properties(self, coordinate_space_3d):
        with tempfile.TemporaryDirectory() as tmpdir:
            props = [AnnotationPropertySpec(id="score", type="float32")]
            writer = _PrecomputedAnnotationWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                properties=props,
            )
            coords = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            scores = np.array([0.5, 0.9], dtype=np.float32)
            writer.set_coordinates(coords)
            writer.set_property("score", scores)
            writer.write(tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            assert len(info["properties"]) == 1
            assert info["properties"][0]["id"] == "score"

    def test_write_points_with_relationships(self, coordinate_space_3d):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = _PrecomputedAnnotationWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                relationships=["segment"],
            )
            coords = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            writer.set_coordinates(coords)
            writer.set_relationship("segment", np.array([100, 200], dtype=np.uint64))
            writer.write(tmpdir)

            # Check relationship dir exists
            rel_dir = os.path.join(tmpdir, "rel_segment")
            assert os.path.isdir(rel_dir)
            # Should have files (either individual segment ID files or shard files)
            rel_files = os.listdir(rel_dir)
            assert len(rel_files) >= 1
            # Files should exist for the segment relationships
            has_individual = any(f in ["100", "200"] for f in rel_files) 
            has_shard = any(f.endswith('.shard') for f in rel_files)
            assert has_individual or has_shard

    def test_write_points_variable_relationships(self, coordinate_space_3d):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = _PrecomputedAnnotationWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                relationships=["segments"],
            )
            coords = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            writer.set_coordinates(coords)
            writer.set_relationship("segments", [[100, 200], [300]])
            writer.write(tmpdir)

            rel_dir = os.path.join(tmpdir, "rel_segments")
            # Should have files (either individual segment ID files or shard files)
            rel_files = os.listdir(rel_dir)
            assert len(rel_files) >= 1
            # Files should exist for the segment relationships
            has_individual = any(f in ["100", "200", "300"] for f in rel_files)
            has_shard = any(f.endswith('.shard') for f in rel_files)
            assert has_individual or has_shard

    def test_per_row_api(self, coordinate_space_3d):
        with tempfile.TemporaryDirectory() as tmpdir:
            props = [AnnotationPropertySpec(id="score", type="float32")]
            writer = _PrecomputedAnnotationWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                relationships=["segment"],
                properties=props,
            )
            writer.add_point([1, 2, 3], score=0.5, segment=100)
            writer.add_point([4, 5, 6], score=0.9, segment=200)
            writer.write(tmpdir)

            by_id_dir = os.path.join(tmpdir, "by_id")
            assert len(os.listdir(by_id_dir)) >= 1

    def test_bulk_and_per_row_produce_same_info(self, coordinate_space_3d):
        coords = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        with (
            tempfile.TemporaryDirectory() as tmp_bulk,
            tempfile.TemporaryDirectory() as tmp_row,
        ):
            # Bulk
            w1 = _PrecomputedAnnotationWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                spatial_hierarchy=_UniformHierarchy(rank=3, chunk_size=1000),
            )
            w1.set_coordinates(coords)
            w1.write(tmp_bulk)

            # Per-row
            w2 = _PrecomputedAnnotationWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                spatial_hierarchy=_UniformHierarchy(rank=3, chunk_size=1000),
            )
            w2.add_point([1, 2, 3])
            w2.add_point([4, 5, 6])
            w2.write(tmp_row)

            with open(os.path.join(tmp_bulk, "info")) as f:
                info_bulk = json.load(f)
            with open(os.path.join(tmp_row, "info")) as f:
                info_row = json.load(f)

            assert info_bulk["annotation_type"] == info_row["annotation_type"]
            assert info_bulk["lower_bound"] == info_row["lower_bound"]
            assert info_bulk["upper_bound"] == info_row["upper_bound"]


# ── DataFrame Writer Tests ──────────────────────────────────────────


class TestPointAnnotationWriter:
    def test_write_xyz_columns(self, coordinate_space_3d, sample_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = PointAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                point_column=["pt_x", "pt_y", "pt_z"],
            )
            writer.write(sample_df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            assert info["annotation_type"] == "point"
            assert len(os.listdir(os.path.join(tmpdir, "by_id"))) >= 1

    def test_write_array_column(self, coordinate_space_3d, sample_df_array_col):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = PointAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                point_column="pt_position",
            )
            writer.write(sample_df_array_col, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            assert info["annotation_type"] == "point"
            assert len(os.listdir(os.path.join(tmpdir, "by_id"))) >= 1

    def test_write_with_properties(self, coordinate_space_3d, sample_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = PointAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                point_column=["pt_x", "pt_y", "pt_z"],
                property_columns=["score"],
            )
            writer.write(sample_df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            assert len(info["properties"]) == 1
            assert info["properties"][0]["id"] == "score"

    def test_write_with_relationships(self, coordinate_space_3d, sample_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = PointAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                point_column=["pt_x", "pt_y", "pt_z"],
                relationship_columns=["segment_id"],
            )
            writer.write(sample_df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            assert len(info["relationships"]) == 1
            assert info["relationships"][0]["id"] == "segment_id"

            rel_dir = os.path.join(tmpdir, "rel_segment_id")
            assert os.path.isdir(rel_dir)
            assert len(os.listdir(rel_dir)) > 0

    def test_xyz_and_array_col_produce_same_output(
        self, coordinate_space_3d, sample_points
    ):
        df_xyz = pd.DataFrame(
            {
                "x": sample_points[:, 0],
                "y": sample_points[:, 1],
                "z": sample_points[:, 2],
            }
        )
        df_xyz.index = np.arange(len(sample_points), dtype=np.uint64)

        df_arr = pd.DataFrame(
            {
                "pt": list(sample_points),
            }
        )
        df_arr.index = np.arange(len(sample_points), dtype=np.uint64)

        with (
            tempfile.TemporaryDirectory() as tmp1,
            tempfile.TemporaryDirectory() as tmp2,
        ):
            w1 = PointAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                point_column=["x", "y", "z"],
                chunk_size=10000,
            )
            w1.write(df_xyz, tmp1)

            w2 = PointAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                point_column="pt",
                chunk_size=10000,
            )
            w2.write(df_arr, tmp2)

            # Both should produce same metadata and structure
            # Compare info files
            with open(os.path.join(tmp1, "info")) as f1, open(os.path.join(tmp2, "info")) as f2:
                info1 = json.load(f1)
                info2 = json.load(f2)
                # Should have same bounds, type, etc. (excluding any randomly generated IDs)
                assert info1["annotation_type"] == info2["annotation_type"]
                assert info1["lower_bound"] == info2["lower_bound"] 
                assert info1["upper_bound"] == info2["upper_bound"]
                assert len(info1["spatial"]) == len(info2["spatial"])
            
            # Both should have same directory structure
            assert os.path.isdir(os.path.join(tmp1, "by_id"))
            assert os.path.isdir(os.path.join(tmp2, "by_id"))
            assert len(os.listdir(os.path.join(tmp1, "by_id"))) >= 1
            assert len(os.listdir(os.path.join(tmp2, "by_id"))) >= 1

    def test_validation_errors(self, coordinate_space_3d):
        with pytest.raises(ValueError, match="point_column"):
            PointAnnotationWriter(
                coordinate_space=coordinate_space_3d,
            )

    def test_write_prefix_expansion(self, coordinate_space_3d, sample_df):
        """Test that point_column='pt' expands to pt_x, pt_y, pt_z."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = PointAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                point_column="pt",
            )
            writer.write(sample_df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            assert info["annotation_type"] == "point"
            assert len(os.listdir(os.path.join(tmpdir, "by_id"))) >= 1

    def test_bounds_match_data(self, coordinate_space_3d, sample_points):
        df = pd.DataFrame({"pt": list(sample_points)})
        df.index = np.arange(len(sample_points), dtype=np.uint64)

        with tempfile.TemporaryDirectory() as tmpdir:
            w = PointAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                point_column="pt",
            )
            w.write(df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)

            lb = np.array(info["lower_bound"])
            ub = np.array(info["upper_bound"])
            data_min = sample_points.min(axis=0)
            data_max = sample_points.max(axis=0)

            # Lower bound should be <= data min
            assert np.all(lb <= data_min + 1e-6)
            # Upper bound should be >= data max
            assert np.all(ub >= data_max - 1e-6)


# ── Two-Position DataFrame Fixtures ─────────────────────────────────


@pytest.fixture
def sample_two_point_df():
    """DataFrame with two sets of xyz columns for line/bbox annotations."""
    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame(
        {
            "pt_a_x": rng.uniform(0, 500, size=n).astype(np.float32),
            "pt_a_y": rng.uniform(0, 500, size=n).astype(np.float32),
            "pt_a_z": rng.uniform(0, 500, size=n).astype(np.float32),
            "pt_b_x": rng.uniform(500, 1000, size=n).astype(np.float32),
            "pt_b_y": rng.uniform(500, 1000, size=n).astype(np.float32),
            "pt_b_z": rng.uniform(500, 1000, size=n).astype(np.float32),
            "score": rng.uniform(0, 1, size=n).astype(np.float32),
            "segment_id": rng.integers(1, 50, size=n).astype(np.uint64),
        }
    )
    df.index = np.arange(n, dtype=np.uint64)
    return df


@pytest.fixture
def sample_two_point_array_df():
    """DataFrame with two array columns for line/bbox annotations."""
    rng = np.random.default_rng(42)
    n = 50
    pts_a = rng.uniform(0, 500, size=(n, 3)).astype(np.float32)
    pts_b = rng.uniform(500, 1000, size=(n, 3)).astype(np.float32)
    df = pd.DataFrame(
        {
            "point_a": list(pts_a),
            "point_b": list(pts_b),
        }
    )
    df.index = np.arange(n, dtype=np.uint64)
    return df


@pytest.fixture
def sample_ellipsoid_df():
    """DataFrame with center and radii columns for ellipsoid annotations."""
    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame(
        {
            "center_x": rng.uniform(0, 1000, size=n).astype(np.float32),
            "center_y": rng.uniform(0, 1000, size=n).astype(np.float32),
            "center_z": rng.uniform(0, 1000, size=n).astype(np.float32),
            "radii_x": rng.uniform(1, 50, size=n).astype(np.float32),
            "radii_y": rng.uniform(1, 50, size=n).astype(np.float32),
            "radii_z": rng.uniform(1, 50, size=n).astype(np.float32),
            "score": rng.uniform(0, 1, size=n).astype(np.float32),
        }
    )
    df.index = np.arange(n, dtype=np.uint64)
    return df


# ── Line Annotation Writer Tests ────────────────────────────────────


class TestLineAnnotationWriter:
    def test_write_xyz_columns(self, coordinate_space_3d, sample_two_point_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LineAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                point_a_column=["pt_a_x", "pt_a_y", "pt_a_z"],
                point_b_column=["pt_b_x", "pt_b_y", "pt_b_z"],
            )
            writer.write(sample_two_point_df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            assert info["annotation_type"] == "line"
            assert len(os.listdir(os.path.join(tmpdir, "by_id"))) >= 1

    def test_write_prefix_expansion(self, coordinate_space_3d, sample_two_point_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LineAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                point_a_column="pt_a",
                point_b_column="pt_b",
            )
            writer.write(sample_two_point_df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            assert info["annotation_type"] == "line"
            assert len(os.listdir(os.path.join(tmpdir, "by_id"))) >= 1

    def test_write_array_columns(self, coordinate_space_3d, sample_two_point_array_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LineAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                point_a_column="point_a",
                point_b_column="point_b",
            )
            writer.write(sample_two_point_array_df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            assert info["annotation_type"] == "line"
            assert len(os.listdir(os.path.join(tmpdir, "by_id"))) >= 1

    def test_write_with_properties(self, coordinate_space_3d, sample_two_point_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LineAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                point_a_column="pt_a",
                point_b_column="pt_b",
                property_columns=["score"],
            )
            writer.write(sample_two_point_df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            assert len(info["properties"]) == 1
            assert info["properties"][0]["id"] == "score"

    def test_write_with_relationships(self, coordinate_space_3d, sample_two_point_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LineAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                point_a_column="pt_a",
                point_b_column="pt_b",
                relationship_columns=["segment_id"],
            )
            writer.write(sample_two_point_df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            assert len(info["relationships"]) == 1
            assert info["relationships"][0]["id"] == "segment_id"

    def test_validation_errors(self, coordinate_space_3d):
        with pytest.raises(ValueError, match="point_a_column and point_b_column"):
            LineAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                point_a_column="pt_a",
            )


# ── Bounding Box Annotation Writer Tests ────────────────────────────


class TestBoundingBoxAnnotationWriter:
    def test_write_xyz_columns(self, coordinate_space_3d, sample_two_point_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = BoundingBoxAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                point_a_column=["pt_a_x", "pt_a_y", "pt_a_z"],
                point_b_column=["pt_b_x", "pt_b_y", "pt_b_z"],
            )
            writer.write(sample_two_point_df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            assert info["annotation_type"] == "axis_aligned_bounding_box"
            assert len(os.listdir(os.path.join(tmpdir, "by_id"))) >= 1

    def test_write_array_columns(self, coordinate_space_3d, sample_two_point_array_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = BoundingBoxAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                point_a_column="point_a",
                point_b_column="point_b",
            )
            writer.write(sample_two_point_array_df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            assert info["annotation_type"] == "axis_aligned_bounding_box"
            assert len(os.listdir(os.path.join(tmpdir, "by_id"))) >= 1

    def test_write_with_properties(self, coordinate_space_3d, sample_two_point_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = BoundingBoxAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                point_a_column="pt_a",
                point_b_column="pt_b",
                property_columns=["score"],
            )
            writer.write(sample_two_point_df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            assert len(info["properties"]) == 1

    def test_validation_errors(self, coordinate_space_3d):
        with pytest.raises(ValueError, match="point_a_column and point_b_column"):
            BoundingBoxAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                point_a_column="pt_a",
            )


# ── Ellipsoid Annotation Writer Tests ───────────────────────────────


class TestEllipsoidAnnotationWriter:
    def test_write_xyz_columns(self, coordinate_space_3d, sample_ellipsoid_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = EllipsoidAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                center_column=["center_x", "center_y", "center_z"],
                radii_column=["radii_x", "radii_y", "radii_z"],
            )
            writer.write(sample_ellipsoid_df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            assert info["annotation_type"] == "ellipsoid"
            assert len(os.listdir(os.path.join(tmpdir, "by_id"))) >= 1

    def test_write_prefix_expansion(self, coordinate_space_3d, sample_ellipsoid_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = EllipsoidAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                center_column="center",
                radii_column="radii",
            )
            writer.write(sample_ellipsoid_df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            assert info["annotation_type"] == "ellipsoid"
            assert len(os.listdir(os.path.join(tmpdir, "by_id"))) >= 1

    def test_write_array_columns(self, coordinate_space_3d):
        rng = np.random.default_rng(42)
        n = 30
        centers = rng.uniform(0, 1000, size=(n, 3)).astype(np.float32)
        radii = rng.uniform(1, 50, size=(n, 3)).astype(np.float32)
        df = pd.DataFrame({"ctr": list(centers), "rad": list(radii)})
        df.index = np.arange(n, dtype=np.uint64)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = EllipsoidAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                center_column="ctr",
                radii_column="rad",
            )
            writer.write(df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            assert info["annotation_type"] == "ellipsoid"
            assert len(os.listdir(os.path.join(tmpdir, "by_id"))) >= 1

    def test_write_with_properties(self, coordinate_space_3d, sample_ellipsoid_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = EllipsoidAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                center_column="center",
                radii_column="radii",
                property_columns=["score"],
            )
            writer.write(sample_ellipsoid_df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            assert len(info["properties"]) == 1
            assert info["properties"][0]["id"] == "score"

    def test_validation_errors(self, coordinate_space_3d):
        with pytest.raises(ValueError, match="center_column and radii_column"):
            EllipsoidAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                center_column="center",
            )


# ── Multi-Cell Spatial Assignment Test ──────────────────────────────


class TestTwoPositionSpatialAssignment:
    def test_compute_chunk_assignments_line_spans_cells(self):
        """compute_chunk_assignments should assign a two-point annotation
        to every cell its bounding box overlaps."""
        # Line from (50,50,50) to (250,250,250), grid with chunk_size=100
        coords = np.array([[50, 50, 50, 250, 250, 250]], dtype=np.float32)
        lower = np.array([0, 0, 0], dtype=np.float64)
        chunk_size = np.array([100, 100, 100], dtype=np.float64)
        num_chunks = np.array([3, 3, 3])

        assignments = compute_chunk_assignments(coords, lower, chunk_size, num_chunks)

        # Annotation 0 should appear in every cell from (0,0,0) to (2,2,2)
        cells_with_ann = [cell for cell, idxs in assignments.items() if 0 in idxs]
        assert len(cells_with_ann) == 27, (
            f"Expected 27 cells (3x3x3 AABB), got {len(cells_with_ann)}: {cells_with_ann}"
        )

    def test_compute_chunk_assignments_bbox_spans_cells(self):
        """Same for bounding box geometry."""
        coords = np.array([[50, 50, 50, 250, 250, 250]], dtype=np.float32)
        lower = np.array([0, 0, 0], dtype=np.float64)
        chunk_size = np.array([100, 100, 100], dtype=np.float64)
        num_chunks = np.array([3, 3, 3])

        assignments = compute_chunk_assignments(coords, lower, chunk_size, num_chunks)
        cells_with_ann = [cell for cell, idxs in assignments.items() if 0 in idxs]
        assert len(cells_with_ann) == 27

    def test_line_writer_multi_cell_integration(self, coordinate_space_3d):
        """Lines spanning multiple cells should appear in multiple spatial
        chunks when there are enough annotations to reach the finest level."""
        rng = np.random.default_rng(42)
        n = 2000
        # Lines that span a wide range (endpoints ~500 apart)
        df = pd.DataFrame(
            {
                "pt_a_x": rng.uniform(0, 500, n).astype(np.float32),
                "pt_a_y": rng.uniform(0, 500, n).astype(np.float32),
                "pt_a_z": rng.uniform(0, 500, n).astype(np.float32),
                "pt_b_x": rng.uniform(500, 1000, n).astype(np.float32),
                "pt_b_y": rng.uniform(500, 1000, n).astype(np.float32),
                "pt_b_z": rng.uniform(500, 1000, n).astype(np.float32),
            }
        )
        df.index = np.arange(n, dtype=np.uint64)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LineAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                point_a_column="pt_a",
                point_b_column="pt_b",
                limit=200,
            )
            writer.write(df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)

            assert info["annotation_type"] == "line"

            # Count total annotation slots across all spatial chunks;
            # for lines spanning cells the total should exceed n
            total = 0
            for level_info in info["spatial"]:
                level_dir = os.path.join(tmpdir, level_info["key"])
                for chunk_file in os.listdir(level_dir):
                    with open(os.path.join(level_dir, chunk_file), "rb") as f:
                        data = f.read()
                    count = struct.unpack_from("<Q", data, 0)[0]
                    total += count

            assert total >= n, (
                f"Expected total slots >= {n} (multi-cell spanning), got {total}"
            )

    def test_bbox_writer_multi_cell_integration(self, coordinate_space_3d):
        """Bounding boxes spanning multiple cells should appear in multiple
        spatial chunks when there are enough annotations."""
        rng = np.random.default_rng(42)
        n = 2000
        df = pd.DataFrame(
            {
                "pt_a_x": rng.uniform(0, 500, n).astype(np.float32),
                "pt_a_y": rng.uniform(0, 500, n).astype(np.float32),
                "pt_a_z": rng.uniform(0, 500, n).astype(np.float32),
                "pt_b_x": rng.uniform(500, 1000, n).astype(np.float32),
                "pt_b_y": rng.uniform(500, 1000, n).astype(np.float32),
                "pt_b_z": rng.uniform(500, 1000, n).astype(np.float32),
            }
        )
        df.index = np.arange(n, dtype=np.uint64)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = BoundingBoxAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                point_a_column="pt_a",
                point_b_column="pt_b",
                limit=200,
            )
            writer.write(df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)

            assert info["annotation_type"] == "axis_aligned_bounding_box"

            total = 0
            for level_info in info["spatial"]:
                level_dir = os.path.join(tmpdir, level_info["key"])
                for chunk_file in os.listdir(level_dir):
                    with open(os.path.join(level_dir, chunk_file), "rb") as f:
                        data = f.read()
                    count = struct.unpack_from("<Q", data, 0)[0]
                    total += count

            assert total >= n


# ── Source Resolution Tests ─────────────────────────────────────────


class TestSourceResolution:
    def test_resolution_parameter(self):
        """Test that resolution= creates a valid writer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = _PrecomputedAnnotationWriter(
                annotation_type="point",
                resolution=[8, 8, 40],
            )
            coords = np.array([[100, 200, 50]], dtype=np.float32)
            writer.set_coordinates(coords)
            writer.write(tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            dims = info["dimensions"]
            assert "x" in dims
            assert "y" in dims
            assert "z" in dims

    def test_resolution_dataframe_writer(self, sample_df):
        """Test AnnotationDataFrameWriter with resolution= parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = PointAnnotationWriter(
                resolution=[1, 1, 1],
                point_column=["pt_x", "pt_y", "pt_z"],
            )
            writer.write(sample_df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            assert info["annotation_type"] == "point"

    def test_no_source_raises(self):
        """Must provide at least one source of coordinate space."""
        with pytest.raises(ValueError, match="Must provide one of"):
            _PrecomputedAnnotationWriter(annotation_type="point")

    def test_multiple_sources_raises(self, coordinate_space_3d):
        """Cannot provide both coordinate_space and resolution."""
        with pytest.raises(ValueError, match="Provide only one of"):
            _PrecomputedAnnotationWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                resolution=[1, 1, 1],
            )


# ── Enum Property Encoding Tests ─────────────────────────────────────


class TestEnumPropertyEncoding:
    """Unit tests for _is_enum_column and _build_enum_property helpers."""

    from nglui.precomputed.dataframe_writer import _build_enum_property, _is_enum_column

    # -- _is_enum_column --

    def test_is_enum_bool_numpy(self):
        s = pd.Series([True, False, True], dtype=bool)
        from nglui.precomputed.dataframe_writer import _is_enum_column

        assert _is_enum_column(s)

    def test_is_enum_bool_nullable(self):
        s = pd.array([True, False, None], dtype=pd.BooleanDtype())
        s = pd.Series(s)
        from nglui.precomputed.dataframe_writer import _is_enum_column

        assert _is_enum_column(s)

    def test_is_enum_categorical(self):
        s = pd.Categorical(["a", "b", "a"])
        from nglui.precomputed.dataframe_writer import _is_enum_column

        assert _is_enum_column(pd.Series(s))

    def test_is_enum_string_object(self):
        s = pd.Series(["foo", "bar", "foo"])
        from nglui.precomputed.dataframe_writer import _is_enum_column

        assert _is_enum_column(s)

    def test_is_not_enum_numeric(self):
        s = pd.Series([1.0, 2.0, 3.0])
        from nglui.precomputed.dataframe_writer import _is_enum_column

        assert not _is_enum_column(s)

    # -- bool branch of _build_enum_property --

    def test_bool_numpy_no_nulls(self):
        from nglui.precomputed.dataframe_writer import _build_enum_property

        s = pd.Series([True, False, True, False], dtype=bool)
        spec, codes = _build_enum_property(s, "flag")

        assert spec.id == "flag"
        assert spec.type == "uint8"
        assert list(spec.enum_labels) == ["False", "True"]
        assert list(spec.enum_values) == [0, 1]
        np.testing.assert_array_equal(codes, [1, 0, 1, 0])
        assert codes.dtype == np.uint8

    def test_bool_nullable_with_nulls(self):
        from nglui.precomputed.dataframe_writer import _build_enum_property

        s = pd.Series(pd.array([True, None, False, True], dtype=pd.BooleanDtype()))
        spec, codes = _build_enum_property(s, "flag")

        assert spec.id == "flag"
        assert spec.type == "uint8"
        assert list(spec.enum_labels) == ["null", "False", "True"]
        assert list(spec.enum_values) == [0, 1, 2]
        # True=2, null=0, False=1, True=2
        np.testing.assert_array_equal(codes, [2, 0, 1, 2])
        assert codes.dtype == np.uint8

    def test_bool_no_nulls_uint_type(self):
        """Two labels always fit in uint8."""
        from nglui.precomputed.dataframe_writer import _build_enum_property

        s = pd.Series([False, False, True], dtype=bool)
        spec, codes = _build_enum_property(s, "x")
        assert spec.type == "uint8"

    # -- write round-trip with bool property --

    def test_write_bool_property(self, coordinate_space_3d, sample_df):
        sample_df = sample_df.copy()
        rng = np.random.default_rng(0)
        sample_df["is_active"] = rng.integers(0, 2, size=len(sample_df)).astype(bool)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = PointAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                point_column=["pt_x", "pt_y", "pt_z"],
                property_columns=["is_active"],
            )
            writer.write(sample_df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)

        prop = info["properties"][0]
        assert prop["id"] == "is_active"
        assert prop["type"] == "uint8"
        assert prop["enum_labels"] == ["False", "True"]
        assert prop["enum_values"] == [0, 1]

    def test_write_bool_nullable_property(self, coordinate_space_3d, sample_df):
        sample_df = sample_df.copy()
        vals = [True, False, None] * (len(sample_df) // 3) + [True] * (
            len(sample_df) % 3
        )
        sample_df["is_active"] = pd.array(
            vals[: len(sample_df)], dtype=pd.BooleanDtype()
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = PointAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                point_column=["pt_x", "pt_y", "pt_z"],
                property_columns=["is_active"],
            )
            writer.write(sample_df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)

        prop = info["properties"][0]
        assert prop["enum_labels"] == ["null", "False", "True"]
        assert prop["enum_values"] == [0, 1, 2]


# ── Multi-Scale Spatial Index Tests ─────────────────────────────────


class TestBuildSpatialLevels:
    def test_single_level_small_data(self):
        """Few annotations → single level with grid [1,1,1]."""
        levels = build_spatial_levels(
            np.array([0, 0, 0.0]),
            np.array([100, 100, 100.0]),
            n_annotations=10,
            limit=5000,
        )
        assert len(levels) == 1
        assert levels[0].key == "spatial0"
        np.testing.assert_array_equal(levels[0].grid_shape, [1, 1, 1])

    def test_multi_level_large_data(self):
        """Many annotations → multiple levels, coarsest is [1,1,1]."""
        levels = build_spatial_levels(
            np.array([0, 0, 0.0]),
            np.array([10000, 10000, 10000.0]),
            n_annotations=100000,
            limit=500,
        )
        assert len(levels) > 1
        # Coarsest level should be [1,1,1]
        np.testing.assert_array_equal(levels[0].grid_shape, [1, 1, 1])
        # Finest level should be the last
        assert np.all(levels[-1].grid_shape >= levels[0].grid_shape)
        # Keys should be sequential
        for i, level in enumerate(levels):
            assert level.key == f"spatial{i}"

    def test_custom_levels_passthrough(self):
        """Explicit levels are returned as-is."""
        custom = [
            SpatialLevel(
                "spatial0", np.array([1, 1, 1]), np.array([100, 100, 100.0]), 1000
            ),
            SpatialLevel(
                "spatial1", np.array([2, 2, 2]), np.array([50, 50, 50.0]), 1000
            ),
        ]
        result = build_spatial_levels(
            np.zeros(3), np.ones(3) * 100, n_annotations=50000, levels=custom
        )
        assert result is custom

    def test_levels_cover_extent(self):
        """Each level's grid_shape * chunk_size should cover the extent."""
        lower = np.array([0, 0, 0.0])
        upper = np.array([1000, 2000, 500.0])
        levels = build_spatial_levels(lower, upper, n_annotations=50000, limit=500)
        extent = upper - lower
        for level in levels:
            coverage = level.grid_shape * level.chunk_size
            np.testing.assert_allclose(coverage, extent, rtol=1e-10)


class TestMultiscaleAssignments:
    def test_every_row_assigned(self):
        """Every annotation must appear in exactly one (level, cell)."""
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 1000, size=(500, 3)).astype(np.float32)
        lower = np.array([0, 0, 0.0])
        upper = np.array([1000, 1000, 1000.0])
        levels = build_spatial_levels(lower, upper, len(coords), limit=100)

        assignment = compute_multiscale_assignments(
            coords, lower, levels, rng=np.random.default_rng(123)
        )

        # Each row should have at least one assignment
        for i, assigns in enumerate(assignment.row_assignments):
            assert len(assigns) >= 1, f"Row {i} has no assignment"

        # Total assignments across all chunks should equal n (for points)
        total_in_chunks = sum(
            len(indices) for indices in assignment.level_chunks.values()
        )
        assert total_in_chunks == len(coords)

    def test_forward_inverse_consistency(self):
        """Forward and inverse mappings must agree."""
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 1000, size=(200, 3)).astype(np.float32)
        lower = np.array([0, 0, 0.0])
        upper = np.array([1000, 1000, 1000.0])
        levels = build_spatial_levels(lower, upper, len(coords), limit=50)

        assignment = compute_multiscale_assignments(
            coords, lower, levels, rng=np.random.default_rng(0)
        )

        # Build inverse from forward and compare
        from collections import defaultdict

        rebuilt_inverse = defaultdict(list)
        for row_idx, assigns in enumerate(assignment.row_assignments):
            for level_idx, cell in assigns:
                rebuilt_inverse[(level_idx, cell)].append(row_idx)

        for key in rebuilt_inverse:
            expected = sorted(rebuilt_inverse[key])
            actual = sorted(assignment.level_chunks[key].tolist())
            assert expected == actual, f"Mismatch at {key}"

    def test_coarse_level_respects_limit(self):
        """Coarsest level chunks should not vastly exceed limit."""
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 1000, size=(10000, 3)).astype(np.float32)
        lower = np.array([0, 0, 0.0])
        upper = np.array([1000, 1000, 1000.0])
        limit = 500
        levels = build_spatial_levels(lower, upper, len(coords), limit=limit)

        assignment = compute_multiscale_assignments(
            coords, lower, levels, rng=np.random.default_rng(99)
        )

        # Check coarsest level (level 0) doesn't have chunks vastly over limit
        for (level_idx, cell), indices in assignment.level_chunks.items():
            if level_idx == 0:
                # Probabilistic: allow 2x margin
                assert len(indices) <= limit * 2, (
                    f"Coarse chunk {cell} has {len(indices)} > {limit * 2}"
                )

    def test_deterministic_with_seed(self):
        """Same seed should produce identical assignments."""
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 1000, size=(200, 3)).astype(np.float32)
        lower = np.array([0, 0, 0.0])
        upper = np.array([1000, 1000, 1000.0])
        levels = build_spatial_levels(lower, upper, len(coords), limit=50)

        a1 = compute_multiscale_assignments(
            coords, lower, levels, rng=np.random.default_rng(7)
        )
        a2 = compute_multiscale_assignments(
            coords, lower, levels, rng=np.random.default_rng(7)
        )

        assert len(a1.level_chunks) == len(a2.level_chunks)
        for key in a1.level_chunks:
            np.testing.assert_array_equal(a1.level_chunks[key], a2.level_chunks[key])


class TestMultiscaleWriter:
    def test_write_produces_multiple_spatial_dirs(self, coordinate_space_3d):
        """Writing many points should produce multiple spatial{i} dirs."""
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 1000, size=(10000, 3)).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = _PrecomputedAnnotationWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                spatial_hierarchy=_IsotropicHierarchy(rank=3, limit=500),
            )
            writer.set_coordinates(coords)
            writer.write(tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)

            # Should have multiple spatial levels
            assert len(info["spatial"]) > 1

            # Each level dir should exist and have chunks
            for level_info in info["spatial"]:
                level_dir = os.path.join(tmpdir, level_info["key"])
                assert os.path.isdir(level_dir), f"Missing {level_info['key']}"
                assert len(os.listdir(level_dir)) > 0

    def test_info_spatial_coarsest_is_1_1_1(self, coordinate_space_3d):
        """Coarsest level should have grid_shape [1,1,1]."""
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 1000, size=(5000, 3)).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = _PrecomputedAnnotationWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                spatial_hierarchy=_IsotropicHierarchy(rank=3, limit=500),
            )
            writer.set_coordinates(coords)
            writer.write(tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)

            assert info["spatial"][0]["grid_shape"] == [1, 1, 1]

    def test_all_annotations_accounted_for(self, coordinate_space_3d):
        """Total annotations across all spatial chunks must equal N."""
        rng = np.random.default_rng(42)
        n = 2000
        coords = rng.uniform(0, 1000, size=(n, 3)).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = _PrecomputedAnnotationWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                spatial_hierarchy=_IsotropicHierarchy(rank=3, limit=200),
            )
            writer.set_coordinates(coords)
            writer.write(tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)

            # Count annotations across all spatial chunks
            total = 0
            for level_info in info["spatial"]:
                level_dir = os.path.join(tmpdir, level_info["key"])
                for chunk_file in os.listdir(level_dir):
                    with open(os.path.join(level_dir, chunk_file), "rb") as f:
                        data = f.read()
                    count = struct.unpack_from("<Q", data, 0)[0]
                    total += count

            assert total == n

    def test_backward_compat_high_limit(self, coordinate_space_3d, sample_points):
        """With limit >= n_annotations, should produce a single level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = _PrecomputedAnnotationWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                spatial_hierarchy=_IsotropicHierarchy(rank=3, limit=100000),
            )
            writer.set_coordinates(sample_points)
            writer.write(tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)

            assert len(info["spatial"]) == 1
            assert info["spatial"][0]["grid_shape"] == [1, 1, 1]

    def test_dataframe_writer_with_limit(self, coordinate_space_3d):
        """AnnotationDataFrameWriter should pass limit through."""
        rng = np.random.default_rng(42)
        n = 5000
        df = pd.DataFrame(
            {
                "x": rng.uniform(0, 1000, n).astype(np.float32),
                "y": rng.uniform(0, 1000, n).astype(np.float32),
                "z": rng.uniform(0, 1000, n).astype(np.float32),
            }
        )
        df.index = np.arange(n, dtype=np.uint64)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = PointAnnotationWriter(
                coordinate_space=coordinate_space_3d,
                point_column=["x", "y", "z"],
                limit=500,
            )
            writer.write(df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)

            assert len(info["spatial"]) > 1


# ── SpatialHierarchy class tests ───────────────────────────────────


class TestSpatialHierarchyBase:
    """Tests for the SpatialHierarchy base class API."""

    def test_construction_defaults(self):
        h = _UniformHierarchy(rank=3)
        assert h.limit == 5000
        assert h.lower_bound is None
        assert h.upper_bound is None
        assert h.bound_padding == 0.0
        assert h.out_of_bounds == "warn"
        assert h.levels_ is None

    def test_assign_before_fit_raises(self):
        h = _UniformHierarchy(rank=3)
        coords = np.array([[1, 2, 3]], dtype=np.float32)
        with pytest.raises(RuntimeError, match="not been fitted"):
            h.assign(coords)

    def test_fit_returns_self(self):
        h = _UniformHierarchy(rank=3)
        coords = np.random.default_rng(0).uniform(0, 100, (50, 3)).astype(np.float32)
        result = h.fit(coords)
        assert result is h
        assert h.levels_ is not None
        assert h.lower_bound_ is not None
        assert h.upper_bound_ is not None

    def test_auto_computed_bounds(self):
        coords = np.array([[10, 20, 30], [100, 200, 300]], dtype=np.float32)
        h = _UniformHierarchy(rank=3).fit(coords)
        np.testing.assert_array_less(h.lower_bound_, coords.min(axis=0) + 1e-6)
        np.testing.assert_array_less(coords.max(axis=0), h.upper_bound_ + 1e-6)

    def test_explicit_bounds_adopted(self):
        lb = np.array([0, 0, 0], dtype=np.float64)
        ub = np.array([500, 500, 500], dtype=np.float64)
        coords = np.array([[10, 20, 30], [100, 200, 300]], dtype=np.float32)
        h = _UniformHierarchy(rank=3, lower_bound=lb, upper_bound=ub).fit(coords)
        np.testing.assert_array_equal(h.lower_bound_, lb)
        # upper_bound_ gets nextafter adjustment
        assert np.all(h.upper_bound_ >= ub)

    def test_bound_padding_auto_only(self):
        coords = np.array([[0, 0, 0], [1000, 1000, 40]], dtype=np.float32)
        h_no_pad = _UniformHierarchy(rank=3).fit(coords)
        h_pad = _UniformHierarchy(rank=3, bound_padding=0.1).fit(coords)

        # Padded bounds should be wider
        assert np.all(h_pad.lower_bound_ <= h_no_pad.lower_bound_)
        assert np.all(h_pad.upper_bound_ >= h_no_pad.upper_bound_)

    def test_bound_padding_not_applied_to_explicit(self):
        lb = np.array([0, 0, 0], dtype=np.float64)
        ub = np.array([500, 500, 500], dtype=np.float64)
        coords = np.array([[10, 20, 30], [100, 200, 300]], dtype=np.float32)
        h = _UniformHierarchy(
            rank=3, lower_bound=lb, upper_bound=ub, bound_padding=0.5
        ).fit(coords)
        np.testing.assert_array_equal(h.lower_bound_, lb)

    def test_oob_error_mode(self):
        coords_fit = np.array([[10, 10, 10], [20, 20, 20]], dtype=np.float32)
        h = _UniformHierarchy(rank=3, out_of_bounds="error").fit(coords_fit)
        coords_oob = np.array([[100, 100, 100]], dtype=np.float32)
        with pytest.raises(ValueError, match="outside"):
            h.assign(coords_oob)

    def test_oob_warn_mode(self):
        coords_fit = np.array([[10, 10, 10], [20, 20, 20]], dtype=np.float32)
        h = _UniformHierarchy(rank=3, out_of_bounds="warn").fit(coords_fit)
        coords_oob = np.array([[100, 100, 100]], dtype=np.float32)
        with pytest.warns(UserWarning, match="outside"):
            h.assign(coords_oob)

    def test_oob_ignore_mode(self):
        coords_fit = np.array([[10, 10, 10], [20, 20, 20]], dtype=np.float32)
        h = _UniformHierarchy(rank=3, out_of_bounds="ignore").fit(coords_fit)
        coords_oob = np.array([[100, 100, 100]], dtype=np.float32)
        # Should not raise or warn
        result = h.assign(coords_oob)
        assert result is not None

    def test_fit_assign_matches_separate(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 1000, (500, 3)).astype(np.float32)

        h1 = _UniformHierarchy(rank=3, limit=100)
        a1 = h1.fit(coords).assign(coords, rng=np.random.default_rng(7))

        h2 = _UniformHierarchy(rank=3, limit=100)
        a2 = h2.fit_assign(coords, rng=np.random.default_rng(7))

        assert len(a1.level_chunks) == len(a2.level_chunks)
        for key in a1.level_chunks:
            np.testing.assert_array_equal(a1.level_chunks[key], a2.level_chunks[key])


class TestUniformHierarchy:
    """Tests for UniformHierarchy matching existing build_spatial_levels."""

    def test_matches_build_spatial_levels(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 1000, (5000, 3)).astype(np.float32)
        lower = coords.min(axis=0).astype(np.float64)
        upper = coords.max(axis=0).astype(np.float64)
        extent = upper - lower
        extent = np.maximum(extent, 1.0)
        upper = lower + extent
        upper = np.nextafter(upper, np.inf)

        old_levels = build_spatial_levels(lower, upper, len(coords), limit=500)

        h = _UniformHierarchy(rank=3, limit=500, lower_bound=lower, upper_bound=upper)
        h.fit(coords)

        assert len(h.levels_) == len(old_levels)
        for new_lev, old_lev in zip(h.levels_, old_levels):
            assert new_lev.key == old_lev.key
            np.testing.assert_array_equal(new_lev.grid_shape, old_lev.grid_shape)
            assert new_lev.limit == old_lev.limit

    def test_explicit_chunk_size(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 1000, (1000, 3)).astype(np.float32)
        h = _UniformHierarchy(rank=3, chunk_size=200).fit(coords)

        # Finest level should have chunk_size close to 200
        finest = h.levels_[-1]
        assert np.all(finest.chunk_size <= 201)

    def test_explicit_chunk_size_scalar(self):
        coords = np.array([[0, 0, 0], [1000, 1000, 1000]], dtype=np.float32)
        h = _UniformHierarchy(rank=3, chunk_size=500.0).fit(coords)
        finest = h.levels_[-1]
        assert np.allclose(finest.chunk_size, finest.chunk_size[0])

    def test_coarsest_is_1_1_1(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 1000, (5000, 3)).astype(np.float32)
        h = _UniformHierarchy(rank=3, limit=500).fit(coords)
        np.testing.assert_array_equal(h.levels_[0].grid_shape, [1, 1, 1])

    def test_single_level_for_few_annotations(self):
        coords = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)
        h = _UniformHierarchy(rank=3, limit=5000).fit(coords)
        assert len(h.levels_) == 1
        np.testing.assert_array_equal(h.levels_[0].grid_shape, [1, 1, 1])


class TestIsotropicHierarchy:
    """Tests for IsotropicHierarchy spec-compliant subdivision."""

    def test_isotropic_data_uniform_subdivision(self):
        """Isotropic extents should halve all dimensions together."""
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 1000, (50000, 3)).astype(np.float32)
        h = _IsotropicHierarchy(rank=3, limit=500).fit(coords)

        for level in h.levels_:
            gs = level.grid_shape
            # All dimensions should be equal (or within 1 due to rounding)
            assert gs.max() - gs.min() <= 1, (
                f"Expected uniform grid for isotropic data, got {gs}"
            )

    def test_anisotropic_data_selective_subdivision(self):
        """Anisotropic extents should halve larger dimensions first."""
        rng = np.random.default_rng(42)
        n = 50000
        coords = np.column_stack(
            [
                rng.uniform(0, 1000, n),
                rng.uniform(0, 1000, n),
                rng.uniform(0, 40, n),
            ]
        ).astype(np.float32)
        h = _IsotropicHierarchy(rank=3, limit=500).fit(coords)

        # Check that early levels (after the [1,1,1] root) don't subdivide z
        if len(h.levels_) > 2:
            level1 = h.levels_[1]
            # x and y should be subdivided before z
            assert level1.grid_shape[2] <= level1.grid_shape[0], (
                f"Expected z not subdivided before x/y: {level1.grid_shape}"
            )

    def test_chunk_size_divisibility(self):
        """Adjacent levels must have chunk_size that evenly divides."""
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 1000, (10000, 3)).astype(np.float32)
        h = _IsotropicHierarchy(rank=3, limit=500).fit(coords)

        for i in range(len(h.levels_) - 1):
            coarser_cs = h.levels_[i].chunk_size
            finer_cs = h.levels_[i + 1].chunk_size
            for d in range(3):
                ratio = coarser_cs[d] / finer_cs[d]
                assert ratio == pytest.approx(1.0) or ratio == pytest.approx(2.0), (
                    f"Level {i}→{i + 1} dim {d}: ratio {ratio}, "
                    f"coarser={coarser_cs[d]}, finer={finer_cs[d]}"
                )

    def test_grid_times_chunk_equals_extent(self):
        """grid_shape * chunk_size must equal extent for every level."""
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 1000, (10000, 3)).astype(np.float32)
        h = _IsotropicHierarchy(rank=3, limit=500).fit(coords)

        extent = h.upper_bound_ - h.lower_bound_
        for level in h.levels_:
            product = level.grid_shape * level.chunk_size
            np.testing.assert_allclose(product, extent, rtol=1e-10)

    def test_single_level_few_annotations(self):
        coords = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)
        h = _IsotropicHierarchy(rank=3, limit=5000).fit(coords)
        assert len(h.levels_) == 1
        np.testing.assert_array_equal(h.levels_[0].grid_shape, [1, 1, 1])

    def test_many_annotations_multiple_levels(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 1000, (100000, 3)).astype(np.float32)
        h = _IsotropicHierarchy(rank=3, limit=5000).fit(coords)
        assert len(h.levels_) > 1

    def test_coarsest_is_1_1_1(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 1000, (10000, 3)).astype(np.float32)
        h = _IsotropicHierarchy(rank=3, limit=500).fit(coords)
        np.testing.assert_array_equal(h.levels_[0].grid_shape, [1, 1, 1])


class TestTwoPointBounds:
    """Tests for hierarchy bounds computation with two-point (line/bbox) data."""

    def test_uniform_fit_two_point_bounds_shape(self):
        """Fitting (N, 6) line coords with rank=3 produces (3,) bounds."""
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 100, (50, 6)).astype(np.float32)
        h = _UniformHierarchy(rank=3).fit(coords)
        assert h.lower_bound_.shape == (3,)
        assert h.upper_bound_.shape == (3,)

    def test_isotropic_fit_two_point_bounds_shape(self):
        """Fitting (N, 6) line coords with rank=3 produces (3,) bounds."""
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 100, (50, 6)).astype(np.float32)
        h = _IsotropicHierarchy(rank=3).fit(coords)
        assert h.lower_bound_.shape == (3,)
        assert h.upper_bound_.shape == (3,)

    def test_two_point_bounds_values(self):
        """Bounds should span the min/max across both point-halves."""
        coords = np.array(
            [
                [10, 20, 30, 90, 80, 70],
                [50, 60, 5, 40, 10, 95],
            ],
            dtype=np.float32,
        )
        h = _UniformHierarchy(rank=3).fit(coords)
        # lower should be <= component-wise min of both halves
        expected_lower = np.array([10, 10, 5], dtype=np.float64)
        expected_upper = np.array([90, 80, 95], dtype=np.float64)
        np.testing.assert_array_less(h.lower_bound_, expected_lower + 1e-6)
        np.testing.assert_array_less(expected_upper, h.upper_bound_ + 1e-6)

    def test_two_point_fit_assign_round_trip(self):
        """fit_assign on (N, 6) data with rank=3 should complete without error."""
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 100, (200, 6)).astype(np.float32)
        h = _UniformHierarchy(rank=3)
        result = h.fit_assign(coords, rng=rng)
        assert len(result.row_assignments) == 200
        assert len(result.level_chunks) > 0

    def test_two_point_levels_grid_shape_is_rank_sized(self):
        """All level grid_shapes should have length rank, not geom_size."""
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 100, (500, 6)).astype(np.float32)
        h = _IsotropicHierarchy(rank=3).fit(coords)
        for level in h.levels_:
            assert len(level.grid_shape) == 3
            assert len(level.chunk_size) == 3
