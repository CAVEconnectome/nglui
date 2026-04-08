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

from nglui.precomputed import AnnotationDataFrameWriter, PrecomputedAnnotationWriter
from nglui.precomputed._encoding import (
    build_dtype,
    encode_by_id_entries,
    encode_fixed_blocks,
    encode_multiple_annotations,
    sort_properties,
)
from nglui.precomputed._sharding import ShardSpec, choose_output_spec
from nglui.precomputed._spatial import (
    MultiscaleAssignment,
    SpatialLevel,
    auto_chunk_size,
    build_spatial_levels,
    compressed_morton_code,
    compute_chunk_assignments,
    compute_multiscale_assignments,
)


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
            writer = PrecomputedAnnotationWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                write_sharded=False,
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

            # Verify by_id files exist
            by_id_dir = os.path.join(tmpdir, "by_id")
            assert os.path.isdir(by_id_dir)
            assert len(os.listdir(by_id_dir)) == 100

            # Verify spatial0 directory has files
            spatial_dir = os.path.join(tmpdir, "spatial0")
            assert os.path.isdir(spatial_dir)
            assert len(os.listdir(spatial_dir)) > 0

    def test_write_points_with_properties(self, coordinate_space_3d):
        with tempfile.TemporaryDirectory() as tmpdir:
            props = [AnnotationPropertySpec(id="score", type="float32")]
            writer = PrecomputedAnnotationWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                properties=props,
                write_sharded=False,
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
            writer = PrecomputedAnnotationWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                relationships=["segment"],
                write_sharded=False,
            )
            coords = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            writer.set_coordinates(coords)
            writer.set_relationship("segment", np.array([100, 200], dtype=np.uint64))
            writer.write(tmpdir)

            # Check relationship dir exists
            rel_dir = os.path.join(tmpdir, "rel_segment")
            assert os.path.isdir(rel_dir)
            # Should have files for segment IDs 100 and 200
            assert os.path.isfile(os.path.join(rel_dir, "100"))
            assert os.path.isfile(os.path.join(rel_dir, "200"))

    def test_write_points_variable_relationships(self, coordinate_space_3d):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = PrecomputedAnnotationWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                relationships=["segments"],
                write_sharded=False,
            )
            coords = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            writer.set_coordinates(coords)
            writer.set_relationship("segments", [[100, 200], [300]])
            writer.write(tmpdir)

            rel_dir = os.path.join(tmpdir, "rel_segments")
            # Should have files for 100, 200, 300
            assert os.path.isfile(os.path.join(rel_dir, "100"))
            assert os.path.isfile(os.path.join(rel_dir, "200"))
            assert os.path.isfile(os.path.join(rel_dir, "300"))

    def test_per_row_api(self, coordinate_space_3d):
        with tempfile.TemporaryDirectory() as tmpdir:
            props = [AnnotationPropertySpec(id="score", type="float32")]
            writer = PrecomputedAnnotationWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                relationships=["segment"],
                properties=props,
                write_sharded=False,
            )
            writer.add_point([1, 2, 3], score=0.5, segment=100)
            writer.add_point([4, 5, 6], score=0.9, segment=200)
            writer.write(tmpdir)

            by_id_dir = os.path.join(tmpdir, "by_id")
            assert len(os.listdir(by_id_dir)) == 2

    def test_bulk_and_per_row_produce_same_info(self, coordinate_space_3d):
        coords = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        with (
            tempfile.TemporaryDirectory() as tmp_bulk,
            tempfile.TemporaryDirectory() as tmp_row,
        ):
            # Bulk
            w1 = PrecomputedAnnotationWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                write_sharded=False,
                chunk_size=1000,
            )
            w1.set_coordinates(coords)
            w1.write(tmp_bulk)

            # Per-row
            w2 = PrecomputedAnnotationWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                write_sharded=False,
                chunk_size=1000,
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


class TestAnnotationDataFrameWriter:
    def test_write_xyz_columns(self, coordinate_space_3d, sample_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = AnnotationDataFrameWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                x_column="pt_x",
                y_column="pt_y",
                z_column="pt_z",
                write_sharded=False,
            )
            writer.write(sample_df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            assert info["annotation_type"] == "point"
            assert len(os.listdir(os.path.join(tmpdir, "by_id"))) == 100

    def test_write_array_column(self, coordinate_space_3d, sample_df_array_col):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = AnnotationDataFrameWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                point_column="pt_position",
                write_sharded=False,
            )
            writer.write(sample_df_array_col, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            assert info["annotation_type"] == "point"
            assert len(os.listdir(os.path.join(tmpdir, "by_id"))) == 100

    def test_write_with_properties(self, coordinate_space_3d, sample_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = AnnotationDataFrameWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                x_column="pt_x",
                y_column="pt_y",
                z_column="pt_z",
                property_columns={"score": "score"},
                write_sharded=False,
            )
            writer.write(sample_df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            assert len(info["properties"]) == 1
            assert info["properties"][0]["id"] == "score"

    def test_write_with_relationships(self, coordinate_space_3d, sample_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = AnnotationDataFrameWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                x_column="pt_x",
                y_column="pt_y",
                z_column="pt_z",
                relationship_columns={"segment": "segment_id"},
                write_sharded=False,
            )
            writer.write(sample_df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            assert len(info["relationships"]) == 1
            assert info["relationships"][0]["id"] == "segment"

            rel_dir = os.path.join(tmpdir, "rel_segment")
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
            w1 = AnnotationDataFrameWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                x_column="x",
                y_column="y",
                z_column="z",
                write_sharded=False,
                chunk_size=10000,
            )
            w1.write(df_xyz, tmp1)

            w2 = AnnotationDataFrameWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                point_column="pt",
                write_sharded=False,
                chunk_size=10000,
            )
            w2.write(df_arr, tmp2)

            # Both should produce same by_id content
            for i in range(len(sample_points)):
                f1 = os.path.join(tmp1, "by_id", str(i))
                f2 = os.path.join(tmp2, "by_id", str(i))
                with open(f1, "rb") as a, open(f2, "rb") as b:
                    assert a.read() == b.read(), f"Mismatch at annotation {i}"

    def test_validation_errors(self, coordinate_space_3d):
        with pytest.raises(ValueError, match="point_column"):
            AnnotationDataFrameWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
            )

        with pytest.raises(ValueError, match="not both"):
            AnnotationDataFrameWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                point_column="pt",
                x_column="x",
                y_column="y",
                z_column="z",
            )

    def test_bounds_match_data(self, coordinate_space_3d, sample_points):
        df = pd.DataFrame({"pt": list(sample_points)})
        df.index = np.arange(len(sample_points), dtype=np.uint64)

        with tempfile.TemporaryDirectory() as tmpdir:
            w = AnnotationDataFrameWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                point_column="pt",
                write_sharded=False,
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


# ── Source Resolution Tests ─────────────────────────────────────────


class TestSourceResolution:
    def test_resolution_parameter(self):
        """Test that resolution= creates a valid writer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = PrecomputedAnnotationWriter(
                annotation_type="point",
                resolution=[8, 8, 40],
                write_sharded=False,
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
            writer = AnnotationDataFrameWriter(
                annotation_type="point",
                resolution=[1, 1, 1],
                x_column="pt_x",
                y_column="pt_y",
                z_column="pt_z",
                write_sharded=False,
            )
            writer.write(sample_df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)
            assert info["annotation_type"] == "point"

    def test_no_source_raises(self):
        """Must provide at least one source of coordinate space."""
        with pytest.raises(ValueError, match="Must provide one of"):
            PrecomputedAnnotationWriter(annotation_type="point")

    def test_multiple_sources_raises(self, coordinate_space_3d):
        """Cannot provide both coordinate_space and resolution."""
        with pytest.raises(ValueError, match="Provide only one of"):
            PrecomputedAnnotationWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                resolution=[1, 1, 1],
            )


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
            writer = PrecomputedAnnotationWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                write_sharded=False,
                limit=500,
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
            writer = PrecomputedAnnotationWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                write_sharded=False,
                limit=500,
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
            writer = PrecomputedAnnotationWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                write_sharded=False,
                limit=200,
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
            writer = PrecomputedAnnotationWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                write_sharded=False,
                limit=100000,
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
            writer = AnnotationDataFrameWriter(
                annotation_type="point",
                coordinate_space=coordinate_space_3d,
                x_column="x",
                y_column="y",
                z_column="z",
                write_sharded=False,
                limit=500,
            )
            writer.write(df, tmpdir)

            with open(os.path.join(tmpdir, "info")) as f:
                info = json.load(f)

            assert len(info["spatial"]) > 1
