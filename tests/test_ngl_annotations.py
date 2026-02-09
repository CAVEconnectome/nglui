from unittest.mock import Mock, patch

import numpy as np
import pytest

from nglui.statebuilder.ngl_annotations import (
    MAX_TAG_COUNT,
    AnnotationBase,
    AnnotationProperty,
    AnnotationTag,
    BoundingBoxAnnotation,
    EllipsoidAnnotation,
    LineAnnotation,
    PointAnnotation,
    PolylineAnnotation,
    TagTool,
    TagToolFactory,
    make_annotation_properties,
    make_bindings,
)


class TestTagTool:
    def test_tag_tool_creation(self):
        tag_tool = TagTool(tag_num=5)
        assert tag_tool.tag_num == 5

    @patch("nglui.statebuilder.ngl_annotations.viewer_state")
    def test_initialize_neuroglancer(self, mock_viewer_state):
        tag_tool = TagTool(tag_num=3)
        tag_tool.initialize_neuroglancer()

        # Should call export_tool decorator
        mock_viewer_state.export_tool.assert_called_once()

    def test_tag_tool_factory(self):
        # Test that TagToolFactory creates tools without errors
        # This is hard to test directly, but we can verify it doesn't crash
        TagToolFactory(3)  # Should not raise


class TestAnnotationProperty:
    def test_annotation_property_default(self):
        prop = AnnotationProperty()
        assert prop.id is None
        assert prop.type == "uint8"
        assert prop.tag is None

    def test_annotation_property_with_values(self):
        prop = AnnotationProperty(id="test_id", type="float32", tag="test_tag")
        assert prop.id == "test_id"
        assert prop.type == "float32"
        assert prop.tag == "test_tag"

    def test_to_neuroglancer(self):
        prop = AnnotationProperty(id="test_id", type="uint16", tag="test_tag")
        result = prop.to_neuroglancer()

        expected = {"id": "test_id", "type": "uint16", "tag": "test_tag"}
        assert result == expected


class TestAnnotationTag:
    def test_annotation_tag_creation(self):
        tag = AnnotationTag(id=1, tag="cell_body")
        assert tag.id == 1
        assert tag.tag == "cell_body"

    def test_to_neuroglancer(self):
        tag = AnnotationTag(id=2, tag="dendrite")
        result = tag.to_neuroglancer(tag_base_number=5)

        expected = {"id": "tag7", "type": "uint8", "tag": "dendrite"}
        assert result == expected

    def test_to_neuroglancer_zero_base(self):
        tag = AnnotationTag(id=0, tag="axon")
        result = tag.to_neuroglancer(tag_base_number=0)

        expected = {"id": "tag0", "type": "uint8", "tag": "axon"}
        assert result == expected


class TestMakeAnnotationProperties:
    def test_make_annotation_properties_basic(self):
        annotations = ["cell_body", "dendrite", "axon"]
        result = make_annotation_properties(annotations, tag_base_number=0)

        assert len(result) == 3
        assert result[0]["id"] == "tag0"
        assert result[0]["tag"] == "cell_body"
        assert result[1]["id"] == "tag1"
        assert result[1]["tag"] == "dendrite"
        assert result[2]["id"] == "tag2"
        assert result[2]["tag"] == "axon"

    def test_make_annotation_properties_with_base_number(self):
        annotations = ["soma", "spine"]
        result = make_annotation_properties(annotations, tag_base_number=10)

        assert len(result) == 2
        assert result[0]["id"] == "tag10"
        assert result[0]["tag"] == "soma"
        assert result[1]["id"] == "tag11"
        assert result[1]["tag"] == "spine"

    def test_make_annotation_properties_empty(self):
        result = make_annotation_properties([])
        assert result == []


class TestMakeBindings:
    def test_make_bindings_basic(self):
        properties = [
            {"id": "tag0", "tag": "cell_body"},
            {"id": "tag1", "tag": "dendrite"},
        ]
        result = make_bindings(properties)

        assert result["Q"] == "tagTool_tag0"
        assert result["W"] == "tagTool_tag1"

    def test_make_bindings_custom_bindings(self):
        properties = [{"id": "tag0", "tag": "test"}]
        custom_bindings = ["X", "Y", "Z"]

        result = make_bindings(properties, bindings=custom_bindings)
        assert result["X"] == "tagTool_tag0"

    def test_make_bindings_too_many_properties(self):
        properties = [{"id": f"tag{i}", "tag": f"tag{i}"} for i in range(15)]

        with pytest.raises(ValueError, match="Too many properties for bindings"):
            make_bindings(properties)

    def test_make_bindings_empty_properties(self):
        result = make_bindings([])
        assert result == {}


class TestAnnotationBase:
    def test_annotation_base_defaults(self):
        # AnnotationBase is abstract, so we need to test via a subclass
        point = PointAnnotation(point=[10, 20, 30])
        assert point.id is not None  # Should auto-generate
        assert point.description is None
        assert point.segments is None
        assert point.tags == []
        assert point.resolution is None
        assert point.props == []

    def test_annotation_base_with_values(self):
        point = PointAnnotation(
            point=[10, 20, 30],
            id="test_id",
            description="test point",
            segments=[123, 456],
            tags=["cell_body", "dendrite"],
            resolution=[4, 4, 40],
        )
        assert point.id == "test_id"
        assert point.description == "test point"
        assert point.segments == [[123, 456]]  # Gets converted by converter
        assert point.tags == ["cell_body", "dendrite"]
        assert np.array_equal(point.resolution, [4, 4, 40])

    def test_initialize_property_list(self):
        point = PointAnnotation(point=[10, 20, 30])
        point.initialize_property_list(5)
        assert point.props == [0, 0, 0, 0, 0]

    def test_set_tag_id(self):
        point = PointAnnotation(point=[10, 20, 30])
        point.initialize_property_list(3)
        point.set_tag_id(1)
        assert point.props == [0, 1, 0]

    def test_set_tags(self):
        point = PointAnnotation(point=[10, 20, 30], tags=["cell_body", "axon"])
        tag_map = {"cell_body": 0, "dendrite": 1, "axon": 2}

        point.set_tags(tag_map)
        assert point.props == [1, 0, 1]  # cell_body and axon are set

    def test_set_tags_missing_tag(self):
        point = PointAnnotation(point=[10, 20, 30], tags=["unknown_tag"])
        tag_map = {"cell_body": 0, "dendrite": 1}

        point.set_tags(tag_map)
        assert point.props == [0, 0]  # unknown_tag not in map

    def test_scale_points(self):
        # Test the base scale_points method with actual scaling behavior
        point = PointAnnotation(point=[100, 200, 300], resolution=[8, 8, 40])
        layer_resolution = [4, 4, 40]

        # Call scale_points and check the result
        point.scale_points(layer_resolution)

        # Should scale by [2.0, 2.0, 1.0] = [8,8,40] / [4,4,40]
        expected_point = [200.0, 400.0, 300.0]
        assert point.point == expected_point

    def test_auto_generated_id(self):
        point1 = PointAnnotation(point=[10, 20, 30])
        point2 = PointAnnotation(point=[40, 50, 60])

        assert point1.id is not None
        assert point2.id is not None
        assert point1.id != point2.id  # Should be unique


class TestPointAnnotation:
    def test_point_annotation_creation(self):
        point = PointAnnotation(point=[100, 200, 300])
        assert np.array_equal(point.point, [100, 200, 300])

    def test_point_annotation_with_numpy_array(self):
        np_point = np.array([100.5, 200.7, 300.2])
        point = PointAnnotation(point=np_point)
        # strip_numpy_types converter should handle this
        assert isinstance(point.point, (list, np.ndarray))

    def test_scale_points(self):
        point = PointAnnotation(point=[100, 200, 300])
        point._scale_points([2.0, 2.0, 1.0])

        # Should scale the point coordinates
        assert point.point == [200.0, 400.0, 300.0]

    @patch("neuroglancer.viewer_state.PointAnnotation")
    def test_to_neuroglancer(self, mock_point_annotation):
        point = PointAnnotation(point=[100, 200, 300])
        tag_map = {"cell_body": 0}
        layer_resolution = [4, 4, 40]

        point.to_neuroglancer(tag_map=tag_map, layer_resolution=layer_resolution)

        # Should call the neuroglancer PointAnnotation constructor
        mock_point_annotation.assert_called_once()


class TestLineAnnotation:
    def test_line_annotation_creation(self):
        line = LineAnnotation(pointA=[10, 20, 30], pointB=[40, 50, 60])
        assert np.array_equal(line.pointA, [10, 20, 30])
        assert np.array_equal(line.pointB, [40, 50, 60])

    def test_scale_points(self):
        line = LineAnnotation(pointA=[100, 200, 300], pointB=[400, 500, 600])
        line._scale_points([2.0, 2.0, 1.0])

        assert line.pointA == [200.0, 400.0, 300.0]
        assert line.pointB == [800.0, 1000.0, 600.0]

    @patch("neuroglancer.viewer_state.LineAnnotation")
    def test_to_neuroglancer(self, mock_line_annotation):
        line = LineAnnotation(pointA=[10, 20, 30], pointB=[40, 50, 60])

        line.to_neuroglancer(tag_map={}, layer_resolution=None)
        mock_line_annotation.assert_called_once()


class TestEllipsoidAnnotation:
    def test_ellipsoid_annotation_creation(self):
        ellipsoid = EllipsoidAnnotation(center=[100, 200, 300], radii=[10, 15, 20])
        assert np.array_equal(ellipsoid.center, [100, 200, 300])
        assert np.array_equal(ellipsoid.radii, [10, 15, 20])

    def test_scale_points(self):
        ellipsoid = EllipsoidAnnotation(center=[100, 200, 300], radii=[10, 15, 20])
        ellipsoid._scale_points([2.0, 2.0, 1.0])

        assert ellipsoid.center == [200.0, 400.0, 300.0]
        assert ellipsoid.radii == [20.0, 30.0, 20.0]

    @patch("neuroglancer.viewer_state.EllipsoidAnnotation")
    def test_to_neuroglancer(self, mock_ellipsoid_annotation):
        ellipsoid = EllipsoidAnnotation(center=[100, 200, 300], radii=[10, 15, 20])

        ellipsoid.to_neuroglancer(tag_map={}, layer_resolution=None)
        mock_ellipsoid_annotation.assert_called_once()


class TestBoundingBoxAnnotation:
    def test_bounding_box_annotation_creation(self):
        bbox = BoundingBoxAnnotation(pointA=[10, 20, 30], pointB=[40, 50, 60])
        assert np.array_equal(bbox.pointA, [10, 20, 30])
        assert np.array_equal(bbox.pointB, [40, 50, 60])

    def test_scale_points(self):
        bbox = BoundingBoxAnnotation(pointA=[100, 200, 300], pointB=[400, 500, 600])
        bbox._scale_points([2.0, 2.0, 1.0])

        assert bbox.pointA == [200.0, 400.0, 300.0]
        assert bbox.pointB == [800.0, 1000.0, 600.0]

    @patch("neuroglancer.viewer_state.AxisAlignedBoundingBoxAnnotation")
    def test_to_neuroglancer(self, mock_bbox_annotation):
        bbox = BoundingBoxAnnotation(pointA=[10, 20, 30], pointB=[40, 50, 60])

        bbox.to_neuroglancer(tag_map={}, layer_resolution=None)
        mock_bbox_annotation.assert_called_once()


class TestPolylineAnnotation:
    def test_polyline_annotation_creation(self):
        polyline = PolylineAnnotation(points=[[10, 20, 30], [40, 50, 60], [70, 80, 90]])
        assert len(polyline.points) == 3
        assert polyline.points[0] == [10, 20, 30]
        assert polyline.points[2] == [70, 80, 90]

    def test_polyline_annotation_with_numpy_array(self):
        np_points = np.array([[100.5, 200.7, 300.2], [400.1, 500.3, 600.9]])
        polyline = PolylineAnnotation(points=np_points)
        assert len(polyline.points) == 2

    def test_scale_points(self):
        polyline = PolylineAnnotation(points=[[100, 200, 300], [400, 500, 600]])
        polyline._scale_points([2.0, 2.0, 1.0])

        expected = [[200.0, 400.0, 300.0], [800.0, 1000.0, 600.0]]
        assert np.allclose(polyline.points, expected)

    @patch("neuroglancer.viewer_state.PolyLineAnnotation")
    def test_to_neuroglancer(self, mock_polyline_annotation):
        polyline = PolylineAnnotation(points=[[10, 20, 30], [40, 50, 60]])

        polyline.to_neuroglancer(tag_map={}, layer_resolution=None)
        mock_polyline_annotation.assert_called_once()

    def test_polyline_with_resolution(self):
        polyline = PolylineAnnotation(
            points=[[100, 200, 300], [400, 500, 600]],
            resolution=[8, 8, 40],
        )
        layer_resolution = [4, 4, 40]
        polyline.scale_points(layer_resolution)

        expected = [[200.0, 400.0, 300.0], [800.0, 1000.0, 600.0]]
        assert np.allclose(polyline.points, expected)

    def test_polyline_with_segments_and_tags(self):
        polyline = PolylineAnnotation(
            points=[[10, 20, 30], [40, 50, 60]],
            segments=[123, 456],
            tags=["dendrite", "axon"],
        )
        assert polyline.segments == [[123, 456]]
        assert polyline.tags == ["dendrite", "axon"]

    def test_polyline_single_point(self):
        polyline = PolylineAnnotation(points=[[10, 20, 30]])
        assert len(polyline.points) == 1


class TestAnnotationIntegration:
    def test_annotation_with_complex_tags(self):
        # Test annotation with multiple tags and properties
        point = PointAnnotation(
            point=[100, 200, 300],
            tags=["cell_body", "dendrite", "spine"],
            segments=[123, 456],
        )

        tag_map = {"cell_body": 0, "dendrite": 1, "spine": 2, "axon": 3}
        point.set_tags(tag_map)

        assert point.props == [1, 1, 1, 0]  # First 3 tags set, axon not

    def test_annotation_scaling_with_resolution(self):
        # Test end-to-end scaling behavior
        point = PointAnnotation(point=[1000, 2000, 3000], resolution=[8, 8, 40])

        # Scale to a different layer resolution
        layer_resolution = [4, 4, 40]
        point.scale_points(layer_resolution)

        # Point should be scaled by [2, 2, 1]
        assert np.allclose(point.point, [2000, 4000, 3000])

    def test_max_tag_count_validation(self):
        # Test that MAX_TAG_COUNT is reasonable
        assert MAX_TAG_COUNT == 10
        assert isinstance(MAX_TAG_COUNT, int)

    def test_annotation_property_filtering(self):
        # Test the _annotation_filter function indirectly
        point = PointAnnotation(
            point=[100, 200, 300], tags=["test"], resolution=[4, 4, 40]
        )

        # When converting to neuroglancer, tags and resolution should be filtered out
        with patch("neuroglancer.viewer_state.PointAnnotation") as mock_point:
            point.to_neuroglancer()

            # The call should not include tags or resolution in the kwargs
            call_kwargs = mock_point.call_args[1] if mock_point.call_args[1] else {}
            assert "tags" not in call_kwargs
            assert "resolution" not in call_kwargs


class TestAnnotationEdgeCases:
    def test_empty_segments_list(self):
        point = PointAnnotation(point=[10, 20, 30], segments=[])
        assert point.segments == []

    def test_none_values(self):
        point = PointAnnotation(
            point=[10, 20, 30],
            id=None,  # Should auto-generate
            description=None,
            segments=None,
            tags=None,  # Should default to []
            resolution=None,
        )

        assert point.id is not None  # Auto-generated
        assert point.description is None
        assert point.segments is None
        assert point.tags == []
        assert point.resolution is None

    def test_numpy_type_conversion(self):
        # Test that numpy types are properly converted
        np_point = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        point = PointAnnotation(point=np_point)

        # strip_numpy_types should handle the conversion
        assert isinstance(point.point, (list, np.ndarray))

    def test_tag_map_with_missing_keys(self):
        point = PointAnnotation(
            point=[10, 20, 30], tags=["existing", "missing", "another_existing"]
        )

        # Tag map only has some of the tags, using consecutive integers from 0
        tag_map = {"existing": 0, "another_existing": 1}  # consecutive 0, 1
        point.set_tags(tag_map)

        # Only the existing tags should be set (missing tag is not in map)
        assert point.props == [1, 1]  # both existing tags are set

    def test_zero_scale_factor(self):
        point = PointAnnotation(point=[100, 200, 300])

        # Test with zero scale factor (edge case)
        point._scale_points([0, 1, 2])
        assert point.point == [0.0, 200.0, 600.0]
