from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import webcolors

from nglui.statebuilder.utils import (
    NamedList,
    is_list_like,
    parse_color,
    split_point_columns,
    strip_layers,
    strip_numpy_types,
    strip_state_properties,
)


class TestStripStateProperties:
    def test_strip_state_properties_basic(self):
        state = {
            "layers": ["layer1", "layer2"],
            "position": [100, 200, 300],  # This is a list, so gets emptied
            "scale": 2.0,  # This is not a list, so gets deleted
            "other": "value",
        }

        result = strip_state_properties(
            state, strip_keys=["layers", "position", "scale"]
        )

        expected = {
            "layers": [],  # List gets emptied
            "position": [],  # List gets emptied
            "other": "value",
        }
        # scale should be deleted entirely since it's not a list
        assert "scale" not in result
        assert result == expected

        # Original state should not be modified
        assert state["layers"] == ["layer1", "layer2"]

    def test_strip_state_properties_empty_strip_keys(self):
        state = {"layers": ["layer1"], "position": [100, 200]}
        result = strip_state_properties(state, strip_keys=[])
        assert result == state
        assert result is not state  # Should be a deep copy

    def test_strip_state_properties_nonexistent_keys(self):
        state = {"layers": ["layer1"], "position": [100, 200]}
        result = strip_state_properties(state, strip_keys=["nonexistent"])
        assert result == state

    def test_strip_state_properties_mixed_types(self):
        state = {
            "list_field": ["item1", "item2"],
            "non_list_field": "single_value",
            "dict_field": {"nested": "value"},
        }

        result = strip_state_properties(
            state, strip_keys=["list_field", "non_list_field", "dict_field"]
        )

        expected = {"list_field": []}  # Only list gets emptied
        assert result == expected


class TestStripLayers:
    def test_strip_layers_basic(self):
        state = {
            "layers": ["layer1", "layer2"],
            "selectedLayer": {"layer": "layer1"},
            "selection": {"segment": 123},
            "position": [100, 200, 300],
            "scale": 2.0,
        }

        result = strip_layers(state)

        expected = {"layers": [], "position": [100, 200, 300], "scale": 2.0}
        assert result == expected

    def test_strip_layers_empty_state(self):
        state = {}
        result = strip_layers(state)
        assert result == {}

    def test_strip_layers_no_layer_fields(self):
        state = {"position": [100, 200], "scale": 1.0}
        result = strip_layers(state)
        assert result == state


class TestNamedList:
    def test_namedlist_creation_empty(self):
        nl = NamedList()
        assert len(nl) == 0
        assert nl._name_map == {}

    def test_namedlist_creation_with_items(self):
        # Mock objects with name attribute
        item1 = Mock()
        item1.name = "first"
        item2 = Mock()
        item2.name = "second"

        nl = NamedList([item1, item2])

        assert len(nl) == 2
        assert nl[0] is item1
        assert nl[1] is item2
        assert nl["first"] is item1
        assert nl["second"] is item2

    def test_namedlist_string_indexing(self):
        item = Mock()
        item.name = "test_item"

        nl = NamedList([item])
        assert nl["test_item"] is item

    def test_namedlist_string_indexing_keyerror(self):
        nl = NamedList()

        with pytest.raises(KeyError):
            _ = nl["nonexistent"]

    def test_namedlist_numeric_indexing(self):
        item1 = Mock()
        item1.name = "first"
        item2 = Mock()
        item2.name = "second"

        nl = NamedList([item1, item2])

        assert nl[0] is item1
        assert nl[1] is item2
        assert nl[-1] is item2

    def test_namedlist_append(self):
        item1 = Mock()
        item1.name = "first"
        item2 = Mock()
        item2.name = "second"

        nl = NamedList([item1])
        nl.append(item2)

        assert len(nl) == 2
        assert nl["first"] is item1
        assert nl["second"] is item2
        assert nl[1] is item2

    def test_namedlist_extend(self):
        item1 = Mock()
        item1.name = "first"
        item2 = Mock()
        item2.name = "second"
        item3 = Mock()
        item3.name = "third"

        nl = NamedList([item1])
        nl.extend([item2, item3])

        assert len(nl) == 3
        assert nl["first"] is item1
        assert nl["second"] is item2
        assert nl["third"] is item3

    def test_namedlist_name_update_on_append_extend(self):
        item1 = Mock()
        item1.name = "original"

        nl = NamedList([item1])
        assert nl["original"] is item1

        # Change name and append again - should update mapping
        item1.name = "changed"
        item2 = Mock()
        item2.name = "new"

        nl.append(item2)

        # Old mapping should still exist (this is current behavior)
        assert nl["original"] is item1
        assert nl["new"] is item2


class TestSplitPointColumns:
    def test_split_point_columns_single_column_exists(self):
        columns = ["x", "y", "z", "point_position", "other"]
        result = split_point_columns("point_position", columns)
        assert result == "point_position"

    def test_split_point_columns_prefix_expansion(self):
        columns = ["prefix_x", "prefix_y", "prefix_z", "other"]
        result = split_point_columns("prefix", columns)
        assert result == ["prefix_x", "prefix_y", "prefix_z"]

    def test_split_point_columns_list_input_valid(self):
        columns = ["x", "y", "z"]
        col_list = ["x", "y", "z"]
        result = split_point_columns(col_list, columns)
        assert result == col_list

    def test_split_point_columns_list_input_invalid(self):
        columns = ["x", "y", "z"]
        col_list = ["x", "y", 123]  # Non-string in list
        result = split_point_columns(col_list, columns)
        assert result is None

    def test_split_point_columns_missing_column(self):
        columns = ["x", "y", "z"]

        with pytest.raises(ValueError, match="Column 'missing' not found"):
            split_point_columns("missing", columns)

    def test_split_point_columns_incomplete_prefix(self):
        columns = ["prefix_x", "prefix_y", "other"]  # Missing prefix_z

        with pytest.raises(
            ValueError, match="'prefix_x', 'prefix_y', 'prefix_z' are not all present"
        ):
            split_point_columns("prefix", columns)

    def test_split_point_columns_empty_columns(self):
        columns = []

        with pytest.raises(ValueError, match="Column 'test' not found"):
            split_point_columns("test", columns)


class TestStripNumpyTypes:
    def test_strip_numpy_types_scalar_numpy(self):
        # Test numpy scalars
        np_int = np.int32(42)
        result = strip_numpy_types(np_int)
        assert result == 42
        assert isinstance(result, int)

        np_float = np.float64(3.14)
        result = strip_numpy_types(np_float)
        assert result == 3.14
        assert isinstance(result, float)

    def test_strip_numpy_types_numpy_array(self):
        np_array = np.array([1, 2, 3])
        result = strip_numpy_types(np_array)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_strip_numpy_types_nested_array(self):
        nested_array = np.array([[1, 2], [3, 4]])
        result = strip_numpy_types(nested_array)
        assert result == [[1, 2], [3, 4]]
        assert isinstance(result, list)
        assert all(isinstance(row, list) for row in result)

    def test_strip_numpy_types_python_list(self):
        python_list = [1, 2, 3]
        result = strip_numpy_types(python_list)
        assert result == [1, 2, 3]
        assert result is not python_list  # Should be a copy

    def test_strip_numpy_types_mixed_list(self):
        mixed_list = [1, np.int32(2), [np.float64(3.0), 4]]
        result = strip_numpy_types(mixed_list)
        assert result == [1, 2, [3.0, 4]]
        assert isinstance(result[1], int)
        assert isinstance(result[2][0], float)

    def test_strip_numpy_types_dict(self):
        np_dict = {
            "numpy_int": np.int32(42),
            "numpy_array": np.array([1, 2]),
            "regular": "string",
            "nested": {"inner": np.float64(3.14)},
        }

        result = strip_numpy_types(np_dict)

        expected = {
            "numpy_int": 42,
            "numpy_array": [1, 2],
            "regular": "string",
            "nested": {"inner": 3.14},
        }
        assert result == expected

    def test_strip_numpy_types_tuple(self):
        # Simple flat tuple with numpy types
        np_tuple = (np.int32(1), np.float64(2.0), np.bool_(True))
        result = strip_numpy_types(np_tuple)
        # Tuples get converted to lists by strip_numpy_types
        assert result == [1, 2.0, True]

    def test_strip_numpy_types_set(self):
        # Sets get converted to lists by strip_numpy_types
        np_set = {np.int32(1), np.int32(2), np.int32(3)}
        result = strip_numpy_types(np_set)
        # Sets get converted to lists, order may vary
        assert set(result) == {1, 2, 3}
        assert isinstance(result, list)

    def test_strip_numpy_types_regular_python_types(self):
        # Should pass through regular Python types unchanged
        test_cases = [42, 3.14, "string", True, None, [1, 2, 3], {"key": "value"}]

        for case in test_cases:
            result = strip_numpy_types(case)
            assert result == case
            assert type(result) == type(case)

    def test_strip_numpy_types_complex_nested_structure(self):
        complex_structure = {
            "data": [
                {
                    "points": np.array([[1.0, 2.0], [3.0, 4.0]]),
                    "metadata": {"count": np.int32(2), "average": np.float64(2.5)},
                }
            ],
            "config": (np.bool_(True), "test"),
        }

        result = strip_numpy_types(complex_structure)

        expected = {
            "data": [
                {
                    "points": [[1.0, 2.0], [3.0, 4.0]],
                    "metadata": {"count": 2, "average": 2.5},
                }
            ],
            "config": [True, "test"],  # Tuples become lists
        }
        assert result == expected

    def test_strip_numpy_types_nan_inf(self):
        # Test handling of special float values
        test_values = [np.nan, np.inf, -np.inf]

        for val in test_values:
            result = strip_numpy_types(val)
            if np.isnan(val):
                assert np.isnan(result)
            else:
                assert result == float(val)


class TestParseColor:
    def test_parse_color_hex_string(self):
        hex_color = "#ff0000"
        result = parse_color(hex_color)
        assert result == "#ff0000"  # Should return as-is for valid hex

    def test_parse_color_hex_string_uppercase(self):
        hex_color = "#FF0000"
        result = parse_color(hex_color)
        assert result == "#FF0000"  # Returns as-is, doesn't normalize case

    def test_parse_color_name_string(self):
        color_name = "red"
        result = parse_color(color_name)
        assert result == "#ff0000"

    def test_parse_color_name_string_case_insensitive(self):
        color_name = "RED"
        result = parse_color(color_name)
        assert result == "#ff0000"

    def test_parse_color_rgb_tuple_float(self):
        rgb_tuple = (1.0, 0.0, 0.0)  # Red in float format
        result = parse_color(rgb_tuple)
        assert result == "#ff0000"

    def test_parse_color_rgb_tuple_partial(self):
        rgb_tuple = (0.5, 0.5, 0.5)  # Gray
        result = parse_color(rgb_tuple)
        # 0.5 * 255 = 127.5 -> int(127.5) = 127 = 0x7f
        assert result == "#7f7f7f"

    def test_parse_color_rgb_list(self):
        rgb_list = [0.0, 1.0, 0.0]  # Green
        result = parse_color(rgb_list)
        assert result == "#00ff00"

    def test_parse_color_invalid_hex(self):
        invalid_hex = "#gggggg"

        with pytest.raises(ValueError):
            parse_color(invalid_hex)

    def test_parse_color_invalid_name(self):
        invalid_name = "not_a_color"

        with pytest.raises(ValueError):
            parse_color(invalid_name)

    def test_parse_color_rgb_out_of_range(self):
        # Test behavior with out-of-range values
        rgb_tuple = (2.0, -1.0, 0.5)

        # webcolors.rgb_to_hex clamps values, so this should work
        result = parse_color(rgb_tuple)
        # 2.0*255=510 -> clamped to 255 = ff
        # -1.0*255=-255 -> clamped to 0 = 00
        # 0.5*255=127.5 -> int(127.5)=127 = 7f
        assert isinstance(result, str)
        assert result.startswith("#")

    def test_parse_color_wrong_tuple_length(self):
        invalid_tuple = (1.0, 0.0)  # Missing blue component

        with pytest.raises(
            TypeError
        ):  # webcolors raises TypeError for wrong tuple length
            parse_color(invalid_tuple)

    def test_parse_color_none(self):
        result = parse_color(None)
        assert result is None

    def test_parse_color_already_parsed(self):
        # Test that already valid hex colors pass through
        hex_color = "#1a2b3c"
        result = parse_color(hex_color)
        assert result == hex_color


class TestIsListLike:
    def test_is_list_like_list(self):
        assert is_list_like([1, 2, 3]) is True

    def test_is_list_like_tuple(self):
        assert is_list_like((1, 2, 3)) is True

    def test_is_list_like_numpy_array(self):
        assert is_list_like(np.array([1, 2, 3])) is True

    def test_is_list_like_pandas_series(self):
        series = pd.Series([1, 2, 3])
        assert is_list_like(series) is True

    def test_is_list_like_string(self):
        # Strings are technically iterable but typically not considered "list-like"
        # This depends on the implementation
        assert is_list_like("string") is False

    def test_is_list_like_dict(self):
        # Dicts are iterable but not list-like
        assert is_list_like({"key": "value"}) is False

    def test_is_list_like_set(self):
        assert is_list_like({1, 2, 3}) is True

    def test_is_list_like_generator(self):
        gen = (x for x in [1, 2, 3])
        # Generators might or might not be considered list-like depending on implementation
        result = is_list_like(gen)
        assert isinstance(result, bool)  # Should return a boolean

    def test_is_list_like_scalar(self):
        assert is_list_like(42) is False
        assert is_list_like(3.14) is False
        assert is_list_like(None) is False


class TestUtilsEdgeCases:
    def test_named_list_with_none_names(self):
        item = Mock()
        item.name = None

        nl = NamedList([item])
        # Should convert None to string "None"
        assert nl["None"] is item

    def test_named_list_with_numeric_names(self):
        item = Mock()
        item.name = 123

        nl = NamedList([item])
        # Should convert number to string "123"
        assert nl["123"] is item

    def test_strip_numpy_types_empty_containers(self):
        assert strip_numpy_types([]) == []
        assert strip_numpy_types({}) == {}
        assert strip_numpy_types(()) == []  # Empty tuple becomes empty list

    def test_strip_numpy_types_deeply_nested(self):
        # Test very deep nesting
        deep_structure = [[[[[np.int32(42)]]]]]
        result = strip_numpy_types(deep_structure)
        assert result == [[[[[42]]]]]

    def test_split_point_columns_case_sensitivity(self):
        columns = ["Prefix_X", "Prefix_Y", "Prefix_Z"]

        # Case sensitive - should fail
        with pytest.raises(ValueError):
            split_point_columns("prefix", columns)

    def test_parse_color_edge_values(self):
        # Test edge values for RGB
        edge_cases = [
            (0.0, 0.0, 0.0),  # Pure black
            (1.0, 1.0, 1.0),  # Pure white
        ]

        for case in edge_cases:
            result = parse_color(case)
            assert isinstance(result, str)
            assert result.startswith("#")
            assert len(result) == 7
