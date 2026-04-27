"""Tests for AnnotationShaderBuilder, SkeletonShaderBuilder, and shader utilities."""

import pytest

from nglui.statebuilder.shaders import (
    AnnotationShaderBuilder,
    SkeletonShaderBuilder,
    _format_float,
    _normalize_color_str,
    parse_color_rgb,
    simple_point_shader,
)

# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestFormatFloat:
    def test_integer_value(self):
        assert _format_float(0.0) == "0.0"
        assert _format_float(1.0) == "1.0"
        assert _format_float(20.0) == "20.0"

    def test_fractional_value(self):
        assert _format_float(0.5) == "0.5"
        assert _format_float(0.0002) == "0.0002"
        assert _format_float(1.5) == "1.5"


class TestNormalizeColorStr:
    def test_string_passthrough(self):
        assert _normalize_color_str("cyan") == "cyan"
        assert _normalize_color_str("#ff00ff") == "#ff00ff"

    def test_0_to_1_tuple(self):
        result = _normalize_color_str((1.0, 0.0, 1.0))
        assert result == "#ff00ff"

    def test_0_to_255_tuple(self):
        result = _normalize_color_str((255, 0, 255))
        assert result == "#ff00ff"


class TestParseColorRgb:
    def test_named_color(self):
        rgb = parse_color_rgb("white")
        assert rgb == [1.0, 1.0, 1.0]

    def test_hex_color(self):
        rgb = parse_color_rgb("#ff0000")
        assert rgb[0] == pytest.approx(1.0)
        assert rgb[1] == pytest.approx(0.0)
        assert rgb[2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# simple_point_shader
# ---------------------------------------------------------------------------


class TestSimplePointShader:
    def test_defaults(self):
        code = simple_point_shader()
        assert 'color(default="tomato")' in code
        assert "slider(min=0.0, max=20.0, default=5.0)" in code
        assert "setPointMarkerSize(markerSize)" in code
        assert "setColor(markerColor)" in code

    def test_custom_color_and_size(self):
        code = simple_point_shader(color="cyan", markersize=10.0)
        assert 'color(default="cyan")' in code
        assert "default=10.0" in code


# ---------------------------------------------------------------------------
# AnnotationShaderBuilder
# ---------------------------------------------------------------------------


class TestAnnotationShaderBuilderBasic:
    def test_empty_build_produces_minimal_shader(self):
        code = AnnotationShaderBuilder().build()
        assert "void main() {" in code
        assert "setColor(defaultColor());" in code

    def test_build_returns_string(self):
        code = AnnotationShaderBuilder().build()
        assert isinstance(code, str)

    def test_str_equals_build(self):
        builder = AnnotationShaderBuilder().opacity()
        assert str(builder) == builder.build()

    def test_repr(self):
        builder = AnnotationShaderBuilder().categorical_color(
            prop="label", categories={0: ("a", "red"), 1: ("b", "blue")}
        )
        r = repr(builder)
        assert "label" in r
        assert "2" in r


class TestOpacity:
    def test_adds_slider_control(self):
        code = AnnotationShaderBuilder().opacity(default=0.8).build()
        assert "#uicontrol float opacity slider" in code
        assert "default=0.8" in code

    def test_adds_alpha_variable(self):
        code = AnnotationShaderBuilder().opacity().build()
        assert "float alpha = opacity;" in code

    def test_custom_name(self):
        code = AnnotationShaderBuilder().opacity(name="transp").build()
        assert "#uicontrol float transp slider" in code
        assert "float alpha = transp;" in code

    def test_custom_range(self):
        code = AnnotationShaderBuilder().opacity(min=0.2, max=0.9, default=0.5).build()
        assert "min=0.2" in code
        assert "max=0.9" in code
        assert "default=0.5" in code


class TestHighlight:
    def test_highlight_without_opacity_uses_1(self):
        code = AnnotationShaderBuilder().highlight(prop="selected", value=1).build()
        assert "prop_selected() == uint(1)" in code
        assert "? 1.0 : 1.0" in code

    def test_highlight_with_opacity(self):
        code = (
            AnnotationShaderBuilder()
            .opacity(default=0.3)
            .highlight(prop="selected", value=1)
            .build()
        )
        assert "? 1.0 : opacity" in code

    def test_custom_highlighted_alpha(self):
        code = (
            AnnotationShaderBuilder()
            .opacity()
            .highlight(prop="sel", value=2, highlighted_alpha=0.9)
            .build()
        )
        assert "? 0.9 : opacity" in code


class TestPointSize:
    def test_fixed_size(self):
        code = AnnotationShaderBuilder().point_size(size=8.0).build()
        assert "setPointMarkerSize(8.0);" in code

    def test_property_based_size(self):
        code = AnnotationShaderBuilder().point_size(prop="radius").build()
        assert "setPointMarkerSize(float(prop_radius()));" in code

    def test_property_with_scale(self):
        code = AnnotationShaderBuilder().point_size(prop="size", scale=0.0002).build()
        assert "setPointMarkerSize(float(prop_size())*0.0002);" in code

    def test_slider_mode(self):
        code = AnnotationShaderBuilder().point_size(size=5.0, slider=True).build()
        assert "#uicontrol float pointSize slider" in code
        assert "setPointMarkerSize(pointSize);" in code

    def test_slider_custom_name(self):
        code = (
            AnnotationShaderBuilder()
            .point_size(size=5.0, slider=True, slider_name="sz")
            .build()
        )
        assert "#uicontrol float sz slider" in code
        assert "setPointMarkerSize(sz);" in code


class TestBorderWidth:
    def test_border_width_line(self):
        code = AnnotationShaderBuilder().border_width(0.0).build()
        assert "setPointMarkerBorderWidth(0.0);" in code

    def test_nonzero_border(self):
        code = AnnotationShaderBuilder().border_width(2.5).build()
        assert "setPointMarkerBorderWidth(2.5);" in code


class TestCategoricalColor:
    def test_dict_input(self):
        code = (
            AnnotationShaderBuilder()
            .categorical_color(
                prop="type",
                categories={0: ("null", "white"), 1: ("soma", "cyan")},
            )
            .build()
        )
        assert '#uicontrol vec3 null color(default="white")' in code
        assert '#uicontrol vec3 soma color(default="cyan")' in code
        assert "#uicontrol bool show_null checkbox" in code
        assert "#uicontrol bool show_soma checkbox" in code

    def test_list_input(self):
        code = (
            AnnotationShaderBuilder()
            .categorical_color(
                prop="type",
                categories=[(0, "null", "white"), (1, "soma", "cyan")],
            )
            .build()
        )
        assert "#uicontrol vec3 null" in code
        assert "#uicontrol vec3 soma" in code

    def test_generates_if_else_block(self):
        code = (
            AnnotationShaderBuilder()
            .categorical_color(
                prop="type",
                categories={0: ("a", "red"), 1: ("b", "blue")},
            )
            .build()
        )
        assert "if (prop_type() == uint(0))" in code
        assert "} else if (prop_type() == uint(1))" in code

    def test_shared_label_merges_values(self):
        """Multiple values sharing a label produce a || condition."""
        code = (
            AnnotationShaderBuilder()
            .categorical_color(
                prop="type",
                categories={1: ("spine", "magenta"), 2: ("spine", "magenta")},
            )
            .build()
        )
        # Only one color control and one checkbox for 'spine'
        assert code.count("#uicontrol vec3 spine") == 1
        assert code.count("#uicontrol bool show_spine") == 1
        # Condition uses || to cover both values
        assert "prop_type() == uint(1) || prop_type() == uint(2)" in code

    def test_visibility_checkbox_default_true(self):
        code = (
            AnnotationShaderBuilder()
            .categorical_color(prop="t", categories={0: ("a", "red")})
            .build()
        )
        assert "checkbox(default=true)" in code

    def test_visibility_checkbox_default_false(self):
        code = (
            AnnotationShaderBuilder()
            .categorical_color(
                prop="t", categories={0: ("a", "red")}, default_visible=False
            )
            .build()
        )
        assert "checkbox(default=false)" in code

    def test_discard_on_hidden(self):
        code = (
            AnnotationShaderBuilder()
            .categorical_color(prop="t", categories={0: ("a", "red")})
            .build()
        )
        assert "if (!show_a) { discard; }" in code

    def test_setcolor_uses_alpha_when_present(self):
        code = (
            AnnotationShaderBuilder()
            .opacity()
            .categorical_color(prop="t", categories={0: ("a", "red")})
            .build()
        )
        assert "setColor(vec4(a, alpha));" in code

    def test_setcolor_uses_1_when_no_alpha(self):
        code = (
            AnnotationShaderBuilder()
            .categorical_color(prop="t", categories={0: ("a", "red")})
            .build()
        )
        assert "setColor(vec4(a, 1.0));" in code

    def test_raises_on_second_call(self):
        builder = AnnotationShaderBuilder().categorical_color(
            prop="t", categories={0: ("a", "red")}
        )
        with pytest.raises(ValueError, match="categorical_color"):
            builder.categorical_color(prop="u", categories={0: ("b", "blue")})

    def test_label_order_follows_first_appearance(self):
        code = (
            AnnotationShaderBuilder()
            .categorical_color(
                prop="t",
                categories={
                    0: ("null", "white"),
                    1: ("soma", "cyan"),
                    2: ("shaft", "yellow"),
                },
            )
            .build()
        )
        null_pos = code.index("vec3 null")
        soma_pos = code.index("vec3 soma")
        shaft_pos = code.index("vec3 shaft")
        assert null_pos < soma_pos < shaft_pos

    def test_control_order_colors_sliders_checkboxes(self):
        """Color controls appear before sliders, sliders before checkboxes."""
        code = (
            AnnotationShaderBuilder()
            .opacity()
            .categorical_color(prop="t", categories={0: ("a", "red")})
            .build()
        )
        color_pos = code.index("#uicontrol vec3")
        slider_pos = code.index("#uicontrol float")
        checkbox_pos = code.index("#uicontrol bool")
        assert color_pos < slider_pos < checkbox_pos


# ===========================================================================
# SkeletonShaderBuilder
# ===========================================================================


class TestSkeletonShaderBuilderBasic:
    def test_empty_build(self):
        code = SkeletonShaderBuilder().build()
        assert "void main() {" in code

    def test_repr(self):
        b = SkeletonShaderBuilder(["compartment"])
        assert "compartment" in repr(b)

    def test_str_equals_build(self):
        b = SkeletonShaderBuilder(["x"]).use_segment_color()
        assert str(b) == b.build()

    def test_constructor_assigns_vcustom_indices(self):
        code = (
            SkeletonShaderBuilder(["compartment", "distance"])
            .categorical_color(attr="compartment", categories={1.0: ("a", "red")})
            .build()
        )
        assert "vCustom1" in code

    def test_vertex_attribute_method(self):
        code = (
            SkeletonShaderBuilder()
            .vertex_attribute("compartment", index=1)
            .categorical_color(attr="compartment", categories={1.0: ("a", "red")})
            .build()
        )
        assert "float compartment = vCustom1;" in code

    def test_unknown_attribute_raises(self):
        with pytest.raises(ValueError, match="not declared"):
            SkeletonShaderBuilder().categorical_color(
                attr="compartment", categories={1.0: ("a", "red")}
            )


class TestSkeletonSegmentColor:
    def test_use_segment_color_emits_rgb(self):
        code = SkeletonShaderBuilder().use_segment_color().build()
        assert "emitRGB(segmentColor().rgb);" in code

    def test_desaturate_requires_segment_color(self):
        with pytest.raises(ValueError, match="use_segment_color"):
            SkeletonShaderBuilder(["c"]).desaturate(attr="c", reference_value=2.0)

    def test_desaturate_generates_hsl_branch(self):
        code = (
            SkeletonShaderBuilder(["compartment"])
            .use_segment_color()
            .desaturate(attr="compartment", reference_value=2.0)
            .build()
        )
        assert "rgbToHsl" in code
        assert "hslToRgb" in code
        assert "compartment == 2.0" in code
        assert "hsl.y *= 0.5;" in code

    def test_desaturate_custom_scale(self):
        code = (
            SkeletonShaderBuilder(["c"])
            .use_segment_color()
            .desaturate(attr="c", reference_value=1.0, saturation_scale=0.25)
            .build()
        )
        assert "hsl.y *= 0.25;" in code

    def test_desaturate_reproduces_compartment_shader(self):
        """Verify the canonical compartment desaturation pattern."""
        code = (
            SkeletonShaderBuilder(["compartment"])
            .use_segment_color()
            .desaturate(attr="compartment", reference_value=2.0)
            .build()
        )
        assert "float compartment = vCustom1;" in code
        assert "segmentColor()" in code
        assert "compartment == 2.0" in code
        assert "emitRGB(uColor.rgb);" in code
        assert "emitRGB(hslToRgb(hsl));" in code


class TestSkeletonCategoricalColor:
    def test_generates_color_controls(self):
        code = (
            SkeletonShaderBuilder(["c"])
            .categorical_color(
                attr="c", categories={1.0: ("axon", "white"), 2.0: ("dendrite", "cyan")}
            )
            .build()
        )
        assert '#uicontrol vec3 axon color(default="white")' in code
        assert '#uicontrol vec3 dendrite color(default="cyan")' in code

    def test_generates_checkbox_controls(self):
        code = (
            SkeletonShaderBuilder(["c"])
            .categorical_color(attr="c", categories={1.0: ("axon", "white")})
            .build()
        )
        assert "#uicontrol bool show_axon checkbox" in code

    def test_float_comparison_in_glsl(self):
        code = (
            SkeletonShaderBuilder(["c"])
            .categorical_color(attr="c", categories={1.0: ("axon", "white")})
            .build()
        )
        assert "c == 1.0" in code

    def test_emit_rgb_not_set_color(self):
        code = (
            SkeletonShaderBuilder(["c"])
            .categorical_color(attr="c", categories={1.0: ("axon", "white")})
            .build()
        )
        assert "emitRGB(axon);" in code
        assert "setColor" not in code

    def test_string_list_categories(self):
        builder = SkeletonShaderBuilder(["c"]).categorical_color(
            attr="c", categories=["axon", "dendrite", "soma"]
        )
        assert builder.label_map == {"axon": 0, "dendrite": 1, "soma": 2}
        code = builder.build()
        assert "c == 0.0" in code
        assert "c == 1.0" in code
        assert "c == 2.0" in code

    def test_dict_str_str_categories(self):
        builder = SkeletonShaderBuilder(["c"]).categorical_color(
            attr="c", categories={"axon": "white", "dendrite": "cyan"}
        )
        assert builder.label_map == {"axon": 0, "dendrite": 1}

    def test_explicit_numeric_keys_label_map_none(self):
        builder = SkeletonShaderBuilder(["c"]).categorical_color(
            attr="c", categories={1.0: ("axon", "white")}
        )
        assert builder.label_map is None

    def test_mutual_exclusion_with_continuous(self):
        with pytest.raises(ValueError):
            (
                SkeletonShaderBuilder(["c"])
                .categorical_color(attr="c", categories={1.0: ("a", "red")})
                .continuous_color(attr="c")
            )


class TestSkeletonContinuousColor:
    def test_cubehelix_default(self):
        code = SkeletonShaderBuilder(["d"]).continuous_color(attr="d").build()
        assert "colormapCubehelix(t)" in code

    def test_jet_colormap(self):
        code = (
            SkeletonShaderBuilder(["d"])
            .continuous_color(attr="d", colormap="jet")
            .build()
        )
        assert "colormapJet(t)" in code

    def test_range_sliders(self):
        code = (
            SkeletonShaderBuilder(["d"])
            .continuous_color(attr="d", range_min=0, range_max=500)
            .build()
        )
        assert "rangeMin" in code
        assert "rangeMax" in code
        assert "default=0.0" in code
        assert "default=500.0" in code

    def test_hardcoded_range(self):
        code = (
            SkeletonShaderBuilder(["d"])
            .continuous_color(attr="d", range_min=10, range_max=100, range_slider=False)
            .build()
        )
        assert "rangeMin" not in code
        assert "10.0" in code
        assert "100.0" in code

    def test_attr_declaration_in_main(self):
        code = (
            SkeletonShaderBuilder(["distance"])
            .continuous_color(attr="distance")
            .build()
        )
        assert "float distance = vCustom1;" in code

    def test_emit_rgb_output(self):
        code = SkeletonShaderBuilder(["d"]).continuous_color(attr="d").build()
        assert "emitRGB(colormapCubehelix(t));" in code

    def test_invalid_colormap_raises(self):
        with pytest.raises(ValueError, match="Unknown colormap"):
            SkeletonShaderBuilder(["d"]).continuous_color(attr="d", colormap="viridis")


class TestContinuousColor:
    def test_default_colormap_cubehelix(self):
        code = AnnotationShaderBuilder().continuous_color(prop="weight").build()
        assert "colormapCubehelix(t)" in code

    def test_jet_colormap(self):
        code = (
            AnnotationShaderBuilder()
            .continuous_color(prop="weight", colormap="jet")
            .build()
        )
        assert "colormapJet(t)" in code

    def test_invalid_colormap_raises(self):
        with pytest.raises(ValueError, match="Unknown colormap"):
            AnnotationShaderBuilder().continuous_color(prop="w", colormap="viridis")

    def test_range_sliders_added_by_default(self):
        code = (
            AnnotationShaderBuilder()
            .continuous_color(prop="weight", range_min=0.0, range_max=100.0)
            .build()
        )
        assert "#uicontrol float rangeMin slider" in code
        assert "#uicontrol float rangeMax slider" in code
        assert "default=0.0" in code
        assert "default=100.0" in code

    def test_range_slider_false_hardcodes_range(self):
        code = (
            AnnotationShaderBuilder()
            .continuous_color(
                prop="weight", range_min=10.0, range_max=50.0, range_slider=False
            )
            .build()
        )
        assert "rangeMin" not in code
        assert "10.0" in code
        assert "50.0" in code

    def test_normalized_t_in_shader(self):
        code = AnnotationShaderBuilder().continuous_color(prop="w").build()
        assert "float t = clamp(" in code
        assert "prop_w()" in code

    def test_uses_alpha_when_opacity_set(self):
        code = (
            AnnotationShaderBuilder()
            .opacity(default=0.8)
            .continuous_color(prop="w")
            .build()
        )
        assert "vec4(colormapCubehelix(t), alpha)" in code

    def test_uses_1_when_no_opacity(self):
        code = AnnotationShaderBuilder().continuous_color(prop="w").build()
        assert "vec4(colormapCubehelix(t), 1.0)" in code

    def test_mutual_exclusion_with_categorical(self):
        builder = AnnotationShaderBuilder().categorical_color(
            prop="t", categories={0: ("a", "red")}
        )
        with pytest.raises(ValueError, match="continuous_color"):
            builder.continuous_color(prop="w")

    def test_mutual_exclusion_other_direction(self):
        builder = AnnotationShaderBuilder().continuous_color(prop="w")
        with pytest.raises(ValueError, match="categorical_color"):
            builder.categorical_color(prop="t", categories={0: ("a", "red")})

    def test_combines_with_point_size_and_opacity(self):
        code = (
            AnnotationShaderBuilder()
            .point_size(prop="radius", scale=0.001)
            .opacity(default=0.9)
            .continuous_color(prop="weight", colormap="jet", range_min=0, range_max=1)
            .build()
        )
        assert "setPointMarkerSize(float(prop_radius())*0.001);" in code
        assert "float alpha = opacity;" in code
        assert "colormapJet(t)" in code


class TestStringCategories:
    """Tests for the string-label auto-mapping inputs."""

    def test_list_of_strings_generates_shader(self):
        code = (
            AnnotationShaderBuilder()
            .categorical_color(
                prop="cell_type",
                categories=["excitatory", "inhibitory"],
            )
            .build()
        )
        assert "#uicontrol vec3 excitatory color" in code
        assert "#uicontrol vec3 inhibitory color" in code
        assert "show_excitatory" in code
        assert "show_inhibitory" in code

    def test_list_of_strings_assigns_sequential_integers(self):
        code = (
            AnnotationShaderBuilder()
            .categorical_color(
                prop="cell_type",
                categories=["excitatory", "inhibitory", "unknown"],
            )
            .build()
        )
        assert "prop_cell_type() == uint(0)" in code
        assert "prop_cell_type() == uint(1)" in code
        assert "prop_cell_type() == uint(2)" in code

    def test_list_of_strings_populates_label_map(self):
        builder = AnnotationShaderBuilder().categorical_color(
            prop="cell_type",
            categories=["excitatory", "inhibitory", "unknown"],
        )
        assert builder.label_map == {"excitatory": 0, "inhibitory": 1, "unknown": 2}

    def test_list_of_strings_deduplicates(self):
        builder = AnnotationShaderBuilder().categorical_color(
            prop="t",
            categories=["a", "b", "a", "c"],
        )
        assert list(builder.label_map.keys()) == ["a", "b", "c"]
        assert builder.label_map == {"a": 0, "b": 1, "c": 2}

    def test_list_of_strings_auto_assigns_colors_from_palette(self):
        code = (
            AnnotationShaderBuilder()
            .categorical_color(
                prop="t",
                categories=["a", "b"],
            )
            .build()
        )
        # Should have hex colors from the default Bold_10 palette
        assert 'color(default="#' in code

    def test_list_of_strings_custom_palette(self):
        code = (
            AnnotationShaderBuilder()
            .categorical_color(
                prop="t",
                categories=["a", "b", "c"],
                palette="Set1",
            )
            .build()
        )
        assert "#uicontrol vec3 a color" in code

    def test_dict_str_str_generates_shader(self):
        code = (
            AnnotationShaderBuilder()
            .categorical_color(
                prop="cell_type",
                categories={"excitatory": "tomato", "inhibitory": "cyan"},
            )
            .build()
        )
        assert '#uicontrol vec3 excitatory color(default="tomato")' in code
        assert '#uicontrol vec3 inhibitory color(default="cyan")' in code

    def test_dict_str_str_assigns_sequential_integers(self):
        code = (
            AnnotationShaderBuilder()
            .categorical_color(
                prop="t",
                categories={"a": "red", "b": "blue"},
            )
            .build()
        )
        assert "prop_t() == uint(0)" in code
        assert "prop_t() == uint(1)" in code

    def test_dict_str_str_populates_label_map(self):
        builder = AnnotationShaderBuilder().categorical_color(
            prop="t",
            categories={"excitatory": "tomato", "inhibitory": "cyan"},
        )
        assert builder.label_map == {"excitatory": 0, "inhibitory": 1}

    def test_explicit_int_keys_label_map_is_none(self):
        builder = AnnotationShaderBuilder().categorical_color(
            prop="t",
            categories={0: ("a", "red"), 1: ("b", "blue")},
        )
        assert builder.label_map is None

    def test_explicit_int_tuples_label_map_is_none(self):
        builder = AnnotationShaderBuilder().categorical_color(
            prop="t",
            categories=[(0, "a", "red"), (1, "b", "blue")],
        )
        assert builder.label_map is None

    def test_arbitrary_iterable_accepted(self):
        """Non-list iterables (e.g. a generator or set-derived sequence) work."""
        labels = iter(["x", "y", "z"])
        builder = AnnotationShaderBuilder().categorical_color(
            prop="t", categories=labels
        )
        assert builder.label_map == {"x": 0, "y": 1, "z": 2}

    def test_label_map_usable_for_dataframe_encoding(self):
        """label_map keys match the categories and values are 0-based ints."""
        categories = ["soma", "axon", "dendrite"]
        builder = AnnotationShaderBuilder().categorical_color(
            prop="compartment", categories=categories
        )
        lm = builder.label_map
        # All labels present, all values are 0-based sequential integers
        assert set(lm.keys()) == set(categories)
        assert sorted(lm.values()) == list(range(len(categories)))


class TestFullExample:
    """Reproduce the synapse-type shader from the module docstring."""

    def setup_method(self):
        self.shader = (
            AnnotationShaderBuilder()
            .point_size(prop="size", scale=0.0002)
            .opacity(default=1.0)
            .highlight(prop="pre_in_selection", value=1)
            .categorical_color(
                prop="tag_detailed",
                categories={
                    0: ("null", "white"),
                    1: ("spine", "magenta"),
                    2: ("spine", "magenta"),
                    3: ("multi_spine", "purple"),
                    4: ("shaft", "yellow"),
                    5: ("soma", "cyan"),
                },
            )
            .border_width(0.0)
            .build()
        )

    def test_has_all_color_controls(self):
        for label in ("null", "spine", "multi_spine", "shaft", "soma"):
            assert f"#uicontrol vec3 {label} color" in self.shader

    def test_no_duplicate_spine_control(self):
        assert self.shader.count("#uicontrol vec3 spine") == 1

    def test_opacity_slider(self):
        assert "#uicontrol float opacity slider" in self.shader
        assert "default=1.0" in self.shader

    def test_all_checkboxes_present(self):
        for label in ("null", "spine", "multi_spine", "shaft", "soma"):
            assert f"show_{label}" in self.shader

    def test_point_size_from_property(self):
        assert "setPointMarkerSize(float(prop_size())*0.0002);" in self.shader

    def test_conditional_alpha(self):
        assert "prop_pre_in_selection() == uint(1)" in self.shader
        assert "? 1.0 : opacity" in self.shader

    def test_spine_condition_covers_both_values(self):
        assert (
            "prop_tag_detailed() == uint(1) || prop_tag_detailed() == uint(2)"
            in self.shader
        )

    def test_border_width(self):
        assert "setPointMarkerBorderWidth(0.0);" in self.shader

    def test_shader_structure_order(self):
        """Controls come before main(), main() contains all body lines."""
        uicontrol_end = self.shader.rindex("#uicontrol")
        main_start = self.shader.index("void main()")
        assert uicontrol_end < main_start
