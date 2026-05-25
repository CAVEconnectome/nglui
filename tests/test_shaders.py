"""Tests for AnnotationShaderBuilder, SkeletonShaderBuilder, and shader utilities."""

import pytest

from nglui.statebuilder.shaders import (
    AnnotationShaderBuilder,
    SkeletonShaderBuilder,
    _format_float,
    _normalize_color_str,
    auto_annotation_shader,
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

    def test_no_scientific_notation_small(self):
        """GLSL ES 1.0 rejects scientific-notation literals — repr(1e-7)
        returns '1e-07' but _format_float must give a fixed-point form."""
        s = _format_float(1e-7)
        assert "e" not in s.lower()
        assert float(s) == 1e-7

    def test_no_scientific_notation_large(self):
        s = _format_float(1e20)
        assert "e" not in s.lower()
        assert float(s) == 1e20

    def test_no_scientific_notation_negative(self):
        s = _format_float(-1e-9)
        assert "e" not in s.lower()
        assert float(s) == -1e-9

    def test_non_finite_raises(self):
        with pytest.raises(ValueError):
            _format_float(float("nan"))
        with pytest.raises(ValueError):
            _format_float(float("inf"))


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
        """An unconfigured annotation builder must produce a valid default
        shader that uses the layer's defaultColor()."""
        code = AnnotationShaderBuilder().build()
        assert "void main() {" in code
        assert "setColor(defaultColor());" in code

    def test_empty_build_has_no_leading_blank_line(self):
        """When no UI controls are declared, no separator blank line should
        precede void main()."""
        code = AnnotationShaderBuilder().build()
        assert not code.startswith("\n")
        assert code.startswith("void main() {")

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

    def test_alpha_applied_to_default_color_fallthrough(self):
        """Regression: opacity must apply to the fallthrough setColor too,
        otherwise points whose property doesn't match any categorical branch
        are drawn full-alpha."""
        code = (
            AnnotationShaderBuilder()
            .opacity(default=0.3)
            .categorical_color(prop="x", categories={0: ("a", "red")})
            .build()
        )
        assert "setColor(vec4(defaultColor().rgb, alpha));" in code
        # Alpha must be declared before the first setColor that uses it.
        alpha_idx = code.index("float alpha = opacity;")
        first_use_idx = code.index("vec4(defaultColor().rgb, alpha)")
        assert alpha_idx < first_use_idx

    def test_no_alpha_keeps_plain_default_color(self):
        """When no opacity/highlight is set, the cheap defaultColor() form
        should still be used."""
        code = AnnotationShaderBuilder().build()
        assert "setColor(defaultColor());" in code
        assert "vec4(defaultColor()" not in code


class TestNameCollision:
    """Across-kind collision detection — NG puts every #uicontrol name into
    one namespace, so e.g. an opacity slider named 'rangeMin' must conflict
    with the rangeMin slider that continuous_color creates."""

    def test_opacity_called_twice_raises(self):
        with pytest.raises(ValueError, match="collides"):
            AnnotationShaderBuilder().opacity().opacity()

    def test_opacity_named_rangeMin_then_continuous_color_raises(self):
        with pytest.raises(ValueError, match="collides"):
            (
                AnnotationShaderBuilder()
                .opacity(name="rangeMin")
                .continuous_color(prop="v")
            )

    def test_label_sanitisation_collision_raises(self):
        """Labels 'a-b' and 'a_b' both sanitize to 'a_b'."""
        with pytest.raises(ValueError, match="duplicate control names"):
            AnnotationShaderBuilder().categorical_color(
                prop="x", categories={0: ("a-b", "red"), 1: ("a_b", "blue")}
            )

    def test_failed_categorical_leaves_builder_usable(self):
        """A failed categorical_color call must not partially mutate the
        builder — the user should be able to retry."""
        b = AnnotationShaderBuilder().opacity()
        with pytest.raises(ValueError):
            b.categorical_color(
                prop="x", categories={0: ("a-b", "red"), 1: ("a_b", "blue")}
            )
        # Retry with non-colliding labels — must succeed.
        b.categorical_color(
            prop="x", categories={0: ("foo", "red"), 1: ("bar", "blue")}
        )
        code = b.build()
        assert "#uicontrol vec3 foo" in code
        assert "#uicontrol vec3 bar" in code

    def test_skeleton_label_sanitisation_collision_raises(self):
        with pytest.raises(ValueError, match="duplicate control names"):
            SkeletonShaderBuilder(["c"]).categorical_color(
                attr="c",
                categories={0.0: ("a-b", "red"), 1.0: ("a_b", "blue")},
            )


class TestInputValidation:
    """Bug-fix regression tests for #7-#11."""

    def test_categorical_dict_value_must_be_two_tuple(self):
        with pytest.raises(ValueError, match="2-tuple"):
            AnnotationShaderBuilder().categorical_color(
                prop="x", categories={0: ("a", "red", "extra")}
            )

    def test_categorical_dict_value_not_a_tuple(self):
        with pytest.raises(ValueError, match="2-tuple"):
            AnnotationShaderBuilder().categorical_color(
                prop="x",
                categories={0: "red"},  # missing label
            )

    def test_categorical_list_entry_must_be_three_tuple(self):
        with pytest.raises(ValueError, match="3-tuple"):
            AnnotationShaderBuilder().categorical_color(
                prop="x", categories=[(0, "label_only")]
            )

    def test_categorical_negative_value_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            AnnotationShaderBuilder().categorical_color(
                prop="x", categories={-1: ("a", "red")}
            )

    def test_highlight_negative_value_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            AnnotationShaderBuilder().highlight(prop="sel", value=-1)

    def test_continuous_color_zero_width_range_raises(self):
        with pytest.raises(ValueError, match="range_min != range_max"):
            AnnotationShaderBuilder().continuous_color(
                prop="x", range_min=5.0, range_max=5.0
            )

    def test_skeleton_continuous_color_zero_width_range_raises(self):
        with pytest.raises(ValueError, match="range_min != range_max"):
            SkeletonShaderBuilder(["d"]).continuous_color(
                attr="d", range_min=0.0, range_max=0.0
            )

    def test_skeleton_categorical_dict_value_must_be_two_tuple(self):
        with pytest.raises(ValueError, match="2-tuple"):
            SkeletonShaderBuilder(["c"]).categorical_color(
                attr="c", categories={1.0: ("a", "red", "extra")}
            )

    def test_skeleton_vertex_attribute_index_zero_raises(self):
        with pytest.raises(ValueError, match=">= 1"):
            SkeletonShaderBuilder().vertex_attribute("x", 0)

    def test_skeleton_vertex_attribute_negative_index_raises(self):
        with pytest.raises(ValueError, match=">= 1"):
            SkeletonShaderBuilder().vertex_attribute("x", -1)

    def test_skeleton_categorical_negative_value_allowed(self):
        """Skeleton attrs are floats; negative values are valid (unlike for
        annotations which compare against uint)."""
        code = (
            SkeletonShaderBuilder(["c"])
            .categorical_color(attr="c", categories={-1.0: ("a", "red")})
            .build()
        )
        assert "c == -1.0" in code

    def test_continuous_color_call_once(self):
        with pytest.raises(ValueError, match="only be called once"):
            (
                AnnotationShaderBuilder()
                .continuous_color(prop="a")
                .continuous_color(prop="b")
            )

    def test_continuous_color_call_once_even_without_sliders(self):
        """range_slider=False registers no sliders, so this case slips past
        the slider name-collision check — must be guarded explicitly."""
        with pytest.raises(ValueError, match="only be called once"):
            (
                AnnotationShaderBuilder()
                .continuous_color(prop="a", range_slider=False)
                .continuous_color(prop="b", range_slider=False)
            )

    def test_skeleton_continuous_color_call_once(self):
        with pytest.raises(ValueError, match="only be called once"):
            (
                SkeletonShaderBuilder(["a", "b"])
                .continuous_color(attr="a")
                .continuous_color(attr="b")
            )

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

    def test_property_with_scale_slider(self):
        """slider=True alongside prop= emits a multiplier slider."""
        code = AnnotationShaderBuilder().point_size(prop="size", slider=True).build()
        # Default slider name in scale mode is 'pointScale'.
        assert "#uicontrol float pointScale slider" in code
        # default=scale (1.0), range=[slider_min, slider_max] (0..20).
        assert "min=0.0, max=20.0, default=1.0" in code
        assert "setPointMarkerSize(float(prop_size())*pointScale);" in code

    def test_property_with_scale_slider_custom(self):
        code = (
            AnnotationShaderBuilder()
            .point_size(
                prop="vol",
                scale=0.0002,
                slider=True,
                slider_min=0.0,
                slider_max=0.001,
                slider_name="volScale",
            )
            .build()
        )
        assert (
            "#uicontrol float volScale slider(min=0.0, max=0.001, default=0.0002)"
            in code
        )
        assert "setPointMarkerSize(float(prop_vol())*volScale);" in code

    def test_slider_default_must_be_within_range(self):
        """A slider's default must satisfy min <= default <= max."""
        b = AnnotationShaderBuilder()
        with pytest.raises(ValueError, match="default .* outside"):
            b.opacity(default=2.0, min=0.0, max=1.0)
        with pytest.raises(ValueError, match="default .* outside"):
            AnnotationShaderBuilder().point_size(
                size=50.0, slider=True, slider_min=0.0, slider_max=20.0
            )
        with pytest.raises(ValueError, match="default .* outside"):
            AnnotationShaderBuilder().point_size(
                prop="vol", scale=1.0, slider=True, slider_min=0.0, slider_max=0.001
            )

    def test_slider_min_cannot_exceed_max(self):
        with pytest.raises(ValueError, match="cannot exceed max"):
            AnnotationShaderBuilder().opacity(default=0.5, min=1.0, max=0.5)

    def test_calling_point_size_twice_replaces_slider(self):
        """Second call shouldn't accumulate stale slider entries."""
        code = (
            AnnotationShaderBuilder()
            .point_size(prop="size", slider=True)
            .point_size(prop="size", slider=True, slider_max=100.0)
            .build()
        )
        # Only one pointScale slider line should be present.
        assert code.count("#uicontrol float pointScale slider") == 1
        assert "max=100.0" in code

    def test_invlerp_mode_emits_uicontrol_invlerp(self):
        code = (
            AnnotationShaderBuilder()
            .point_size(
                prop="vol",
                slider=True,
                slider_mode="invlerp",
                slider_min=2.0,
                slider_max=30.0,
                invlerp_range=(0.0, 1000.0),
            )
            .build()
        )
        # One invlerp uicontrol, no plain slider for pointScale.
        assert (
            '#uicontrol invlerp pointScale(property="vol", '
            "range=[0.0, 1000.0], clamp=true)"
        ) in code
        assert "#uicontrol float pointScale slider" not in code
        # GLSL maps invlerp output [0,1] to [slider_min, slider_max].
        assert "setPointMarkerSize(2.0 + (30.0 - 2.0) * pointScale());" in code

    def test_invlerp_mode_zero_min_simplifies(self):
        code = (
            AnnotationShaderBuilder()
            .point_size(
                prop="vol",
                slider=True,
                slider_mode="invlerp",
                slider_min=0.0,
                slider_max=20.0,
            )
            .build()
        )
        # When min=0 the GLSL collapses to a simple multiply.
        assert "setPointMarkerSize(pointScale()*20.0);" in code

    def test_invlerp_mode_clamp_false(self):
        code = (
            AnnotationShaderBuilder()
            .point_size(
                prop="vol",
                slider=True,
                slider_mode="invlerp",
                invlerp_clamp=False,
            )
            .build()
        )
        assert "clamp=false" in code

    def test_invlerp_mode_requires_slider_and_prop(self):
        with pytest.raises(ValueError, match="invlerp"):
            AnnotationShaderBuilder().point_size(slider_mode="invlerp")
        with pytest.raises(ValueError, match="invlerp"):
            AnnotationShaderBuilder().point_size(
                prop="x", slider=False, slider_mode="invlerp"
            )

    def test_invlerp_mode_unknown_value_raises(self):
        with pytest.raises(ValueError, match="must be 'linear' or 'invlerp'"):
            AnnotationShaderBuilder().point_size(slider_mode="log")

    def test_from_info_invlerp_mode(self):
        info = {
            "@type": "neuroglancer_annotations_v1",
            "annotation_type": "point",
            "properties": [{"id": "size", "type": "float32"}],
        }
        shader = auto_annotation_shader(
            info,
            size_slider_mode="invlerp",
            size_slider_min=1.0,
            size_slider_max=30.0,
            size_invlerp_range=(0.0, 1e6),
        )
        assert (
            '#uicontrol invlerp pointScale(property="size", '
            "range=[0.0, 1000000.0], clamp=true)"
        ) in shader
        assert "1.0 + (30.0 - 1.0) * pointScale()" in shader

    def test_from_info_invlerp_replacing_slider_after_double_call(self):
        """Switching modes via a second call cleans up the prior control."""
        b = AnnotationShaderBuilder()
        b.point_size(prop="vol", slider=True)  # linear
        b.point_size(prop="vol", slider=True, slider_mode="invlerp")  # invlerp
        code = b.build()
        # Only the invlerp control survives.
        assert "#uicontrol invlerp pointScale" in code
        assert "#uicontrol float pointScale slider" not in code


class TestBorderWidth:
    def test_border_width_line(self):
        code = AnnotationShaderBuilder().border_width(0.0).build()
        assert "setPointMarkerBorderWidth(0.0);" in code

    def test_nonzero_border(self):
        code = AnnotationShaderBuilder().border_width(2.5).build()
        assert "setPointMarkerBorderWidth(2.5);" in code


class TestCategoricalColor:
    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            AnnotationShaderBuilder().categorical_color(prop="x", categories={})

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            AnnotationShaderBuilder().categorical_color(prop="x", categories=[])

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

    def test_with_show_checkboxes_false_omits_checkbox_controls(self):
        code = (
            AnnotationShaderBuilder()
            .categorical_color(
                prop="t",
                categories={0: ("a", "red"), 1: ("b", "blue")},
                with_show_checkboxes=False,
            )
            .build()
        )
        assert "#uicontrol bool show_" not in code
        assert "discard" not in code
        # Colors and dispatch still emitted.
        assert "#uicontrol vec3 a color" in code
        assert "#uicontrol vec3 b color" in code
        assert "setColor(vec4(a, 1.0));" in code

    def test_uppercase_label_sanitized_for_color_preserved_for_show(self):
        """categorical_color sanitizes labels per-purpose."""
        code = (
            AnnotationShaderBuilder()
            .categorical_color(prop="t", categories={0: ("PV", "red")})
            .build()
        )
        # color id is lowercased-first, show id preserves casing.
        assert "#uicontrol vec3 pV color" in code
        assert "#uicontrol bool show_PV checkbox" in code
        assert "if (!show_PV) { discard; }" in code
        assert "setColor(vec4(pV, 1.0));" in code

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
    def test_empty_build_emits_segment_color_default(self):
        """An unconfigured skeleton builder must still produce a valid shader
        that emits a colour, otherwise Neuroglancer renders nothing."""
        code = SkeletonShaderBuilder().build()
        assert "void main() {" in code
        assert "emitRGB(segmentColor().rgb);" in code

    def test_empty_build_with_attrs_still_defaults_to_segment_color(self):
        """Declaring vertex attributes alone is not a colour rule — the
        fallback default still applies."""
        code = SkeletonShaderBuilder(["compartment"]).build()
        assert "emitRGB(segmentColor().rgb);" in code
        # No attribute decls should be emitted since nothing references them.
        assert "vCustom" not in code

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
    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            SkeletonShaderBuilder(["c"]).categorical_color(attr="c", categories={})

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            SkeletonShaderBuilder(["c"]).categorical_color(attr="c", categories=[])

    def test_categorical_emits_segment_color_fallback(self):
        """Regression: vertices that don't match any category must still
        receive a defined colour. Without the base emit, GLSL output is
        undefined."""
        code = (
            SkeletonShaderBuilder(["c"])
            .categorical_color(attr="c", categories={1.0: ("a", "red")})
            .build()
        )
        # Base emit must come before the if/else chain.
        base_idx = code.index("emitRGB(segmentColor().rgb);")
        cat_idx = code.index("if (c == 1.0)")
        assert base_idx < cat_idx

    def test_continuous_emits_segment_color_fallback(self):
        """Continuous colormaps clamp to [0,1] so all vertices receive a
        colour, but the base emit before the colormap is still required so
        output is defined when range_min == range_max etc."""
        code = SkeletonShaderBuilder(["d"]).continuous_color(attr="d").build()
        assert "emitRGB(segmentColor().rgb);" in code

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

    def test_with_show_checkboxes_false_omits_checkbox_controls(self):
        code = (
            SkeletonShaderBuilder(["c"])
            .categorical_color(
                attr="c",
                categories={1.0: ("axon", "white"), 2.0: ("dendrite", "cyan")},
                with_show_checkboxes=False,
            )
            .build()
        )
        assert "#uicontrol bool show_" not in code
        assert "discard" not in code
        # Colors and dispatch still emitted.
        assert "#uicontrol vec3 axon color" in code
        assert "#uicontrol vec3 dendrite color" in code
        assert "emitRGB(axon);" in code

    def test_uppercase_label_sanitized_for_color_preserved_for_show(self):
        code = (
            SkeletonShaderBuilder(["c"])
            .categorical_color(attr="c", categories={1.0: ("PV", "red")})
            .build()
        )
        # color id is lowercased-first, show id preserves casing.
        assert "#uicontrol vec3 pV color" in code
        assert "#uicontrol bool show_PV checkbox" in code
        assert "if (!show_PV) { discard; }" in code
        assert "emitRGB(pV);" in code

    def test_special_chars_sanitized(self):
        code = (
            SkeletonShaderBuilder(["c"])
            .categorical_color(attr="c", categories={1.0: ("L2/3-int", "red")})
            .build()
        )
        # color id: lowercase-first + non-alnum→_
        assert "#uicontrol vec3 l2_3_int color" in code
        # show id: original casing + non-alnum→_
        assert "#uicontrol bool show_L2_3_int checkbox" in code


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


# ---------------------------------------------------------------------------
# AnnotationShaderBuilder.from_info / auto_annotation_shader
# ---------------------------------------------------------------------------


class TestFromInfo:
    """Auto-configure a shader from a Neuroglancer annotation `info` dict."""

    def _info(self, properties):
        return {
            "@type": "neuroglancer_annotations_v1",
            "annotation_type": "point",
            "properties": properties,
        }

    def test_categorical_uint_with_enum(self):
        info = self._info(
            [
                {
                    "id": "cell_type",
                    "type": "uint8",
                    "enum_values": [0, 1, 2],
                    "enum_labels": ["exc", "inh", "unknown"],
                }
            ]
        )
        shader = auto_annotation_shader(info)
        for label in ("exc", "inh", "unknown"):
            assert f"#uicontrol vec3 {label} color" in shader
            assert f"show_{label}" in shader
        assert "prop_cell_type() == uint(0)" in shader
        assert "#uicontrol float opacity slider" in shader

    def test_continuous_float_only(self):
        info = self._info([{"id": "score", "type": "float32"}])
        shader = auto_annotation_shader(info)
        assert "colormapCubehelix" in shader
        assert "prop_score()" in shader
        assert "#uicontrol float rangeMin slider" in shader
        assert "#uicontrol float rangeMax slider" in shader

    def test_categorical_wins_over_float(self):
        info = self._info(
            [
                {"id": "score", "type": "float32"},
                {
                    "id": "cell_type",
                    "type": "uint8",
                    "enum_values": [0, 1],
                    "enum_labels": ["a", "b"],
                },
            ]
        )
        shader = auto_annotation_shader(info)
        assert "colormapCubehelix" not in shader
        assert "#uicontrol vec3 a color" in shader
        assert "#uicontrol vec3 b color" in shader

    def test_size_property_picked_by_name_default_invlerp(self):
        info = self._info(
            [
                {"id": "score", "type": "float32"},
                {"id": "radius", "type": "float32"},
            ]
        )
        shader = auto_annotation_shader(info)
        # Default size_slider_mode is invlerp.
        assert '#uicontrol invlerp pointScale(property="radius"' in shader
        assert "pointScale()" in shader

    def test_size_property_picked_by_name_linear_mode(self):
        info = self._info(
            [
                {"id": "score", "type": "float32"},
                {"id": "radius", "type": "float32"},
            ]
        )
        shader = auto_annotation_shader(info, size_slider_mode="linear")
        assert "setPointMarkerSize(float(prop_radius())*pointScale);" in shader
        assert "#uicontrol float pointScale slider" in shader

    def test_explicit_size_prop_linear(self):
        info = self._info(
            [
                {"id": "score", "type": "float32"},
                {"id": "weight", "type": "float32"},
            ]
        )
        shader = auto_annotation_shader(
            info, size_prop="weight", size_slider_mode="linear"
        )
        assert "setPointMarkerSize(float(prop_weight())*pointScale);" in shader

    def test_size_scale_slider_defaults_linear(self):
        info = self._info([{"id": "size", "type": "float32"}])
        shader = auto_annotation_shader(info, size_slider_mode="linear")
        # Linear default slider: range [0, 20], default 1.0
        assert (
            "#uicontrol float pointScale slider(min=0.0, max=20.0, default=1.0)"
            in shader
        )

    def test_size_slider_narrow_range_auto_picks_midpoint_linear(self):
        """When size_scale isn't passed and 1.0 is outside the slider range,
        the default auto-picks the midpoint (linear mode)."""
        info = self._info([{"id": "size", "type": "float32"}])
        shader = auto_annotation_shader(
            info, size_slider_mode="linear", size_slider_max=0.001
        )
        # Midpoint of [0.0, 0.001] = 0.0005
        assert (
            "#uicontrol float pointScale slider(min=0.0, max=0.001, default=0.0005)"
            in shader
        )

    def test_explicit_size_scale_outside_range_still_raises_linear(self):
        """An explicitly-passed size_scale must satisfy the slider range."""
        info = self._info([{"id": "size", "type": "float32"}])
        with pytest.raises(ValueError, match="default .* outside"):
            auto_annotation_shader(
                info,
                size_slider_mode="linear",
                size_scale=5.0,
                size_slider_max=1.0,
            )

    def test_size_scale_slider_custom_range_linear(self):
        info = self._info([{"id": "size", "type": "float32"}])
        shader = auto_annotation_shader(
            info,
            size_slider_mode="linear",
            size_scale=0.0002,
            size_slider_min=0.0,
            size_slider_max=0.001,
        )
        assert (
            "#uicontrol float pointScale slider(min=0.0, max=0.001, default=0.0002)"
            in shader
        )
        assert "setPointMarkerSize(float(prop_size())*pointScale);" in shader

    def test_explicit_color_prop(self):
        info = self._info(
            [
                {
                    "id": "primary",
                    "type": "uint8",
                    "enum_values": [0, 1],
                    "enum_labels": ["a", "b"],
                },
                {
                    "id": "secondary",
                    "type": "uint8",
                    "enum_values": [0, 1],
                    "enum_labels": ["x", "y"],
                },
            ]
        )
        shader = auto_annotation_shader(info, color_prop="secondary")
        assert "#uicontrol vec3 x color" in shader
        assert "#uicontrol vec3 y color" in shader
        assert "#uicontrol vec3 a color" not in shader

    def test_unknown_color_prop_raises(self):
        info = self._info([{"id": "score", "type": "float32"}])
        with pytest.raises(ValueError, match="not found"):
            AnnotationShaderBuilder.from_info(info, color_prop="missing")

    def test_unknown_size_prop_raises(self):
        info = self._info([{"id": "score", "type": "float32"}])
        with pytest.raises(ValueError, match="not found"):
            AnnotationShaderBuilder.from_info(info, size_prop="missing")

    def test_rgb_property_is_skipped_with_warning(self):
        info = self._info(
            [
                {"id": "marker_color", "type": "rgb"},
                {"id": "score", "type": "float32"},
            ]
        )
        with pytest.warns(UserWarning, match="rgb/rgba"):
            shader = auto_annotation_shader(info)
        # Float still drives color; rgb prop produces no GLSL.
        assert "colormapCubehelix" in shader
        assert "marker_color" not in shader

    def test_uncategorized_integer_warns(self):
        info = self._info([{"id": "raw_id", "type": "uint32"}])
        with pytest.warns(UserWarning, match="enum_values"):
            shader = auto_annotation_shader(info)
        # No color encoding possible; only opacity slider remains.
        assert "#uicontrol float opacity slider" in shader
        assert "prop_raw_id" not in shader

    def test_empty_properties(self):
        shader = auto_annotation_shader(self._info([]))
        assert "#uicontrol float opacity slider" in shader
        assert "void main()" in shader

    def test_enum_length_mismatch_raises(self):
        info = self._info(
            [
                {
                    "id": "cell_type",
                    "type": "uint8",
                    "enum_values": [0, 1, 2],
                    "enum_labels": ["a", "b"],
                }
            ]
        )
        with pytest.raises(ValueError, match="mismatched"):
            AnnotationShaderBuilder.from_info(info)

    def test_enum_labels_with_special_chars_are_sanitized(self):
        # GLSL identifiers can't have spaces or hyphens
        info = self._info(
            [
                {
                    "id": "cell_type",
                    "type": "uint8",
                    "enum_values": [0, 1],
                    "enum_labels": ["pyramidal cell", "L2/3-int"],
                }
            ]
        )
        shader = auto_annotation_shader(info)
        # Sanitized labels are valid Neuroglancer #uicontrol names: special
        # chars become underscores and the first letter is lowercased.
        assert "pyramidal_cell" in shader
        assert "l2_3_int" in shader

    def test_show_checkboxes_preserve_original_casing(self):
        """show_<X> uses original label casing; only color names are lowercased."""
        info = self._info(
            [
                {
                    "id": "cell_type",
                    "type": "uint32",
                    "enum_values": [0, 1, 2],
                    "enum_labels": ["AltBasket", "ChC", "PV"],
                }
            ]
        )
        shader = auto_annotation_shader(info, color_prop="cell_type")
        # Color controls use the lowercased-first form …
        assert "#uicontrol vec3 altBasket color" in shader
        assert "#uicontrol vec3 chC color" in shader
        assert "#uicontrol vec3 pV color" in shader
        # … while show_<X> preserves the original casing.
        assert "#uicontrol bool show_AltBasket checkbox" in shader
        assert "#uicontrol bool show_ChC checkbox" in shader
        assert "#uicontrol bool show_PV checkbox" in shader
        # And the GLSL body references both names correctly.
        assert "if (!show_AltBasket) { discard; }" in shader
        assert "setColor(vec4(altBasket," in shader

    def test_show_checkboxes_disabled_omits_controls_and_discard(self):
        info = self._info(
            [
                {
                    "id": "ct",
                    "type": "uint8",
                    "enum_values": [0, 1],
                    "enum_labels": ["a", "b"],
                }
            ]
        )
        shader = auto_annotation_shader(info, show_checkboxes=False)
        assert "#uicontrol bool show_" not in shader
        assert "discard" not in shader
        # Color setting still works.
        assert "setColor(vec4(a," in shader
        assert "setColor(vec4(b," in shader

    def test_show_checkboxes_special_chars_sanitized(self):
        """Non-alphanumeric chars become underscores; casing preserved."""
        info = self._info(
            [
                {
                    "id": "ct",
                    "type": "uint8",
                    "enum_values": [0],
                    "enum_labels": ["L2/3-int"],
                }
            ]
        )
        shader = auto_annotation_shader(info)
        # show_<X> preserves capital L
        assert "show_L2_3_int" in shader
        # color id uses lowercase-first
        assert "#uicontrol vec3 l2_3_int" in shader

    def test_uppercase_labels_lowercased_for_ng_uicontrol_regex(self):
        """Regression test: NG's #uicontrol parser requires names to match
        [a-z][a-zA-Z0-9_]* — labels with uppercase first letters used to
        produce shader source NG rejected as invalid syntax."""
        info = self._info(
            [
                {
                    "id": "cell_type",
                    "type": "uint32",
                    "enum_values": list(range(4)),
                    "enum_labels": ["AltBasket", "ChC", "L1", "PV"],
                }
            ]
        )
        shader = auto_annotation_shader(info, color_prop="cell_type")
        # Validate every #uicontrol line against the exact regex NG uses.
        import re as _re

        ng_inner = _re.compile(
            r"^([_a-zA-Z][_a-zA-Z0-9]*)[ \t]+([a-z][a-zA-Z0-9_]*)(?:[ \t]+([a-z]+))?"
        )
        for line in shader.splitlines():
            if line.startswith("#uicontrol"):
                inner = line[len("#uicontrol") :].strip()
                assert ng_inner.match(inner), f"NG would reject: {line!r}"
        # Spot-check the lowercased forms
        assert "altBasket" in shader
        assert "chC" in shader
        assert "l1" in shader
        assert "pV" in shader

    def test_palette_dict_full_override(self):
        info = self._info(
            [
                {
                    "id": "ct",
                    "type": "uint8",
                    "enum_values": [0, 1, 2],
                    "enum_labels": ["AltBasket", "ChC", "PV"],
                }
            ]
        )
        shader = auto_annotation_shader(
            info,
            palette={"AltBasket": "red", "ChC": "#00ff00", "PV": "orange"},
        )
        # Dict keys match the ORIGINAL labels, not the sanitized identifiers.
        assert '#uicontrol vec3 altBasket color(default="red")' in shader
        assert '#uicontrol vec3 chC color(default="#00ff00")' in shader
        assert '#uicontrol vec3 pV color(default="orange")' in shader

    def test_palette_dict_partial_uses_default_color(self):
        info = self._info(
            [
                {
                    "id": "ct",
                    "type": "uint8",
                    "enum_values": [0, 1, 2],
                    "enum_labels": ["a", "b", "c"],
                }
            ]
        )
        shader = auto_annotation_shader(info, palette={"b": "red"})
        # Default fallback is medium gray (#808080)
        assert '#uicontrol vec3 a color(default="#808080")' in shader
        assert '#uicontrol vec3 b color(default="red")' in shader
        assert '#uicontrol vec3 c color(default="#808080")' in shader

    def test_palette_dict_with_custom_default_color(self):
        info = self._info(
            [
                {
                    "id": "ct",
                    "type": "uint8",
                    "enum_values": [0, 1],
                    "enum_labels": ["a", "b"],
                }
            ]
        )
        shader = auto_annotation_shader(
            info, palette={"a": "red"}, default_color="#444444"
        )
        assert '#uicontrol vec3 a color(default="red")' in shader
        assert '#uicontrol vec3 b color(default="#444444")' in shader

    def test_palette_string_ignores_default_color(self):
        """default_color is only consulted for dict palettes."""
        info = self._info(
            [
                {
                    "id": "ct",
                    "type": "uint8",
                    "enum_values": [0],
                    "enum_labels": ["only"],
                }
            ]
        )
        shader = auto_annotation_shader(info, default_color="#abcdef")
        # Color comes from the palette (Tableau_10), not from default_color.
        assert "#abcdef" not in shader.lower()

    def test_palette_dict_extra_keys_ignored(self):
        info = self._info(
            [
                {
                    "id": "ct",
                    "type": "uint8",
                    "enum_values": [0, 1],
                    "enum_labels": ["a", "b"],
                }
            ]
        )
        # 'unused' is not in enum_labels — silently ignored.
        shader = auto_annotation_shader(
            info, palette={"a": "red", "b": "blue", "unused": "magenta"}
        )
        assert "magenta" not in shader

    def test_leading_digit_label_is_prefixed(self):
        """A label that starts with something not in [a-z] gets a 'c_' prefix."""
        info = self._info(
            [
                {
                    "id": "stage",
                    "type": "uint8",
                    "enum_values": [0, 1],
                    "enum_labels": ["1st-pass", "2nd-pass"],
                }
            ]
        )
        shader = auto_annotation_shader(info)
        assert "c_1st_pass" in shader
        assert "c_2nd_pass" in shader

    def test_returns_unbuilt_builder(self):
        """from_info returns a builder so the caller can override."""
        info = self._info([{"id": "score", "type": "float32"}])
        builder = AnnotationShaderBuilder.from_info(info)
        assert isinstance(builder, AnnotationShaderBuilder)
        # Caller can chain more methods:
        builder.border_width(2.0)
        shader = builder.build()
        assert "setPointMarkerBorderWidth(2.0)" in shader

    def test_build_to_clipboard(self, monkeypatch):
        captured = {}

        def fake_copy(text):
            captured["text"] = text

        monkeypatch.setattr("nglui.statebuilder.shaders.pyperclip.copy", fake_copy)
        shader = AnnotationShaderBuilder().opacity(default=0.5).build(to_clipboard=True)
        assert captured["text"] == shader
        assert "#uicontrol float opacity slider" in shader

    def test_build_default_does_not_touch_clipboard(self, monkeypatch):
        called = {"n": 0}

        def fake_copy(text):
            called["n"] += 1

        monkeypatch.setattr("nglui.statebuilder.shaders.pyperclip.copy", fake_copy)
        AnnotationShaderBuilder().opacity().build()
        assert called["n"] == 0

    def test_skeleton_build_to_clipboard(self, monkeypatch):
        captured = {}

        def fake_copy(text):
            captured["text"] = text

        monkeypatch.setattr("nglui.statebuilder.shaders.pyperclip.copy", fake_copy)
        shader = SkeletonShaderBuilder().use_segment_color().build(to_clipboard=True)
        assert captured["text"] == shader

    def test_url_input_calls_get_annotation_info(self, monkeypatch):
        captured = {}
        sample_info = self._info(
            [
                {
                    "id": "ct",
                    "type": "uint8",
                    "enum_values": [0],
                    "enum_labels": ["only"],
                }
            ]
        )

        def fake_get(url):
            captured["url"] = url
            return sample_info

        # Patch where the helper looks it up.
        from nglui.parser import info as info_module

        monkeypatch.setattr(info_module, "get_annotation_info", fake_get)
        shader = auto_annotation_shader("https://example.com/anno-source")
        assert captured["url"] == "https://example.com/anno-source"
        assert "#uicontrol vec3 only color" in shader


# ---------------------------------------------------------------------------
# SkeletonShaderBuilder.from_info / auto_skeleton_shader
# ---------------------------------------------------------------------------


class TestSkeletonFromInfo:
    """Auto-configure a skeleton shader from a Neuroglancer skeleton info dict.

    The skeleton API differs from the annotation API: skeleton info has no
    enum metadata, so the builder is configured around picking *which*
    attribute drives colour, with the visualisation rule chosen from that
    attribute's name and ``data_type``.
    """

    def _info(self, vertex_attributes):
        return {
            "@type": "neuroglancer_skeletons",
            "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            "vertex_attributes": vertex_attributes,
        }

    # ---- auto-pick ----------------------------------------------------

    def test_auto_pick_compartment_triggers_swc_desaturate(self):
        from nglui.statebuilder.shaders import SkeletonShaderBuilder

        info = self._info(
            [{"id": "compartment", "data_type": "uint8", "num_components": 1}]
        )
        shader = SkeletonShaderBuilder.from_info(info).build()
        assert "rgbToHsl" in shader
        assert "compartment == 2.0" in shader
        assert "emitRGB(uColor.rgb)" in shader  # axon kept saturated

    def test_auto_pick_float_when_no_compartment(self):
        from nglui.statebuilder.shaders import SkeletonShaderBuilder

        info = self._info(
            [{"id": "distance", "data_type": "float32", "num_components": 1}]
        )
        shader = SkeletonShaderBuilder.from_info(info).build()
        assert "colormapCubehelix" in shader
        assert "float distance = vCustom1;" in shader

    def test_auto_pick_compartment_wins_over_float(self):
        from nglui.statebuilder.shaders import SkeletonShaderBuilder

        info = self._info(
            [
                {"id": "distance", "data_type": "float32", "num_components": 1},
                {"id": "compartment", "data_type": "uint8", "num_components": 1},
            ]
        )
        shader = SkeletonShaderBuilder.from_info(info).build()
        # Compartment wins even when float comes first.
        assert "rgbToHsl" in shader
        assert "compartment == 2.0" in shader
        assert "colormapCubehelix" not in shader

    def test_auto_pick_no_attributes_falls_back_to_segment_color(self):
        from nglui.statebuilder.shaders import SkeletonShaderBuilder

        shader = SkeletonShaderBuilder.from_info(self._info([])).build()
        assert "emitRGB(segmentColor().rgb);" in shader
        assert "rgbToHsl" not in shader

    def test_auto_pick_integer_only_falls_back_to_segment_color(self):
        """Integer attribute that isn't named 'compartment' has no metadata
        we can use — fall back to segment colour."""
        from nglui.statebuilder.shaders import SkeletonShaderBuilder

        info = self._info(
            [{"id": "branch_id", "data_type": "uint16", "num_components": 1}]
        )
        shader = SkeletonShaderBuilder.from_info(info).build()
        assert "emitRGB(segmentColor().rgb);" in shader
        assert "colormapCubehelix" not in shader

    # ---- explicit color_attr ------------------------------------------

    def test_explicit_color_attr_compartment(self):
        from nglui.statebuilder.shaders import SkeletonShaderBuilder

        info = self._info(
            [
                {"id": "compartment", "data_type": "uint8", "num_components": 1},
                {"id": "distance", "data_type": "float32", "num_components": 1},
            ]
        )
        shader = SkeletonShaderBuilder.from_info(info, color_attr="compartment").build()
        assert "rgbToHsl" in shader
        assert "compartment == 2.0" in shader

    def test_explicit_color_attr_float(self):
        """Choosing the float attribute forces continuous_color even when a
        compartment attribute is also present."""
        from nglui.statebuilder.shaders import SkeletonShaderBuilder

        info = self._info(
            [
                {"id": "compartment", "data_type": "uint8", "num_components": 1},
                {"id": "distance", "data_type": "float32", "num_components": 1},
            ]
        )
        shader = SkeletonShaderBuilder.from_info(info, color_attr="distance").build()
        assert "colormapCubehelix" in shader
        assert "float distance = vCustom2;" in shader
        assert "rgbToHsl" not in shader

    def test_explicit_color_attr_custom_reference_compartment(self):
        from nglui.statebuilder.shaders import SkeletonShaderBuilder

        info = self._info(
            [{"id": "compartment", "data_type": "uint8", "num_components": 1}]
        )
        shader = SkeletonShaderBuilder.from_info(
            info, color_attr="compartment", compartment_reference=1.0
        ).build()
        assert "compartment == 1.0" in shader

    def test_explicit_color_attr_integer_warns_and_falls_back(self):
        import warnings

        from nglui.statebuilder.shaders import SkeletonShaderBuilder

        info = self._info(
            [{"id": "branch_id", "data_type": "uint16", "num_components": 1}]
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            shader = SkeletonShaderBuilder.from_info(
                info, color_attr="branch_id"
            ).build()
        messages = [str(w.message) for w in captured]
        assert any("branch_id" in m and "no enum metadata" in m for m in messages)
        assert "emitRGB(segmentColor().rgb);" in shader

    def test_explicit_color_attr_unknown_raises(self):
        from nglui.statebuilder.shaders import SkeletonShaderBuilder

        info = self._info(
            [{"id": "compartment", "data_type": "uint8", "num_components": 1}]
        )
        with pytest.raises(ValueError, match="not found"):
            SkeletonShaderBuilder.from_info(info, color_attr="nope")

    # ---- multi-component / indexing ----------------------------------

    def test_multi_component_attribute_is_skipped_with_warning(self):
        import warnings

        from nglui.statebuilder.shaders import SkeletonShaderBuilder

        info = self._info(
            [
                {"id": "compartment", "data_type": "uint8", "num_components": 1},
                {"id": "orientation", "data_type": "float32", "num_components": 3},
            ]
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            builder = SkeletonShaderBuilder.from_info(info)
        messages = [str(w.message) for w in captured]
        assert any("orientation" in m and "num_components" in m for m in messages)
        assert "orientation" not in builder._attributes
        assert "compartment" in builder._attributes

    def test_attribute_index_is_one_based(self):
        from nglui.statebuilder.shaders import SkeletonShaderBuilder

        info = self._info(
            [
                {"id": "first", "data_type": "float32", "num_components": 1},
                {"id": "second", "data_type": "float32", "num_components": 1},
            ]
        )
        # Pick second explicitly so the GLSL clearly references vCustom2.
        shader = SkeletonShaderBuilder.from_info(info, color_attr="second").build()
        assert "float second = vCustom2;" in shader

    # ---- SWC compartment constant ------------------------------------

    def test_swc_compartments_constant(self):
        """Documented SWC mapping is exposed for callers building a
        categorical_color() manually with named labels."""
        from nglui.statebuilder.shaders import SkeletonShaderBuilder

        assert SkeletonShaderBuilder.SWC_COMPARTMENTS == {
            1: "soma",
            2: "axon",
            3: "dendrite",
            4: "apical_dendrite",
        }


class TestAutoSkeletonShader:
    def _info(self, vertex_attributes):
        return {
            "@type": "neuroglancer_skeletons",
            "vertex_attributes": vertex_attributes,
        }

    def test_dict_input(self):
        from nglui.statebuilder.shaders import auto_skeleton_shader

        info = self._info(
            [{"id": "compartment", "data_type": "uint8", "num_components": 1}]
        )
        shader = auto_skeleton_shader(info)
        assert "compartment == 2.0" in shader

    def test_url_input_calls_get_skeleton_info(self, monkeypatch):
        from nglui.parser import info as info_module
        from nglui.statebuilder.shaders import auto_skeleton_shader

        captured = {}
        sample_info = self._info(
            [{"id": "compartment", "data_type": "uint8", "num_components": 1}]
        )

        def fake_get(url):
            captured["url"] = url
            return sample_info

        monkeypatch.setattr(info_module, "get_skeleton_info", fake_get)
        shader = auto_skeleton_shader("https://example.com/skel-source")
        assert captured["url"] == "https://example.com/skel-source"
        assert "compartment == 2.0" in shader

    def test_url_input_forwards_color_attr(self, monkeypatch):
        from nglui.parser import info as info_module
        from nglui.statebuilder.shaders import auto_skeleton_shader

        sample_info = self._info(
            [
                {"id": "compartment", "data_type": "uint8", "num_components": 1},
                {"id": "distance", "data_type": "float32", "num_components": 1},
            ]
        )
        monkeypatch.setattr(info_module, "get_skeleton_info", lambda url: sample_info)
        shader = auto_skeleton_shader(
            "https://example.com/skel-source", color_attr="distance"
        )
        assert "colormapCubehelix" in shader
        assert "rgbToHsl" not in shader


# ---------------------------------------------------------------------------
# Multi-rule skeleton shader: add_color_rule + cascade emitter
# ---------------------------------------------------------------------------


class TestAddColorRule:
    """Multi-rule mode: each colour rule is gated by a per-attribute
    checkbox; the first checked rule wins via an if/else-if cascade."""

    def test_compartment_rule_emits_desaturate_in_cascade(self):
        b = SkeletonShaderBuilder(["compartment"]).add_color_rule(
            "compartment", "compartment_desaturate"
        )
        code = b.build()
        assert "#uicontrol bool color_by_compartment checkbox(default=true)" in code
        assert "if (color_by_compartment) {" in code
        assert "compartment == 2.0" in code
        # HSL helpers are inlined when desaturate is in the cascade.
        assert "rgbToHsl" in code

    def test_continuous_rule_uses_qualified_slider_names(self):
        b = SkeletonShaderBuilder(["distance"]).add_color_rule(
            "distance", "continuous", colormap="cubehelix"
        )
        code = b.build()
        assert "#uicontrol float distance_rangeMin slider" in code
        assert "#uicontrol float distance_rangeMax slider" in code
        assert "if (color_by_distance) {" in code
        assert (
            "(distance - distance_rangeMin) / (distance_rangeMax - distance_rangeMin)"
            in code
        )

    def test_two_continuous_rules_dont_collide_on_slider_names(self):
        """Each continuous rule's range sliders are qualified by attribute,
        so two of them coexist without registry collision."""
        b = (
            SkeletonShaderBuilder(["a", "b"])
            .add_color_rule("a", "continuous")
            .add_color_rule("b", "continuous")
        )
        code = b.build()
        for name in ("a_rangeMin", "a_rangeMax", "b_rangeMin", "b_rangeMax"):
            assert f"#uicontrol float {name} slider" in code

    def test_first_rule_defaults_active_others_inactive(self):
        b = (
            SkeletonShaderBuilder(["compartment", "distance"])
            .add_color_rule("compartment", "compartment_desaturate")
            .add_color_rule("distance", "continuous")
        )
        code = b.build()
        assert "color_by_compartment checkbox(default=true)" in code
        assert "color_by_distance checkbox(default=false)" in code

    def test_cascade_uses_else_if_after_first(self):
        b = (
            SkeletonShaderBuilder(["compartment", "distance"])
            .add_color_rule("compartment", "compartment_desaturate")
            .add_color_rule("distance", "continuous")
        )
        code = b.build()
        # First rule is `if`, second is `} else if`.
        assert "if (color_by_compartment) {" in code
        assert "} else if (color_by_distance) {" in code

    def test_base_segment_color_emit_precedes_cascade(self):
        """The base emitRGB(segmentColor().rgb) must come before the cascade
        so all-checkboxes-off renders as segment colour."""
        b = SkeletonShaderBuilder(["compartment"]).add_color_rule(
            "compartment", "compartment_desaturate"
        )
        code = b.build()
        base_idx = code.index("emitRGB(segmentColor().rgb);")
        cascade_idx = code.index("if (color_by_compartment) {")
        assert base_idx < cascade_idx

    def test_unknown_kind_raises(self):
        b = SkeletonShaderBuilder(["a"])
        with pytest.raises(ValueError, match="kind must be"):
            b.add_color_rule("a", "categorical")

    def test_unknown_attribute_raises(self):
        b = SkeletonShaderBuilder(["a"])
        with pytest.raises(ValueError, match="not declared"):
            b.add_color_rule("nope", "continuous")

    def test_duplicate_rule_for_attr_raises(self):
        b = SkeletonShaderBuilder(["compartment"]).add_color_rule(
            "compartment", "compartment_desaturate"
        )
        with pytest.raises(ValueError, match="already has a colour rule"):
            b.add_color_rule("compartment", "continuous")

    def test_continuous_rule_zero_range_raises(self):
        b = SkeletonShaderBuilder(["a"])
        with pytest.raises(ValueError, match="range_min != range_max"):
            b.add_color_rule("a", "continuous", range_min=5.0, range_max=5.0)

    def test_continuous_rule_unknown_colormap_raises(self):
        b = SkeletonShaderBuilder(["a"])
        with pytest.raises(ValueError, match="Unknown colormap"):
            b.add_color_rule("a", "continuous", colormap="viridis")

    def test_mutual_exclusion_with_use_segment_color(self):
        b = SkeletonShaderBuilder(["a"]).add_color_rule("a", "continuous")
        with pytest.raises(ValueError, match="conflicts with add_color_rule"):
            b.use_segment_color()

        b2 = SkeletonShaderBuilder(["a"]).use_segment_color()
        with pytest.raises(ValueError, match="conflicts with single-rule"):
            b2.add_color_rule("a", "continuous")

    def test_mutual_exclusion_with_categorical(self):
        b = SkeletonShaderBuilder(["a"]).add_color_rule("a", "continuous")
        with pytest.raises(ValueError, match="conflicts with add_color_rule"):
            b.categorical_color(attr="a", categories={1.0: ("x", "red")})

    def test_mutual_exclusion_with_continuous(self):
        b = SkeletonShaderBuilder(["a", "b"]).add_color_rule("a", "continuous")
        with pytest.raises(ValueError, match="conflicts with add_color_rule"):
            b.continuous_color(attr="b")


class TestSkeletonFromInfoMultiRule:
    """from_info() in multi-rule mode: color_attr=list[str] or 'all'."""

    def _info(self, vertex_attributes):
        return {
            "@type": "neuroglancer_skeletons",
            "vertex_attributes": vertex_attributes,
        }

    def test_list_color_attr_emits_one_rule_per_entry(self):
        info = self._info(
            [
                {"id": "compartment", "data_type": "uint8", "num_components": 1},
                {"id": "distance", "data_type": "float32", "num_components": 1},
            ]
        )
        code = SkeletonShaderBuilder.from_info(
            info, color_attr=["compartment", "distance"]
        ).build()
        assert "color_by_compartment" in code
        assert "color_by_distance" in code
        # Cascade order matches the list.
        assert code.index("color_by_compartment") < code.index("color_by_distance")
        # First entry defaults active.
        assert "color_by_compartment checkbox(default=true)" in code
        assert "color_by_distance checkbox(default=false)" in code

    def test_list_order_controls_default_active(self):
        info = self._info(
            [
                {"id": "compartment", "data_type": "uint8", "num_components": 1},
                {"id": "distance", "data_type": "float32", "num_components": 1},
            ]
        )
        # Reverse the list: distance becomes the default-active rule.
        code = SkeletonShaderBuilder.from_info(
            info, color_attr=["distance", "compartment"]
        ).build()
        assert "color_by_distance checkbox(default=true)" in code
        assert "color_by_compartment checkbox(default=false)" in code

    def test_all_sentinel_includes_every_supportable_attr(self):
        info = self._info(
            [
                {"id": "compartment", "data_type": "uint8", "num_components": 1},
                {"id": "distance", "data_type": "float32", "num_components": 1},
            ]
        )
        code = SkeletonShaderBuilder.from_info(info, color_attr="all").build()
        assert "color_by_compartment" in code
        assert "color_by_distance" in code

    def test_all_sentinel_skips_integer_non_compartment_with_warning(self):
        import warnings

        info = self._info(
            [
                {"id": "compartment", "data_type": "uint8", "num_components": 1},
                {"id": "distance", "data_type": "float32", "num_components": 1},
                {"id": "branch_id", "data_type": "uint16", "num_components": 1},
            ]
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            code = SkeletonShaderBuilder.from_info(info, color_attr="all").build()
        messages = [str(w.message) for w in captured]
        assert any("branch_id" in m and "no enum metadata" in m for m in messages)
        # No checkbox emitted for the skipped attribute.
        assert "color_by_branch_id" not in code

    def test_list_with_unknown_attr_raises(self):
        info = self._info(
            [{"id": "compartment", "data_type": "uint8", "num_components": 1}]
        )
        with pytest.raises(ValueError, match="not found"):
            SkeletonShaderBuilder.from_info(info, color_attr=["nope"])

    def test_all_with_no_supportable_attrs_falls_back_to_segment_color(self):
        """If 'all' resolves to zero supportable rules, the build still
        produces a valid shader (just the base segment-colour emit)."""
        import warnings

        info = self._info(
            [{"id": "branch_id", "data_type": "uint16", "num_components": 1}]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            code = SkeletonShaderBuilder.from_info(info, color_attr="all").build()
        assert "emitRGB(segmentColor().rgb);" in code
        # No cascade — no checkboxes.
        assert "color_by_" not in code


class TestAutoSkeletonShaderMultiRule:
    def test_passthrough_list_color_attr(self, monkeypatch):
        from nglui.parser import info as info_module

        sample_info = {
            "@type": "neuroglancer_skeletons",
            "vertex_attributes": [
                {"id": "compartment", "data_type": "uint8", "num_components": 1},
                {"id": "distance", "data_type": "float32", "num_components": 1},
            ],
        }
        monkeypatch.setattr(info_module, "get_skeleton_info", lambda url: sample_info)
        from nglui.statebuilder.shaders import auto_skeleton_shader

        code = auto_skeleton_shader(
            "https://example.com/skel-source",
            color_attr=["distance", "compartment"],
        )
        assert "color_by_distance checkbox(default=true)" in code
        assert "color_by_compartment checkbox(default=false)" in code
