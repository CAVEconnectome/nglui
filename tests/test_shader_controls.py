"""Shader #uicontrol directives must be syntax Neuroglancer accepts."""

import re

import pytest

from nglui.statebuilder.shaders import (
    Checkbox,
    ColorControl,
    InverlpControl,
    InvlerpControl,
    Slider,
    shader_base,
)

# Neuroglancer's own directive grammar (src/webgl/shader_ui_controls.ts)
DIRECTIVE = re.compile(r"^[ \t]*#[ \t]*uicontrol[ \t]+(.*)$", re.M)
INNER = re.compile(
    r"^([_a-zA-Z][_a-zA-Z0-9]*)[ \t]+([a-z][a-zA-Z0-9_]*)"
    r"(?:[ \t]+([a-z]+))?[ \t]*(?:\([ \t]*(.*)\)[ \t]*)?"
)
PARAM = re.compile(r"^([_a-z][_a-zA-Z0-9]*)[ \t]*=")


def parse(directive: str) -> tuple[str, str, dict]:
    """Parse one directive as Neuroglancer does: (control type, name, params)."""
    (inner,) = DIRECTIVE.findall(directive)
    m = INNER.match(inner)
    assert m is not None, f"Neuroglancer would reject: {directive}"
    type_name, name, control, params = m.groups()
    assert not inner[m.end() :].strip(" \t;"), f"trailing junk in {directive}"
    parsed = {}
    for part in re.split(r",\s*(?![^\[]*\])", params or ""):
        if part:
            key = PARAM.match(part)
            assert key, f"bad parameter {part!r} in {directive}"
            parsed[key.group(1)] = part[key.end() :].strip()
    return control or type_name, name, parsed


class TestInvlerpControl:
    def test_bare(self):
        assert str(InvlerpControl("contrast")) == "#uicontrol invlerp contrast;"

    def test_all_image_parameters(self):
        control = InvlerpControl(
            "contrast", range=[30, 220], window=[0, 255], channel=[1, 2], clamp=False
        )
        assert parse(str(control)) == (
            "invlerp",
            "contrast",
            {
                "range": "[30, 220]",
                "window": "[0, 255]",
                "channel": "[1, 2]",
                "clamp": "false",
            },
        )

    def test_property_parameter(self):
        _, _, params = parse(str(InvlerpControl("size", property="volume")))
        assert params == {"property": '"volume"'}

    def test_float_and_inverted_range(self):
        _, _, params = parse(str(InvlerpControl("c", range=(0.25, 0.1))))
        assert params["range"] == "[0.25, 0.1]"

    def test_single_channel(self):
        assert parse(str(InvlerpControl("c", channel=2)))[2] == {"channel": "2"}

    def test_clamp_true_is_default_and_omitted(self):
        assert "clamp" not in str(InvlerpControl("c"))

    def test_range_must_be_a_pair(self):
        with pytest.raises(ValueError, match="two numbers"):
            InvlerpControl("c", range=[1, 2, 3])

    def test_channel_and_property_exclusive(self):
        with pytest.raises(ValueError, match="not both"):
            InvlerpControl("c", channel=0, property="p")

    def test_misspelled_alias_warns_and_works(self):
        with pytest.warns(DeprecationWarning, match="InvlerpControl"):
            control = InverlpControl("contrast", range=[0, 255])
        assert parse(str(control))[0] == "invlerp"

    def test_in_a_shader(self):
        shader = shader_base(
            uicontrols=[InvlerpControl("contrast", range=[40, 200])],
            body="  emitGrayscale(contrast());",
        )
        (directive,) = [line for line in shader.splitlines() if "#uicontrol" in line]
        assert parse(directive)[:2] == ("invlerp", "contrast")


@pytest.mark.parametrize(
    "control, expected_type",
    [
        (Checkbox("show"), "checkbox"),
        (Slider("size", min=0, max=10, default=2), "slider"),
        (ColorControl("tint", color="red"), "color"),
        (InvlerpControl("contrast"), "invlerp"),
    ],
)
def test_every_control_parses(control, expected_type):
    assert parse(str(control))[0] == expected_type
