"""Tool bindings and palettes, with keys allocated across the whole viewer."""

import warnings

import neuroglancer
import pytest

from nglui.statebuilder import (
    AnnotationLayer,
    ImageLayer,
    PointAnnotation,
    RawLayer,
    SegmentationLayer,
    ViewerState,
    tools,
)

IMG_SRC = "precomputed://gs://example/img"
SEG_SRC = "precomputed://gs://example/seg"


def _anno(name="anno", **kwargs):
    return AnnotationLayer(
        name=name,
        resolution=[4, 4, 40],
        annotations=[PointAnnotation(point=[1, 2, 3])],
        set_position=False,
        **kwargs,
    )


def _state(*layers, **kwargs):
    kwargs.setdefault("dimensions", [4, 4, 40])
    kwargs.setdefault("infer_coordinates", False)
    return ViewerState(layers=list(layers), **kwargs)


def _layer(d, name):
    return next(layer for layer in d["layers"] if layer["name"] == name)


class TestToolJson:
    @pytest.mark.parametrize(
        "tool, expected",
        [
            (tools.AnnotatePoint(), {"type": "annotatePoint"}),
            (tools.AnnotateLine(), {"type": "annotateLine"}),
            (tools.AnnotateBoundingBox(), {"type": "annotateBoundingBox"}),
            (tools.AnnotateEllipsoid(), {"type": "annotateSphere"}),
            (tools.AnnotatePolyline(), {"type": "annotatePolyline"}),
            (tools.SelectSegments(), {"type": "selectSegments"}),
            (tools.MergeSegments(), {"type": "mergeSegments"}),
            (tools.SplitSegments(), {"type": "splitSegments"}),
            (
                tools.ShaderControl(control="gain"),
                {"type": "shaderControl", "control": "gain"},
            ),
            (tools.LayerSetting(setting="objectAlpha"), {"type": "objectAlpha"}),
            (
                tools.Dimension(dimension="z"),
                {"type": "dimension", "dimension": "z"},
            ),
        ],
    )
    def test_json(self, tool, expected):
        assert tool.to_json() == expected

    def test_palette_json_carries_layer(self):
        tool = tools.LayerSetting(setting="opacity", layer="img")
        assert tool.to_json(layer_name="img") == {"type": "opacity", "layer": "img"}

    def test_layer_object_becomes_name(self):
        img = ImageLayer(name="em", source=IMG_SRC)
        assert tools.ShaderControl(layer=img).layer == "em"

    @pytest.mark.parametrize("key", ["p", "PP", "1", ""])
    def test_invalid_key(self, key):
        with pytest.raises(ValueError, match="single capital letter"):
            tools.AnnotatePoint(key=key)

    def test_invalid_setting(self):
        with pytest.raises(ValueError):
            tools.LayerSetting(setting="brightness")


class TestBinding:
    def test_explicit_key_on_layer(self):
        vs = _state(_anno()).add_tools(tools.AnnotatePoint(layer="anno", key="P"))
        assert _layer(vs.to_dict(), "anno")["toolBindings"] == {
            "P": {"type": "annotatePoint"}
        }

    def test_auto_keys_are_distinct(self):
        vs = _state(ImageLayer(source=IMG_SRC), SegmentationLayer(source=SEG_SRC))
        vs.add_tools(
            tools.ShaderControl(layer="img"),
            tools.SelectSegments(layer="seg"),
            tools.LayerSetting(layer="seg", setting="objectAlpha"),
        )
        d = vs.to_dict()
        keys = list(_layer(d, "img")["toolBindings"]) + list(
            _layer(d, "seg")["toolBindings"]
        )
        assert keys == ["Z", "X", "C"]

    def test_explicit_keys_claimed_before_auto(self):
        vs = _state(_anno())
        vs.add_tools(
            tools.AnnotateLine(layer="anno"), tools.AnnotatePoint(layer="anno", key="Z")
        )
        bindings = _layer(vs.to_dict(), "anno")["toolBindings"]
        assert bindings == {
            "Z": {"type": "annotatePoint"},
            "X": {"type": "annotateLine"},
        }

    def test_key_false_is_not_bound(self):
        vs = _state(_anno()).add_tools(tools.AnnotatePoint(layer="anno", key=False))
        # Local annotation layers always carry a (here empty) toolBindings map
        assert not _layer(vs.to_dict(), "anno").get("toolBindings")

    def test_viewer_tool(self):
        vs = _state().add_tools(tools.Dimension(dimension="z", key="D"))
        assert vs.to_dict()["toolBindings"] == {
            "D": {"type": "dimension", "dimension": "z"}
        }

    def test_layer_tools_field(self):
        seg = SegmentationLayer(source=SEG_SRC, tools=[tools.SelectSegments(key="S")])
        d = _state(seg).to_dict()
        assert _layer(d, "seg")["toolBindings"] == {"S": {"type": "selectSegments"}}

    def test_layer_tools_are_not_in_standalone_layer_json(self):
        seg = SegmentationLayer(source=SEG_SRC, tools=[tools.SelectSegments(key="S")])
        assert "toolBindings" not in seg.to_dict()

    def test_layer_given_as_object(self):
        img = ImageLayer(name="em", source=IMG_SRC)
        vs = _state(img).add_tools(tools.ShaderControl(layer=img, key="B"))
        assert _layer(vs.to_dict(), "em")["toolBindings"]["B"]["control"] == (
            "normalized"
        )


class TestBindingErrors:
    def test_missing_layer(self):
        vs = _state().add_tools(tools.AnnotatePoint())
        with pytest.raises(ValueError, match="pass layer="):
            vs.to_dict()

    def test_unknown_layer(self):
        vs = _state(_anno()).add_tools(tools.AnnotatePoint(layer="nope"))
        with pytest.raises(ValueError, match="not in the viewer"):
            vs.to_dict()

    def test_wrong_layer_type(self):
        vs = _state(ImageLayer(source=IMG_SRC)).add_tools(
            tools.SelectSegments(layer="img")
        )
        with pytest.raises(ValueError, match="segmentation layers"):
            vs.to_dict()

    def test_explicit_key_collision(self):
        vs = _state(_anno(), SegmentationLayer(source=SEG_SRC)).add_tools(
            tools.AnnotatePoint(layer="anno", key="P"),
            tools.SelectSegments(layer="seg", key="P"),
        )
        with pytest.raises(ValueError, match="already bound"):
            vs.to_dict()

    def test_not_a_tool(self):
        with pytest.raises(TypeError):
            _state().add_tools("annotatePoint")


class TestCoexistence:
    def test_tag_keys_are_respected(self):
        """Auto keys avoid the keys tag tools already hold."""
        anno = _anno(tags=["axon", "dendrite"])
        vs = _state(anno, capabilities="main")
        vs.add_tools(tools.AnnotatePoint(layer="anno"))
        bindings = _layer(vs.to_dict(), "anno")["toolBindings"]
        assert set(bindings) == {"Q", "W", "Z"}
        assert bindings["Z"] == {"type": "annotatePoint"}

    def test_explicit_key_moves_tag_tool(self):
        """A key the user names wins; the generated tag tool moves to a free one."""
        vs = _state(_anno(tags=["axon"]), capabilities="main")
        vs.add_tools(tools.AnnotatePoint(layer="anno", key="Q"))
        bindings = _layer(vs.to_dict(), "anno")["toolBindings"]
        assert bindings["Q"] == {"type": "annotatePoint"}
        assert bindings["W"] == {"type": "toggleBoolProperty", "property": "axon"}

    def test_two_tagged_layers_get_distinct_keys(self):
        vs = _state(
            _anno("a", tags=["axon", "soma"]),
            _anno("b", tags=["spine"]),
            capabilities="main",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            d = vs.to_dict()
        assert list(_layer(d, "a")["toolBindings"]) == ["Q", "W"]
        assert _layer(d, "b")["toolBindings"] == {
            "E": {"type": "toggleBoolProperty", "property": "spine"}
        }

    def test_legacy_tag_bindings_stay_strings_when_moved(self):
        vs = _state(
            _anno("a", tags=["axon"]), _anno("b", tags=["soma"]), capabilities="legacy"
        )
        d = vs.to_dict()
        assert _layer(d, "a")["toolBindings"] == {"Q": "tagTool_tag0"}
        assert _layer(d, "b")["toolBindings"] == {"W": "tagTool_tag0"}

    def test_tag_tool_moves_off_raw_layer_key(self):
        raw = RawLayer(
            name="old",
            json_data={
                "type": "segmentation",
                "source": SEG_SRC,
                "toolBindings": {"Q": "selectSegments"},
            },
        )
        d = _state(raw, _anno(tags=["axon"]), capabilities="main").to_dict()
        assert _layer(d, "old")["toolBindings"] == {"Q": "selectSegments"}
        assert list(_layer(d, "anno")["toolBindings"]) == ["W"]

    def test_explicit_key_colliding_with_raw_layer_raises(self):
        raw = RawLayer(
            name="old",
            json_data={
                "type": "segmentation",
                "source": SEG_SRC,
                "toolBindings": {"Q": "selectSegments"},
            },
        )
        vs = _state(raw, SegmentationLayer(source=SEG_SRC))
        vs.add_tools(tools.SelectSegments(layer="seg", key="Q"))
        with pytest.raises(ValueError, match="already bound"):
            vs.to_dict()

    def test_raw_layer_keys_are_respected(self):
        raw = RawLayer(
            name="old",
            json_data={
                "type": "segmentation",
                "source": SEG_SRC,
                "toolBindings": {"Z": "selectSegments"},
            },
        )
        vs = _state(raw).add_tools(tools.SelectSegments(layer="old"))
        assert set(_layer(vs.to_dict(), "old")["toolBindings"]) == {"Z", "X"}

    def test_base_state_keys_are_respected(self):
        base = {"toolBindings": {"Z": {"type": "dimension", "dimension": "z"}}}
        vs = _state(base_state=base).add_tools(tools.Dimension(dimension="x"))
        assert set(vs.to_dict()["toolBindings"]) == {"Z", "X"}

    def test_duplicate_fixed_keys_warn(self):
        """Two raw layers both claim Q; nglui cannot move them, so it warns."""
        raws = [
            RawLayer(
                name=name,
                json_data={
                    "type": "segmentation",
                    "source": SEG_SRC,
                    "toolBindings": {"Q": "selectSegments"},
                },
            )
            for name in ("r1", "r2")
        ]
        vs = _state(*raws).add_tools(tools.Dimension(dimension="z"))
        with pytest.warns(UserWarning, match="bound both"):
            vs.to_dict()

    def test_no_tools_no_warning(self):
        vs = _state(_anno("a", tags=["axon"]), _anno("b", tags=["soma"]))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            vs.to_dict()


class TestPalettes:
    def test_palette_from_add_tools(self):
        vs = _state(_anno(), SegmentationLayer(source=SEG_SRC)).add_tools(
            tools.AnnotatePoint(layer="anno", key="P"),
            tools.SelectSegments(layer="seg", key=False),
            palette=tools.ToolPalette("Review", side="right", size=250),
        )
        d = vs.to_dict()
        assert d["toolPalettes"] == {
            "Review": {
                "tools": [
                    {"type": "annotatePoint", "layer": "anno"},
                    {"type": "selectSegments", "layer": "seg"},
                ],
                "side": "right",
                "size": 250,
                "visible": True,
            }
        }
        assert "toolBindings" not in _layer(d, "seg")

    def test_palette_by_name_accumulates(self):
        vs = _state(_anno()).add_tools(
            tools.AnnotatePoint(layer="anno", key=False), palette="P1"
        )
        vs.add_tools(tools.AnnotateLine(layer="anno", key=False), palette="P1")
        entries = vs.to_dict()["toolPalettes"]["P1"]["tools"]
        assert [e["type"] for e in entries] == ["annotatePoint", "annotateLine"]

    def test_add_tool_palette_only_lists(self):
        palette = tools.ToolPalette(
            "Display",
            tools=[tools.LayerSetting(layer="img", setting="opacity")],
            query="type:opacity",
        )
        vs = _state(ImageLayer(source=IMG_SRC)).add_tool_palette(palette)
        d = vs.to_dict()
        assert d["toolPalettes"]["Display"]["query"] == "type:opacity"
        assert "toolBindings" not in _layer(d, "img")

    def test_palette_viewer_tool_has_no_layer(self):
        vs = _state().add_tools(
            tools.Dimension(dimension="z", key=False), palette="Nav"
        )
        assert vs.to_dict()["toolPalettes"]["Nav"]["tools"] == [
            {"type": "dimension", "dimension": "z"}
        ]

    def test_palette_checks_layer(self):
        palette = tools.ToolPalette("X", tools=[tools.SelectSegments(layer="nope")])
        with pytest.raises(ValueError, match="not in the viewer"):
            _state().add_tool_palette(palette).to_dict()


class TestProofreadingWorkflow:
    """A synapse-review state: points to place, segments to select, a palette."""

    def test_round_trip(self):
        img = ImageLayer(source=IMG_SRC)
        seg = SegmentationLayer(source=SEG_SRC, segments=[864691135368056201])
        syn = _anno("synapses", linked_segmentation="seg", tags=["good", "bad"])
        vs = _state(
            img, seg, syn, capabilities="main", title="Synapse review"
        ).add_tools(
            tools.AnnotatePoint(layer=syn, key="P"),
            tools.SelectSegments(layer=seg),
            tools.ShaderControl(layer=img),
            tools.Dimension(dimension="z", key="J"),
            palette=tools.ToolPalette("Review", side="right"),
        )
        url = vs.to_url(target_url="https://neuroglancer-demo.appspot.com")
        parsed = neuroglancer.parse_url(url).to_json()

        layers = {layer["name"]: layer for layer in parsed["layers"]}
        syn_keys = layers["synapses"]["toolBindings"]
        assert syn_keys["P"] == {"type": "annotatePoint"}
        assert {"Q", "W"} <= set(syn_keys)  # the tag tools are untouched
        assert list(layers["seg"]["toolBindings"]) == ["Z"]
        assert list(layers["img"]["toolBindings"]) == ["X"]
        assert parsed["toolBindings"] == {"J": {"type": "dimension", "dimension": "z"}}
        assert [t["type"] for t in parsed["toolPalettes"]["Review"]["tools"]] == [
            "annotatePoint",
            "selectSegments",
            "shaderControl",
            "dimension",
        ]
        all_keys = [
            k for layer in layers.values() for k in layer.get("toolBindings", {})
        ]
        all_keys += list(parsed["toolBindings"])
        assert len(all_keys) == len(set(all_keys))
