"""Tests for the two annotation tag encodings and property id sanitization."""

import pytest

from nglui.statebuilder.capabilities import (
    LEGACY_CAPABILITIES,
    MAIN_CAPABILITIES,
    Capabilities,
)
from nglui.statebuilder.ngl_annotations import (
    BOOL_PROPERTY_TAG_STRATEGY,
    LEGACY_TAG_STRATEGY,
    MAX_TAG_COUNT,
    PROPERTY_ID_PATTERN,
    BoolPropertyStrategy,
    LegacyTagStrategy,
    PointAnnotation,
    build_property_ids,
    register_property_tools,
    sanitize_property_id,
    strategy_for_capabilities,
)


class TestSanitizePropertyId:
    @pytest.mark.parametrize(
        "label,expected",
        [
            ("axon", "axon"),
            ("Cell Body", "cell_body"),
            ("post-synaptic", "post_synaptic"),
            ("Is Axon", "is_axon"),
            ("already_valid2", "already_valid2"),
            ("multiple   spaces", "multiple_spaces"),
            ("trailing_", "trailing"),
            ("MiXeD", "mixed"),
        ],
    )
    def test_readable_transformations(self, label, expected):
        """Ids are user-visible in the schema tab, so they must stay readable."""
        assert sanitize_property_id(label) == expected

    @pytest.mark.parametrize(
        "label", ["1st pass", "42", "_leading", "Cell Body", "post-synaptic", "!!!", ""]
    )
    def test_output_is_always_valid(self, label):
        assert PROPERTY_ID_PATTERN.match(sanitize_property_id(label))

    def test_leading_digit_gets_prefix(self):
        """Neuroglancer accepts this on entry but rejects it on reload."""
        assert sanitize_property_id("1st pass") == "tag_1st_pass"

    def test_empty_falls_back_to_index(self):
        assert sanitize_property_id("!!!", fallback_index=3) == "tag_3"


class TestBuildPropertyIds:
    def test_valid_labels_pass_through_silently(self, recwarn):
        assert build_property_ids(["axon", "soma"]) == {"axon": "axon", "soma": "soma"}
        assert len(recwarn) == 0

    def test_collisions_are_disambiguated(self):
        """Neuroglancer rejects a layer with duplicate property ids."""
        with pytest.warns(UserWarning):
            ids = build_property_ids(["Cell Body", "cell body", "CELL-BODY"])
        assert sorted(ids.values()) == ["cell_body", "cell_body_2", "cell_body_3"]

    def test_renames_warn_once_with_full_map(self):
        with pytest.warns(UserWarning, match="were renamed") as record:
            build_property_ids(["Cell Body", "post-synaptic"])
        assert len(record) == 1
        assert "cell_body" in str(record[0].message)
        assert "post_synaptic" in str(record[0].message)

    def test_explicit_ids_are_respected(self, recwarn):
        ids = build_property_ids(["Cell Body"], tag_ids={"Cell Body": "soma"})
        assert ids == {"Cell Body": "soma"}
        assert len(recwarn) == 0

    def test_invalid_explicit_id_raises(self):
        with pytest.raises(ValueError, match="is not valid"):
            build_property_ids(["a"], tag_ids={"a": "Not Valid"})

    def test_strict_mode_raises_instead_of_renaming(self):
        with pytest.raises(ValueError, match="not a valid annotation property id"):
            build_property_ids(["Cell Body"], strict=True)

    def test_order_follows_tags(self):
        ids = build_property_ids(["zebra", "apple"])
        assert list(ids) == ["zebra", "apple"]


class TestLegacyStrategy:
    def test_property_specs(self):
        assert LEGACY_TAG_STRATEGY.property_specs(["a", "b"]) == [
            {"id": "tag0", "type": "uint8", "tag": "a"},
            {"id": "tag1", "type": "uint8", "tag": "b"},
        ]

    def test_tool_bindings(self):
        specs = LEGACY_TAG_STRATEGY.property_specs(["a", "b"])
        assert LEGACY_TAG_STRATEGY.tool_bindings(specs) == {
            "Q": "tagTool_tag0",
            "W": "tagTool_tag1",
        }

    def test_encode_is_a_bit_vector(self):
        anno = PointAnnotation(point=[1, 2, 3], tags=["b"])
        assert LEGACY_TAG_STRATEGY.encode(anno, ["a", "b", "c"]) == [0, 1, 0]

    def test_labels_need_no_sanitization(self):
        """The label lives in the `tag` key, so it is unconstrained."""
        specs = LEGACY_TAG_STRATEGY.property_specs(["Cell Body!"])
        assert specs[0]["tag"] == "Cell Body!"

    def test_tag_limit_is_enforced(self):
        with pytest.raises(ValueError, match="Spelunker tag encoding"):
            LEGACY_TAG_STRATEGY.property_specs(
                [f"tag{i}" for i in range(MAX_TAG_COUNT + 1)]
            )


class TestBoolPropertyStrategy:
    def test_property_specs(self):
        assert BOOL_PROPERTY_TAG_STRATEGY.property_specs(["axon", "soma"]) == [
            {"id": "axon", "type": "bool"},
            {"id": "soma", "type": "bool"},
        ]

    def test_renamed_label_is_kept_as_description(self):
        with pytest.warns(UserWarning):
            specs = BOOL_PROPERTY_TAG_STRATEGY.property_specs(["Cell Body"])
        assert specs == [
            {"id": "cell_body", "type": "bool", "description": "Cell Body"}
        ]

    def test_tool_bindings_use_object_form(self):
        specs = BOOL_PROPERTY_TAG_STRATEGY.property_specs(["axon"])
        assert BOOL_PROPERTY_TAG_STRATEGY.tool_bindings(specs) == {
            "Q": {"type": "toggleBoolProperty", "property": "axon"}
        }

    def test_encode_uses_booleans(self):
        anno = PointAnnotation(point=[1, 2, 3], tags=["soma"])
        assert BOOL_PROPERTY_TAG_STRATEGY.encode(anno, ["axon", "soma"]) == [
            False,
            True,
        ]

    def test_encode_is_full_length(self):
        """Neuroglancer rejects a props array that is not exactly one per property."""
        anno = PointAnnotation(point=[1, 2, 3], tags=[])
        tags = ["a", "b", "c", "d"]
        assert BOOL_PROPERTY_TAG_STRATEGY.encode(anno, tags) == [False] * 4

    def test_no_tag_count_limit(self):
        """Unlike the legacy encoding, only keybindings are limited, not properties."""
        tags = [f"tag{i}" for i in range(MAX_TAG_COUNT + 5)]
        specs = BOOL_PROPERTY_TAG_STRATEGY.property_specs(tags)
        assert len(specs) == MAX_TAG_COUNT + 5

    def test_excess_tags_warn_rather_than_raise(self):
        tags = [f"tag{i}" for i in range(MAX_TAG_COUNT + 5)]
        specs = BOOL_PROPERTY_TAG_STRATEGY.property_specs(tags)
        with pytest.warns(UserWarning, match="keyboard shortcuts"):
            bindings = BOOL_PROPERTY_TAG_STRATEGY.tool_bindings(specs)
        assert len(bindings) == MAX_TAG_COUNT

    def test_explicit_ids_flow_through(self, recwarn):
        specs = BOOL_PROPERTY_TAG_STRATEGY.property_specs(
            ["Cell Body"], tag_ids={"Cell Body": "soma"}
        )
        assert specs == [{"id": "soma", "type": "bool", "description": "Cell Body"}]
        assert len(recwarn) == 0


class TestStrategySelection:
    def test_modern_capabilities_use_bool_properties(self):
        assert isinstance(
            strategy_for_capabilities(MAIN_CAPABILITIES), BoolPropertyStrategy
        )

    def test_legacy_capabilities_use_tag_properties(self):
        assert isinstance(
            strategy_for_capabilities(LEGACY_CAPABILITIES), LegacyTagStrategy
        )

    def test_unset_capabilities_follow_the_module_default(self):
        from nglui.statebuilder.capabilities import get_default_capabilities

        expected = (
            BoolPropertyStrategy
            if get_default_capabilities().annotation_bool_properties
            else LegacyTagStrategy
        )
        assert isinstance(strategy_for_capabilities(None), expected)

    def test_explicitly_empty_capabilities_use_the_legacy_encoding(self):
        """A deployment known to support nothing is not the same as an unknown one."""
        assert isinstance(strategy_for_capabilities(Capabilities()), LegacyTagStrategy)


class TestToolRegistration:
    def test_property_tools_are_registered(self):
        from neuroglancer import viewer_state

        for tool_type in (
            "toggleBoolProperty",
            "annotateEnumProperty",
            "annotateNumberProperty",
            "selectPreviousAnnotation",
            "selectNextAnnotation",
        ):
            assert tool_type in viewer_state.tool_types

    def test_registration_is_idempotent(self):
        from neuroglancer import viewer_state

        before = viewer_state.tool_types["toggleBoolProperty"]
        register_property_tools()
        assert viewer_state.tool_types["toggleBoolProperty"] is before

    def test_bound_tool_serializes(self):
        """Bindings are validated against the registry, so this must round-trip."""
        from neuroglancer import viewer_state

        layer = viewer_state.LocalAnnotationLayer(
            dimensions=viewer_state.CoordinateSpace(
                names=["x", "y", "z"], units=["nm"] * 3, scales=[1, 1, 1]
            ),
            tool_bindings={
                "Q": {"type": "toggleBoolProperty", "property": "axon"},
                "N": {"type": "selectNextAnnotation"},
            },
        )
        assert layer.to_json()["toolBindings"]["Q"] == {
            "type": "toggleBoolProperty",
            "property": "axon",
        }


class TestNonAsciiLabels:
    """Tag labels come from dataframe columns and are not always ASCII."""

    @pytest.mark.parametrize(
        "label,expected",
        [
            # Accented Latin: dropping the mark leaves a plausible but wrong id.
            ("café", "cafe"),
            ("Müller", "muller"),
            ("naïve", "naive"),
            ("Ångström", "angstrom"),
            # Greek reads as a symbol in scientific labels, so its name substitutes.
            ("β-cell", "beta_cell"),
            ("α7 receptor", "alpha_7_receptor"),
            ("αβγ", "alpha_beta_gamma"),
            # Compatibility forms decompose.
            ("type-Ⅳ", "type_iv"),
            # Unchanged.
            ("axon", "axon"),
            ("Cell Body", "cell_body"),
        ],
    )
    def test_meaning_survives(self, label, expected):
        assert sanitize_property_id(label) == expected

    def test_greek_cell_types_stay_distinct(self):
        """The case this exists for: these all sanitized to `cell` and collided."""
        with pytest.warns(UserWarning):
            ids = build_property_ids(["α-cell", "β-cell", "γ-cell"])
        assert sorted(ids.values()) == ["alpha_cell", "beta_cell", "gamma_cell"]

    @pytest.mark.parametrize("label", ["日本", "нейрон", "עצב"])
    def test_scripts_without_a_latin_reading_fall_back(self, label):
        """Substituting letter names here would produce nonsense, so it is not done.

        The original is preserved in the property description, and `tag_ids` gives
        exact control.
        """
        assert PROPERTY_ID_PATTERN.match(sanitize_property_id(label))
        assert sanitize_property_id(label).startswith("tag_")

    def test_underscores_do_not_pile_up(self):
        assert sanitize_property_id("Cell  Body") == "cell_body"
        assert sanitize_property_id("a--b") == "a_b"
