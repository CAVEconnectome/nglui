from __future__ import annotations

import re
from collections import namedtuple
from itertools import cycle, islice
from typing import Optional, Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import palettable
import webcolors

# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------


def _rgb_to_triple(rgb: webcolors.IntegerRGB) -> list:
    return [rgb.red / 255.0, rgb.green / 255.0, rgb.blue / 255.0]


def parse_color_rgb(clr: Union[str, tuple, list]) -> list:
    """Parse a color into a [r, g, b] list in [0, 1] range."""
    if isinstance(clr, str):
        hex_match = r"^#[0-9a-fA-F]{6}$"
        if re.match(hex_match, clr):
            return _rgb_to_triple(webcolors.hex_to_rgb(clr))
        else:
            return _rgb_to_triple(webcolors.name_to_rgb(clr))
    else:
        return _rgb_to_triple(webcolors.IntegerRGB(*[int(255 * x) for x in clr]))


def color_to_vec3(clr: Union[str, tuple, list]) -> str:
    """Convert a color to a GLSL vec3 string literal."""
    rgb = parse_color_rgb(clr)
    return f"vec3({rgb[0]}, {rgb[1]}, {rgb[2]})"


def _normalize_color_str(color: Union[str, tuple, list]) -> str:
    """Return a color string suitable for a Neuroglancer uicontrol default."""
    if isinstance(color, str):
        return color
    r, g, b = color[0], color[1], color[2]
    if any(c > 1.0 for c in (r, g, b)):
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _format_float(f: float) -> str:
    """Format a float for GLSL: no scientific notation, always has a decimal point."""
    if f == int(f):
        return f"{int(f)}.0"
    return str(f)


def _palette_colors(n: int, palette: Optional[str] = None) -> list[str]:
    """Return *n* hex color strings from a palettable colormap, cycling if needed.

    Parameters
    ----------
    n : int
        Number of colors needed.
    palette : str, optional
        Name of a palettable colormap (e.g. ``'Set1'``, ``'Tableau_10'``).
        Searched across all palettable modules. Defaults to ``'Tableau_10'``.

    Returns
    -------
    list[str]
        Hex color strings like ``'#4e79a7'``, length *n*.
    """
    if palette is None:
        colors = palettable.cartocolors.qualitative.Bold_10.hex_colors
    else:
        # Search all palettable modules for the named colormap
        module_paths = [
            ("colorbrewer", "diverging"),
            ("colorbrewer", "qualitative"),
            ("colorbrewer", "sequential"),
            ("cartocolors", "diverging"),
            ("cartocolors", "qualitative"),
            ("cartocolors", "sequential"),
            ("cmocean", "diverging"),
            ("cmocean", "sequential"),
            ("scientific", "diverging"),
            ("scientific", "sequential"),
            ("lightbartlein", "diverging"),
            ("lightbartlein", "sequential"),
            ("matplotlib", None),
            ("mycarta", None),
            ("tableau", None),
            ("wesanderson", None),
        ]
        found = None
        for mod_name, subcat in module_paths:
            try:
                mod = getattr(palettable, mod_name)
                target = getattr(mod, subcat) if subcat else mod
                if hasattr(target, palette):
                    found = getattr(target, palette)
                    break
                # Try with a size suffix: find smallest that fits
                candidates = [
                    a
                    for a in dir(target)
                    if a.startswith(palette + "_") and a.split("_")[-1].isdigit()
                ]
                if candidates:
                    best = min(candidates, key=lambda a: abs(int(a.split("_")[-1]) - n))
                    found = getattr(target, best)
                    break
            except AttributeError:
                continue
        if found is None:
            raise ValueError(
                f"Palette '{palette}' not found in palettable. "
                "Check https://jiffyclub.github.io/palettable/ for available names."
            )
        colors = found.hex_colors

    return list(islice(cycle(colors), n))


# ---------------------------------------------------------------------------
# Reusable GLSL helper functions (for skeleton shaders)
# ---------------------------------------------------------------------------

ShaderFunction = namedtuple("shaderFunction", ["name", "code"])

rgb_to_hsl = ShaderFunction(
    name="rgbToHsl",
    code="""
vec3 rgbToHsl(vec3 color) {
    float maxC = max(max(color.r, color.g), color.b);
    float minC = min(min(color.r, color.g), color.b);
    float delta = maxC - minC;

    float h = 0.0;
    float s = 0.0;
    float l = (maxC + minC) / 2.0;

    if (delta > 0.0) {
        s = (l < 0.5) ? (delta / (maxC + minC)) : (delta / (2.0 - maxC - minC));

        if (maxC == color.r) {
            h = (color.g - color.b) / delta + (color.g < color.b ? 6.0 : 0.0);
        } else if (maxC == color.g) {
            h = (color.b - color.r) / delta + 2.0;
        } else {
            h = (color.r - color.g) / delta + 4.0;
        }
        h /= 6.0;
    }

    return vec3(h, s, l);
}
""",
)

hue_to_rgb = ShaderFunction(
    name="hue2rgb",
    code="""
float hue2rgb(float p, float q, float t) {
    if (t < 0.0) t += 1.0;
    if (t > 1.0) t -= 1.0;
    if (t < 1.0 / 6.0) return p + (q - p) * 6.0 * t;
    if (t < 1.0 / 2.0) return q;
    if (t < 2.0 / 3.0) return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    return p;
}
""",
)

hsl_to_rgb = ShaderFunction(
    name="hslToRgb",
    code=f"""
vec3 hslToRgb(vec3 hsl) {{
    float h = hsl.x;
    float s = hsl.y;
    float l = hsl.z;

    float r, g, b;

    if (s == 0.0) {{
        r = g = b = l;
    }} else {{
        float q = (l < 0.5) ? (l * (1.0 + s)) : (l + s - l * s);
        float p = 2.0 * l - q;
        r = {hue_to_rgb.name}(p, q, h + 1.0 / 3.0);
        g = {hue_to_rgb.name}(p, q, h);
        b = {hue_to_rgb.name}(p, q, h - 1.0 / 3.0);
    }}

    return vec3(r, g, b);
}}
""",
)


# ---------------------------------------------------------------------------
# Pre-built shader strings
# ---------------------------------------------------------------------------

simple_compartment_skeleton_shader = f"""
{rgb_to_hsl.code}
{hue_to_rgb.code}
{hsl_to_rgb.code}

void main() {{

  float compartment =  vCustom2;
  vec4 uColor = segmentColor();
  vec3 hsl = {rgb_to_hsl.name}(uColor.rgb);

  if (compartment == 2.0) {{
    emitRGB(uColor.rgb);
  }} else {{
    hsl.y *= 0.5;
    vec3 desaturatedColor = {hsl_to_rgb.name}(hsl);
    emitRGB(desaturatedColor);
  }}
}}
"""

basic_shader = """
void main() {
  setColor(defaultColor());
}
"""


def simple_point_shader(color: str = "tomato", markersize: float = 5.0) -> str:
    """Generate a minimal point shader with a color picker and size slider.

    Parameters
    ----------
    color : str
        Default marker color. Default is 'tomato'.
    markersize : float
        Default marker size. Default is 5.0.

    Returns
    -------
    str
        GLSL shader code.
    """
    return (
        f'#uicontrol vec3 markerColor color(default="{color}")\n'
        f"#uicontrol float markerSize slider(min=0.0, max=20.0, default={_format_float(markersize)})\n"
        "void main() {\n"
        "  setPointMarkerSize(markerSize);\n"
        "  setColor(markerColor);\n"
        "}"
    )


# ---------------------------------------------------------------------------
# AnnotationShaderBuilder
# ---------------------------------------------------------------------------


class AnnotationShaderBuilder:
    """Build Neuroglancer annotation layer GLSL shaders programmatically.

    Provides a high-level interface for constructing annotation shaders, similar
    in spirit to seaborn's ``scatterplot`` API: declare what visual encoding you
    want (color by category, opacity slider, point size from a property) and let
    the builder produce the GLSL.

    UI controls are declared as ``#uicontrol`` directives and exposed as
    interactive widgets in the Neuroglancer side panel. Annotation properties
    are read in GLSL via ``prop_<name>()``.

    All methods return ``self`` for chaining. Call :meth:`build` to get the
    final GLSL string.

    Examples
    --------
    Categorical synapse-type shader with per-category color pickers,
    show/hide checkboxes, an opacity slider, and property-driven point size::

        shader = (
            AnnotationShaderBuilder()
            .point_size(prop="size", scale=0.0002)
            .opacity(default=1.0)
            .highlight(prop="pre_in_selection", value=1)
            .categorical_color(
                prop="tag_detailed",
                categories={
                    0: ("null", "white"),
                    1: ("spine", "magenta"),
                    2: ("spine", "magenta"),   # shares 'spine' controls with value 1
                    3: ("multi_spine", "purple"),
                    4: ("shaft", "yellow"),
                    5: ("soma", "cyan"),
                },
            )
            .border_width(0.0)
            .build()
        )

    Simple two-category shader with an opacity slider::

        shader = (
            AnnotationShaderBuilder()
            .opacity(default=0.8)
            .categorical_color(
                prop="label",
                categories=[(0, "excitatory", "tomato"), (1, "inhibitory", "cyan")],
            )
            .build()
        )
    """

    def __init__(self) -> None:
        # UI control lists — order within each group is insertion order
        self._color_controls: list[tuple[str, str]] = []  # [(label, color_str)]
        self._slider_configs: list[dict] = []  # slider param dicts
        self._checkbox_configs: list[tuple[str, bool]] = []  # [(name, default)]

        # Main body configuration
        self._point_size_config: Optional[dict] = None
        self._border_width: Optional[float] = None
        self._opacity_name: Optional[str] = None
        self._highlight_config: Optional[dict] = None
        self._categorical_config: Optional[dict] = None

        # Populated when string labels are auto-mapped to integers
        self._label_map: Optional[dict[str, int]] = None

        # Continuous colormap configuration
        self._continuous_config: Optional[dict] = None

    # ------------------------------------------------------------------
    # Fluent configuration methods
    # ------------------------------------------------------------------

    def opacity(
        self,
        default: float = 1.0,
        name: str = "opacity",
        min: float = 0.0,
        max: float = 1.0,
    ) -> Self:
        """Add a global opacity slider UI control.

        Parameters
        ----------
        default : float
            Default opacity (0–1). Default is 1.0.
        name : str
            GLSL variable name for the slider. Default is ``'opacity'``.
        min : float
            Minimum slider value. Default is 0.0.
        max : float
            Maximum slider value. Default is 1.0.
        """
        self._opacity_name = name
        self._slider_configs.append(
            {"name": name, "type": "float", "min": min, "max": max, "default": default}
        )
        return self

    def highlight(
        self,
        prop: str,
        value: int,
        highlighted_alpha: float = 1.0,
    ) -> Self:
        """Give points where an annotation property equals a value a fixed alpha.

        Points where ``prop_<prop>() == value`` receive *highlighted_alpha*;
        all other points receive the opacity slider value (if :meth:`opacity`
        was called) or ``1.0``.

        This is useful for e.g. keeping selected annotations fully opaque while
        fading the rest via the opacity slider.

        Parameters
        ----------
        prop : str
            Annotation property name (accessed as ``prop_<prop>()`` in GLSL).
        value : int
            Property value that triggers the highlight.
        highlighted_alpha : float
            Alpha assigned to highlighted points. Default is 1.0.
        """
        self._highlight_config = {
            "prop": prop,
            "value": int(value),
            "highlighted_alpha": float(highlighted_alpha),
        }
        return self

    def point_size(
        self,
        size: float = 5.0,
        *,
        prop: Optional[str] = None,
        scale: float = 1.0,
        slider: bool = False,
        slider_name: str = "pointSize",
        slider_min: float = 0.0,
        slider_max: float = 20.0,
    ) -> Self:
        """Set point marker size.

        Parameters
        ----------
        size : float
            Fixed size when *prop* is ``None`` and *slider* is ``False``.
            Default is 5.0.
        prop : str, optional
            Annotation property name to read size from. When set, the marker
            size in GLSL is ``float(prop_<prop>()) * scale``.
        scale : float
            Multiplier applied to the property value. Default is 1.0.
        slider : bool
            If ``True`` and *prop* is ``None``, expose a UI slider for size
            instead of using a fixed value. Default is ``False``.
        slider_name : str
            Name of the size slider control. Default is ``'pointSize'``.
        slider_min : float
            Minimum slider value. Default is 0.0.
        slider_max : float
            Maximum slider value. Default is 20.0.
        """
        self._point_size_config = {
            "size": size,
            "prop": prop,
            "scale": scale,
            "slider": slider,
            "slider_name": slider_name,
        }
        if slider and prop is None:
            self._slider_configs.append(
                {
                    "name": slider_name,
                    "type": "float",
                    "min": slider_min,
                    "max": slider_max,
                    "default": size,
                }
            )
        return self

    def border_width(self, width: float = 0.0) -> Self:
        """Set point marker border width.

        Parameters
        ----------
        width : float
            Border width in pixels. Default is 0.0.
        """
        self._border_width = float(width)
        return self

    # Neuroglancer built-in GLSL colormap functions (colormaps.ts)
    _COLORMAPS: dict[str, str] = {
        "jet": "colormapJet",
        "cubehelix": "colormapCubehelix",
    }

    def continuous_color(
        self,
        prop: str,
        colormap: str = "cubehelix",
        range_min: float = 0.0,
        range_max: float = 1.0,
        range_slider: bool = True,
    ) -> Self:
        """Color annotations by mapping a numeric property through a colormap.

        Uses Neuroglancer's built-in GLSL colormap functions, which take a
        value normalized to [0, 1] and return a ``vec3`` RGB color.

        Available colormaps (from ``neuroglancer/src/webgl/colormaps.ts``):

        - ``'cubehelix'`` — perceptually uniform, works well on dark backgrounds
        - ``'jet'`` — classic rainbow ramp (blue→cyan→green→yellow→red)

        Parameters
        ----------
        prop : str
            Annotation property name (accessed as ``prop_<prop>()`` in GLSL).
            May be a float or integer property; it is cast to ``float``
            automatically.
        colormap : str
            Name of the Neuroglancer built-in colormap. Default is
            ``'cubehelix'``.
        range_min : float
            Property value that maps to 0 (the low end of the colormap).
            Default is 0.0.
        range_max : float
            Property value that maps to 1 (the high end of the colormap).
            Default is 1.0.
        range_slider : bool
            If ``True``, expose ``rangeMin`` and ``rangeMax`` as slider UI
            controls so the range can be adjusted interactively. Default is
            ``True``.

        Raises
        ------
        ValueError
            If *colormap* is not a recognised Neuroglancer built-in, or if
            this method is called alongside :meth:`categorical_color`.
        """
        if colormap not in self._COLORMAPS:
            raise ValueError(
                f"Unknown colormap '{colormap}'. "
                f"Available: {list(self._COLORMAPS.keys())}"
            )
        if self._categorical_config is not None:
            raise ValueError(
                "continuous_color() and categorical_color() cannot both be set "
                "on the same AnnotationShaderBuilder."
            )

        self._continuous_config = {
            "prop": prop,
            "glsl_fn": self._COLORMAPS[colormap],
            "range_min": range_min,
            "range_max": range_max,
            "range_slider": range_slider,
        }

        if range_slider:
            # Use the same slider range for both controls, wide enough to cover
            # the requested range with some room on either side.
            span = abs(range_max - range_min) or 1.0
            slider_min = range_min - span
            slider_max = range_max + span
            self._slider_configs.append(
                {
                    "name": "rangeMin",
                    "type": "float",
                    "min": slider_min,
                    "max": slider_max,
                    "default": range_min,
                }
            )
            self._slider_configs.append(
                {
                    "name": "rangeMax",
                    "type": "float",
                    "min": slider_min,
                    "max": slider_max,
                    "default": range_max,
                }
            )
        return self

    def categorical_color(
        self,
        prop: str,
        categories: Union[
            list[str],
            dict[str, str],
            dict[int, tuple[str, str]],
            list[tuple[int, str, str]],
        ],
        palette: Optional[str] = None,
        default_visible: bool = True,
    ) -> Self:
        """Color annotations by a categorical uint annotation property.

        Generates per-category color picker and show/hide checkbox UI controls,
        and a cascading ``if / else if`` block in ``main()``.

        Multiple property values can share the same category *label* — they
        will share one color picker and one visibility checkbox, and their
        branch condition will use ``||``.

        When string labels are provided (without explicit integer keys), the
        builder automatically assigns integer IDs (0, 1, 2, …) and stores the
        mapping in :attr:`label_map` so you can encode your DataFrame column.

        Parameters
        ----------
        prop : str
            Annotation property name (accessed as ``prop_<prop>()`` in GLSL).
        categories : list or dict
            Supported formats:

            **String-label formats** (integer IDs auto-assigned, :attr:`label_map`
            populated):

            - ``list[str]``: ordered unique label strings; colors drawn from
              *palette* automatically.
              Example: ``["excitatory", "inhibitory", "unknown"]``
            - ``dict[str, str]``: ``{label: color}`` mapping; integer IDs
              assigned in dict order.
              Example: ``{"excitatory": "tomato", "inhibitory": "cyan"}``

            **Explicit integer-key formats** (existing behaviour, :attr:`label_map`
            is ``None``):

            - ``dict[int, tuple[str, str]]``: ``{value: (label, color)}``
            - ``list[tuple[int, str, str]]``: ``[(value, label, color), …]``

            In all formats *label* becomes the GLSL ``vec3`` variable name and
            the ``show_<label>`` checkbox name. *color* is any CSS color name,
            hex string ``"#rrggbb"``, or RGB tuple in [0, 1] or [0, 255].

        palette : str, optional
            palettable colormap name (e.g. ``'Set1'``, ``'Tableau_10'``) used
            to auto-assign colors when ``categories`` is a ``list[str]``.
            Ignored for all other input formats. Defaults to Tableau 10.
        default_visible : bool
            Default state for all ``show_<label>`` checkboxes. Default is
            ``True``.

        Raises
        ------
        ValueError
            If called more than once on the same builder instance.

        See Also
        --------
        label_map : The string→integer mapping produced for string-label inputs.
        """
        if self._categorical_config is not None:
            raise ValueError(
                "categorical_color() can only be called once per AnnotationShaderBuilder. "
                "Create a new builder for a different categorical property."
            )
        if self._continuous_config is not None:
            raise ValueError(
                "categorical_color() and continuous_color() cannot both be set "
                "on the same AnnotationShaderBuilder."
            )

        # Accept any iterable (e.g. pandas Series) by converting to list first
        if not isinstance(categories, (dict, list)):
            categories = list(categories)

        cat_list: list[tuple[int, str, str]]  # (int_value, label, color)
        label_map: Optional[dict[str, int]] = None

        if isinstance(categories, dict):
            first_key = next(iter(categories)) if categories else None
            if first_key is None or isinstance(first_key, str):
                # {label: color} format — auto-assign integer IDs
                unique_labels = list(categories.keys())
                label_map = {label: i for i, label in enumerate(unique_labels)}
                cat_list = [
                    (i, label, categories[label])
                    for i, label in enumerate(unique_labels)
                ]
            else:
                # {int: (label, color)} format — existing behaviour
                cat_list = [
                    (int(v), str(info[0]), info[1]) for v, info in categories.items()
                ]
        else:
            # list branch
            first = categories[0] if categories else None
            if first is None or isinstance(first, str):
                # list[str] — deduplicate preserving order, auto colors + IDs
                unique_labels: list[str] = list(dict.fromkeys(categories))
                colors = _palette_colors(len(unique_labels), palette)
                label_map = {label: i for i, label in enumerate(unique_labels)}
                cat_list = [
                    (i, label, color)
                    for i, (label, color) in enumerate(zip(unique_labels, colors))
                ]
            else:
                # list[tuple[int, str, str]] — existing behaviour
                cat_list = [
                    (int(v), str(label), color) for v, label, color in categories
                ]

        self._label_map = label_map

        # Build label → color and label → [values] mappings, preserving
        # insertion order (first appearance of each label).
        label_color: dict[str, str] = {}
        label_values: dict[str, list[int]] = {}
        label_order: list[str] = []

        for value, label, color in cat_list:
            if label not in label_color:
                label_color[label] = _normalize_color_str(color)
                label_values[label] = []
                label_order.append(label)
            label_values[label].append(value)

        # Register UI controls: colors first, then checkboxes (appended after
        # any sliders at build time because sliders are output between them).
        for label in label_order:
            self._color_controls.append((label, label_color[label]))
        for label in label_order:
            self._checkbox_configs.append((f"show_{label}", default_visible))

        self._categorical_config = {
            "prop": prop,
            "groups": [
                (label, label_color[label], label_values[label])
                for label in label_order
            ],
        }
        return self

    @property
    def label_map(self) -> Optional[dict[str, int]]:
        """String label → integer mapping for string-based category inputs.

        Populated when :meth:`categorical_color` is called with a ``list[str]``
        or ``dict[str, str]`` *categories* argument. ``None`` when explicit
        integer keys were provided.

        Use this to encode a string column in your DataFrame before passing it
        as an annotation property::

            builder = (
                AnnotationShaderBuilder()
                .categorical_color(
                    prop="cell_type",
                    categories=["excitatory", "inhibitory", "unknown"],
                )
            )
            df["cell_type_encoded"] = df["cell_type"].map(builder.label_map)

        Returns
        -------
        dict[str, int] or None
        """
        return self._label_map

    # ------------------------------------------------------------------
    # Internal GLSL generation helpers
    # ------------------------------------------------------------------

    def _alpha_line(self) -> Optional[str]:
        """Return the ``float alpha = ...;`` line, or None if no alpha control."""
        opacity = self._opacity_name
        highlight = self._highlight_config

        if opacity is None and highlight is None:
            return None

        base = opacity if opacity is not None else "1.0"

        if highlight is not None:
            ha = _format_float(highlight["highlighted_alpha"])
            return (
                f"  float alpha = "
                f"(prop_{highlight['prop']}() == uint({highlight['value']})) "
                f"? {ha} : {base};"
            )
        return f"  float alpha = {base};"

    def _point_size_line(self) -> Optional[str]:
        """Return the ``setPointMarkerSize(...)`` line, or None."""
        if self._point_size_config is None:
            return None
        cfg = self._point_size_config
        if cfg["prop"] is not None:
            expr = f"float(prop_{cfg['prop']}())"
            if cfg["scale"] != 1.0:
                expr += f"*{_format_float(cfg['scale'])}"
            return f"  setPointMarkerSize({expr});"
        elif cfg["slider"]:
            return f"  setPointMarkerSize({cfg['slider_name']});"
        else:
            return f"  setPointMarkerSize({_format_float(cfg['size'])});"

    def _continuous_block(self, has_alpha: bool) -> Optional[str]:
        """Return the continuous colormap GLSL lines, or None."""
        if self._continuous_config is None:
            return None
        cfg = self._continuous_config
        alpha_val = "alpha" if has_alpha else "1.0"
        if cfg["range_slider"]:
            t_expr = (
                f"clamp((float(prop_{cfg['prop']}()) - rangeMin) "
                f"/ (rangeMax - rangeMin), 0.0, 1.0)"
            )
        else:
            mn = _format_float(cfg["range_min"])
            mx = _format_float(cfg["range_max"])
            t_expr = (
                f"clamp((float(prop_{cfg['prop']}()) - {mn}) / ({mx} - {mn}), 0.0, 1.0)"
            )
        return (
            f"  float t = {t_expr};\n"
            f"  setColor(vec4({cfg['glsl_fn']}(t), {alpha_val}));"
        )

    def _categorical_block(self, has_alpha: bool) -> Optional[str]:
        """Return the if/else if color-dispatch block, or None."""
        if self._categorical_config is None:
            return None

        prop = self._categorical_config["prop"]
        groups = self._categorical_config["groups"]  # [(label, color, [values])]
        alpha_val = "alpha" if has_alpha else "1.0"

        lines: list[str] = []
        for i, (label, _color, values) in enumerate(groups):
            conds = [f"prop_{prop}() == uint({v})" for v in sorted(values)]
            condition = " || ".join(conds)
            keyword = "if" if i == 0 else "} else if"
            lines.append(f"  {keyword} ({condition}) {{")
            lines.append(f"    if (!show_{label}) {{ discard; }}")
            lines.append(f"    setColor(vec4({label}, {alpha_val}));")
        lines.append("  }")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> str:
        """Generate the complete GLSL annotation shader string.

        UI controls are emitted in this order: color pickers, sliders,
        checkboxes. Within the ``void main()`` body: default color, point size,
        alpha expression, categorical color block, border width.

        Returns
        -------
        str
            GLSL shader code ready to pass to
            :meth:`~nglui.statebuilder.AnnotationLayer.add_shader`.
        """
        parts: list[str] = []

        # UI controls: colors → sliders → checkboxes
        for label, color in self._color_controls:
            parts.append(f'#uicontrol vec3 {label} color(default="{color}")')
        for cfg in self._slider_configs:
            mn = _format_float(cfg["min"])
            mx = _format_float(cfg["max"])
            df = _format_float(cfg["default"])
            parts.append(
                f"#uicontrol {cfg['type']} {cfg['name']} "
                f"slider(min={mn}, max={mx}, default={df})"
            )
        for name, default in self._checkbox_configs:
            parts.append(
                f"#uicontrol bool {name} checkbox(default={str(default).lower()})"
            )

        # void main()
        parts.append("")
        parts.append("void main() {")
        parts.append("  setColor(defaultColor());")

        size_line = self._point_size_line()
        if size_line:
            parts.append(size_line)

        has_alpha = self._opacity_name is not None or self._highlight_config is not None
        alpha_line = self._alpha_line()
        if alpha_line:
            parts.append(alpha_line)

        cat_block = self._categorical_block(has_alpha)
        if cat_block:
            parts.append(cat_block)

        cont_block = self._continuous_block(has_alpha)
        if cont_block:
            parts.append(cont_block)

        if self._border_width is not None:
            parts.append(
                f"  setPointMarkerBorderWidth({_format_float(self._border_width)});"
            )

        parts.append("}")

        return "\n".join(parts)

    def __str__(self) -> str:
        return self.build()

    def __repr__(self) -> str:
        prop = self._categorical_config["prop"] if self._categorical_config else None
        n = len(self._categorical_config["groups"]) if self._categorical_config else 0
        return f"AnnotationShaderBuilder(prop={prop!r}, n_categories={n})"


# ---------------------------------------------------------------------------
# SkeletonShaderBuilder
# ---------------------------------------------------------------------------


class SkeletonShaderBuilder:
    """Build Neuroglancer skeleton layer GLSL shaders programmatically.

    Skeleton shaders colour per-vertex float attributes stored as ``vCustom1``,
    ``vCustom2``, … in the order they appear in the precomputed skeleton info.
    The builder maps attribute *names* to those indices automatically.

    Pass the ordered attribute name list at construction — conveniently the
    same list as :attr:`~nglui.skeletons.SkeletonManager.vertex_attribute_names`::

        builder = SkeletonShaderBuilder(["compartment", "distance"])

    or declare attributes individually::

        builder = SkeletonShaderBuilder()
        builder.vertex_attribute("compartment", index=1)

    All methods return ``self`` for chaining. Call :meth:`build` to get the
    GLSL string.

    Examples
    --------
    Segment colour with desaturated non-axon compartments (reproduces the
    built-in ``skeleton_compartments`` shader)::

        shader = (
            SkeletonShaderBuilder(["compartment"])
            .use_segment_color()
            .desaturate(attr="compartment", reference_value=2.0)
            .build()
        )

    Per-compartment fixed colours::

        shader = (
            SkeletonShaderBuilder(["compartment"])
            .categorical_color(
                attr="compartment",
                categories={1: ("axon", "white"), 2: ("dendrite", "cyan"), 3: ("soma", "yellow")},
            )
            .build()
        )

    Continuous colormap over a float property::

        shader = (
            SkeletonShaderBuilder(["distance_from_soma"])
            .continuous_color(attr="distance_from_soma", colormap="cubehelix",
                              range_min=0, range_max=500)
            .build()
        )
    """

    _COLORMAPS: dict[str, str] = {
        "jet": "colormapJet",
        "cubehelix": "colormapCubehelix",
    }

    def __init__(self, vertex_attributes: Optional[list[str]] = None) -> None:
        """
        Parameters
        ----------
        vertex_attributes : list[str], optional
            Ordered list of vertex attribute names. Position in the list
            determines the ``vCustomN`` index (1-based). Equivalent to passing
            :attr:`~nglui.skeletons.SkeletonManager.vertex_attribute_names`.
        """
        # name → vCustomN index (1-based)
        self._attributes: dict[str, int] = {}
        if vertex_attributes:
            for i, name in enumerate(vertex_attributes, start=1):
                self._attributes[name] = i

        self._color_controls: list[tuple[str, str]] = []
        self._slider_configs: list[dict] = []
        self._checkbox_configs: list[tuple[str, bool]] = []

        self._segment_color: bool = False
        self._desaturate_config: Optional[dict] = None
        self._categorical_config: Optional[dict] = None
        self._continuous_config: Optional[dict] = None
        self._label_map: Optional[dict[str, int]] = None

    # ------------------------------------------------------------------
    # Attribute declaration
    # ------------------------------------------------------------------

    def vertex_attribute(self, name: str, index: int) -> Self:
        """Declare a vertex attribute by name and ``vCustomN`` index.

        Only needed when the attribute list was not passed to the constructor.

        Parameters
        ----------
        name : str
            Attribute name used in the rest of the builder.
        index : int
            The ``vCustomN`` index (1-based) as defined in the precomputed
            skeleton info.
        """
        self._attributes[name] = index
        return self

    def _require_attr(self, name: str) -> int:
        """Return the vCustomN index for *name*, raising if unknown."""
        if name not in self._attributes:
            raise ValueError(
                f"Vertex attribute '{name}' not declared. "
                "Pass it to the constructor or call vertex_attribute()."
            )
        return self._attributes[name]

    # ------------------------------------------------------------------
    # Segment-colour mode
    # ------------------------------------------------------------------

    def use_segment_color(self) -> Self:
        """Use each segment's assigned colour as the base emitted colour.

        May be combined with :meth:`desaturate` to recolour non-reference
        compartments while keeping the segment colour for the reference
        compartment.
        """
        self._segment_color = True
        return self

    def desaturate(
        self,
        attr: str,
        reference_value: float,
        saturation_scale: float = 0.5,
    ) -> Self:
        """Desaturate vertices whose attribute value differs from *reference_value*.

        Requires :meth:`use_segment_color`. Vertices where
        ``attr == reference_value`` emit the full segment colour; all others
        have their HSL saturation multiplied by *saturation_scale*.

        Parameters
        ----------
        attr : str
            Vertex attribute name to branch on.
        reference_value : float
            The attribute value that gets the full segment colour.
        saturation_scale : float
            Factor applied to HSL saturation for non-reference vertices.
            0.0 = greyscale, 1.0 = no change. Default is 0.5.
        """
        if not self._segment_color:
            raise ValueError(
                "desaturate() requires use_segment_color() to be called first."
            )
        self._require_attr(attr)
        self._desaturate_config = {
            "attr": attr,
            "reference_value": float(reference_value),
            "saturation_scale": float(saturation_scale),
        }
        return self

    # ------------------------------------------------------------------
    # Categorical and continuous colour
    # ------------------------------------------------------------------

    def categorical_color(
        self,
        attr: str,
        categories: Union[
            list[str],
            dict[str, str],
            dict[float, tuple[str, str]],
            list[tuple[float, str, str]],
        ],
        palette: Optional[str] = None,
        default_visible: bool = True,
    ) -> Self:
        """Colour skeleton vertices by discrete float attribute values.

        Generates per-category colour picker and show/hide checkbox UI controls
        and a cascading ``if / else if`` block in ``main()``.

        String-label inputs (``list[str]`` or ``dict[str, str]``) auto-assign
        float values 0.0, 1.0, 2.0, … and populate :attr:`label_map`.

        Parameters
        ----------
        attr : str
            Declared vertex attribute name.
        categories : list or dict
            Supported formats — same as :meth:`AnnotationShaderBuilder.categorical_color`
            except that explicit keys are ``float`` rather than ``int``:

            - ``list[str]``: label strings; colors from *palette*.
            - ``dict[str, str]``: ``{label: color}``.
            - ``dict[float, tuple[str, str]]``: ``{value: (label, color)}``.
            - ``list[tuple[float, str, str]]``: ``[(value, label, color)]``.
        palette : str, optional
            palettable colormap for auto-color assignment. Defaults to Bold_10.
        default_visible : bool
            Default for show/hide checkboxes. Default is ``True``.

        Raises
        ------
        ValueError
            If called alongside :meth:`continuous_color` or more than once.
        """
        if self._categorical_config is not None:
            raise ValueError("categorical_color() can only be called once.")
        if self._continuous_config is not None:
            raise ValueError(
                "categorical_color() and continuous_color() cannot both be set."
            )
        self._require_attr(attr)

        if not isinstance(categories, (dict, list)):
            categories = list(categories)

        cat_list: list[tuple[float, str, str]]
        label_map: Optional[dict[str, int]] = None

        if isinstance(categories, dict):
            first_key = next(iter(categories)) if categories else None
            if first_key is None or isinstance(first_key, str):
                unique_labels = list(categories.keys())
                label_map = {label: i for i, label in enumerate(unique_labels)}
                cat_list = [
                    (float(i), label, categories[label])
                    for i, label in enumerate(unique_labels)
                ]
            else:
                cat_list = [
                    (float(v), str(info[0]), info[1]) for v, info in categories.items()
                ]
        else:
            first = categories[0] if categories else None
            if first is None or isinstance(first, str):
                unique_labels = list(dict.fromkeys(categories))
                colors = _palette_colors(len(unique_labels), palette)
                label_map = {label: i for i, label in enumerate(unique_labels)}
                cat_list = [
                    (float(i), label, color)
                    for i, (label, color) in enumerate(zip(unique_labels, colors))
                ]
            else:
                cat_list = [
                    (float(v), str(label), color) for v, label, color in categories
                ]

        self._label_map = label_map

        label_color: dict[str, str] = {}
        label_values: dict[str, list[float]] = {}
        label_order: list[str] = []

        for value, label, color in cat_list:
            if label not in label_color:
                label_color[label] = _normalize_color_str(color)
                label_values[label] = []
                label_order.append(label)
            label_values[label].append(value)

        for label in label_order:
            self._color_controls.append((label, label_color[label]))
        for label in label_order:
            self._checkbox_configs.append((f"show_{label}", default_visible))

        self._categorical_config = {
            "attr": attr,
            "groups": [
                (label, label_color[label], label_values[label])
                for label in label_order
            ],
        }
        return self

    def continuous_color(
        self,
        attr: str,
        colormap: str = "cubehelix",
        range_min: float = 0.0,
        range_max: float = 1.0,
        range_slider: bool = True,
    ) -> Self:
        """Colour skeleton vertices by mapping a float attribute through a colormap.

        Uses the same Neuroglancer built-in GLSL colormaps as
        :meth:`AnnotationShaderBuilder.continuous_color`.

        Parameters
        ----------
        attr : str
            Declared vertex attribute name.
        colormap : str
            ``'cubehelix'`` (default) or ``'jet'``.
        range_min : float
            Attribute value mapped to 0 (low end of colormap).
        range_max : float
            Attribute value mapped to 1 (high end of colormap).
        range_slider : bool
            Expose range sliders in the Neuroglancer UI. Default is ``True``.

        Raises
        ------
        ValueError
            If *colormap* is unknown, attribute is undeclared, or called
            alongside :meth:`categorical_color`.
        """
        if colormap not in self._COLORMAPS:
            raise ValueError(
                f"Unknown colormap '{colormap}'. "
                f"Available: {list(self._COLORMAPS.keys())}"
            )
        if self._categorical_config is not None:
            raise ValueError(
                "continuous_color() and categorical_color() cannot both be set."
            )
        self._require_attr(attr)

        self._continuous_config = {
            "attr": attr,
            "glsl_fn": self._COLORMAPS[colormap],
            "range_min": range_min,
            "range_max": range_max,
            "range_slider": range_slider,
        }

        if range_slider:
            span = abs(range_max - range_min) or 1.0
            slider_min = range_min - span
            slider_max = range_max + span
            self._slider_configs.append(
                {
                    "name": "rangeMin",
                    "type": "float",
                    "min": slider_min,
                    "max": slider_max,
                    "default": range_min,
                }
            )
            self._slider_configs.append(
                {
                    "name": "rangeMax",
                    "type": "float",
                    "min": slider_min,
                    "max": slider_max,
                    "default": range_max,
                }
            )
        return self

    @property
    def label_map(self) -> Optional[dict[str, int]]:
        """String label → integer mapping when string-label categories were used.

        The integer values correspond to the float attribute values assigned in
        the shader (0, 1, 2, …). Use this to encode a string column before
        storing it as a skeleton vertex attribute.

        Returns ``None`` when explicit numeric keys were provided.
        """
        return self._label_map

    # ------------------------------------------------------------------
    # Internal GLSL generation
    # ------------------------------------------------------------------

    def _attr_decls(self) -> list[str]:
        """Return float declarations for all referenced attributes."""
        used: set[str] = set()
        if self._desaturate_config:
            used.add(self._desaturate_config["attr"])
        if self._categorical_config:
            used.add(self._categorical_config["attr"])
        if self._continuous_config:
            used.add(self._continuous_config["attr"])
        return [
            f"  float {name} = vCustom{self._attributes[name]};"
            for name in self._attributes
            if name in used
        ]

    def _segment_color_block(self) -> Optional[str]:
        if not self._segment_color:
            return None
        if self._desaturate_config is None:
            return "  emitRGB(segmentColor().rgb);"

        cfg = self._desaturate_config
        ref = _format_float(cfg["reference_value"])
        sat = _format_float(cfg["saturation_scale"])
        return (
            f"  vec4 uColor = segmentColor();\n"
            f"  vec3 hsl = rgbToHsl(uColor.rgb);\n"
            f"  if ({cfg['attr']} == {ref}) {{\n"
            f"    emitRGB(uColor.rgb);\n"
            f"  }} else {{\n"
            f"    hsl.y *= {sat};\n"
            f"    emitRGB(hslToRgb(hsl));\n"
            f"  }}"
        )

    def _categorical_block(self) -> Optional[str]:
        if self._categorical_config is None:
            return None
        attr = self._categorical_config["attr"]
        groups = self._categorical_config["groups"]

        lines: list[str] = []
        for i, (label, _color, values) in enumerate(groups):
            conds = [f"{attr} == {_format_float(v)}" for v in sorted(values)]
            condition = " || ".join(conds)
            keyword = "if" if i == 0 else "} else if"
            lines.append(f"  {keyword} ({condition}) {{")
            lines.append(f"    if (!show_{label}) {{ discard; }}")
            lines.append(f"    emitRGB({label});")
        lines.append("  }")
        return "\n".join(lines)

    def _continuous_block(self) -> Optional[str]:
        if self._continuous_config is None:
            return None
        cfg = self._continuous_config
        if cfg["range_slider"]:
            t_expr = (
                f"clamp(({cfg['attr']} - rangeMin) "
                f"/ (rangeMax - rangeMin), 0.0, 1.0)"
            )
        else:
            mn = _format_float(cfg["range_min"])
            mx = _format_float(cfg["range_max"])
            t_expr = f"clamp(({cfg['attr']} - {mn}) / ({mx} - {mn}), 0.0, 1.0)"
        return f"  float t = {t_expr};\n" f"  emitRGB({cfg['glsl_fn']}(t));"

    def _needs_hsl(self) -> bool:
        return self._desaturate_config is not None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> str:
        """Generate the complete GLSL skeleton shader string.

        Returns
        -------
        str
            GLSL shader code ready to pass to
            :meth:`~nglui.statebuilder.SegmentationLayer.add_shader`.
        """
        parts: list[str] = []

        # Inline HSL helpers only when desaturation is used
        if self._needs_hsl():
            parts.append(hue_to_rgb.code.strip())
            parts.append(rgb_to_hsl.code.strip())
            parts.append(hsl_to_rgb.code.strip())
            parts.append("")

        # UI controls: colors → sliders → checkboxes
        for label, color in self._color_controls:
            parts.append(f'#uicontrol vec3 {label} color(default="{color}")')
        for cfg in self._slider_configs:
            mn = _format_float(cfg["min"])
            mx = _format_float(cfg["max"])
            df = _format_float(cfg["default"])
            parts.append(
                f"#uicontrol {cfg['type']} {cfg['name']} "
                f"slider(min={mn}, max={mx}, default={df})"
            )
        for name, default in self._checkbox_configs:
            parts.append(
                f"#uicontrol bool {name} checkbox(default={str(default).lower()})"
            )

        # void main()
        if parts:
            parts.append("")
        parts.append("void main() {")

        decls = self._attr_decls()
        if decls:
            parts.extend(decls)

        seg_block = self._segment_color_block()
        if seg_block:
            parts.append(seg_block)

        cat_block = self._categorical_block()
        if cat_block:
            parts.append(cat_block)

        cont_block = self._continuous_block()
        if cont_block:
            parts.append(cont_block)

        parts.append("}")
        return "\n".join(parts)

    def __str__(self) -> str:
        return self.build()

    def __repr__(self) -> str:
        attrs = list(self._attributes.keys())
        return f"SkeletonShaderBuilder(vertex_attributes={attrs!r})"


# ---------------------------------------------------------------------------
# Default shader map (referenced by layer defaults)
# ---------------------------------------------------------------------------

DEFAULT_SHADER_MAP = {
    "skeleton_compartments": simple_compartment_skeleton_shader,
    "points": simple_point_shader(),
    "basic": basic_shader,
}
