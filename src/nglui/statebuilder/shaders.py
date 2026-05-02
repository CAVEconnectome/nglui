from __future__ import annotations

import re
import warnings
from collections import namedtuple
from itertools import cycle, islice
from typing import Optional, Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import palettable
import pyperclip
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
    """Format a float for GLSL: no scientific notation, always has a decimal point.

    GLSL ES 1.0 (used by WebGL 1) does not accept scientific notation in
    float literals, and even strict GLSL ES 3.0 parsers in some browsers
    reject it. ``str(1e-7)`` would normally return ``'1e-07'``, so we
    explicitly avoid that.
    """
    import math

    if not math.isfinite(f):
        raise ValueError(f"Cannot format non-finite float {f!r} as a GLSL literal.")
    if f == int(f):
        return f"{int(f)}.0"
    s = repr(f)
    if "e" not in s and "E" not in s:
        return s
    # Fall back to a wide fixed-point representation, then trim padding zeros.
    s = f"{f:.20f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return s


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
        self._invlerp_configs: list[dict] = []  # invlerp control dicts
        self._checkbox_configs: list[tuple[str, bool]] = []  # [(name, default)]

        # Tracks every #uicontrol identifier the builder has emitted. NG's
        # shader parser puts all control kinds (slider, color, checkbox,
        # invlerp) into a single namespace, so we have to detect collisions
        # across kinds — not just within kind.
        self._used_names: set[str] = set()

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

    def _claim_name(self, name: str, *, kind: str) -> None:
        """Register a control identifier or raise if it's already in use.

        *kind* is purely for the error message ("slider", "color", etc).
        """
        if name in self._used_names:
            raise ValueError(
                f"Control name {name!r} ({kind}) collides with an existing "
                "#uicontrol declaration on this builder."
            )
        self._used_names.add(name)

    def _release_name(self, name: str) -> None:
        """Drop a name from the used-set (for replace-style methods)."""
        self._used_names.discard(name)

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
        self._register_slider(
            name=name, type="float", min=min, max=max, default=default
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
            Property value that triggers the highlight. Must be non-negative
            because annotation properties are compared as ``uint`` in the
            generated GLSL.
        highlighted_alpha : float
            Alpha assigned to highlighted points. Default is 1.0.
        """
        ivalue = int(value)
        if ivalue < 0:
            raise ValueError(
                f"highlight() value must be non-negative; got {value!r}. "
                "Annotation properties are compared as GLSL uint, where "
                "negative literals wrap to large positive numbers and won't "
                "match what you expect."
            )
        self._highlight_config = {
            "prop": prop,
            "value": ivalue,
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
        slider_mode: str = "linear",
        slider_name: Optional[str] = None,
        slider_min: float = 0.0,
        slider_max: float = 20.0,
        invlerp_range: tuple[float, float] = (0.0, 1.0),
        invlerp_clamp: bool = True,
    ) -> Self:
        """Set point marker size.

        Modes (depending on *prop*, *slider*, *slider_mode*):

        - ``prop=None, slider=False``: fixed *size*.
        - ``prop=None, slider=True``: a UI slider directly drives the size.
          The slider's default is *size* and its range is
          ``[slider_min, slider_max]``.
        - ``prop=X, slider=False``: marker size is
          ``float(prop_X()) * scale``.
        - ``prop=X, slider=True, slider_mode="linear"``: scale is exposed
          as an interactive slider; GLSL becomes
          ``float(prop_X()) * <slider_name>``.
        - ``prop=X, slider=True, slider_mode="invlerp"``: the property value
          is normalized to ``[0, 1]`` via Neuroglancer's built-in
          ``invlerp`` control (which provides its own range/window
          UI elements and histogram), then linearly mapped to
          ``[slider_min, slider_max]``. GLSL becomes
          ``slider_min + (slider_max - slider_min) * <slider_name>()``.

        Parameters
        ----------
        size : float
            Fixed size when *prop* is ``None`` and *slider* is ``False``.
            Also the slider default in pure-slider mode. Default is 5.0.
        prop : str, optional
            Annotation property name to read size from.
        scale : float
            Multiplier applied to the property value. Acts as the slider
            default in linear scale-slider mode. Default is 1.0.
        slider : bool
            If ``True``, expose a UI control — see modes above. Default is
            ``False``.
        slider_mode : {"linear", "invlerp"}
            How the scale slider is exposed when ``slider=True`` and *prop*
            is set. ``"linear"`` (default) is a plain float slider used as
            a multiplier. ``"invlerp"`` uses Neuroglancer's invlerp control
            with its own range/window UI. Ignored when *prop* is ``None``.
        slider_name : str, optional
            Name of the control. Defaults to ``'pointSize'`` in pure-size
            mode and ``'pointScale'`` otherwise.
        slider_min, slider_max : float
            Linear mode: range of the slider (default ``0..20``).
            Invlerp mode: output range that ``invlerp()`` ``[0, 1]`` is
            linearly mapped to (e.g. minimum and maximum pixel size).
        invlerp_range : tuple of float
            Initial property-value range for the invlerp control's
            ``range=[lo, hi]`` parameter. The user can adjust this
            interactively in the side panel. Default ``(0.0, 1.0)``.
        invlerp_clamp : bool
            Whether the invlerp output is clamped to ``[0, 1]``. Default
            ``True``.
        """
        if slider_mode not in ("linear", "invlerp"):
            raise ValueError(
                f"slider_mode must be 'linear' or 'invlerp', got {slider_mode!r}."
            )
        if slider_mode == "invlerp" and (not slider or prop is None):
            raise ValueError(
                "slider_mode='invlerp' requires both slider=True and a prop."
            )
        if slider_min > slider_max:
            raise ValueError(
                f"slider_min ({slider_min}) cannot exceed slider_max ({slider_max})."
            )

        if slider_name is None:
            slider_name = "pointScale" if prop is not None else "pointSize"

        # Drop any UI control previously registered by this method so
        # calling point_size() twice replaces rather than accumulates.
        prev = self._point_size_config
        if prev is not None and prev.get("slider"):
            prev_name = prev.get("slider_name")
            self._slider_configs = [
                s for s in self._slider_configs if s.get("name") != prev_name
            ]
            self._invlerp_configs = [
                s for s in self._invlerp_configs if s.get("name") != prev_name
            ]
            if prev_name is not None:
                self._release_name(prev_name)

        self._point_size_config = {
            "size": size,
            "prop": prop,
            "scale": scale,
            "slider": slider,
            "slider_mode": slider_mode,
            "slider_name": slider_name,
            "slider_min": slider_min,
            "slider_max": slider_max,
        }
        if slider:
            if slider_mode == "invlerp":
                self._claim_name(slider_name, kind="invlerp")
                self._invlerp_configs.append(
                    {
                        "name": slider_name,
                        "property": prop,
                        "range": tuple(invlerp_range),
                        "clamp": bool(invlerp_clamp),
                    }
                )
            else:
                self._register_slider(
                    name=slider_name,
                    type="float",
                    min=slider_min,
                    max=slider_max,
                    default=scale if prop is not None else size,
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
        if self._continuous_config is not None:
            raise ValueError(
                "continuous_color() can only be called once per "
                "AnnotationShaderBuilder. Create a new builder for a "
                "different continuous-color rule."
            )
        if self._categorical_config is not None:
            raise ValueError(
                "continuous_color() and categorical_color() cannot both be set "
                "on the same AnnotationShaderBuilder."
            )
        if range_min == range_max:
            raise ValueError(
                f"continuous_color() requires range_min != range_max; got both "
                f"= {range_min!r}. The colormap-domain transform would divide "
                "by zero."
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
            self._register_slider(
                name="rangeMin",
                type="float",
                min=slider_min,
                max=slider_max,
                default=range_min,
            )
            self._register_slider(
                name="rangeMax",
                type="float",
                min=slider_min,
                max=slider_max,
                default=range_max,
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
        with_show_checkboxes: bool = True,
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
        with_show_checkboxes : bool
            If ``True`` (default), per-label ``show_<label>`` visibility
            checkboxes are emitted alongside the color pickers. Set to
            ``False`` for properties with many categories where the
            individual show/hide toggles would clutter the side panel —
            colors will still update interactively, but visibility cannot
            be toggled per category.

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

        if not categories:
            raise ValueError(
                "categorical_color() requires a non-empty `categories` argument."
            )

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
                # {int: (label, color)} format. Validate the value-tuple shape
                # before extracting — silently dropping a 3rd element used to
                # mask user errors like passing (label, color, metadata).
                cat_list = []
                for v, info in categories.items():
                    if not isinstance(info, tuple) or len(info) != 2:
                        raise ValueError(
                            f"categorical_color() dict value for key {v!r} must "
                            f"be a 2-tuple of (label, color); got {info!r}."
                        )
                    cat_list.append((int(v), str(info[0]), info[1]))
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
                # list[tuple[int, str, str]] — validate triple shape.
                cat_list = []
                for entry in categories:
                    if not isinstance(entry, tuple) or len(entry) != 3:
                        raise ValueError(
                            "categorical_color() list entries must be 3-tuples "
                            f"of (value, label, color); got {entry!r}."
                        )
                    v, label, color = entry
                    cat_list.append((int(v), str(label), color))

        # Reject negative integer values: annotation properties are compared
        # as uint in the generated GLSL, so negative literals would silently
        # wrap to 0xFFFFFFFF-ish and fail to match anything sensible.
        for value, _, _ in cat_list:
            if value < 0:
                raise ValueError(
                    f"categorical_color() value must be non-negative; got "
                    f"{value!r}. Annotation properties are compared as GLSL "
                    "uint and negative literals wrap to large positive numbers."
                )

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

        # Sanitize each label twice:
        #   color_id — used as the vec3 control name and inside setColor(),
        #              must satisfy NG's [a-z][a-zA-Z0-9_]* rule.
        #   show_id  — used after the literal `show_` prefix; only needs
        #              non-alphanumeric replacement since `show_` already
        #              supplies the leading lowercase letter.
        color_id_for: dict[str, str] = {}
        show_id_for: dict[str, str] = {}
        for label in label_order:
            color_id_for[label] = self._sanitize_label(label)
            show_id_for[label] = self._sanitize_show_label(label)

        # Register UI controls: colors first, then checkboxes (appended after
        # any sliders at build time because sliders are output between them).
        # Detect collisions before mutating any state so the builder isn't
        # left half-configured if two distinct labels sanitize to the same id.
        color_names = [color_id_for[label] for label in label_order]
        checkbox_names = (
            [f"show_{show_id_for[label]}" for label in label_order]
            if with_show_checkboxes
            else []
        )
        new_names = color_names + checkbox_names
        if len(set(new_names)) != len(new_names):
            dupes = sorted({n for n in new_names if new_names.count(n) > 1})
            raise ValueError(
                "categorical_color() produced duplicate control names "
                f"{dupes!r} — distinct labels sanitized to the same identifier. "
                "Rename one of the colliding labels."
            )
        for name in new_names:
            self._claim_name(name, kind="categorical")
        for label in label_order:
            self._color_controls.append((color_id_for[label], label_color[label]))
        if with_show_checkboxes:
            for label in label_order:
                self._checkbox_configs.append(
                    (f"show_{show_id_for[label]}", default_visible)
                )

        self._categorical_config = {
            "prop": prop,
            "groups": [
                (
                    color_id_for[label],
                    show_id_for[label],
                    label_color[label],
                    label_values[label],
                )
                for label in label_order
            ],
            "with_show_checkboxes": with_show_checkboxes,
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

    def _register_slider(
        self, *, name: str, type: str, min: float, max: float, default: float
    ) -> None:
        """Validate and append a slider config.

        Raises
        ------
        ValueError
            If *min > max*, or if *default* is outside ``[min, max]``,
            or if *name* collides with another control on this builder.
        """
        if min > max:
            raise ValueError(f"Slider {name!r}: min ({min}) cannot exceed max ({max}).")
        if not (min <= default <= max):
            raise ValueError(
                f"Slider {name!r}: default ({default}) is outside "
                f"[min, max] = [{min}, {max}]."
            )
        self._claim_name(name, kind="slider")
        self._slider_configs.append(
            {"name": name, "type": type, "min": min, "max": max, "default": default}
        )

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
            if cfg["slider"] and cfg.get("slider_mode") == "invlerp":
                lo = _format_float(cfg["slider_min"])
                hi = _format_float(cfg["slider_max"])
                # invlerp returns [0,1]; remap to [slider_min, slider_max].
                if cfg["slider_min"] == 0:
                    expr = f"{cfg['slider_name']}()*{hi}"
                else:
                    expr = f"{lo} + ({hi} - {lo}) * {cfg['slider_name']}()"
                return f"  setPointMarkerSize({expr});"
            expr = f"float(prop_{cfg['prop']}())"
            if cfg["slider"]:
                expr += f"*{cfg['slider_name']}"
            elif cfg["scale"] != 1.0:
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
        groups = self._categorical_config["groups"]
        with_checkboxes = self._categorical_config.get("with_show_checkboxes", True)
        alpha_val = "alpha" if has_alpha else "1.0"

        lines: list[str] = []
        for i, (color_id, show_id, _color, values) in enumerate(groups):
            conds = [f"prop_{prop}() == uint({v})" for v in sorted(values)]
            condition = " || ".join(conds)
            keyword = "if" if i == 0 else "} else if"
            lines.append(f"  {keyword} ({condition}) {{")
            if with_checkboxes:
                lines.append(f"    if (!show_{show_id}) {{ discard; }}")
            lines.append(f"    setColor(vec4({color_id}, {alpha_val}));")
        lines.append("  }")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, to_clipboard: bool = False) -> str:
        """Generate the complete GLSL annotation shader string.

        Calling ``build()`` on a freshly constructed builder produces a minimal
        but valid Neuroglancer annotation shader (``setColor(defaultColor())``)
        so the layer renders sensibly without any explicit configuration.

        UI controls are emitted in this order: color pickers, sliders,
        checkboxes. Within the ``void main()`` body: default color, point size,
        alpha expression, categorical color block, border width.

        Parameters
        ----------
        to_clipboard : bool, optional
            If ``True``, also copy the shader source to the system clipboard
            via ``pyperclip``. Default is ``False``.

        Returns
        -------
        str
            GLSL shader code ready to pass to
            :meth:`~nglui.statebuilder.AnnotationLayer.add_shader`.
        """
        parts: list[str] = []

        # UI controls: colors → sliders → invlerps → checkboxes
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
        for cfg in self._invlerp_configs:
            lo = _format_float(cfg["range"][0])
            hi = _format_float(cfg["range"][1])
            clamp_str = "true" if cfg["clamp"] else "false"
            parts.append(
                f"#uicontrol invlerp {cfg['name']}("
                f'property="{cfg["property"]}", '
                f"range=[{lo}, {hi}], clamp={clamp_str})"
            )
        for name, default in self._checkbox_configs:
            parts.append(
                f"#uicontrol bool {name} checkbox(default={str(default).lower()})"
            )

        # void main() — separate UI controls from body with a blank line
        # only when there are any controls to separate.
        if parts:
            parts.append("")
        parts.append("void main() {")

        # Alpha is declared first so the fallthrough setColor below can apply
        # it to defaultColor() — otherwise points whose property doesn't match
        # any categorical branch would be drawn at full opacity, ignoring the
        # configured opacity slider.
        has_alpha = self._opacity_name is not None or self._highlight_config is not None
        alpha_line = self._alpha_line()
        if alpha_line:
            parts.append(alpha_line)

        if has_alpha:
            parts.append("  setColor(vec4(defaultColor().rgb, alpha));")
        else:
            parts.append("  setColor(defaultColor());")

        size_line = self._point_size_line()
        if size_line:
            parts.append(size_line)

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

        shader = "\n".join(parts)
        if to_clipboard:
            pyperclip.copy(shader)
        return shader

    def __str__(self) -> str:
        return self.build()

    def __repr__(self) -> str:
        prop = self._categorical_config["prop"] if self._categorical_config else None
        n = len(self._categorical_config["groups"]) if self._categorical_config else 0
        return f"AnnotationShaderBuilder(prop={prop!r}, n_categories={n})"

    # ------------------------------------------------------------------
    # Auto-configuration from an annotation info file
    # ------------------------------------------------------------------

    _INTEGER_TYPES = frozenset({"uint8", "uint16", "uint32", "int8", "int16", "int32"})
    _FLOAT_TYPES = frozenset({"float32"})
    _COLOR_TYPES = frozenset({"rgb", "rgba"})
    _SIZE_PROP_NAMES = ("size", "radius", "diameter")

    DEFAULT_CATEGORICAL_FALLBACK_COLOR: str = "#808080"

    @classmethod
    def from_info(
        cls,
        info: dict,
        *,
        color_prop: Optional[str] = None,
        size_prop: Optional[str] = None,
        size_scale: Optional[float] = None,
        size_slider_min: float = 0.0,
        size_slider_max: float = 20.0,
        size_slider_mode: str = "invlerp",
        size_invlerp_range: tuple[float, float] = (0.0, 1.0),
        opacity: float = 0.8,
        palette: Union[str, dict[str, str]] = "Tableau_10",
        default_color: Optional[str] = None,
        colormap: str = "cubehelix",
        show_checkboxes: bool = True,
    ) -> Self:
        """Configure a builder from a precomputed annotation `info` file.

        Inspects ``info["properties"]`` and applies sensible defaults so that
        a typical annotation source gets a reasonable shader without manual
        configuration. The returned builder is **not yet built** — chain
        further configuration before calling :meth:`build`.

        Mapping rules:

        - **Color**: if *color_prop* is given, that property is used. Otherwise
          the first integer-typed property with ``enum_values`` and
          ``enum_labels`` becomes a categorical color. Otherwise the first
          ``float32`` property becomes a continuous colormap.
        - **Size**: if *size_prop* is given, that property is used. Otherwise,
          a property named (case-insensitive) ``size`` / ``radius`` /
          ``diameter`` is used. Otherwise no point-size configuration is
          added.
        - **Opacity**: an opacity slider is always added.
        - ``rgb`` / ``rgba`` properties and integer properties without
          ``enum_values`` are skipped (with a warning) — the builder has no
          direct path for those today.

        Parameters
        ----------
        info : dict
            Parsed precomputed annotation info file. Must have a
            ``"properties"`` list per the Neuroglancer schema.
        color_prop : str, optional
            Force a specific property to drive color. If it is integer-typed
            with ``enum_values``, ``categorical_color`` is used; if
            ``float32``, ``continuous_color`` is used.
        size_prop : str, optional
            Force a specific property to drive marker size.
        size_scale : float, optional
            Default value of the size scale slider (used as the multiplier
            on top of the property value). When ``None`` (default), it is
            ``1.0`` if that lies within the slider range, otherwise the
            midpoint of ``[size_slider_min, size_slider_max]``. An explicit
            value is validated against the slider range.
        size_slider_min, size_slider_max : float
            Range of the size scale slider. Default ``[0.0, 20.0]``.
            In ``size_slider_mode="invlerp"``, these are the output pixel
            sizes that the invlerp's ``[0, 1]`` is mapped to.
        size_slider_mode : {"linear", "invlerp"}
            Linear (default) is a plain multiplier slider. ``"invlerp"`` uses
            Neuroglancer's built-in invlerp control, which provides its own
            range/window UI and histogram; the property is normalized to
            ``[0, 1]`` and mapped to ``[size_slider_min, size_slider_max]``
            pixels.
        size_invlerp_range : tuple of float
            Initial property-value range for the invlerp control's
            ``range=[lo, hi]``. The user can adjust this live in the side
            panel. Default ``(0.0, 1.0)``. Ignored in linear mode.
        opacity : float
            Default opacity slider value. Default 0.8.
        palette : str or dict[str, str]
            Either a palettable colormap name (e.g. ``"Tableau_10"``) used to
            auto-assign categorical colors, *or* a ``{label: color}`` dict
            mapping original (un-sanitized) ``enum_labels`` entries to
            specific color strings (any form ``parse_color_rgb`` accepts:
            CSS name, ``"#rrggbb"``, RGB tuple). Labels not present in the
            dict fall back to *default_color*. Default ``"Tableau_10"``.
        default_color : str, optional
            Fallback color used for categorical labels missing from a
            *palette* dict. Default is medium gray (``"#808080"``).
            Ignored when *palette* is a string.
        colormap : str
            Neuroglancer built-in colormap for continuous color. Default
            ``"cubehelix"``.
        show_checkboxes : bool
            If ``True`` (default), per-category ``show_<label>`` visibility
            checkboxes are emitted alongside the color pickers. Set to
            ``False`` for properties with many categories where the
            checkboxes would clutter the side panel.

        Returns
        -------
        AnnotationShaderBuilder
            Configured (but un-built) builder.

        See Also
        --------
        auto_annotation_shader : module-level helper that fetches the info
            file by URL and returns the built shader string in one call.
        """
        if default_color is None:
            default_color = cls.DEFAULT_CATEGORICAL_FALLBACK_COLOR

        # Linear mode: pick a sensible scale-slider default when the caller
        # didn't supply one — 1.0 if it fits, otherwise the midpoint. Invlerp
        # mode doesn't use size_scale (its UI exposes the property range
        # directly).
        if size_scale is None and size_slider_mode != "invlerp":
            if size_slider_min <= 1.0 <= size_slider_max:
                size_scale = 1.0
            else:
                size_scale = (size_slider_min + size_slider_max) / 2
        properties = info.get("properties") or []
        by_id = {p["id"]: p for p in properties if isinstance(p, dict) and "id" in p}

        builder = cls()
        builder.opacity(default=opacity)

        # ---- Size -----------------------------------------------------
        chosen_size = None
        if size_prop is not None:
            if size_prop not in by_id:
                raise ValueError(
                    f"size_prop={size_prop!r} not found in info properties."
                )
            chosen_size = size_prop
        else:
            for prop in properties:
                if prop.get("id", "").lower() in cls._SIZE_PROP_NAMES:
                    chosen_size = prop["id"]
                    break
        if chosen_size is not None:
            builder.point_size(
                prop=chosen_size,
                scale=size_scale if size_scale is not None else 1.0,
                slider=True,
                slider_mode=size_slider_mode,
                slider_min=size_slider_min,
                slider_max=size_slider_max,
                invlerp_range=size_invlerp_range,
            )

        # ---- Color ----------------------------------------------------
        color_kwargs = {
            "palette": palette,
            "default_color": default_color,
            "colormap": colormap,
            "with_show_checkboxes": show_checkboxes,
        }
        if color_prop is not None:
            if color_prop not in by_id:
                raise ValueError(
                    f"color_prop={color_prop!r} not found in info properties."
                )
            cls._apply_color_from_property(builder, by_id[color_prop], **color_kwargs)
        else:
            # Prefer first categorical (integer + enum_values + enum_labels);
            # else first float32.
            categorical = next(
                (
                    p
                    for p in properties
                    if p.get("type") in cls._INTEGER_TYPES
                    and p.get("enum_values")
                    and p.get("enum_labels")
                ),
                None,
            )
            if categorical is not None:
                cls._apply_color_from_property(builder, categorical, **color_kwargs)
            else:
                continuous = next(
                    (p for p in properties if p.get("type") in cls._FLOAT_TYPES),
                    None,
                )
                if continuous is not None:
                    cls._apply_color_from_property(builder, continuous, **color_kwargs)

        # ---- Warn about unsupported types ----------------------------
        for prop in properties:
            ptype = prop.get("type")
            pid = prop.get("id", "<unknown>")
            if ptype in cls._COLOR_TYPES:
                warnings.warn(
                    f"Property {pid!r} has type {ptype!r}; AnnotationShaderBuilder "
                    "does not yet support direct rgb/rgba properties — skipping.",
                    stacklevel=2,
                )
            elif ptype in cls._INTEGER_TYPES and not (
                prop.get("enum_values") and prop.get("enum_labels")
            ):
                if pid != chosen_size:
                    warnings.warn(
                        f"Integer property {pid!r} has no enum_values/enum_labels; "
                        "skipping (categorical color requires explicit category labels).",
                        stacklevel=2,
                    )

        return builder

    @classmethod
    def _apply_color_from_property(
        cls,
        builder: "AnnotationShaderBuilder",
        prop: dict,
        *,
        palette: Union[str, dict[str, str]],
        default_color: str,
        colormap: str,
        with_show_checkboxes: bool = True,
    ) -> None:
        """Configure either categorical or continuous color from a property dict."""
        ptype = prop.get("type")
        pid = prop["id"]
        if ptype in cls._INTEGER_TYPES:
            values = prop.get("enum_values")
            labels = prop.get("enum_labels")
            if not values or not labels:
                raise ValueError(
                    f"Property {pid!r} cannot drive categorical color without "
                    "enum_values and enum_labels."
                )
            if len(values) != len(labels):
                raise ValueError(
                    f"Property {pid!r} has mismatched enum_values "
                    f"({len(values)}) and enum_labels ({len(labels)})."
                )
            colors = cls._resolve_categorical_colors(labels, palette, default_color)
            # Pass ORIGINAL labels — categorical_color sanitizes appropriately
            # for the color control vs. the show_<X> checkbox.
            categories = {
                int(v): (str(label), color)
                for v, label, color in zip(values, labels, colors)
            }
            builder.categorical_color(
                prop=pid,
                categories=categories,
                with_show_checkboxes=with_show_checkboxes,
            )
        elif ptype in cls._FLOAT_TYPES:
            builder.continuous_color(prop=pid, colormap=colormap)
        else:
            raise ValueError(
                f"Property {pid!r} has type {ptype!r}, which cannot drive color."
            )

    @staticmethod
    def _resolve_categorical_colors(
        labels: list[str],
        palette: Union[str, dict[str, str]],
        default_color: str,
    ) -> list[str]:
        """Map enum labels to color strings.

        - If *palette* is a dict, look each original label up; missing labels
          get *default_color*.
        - If *palette* is a string, draw colors from the named palettable
          colormap (cycling as needed).
        """
        if isinstance(palette, dict):
            return [palette.get(label, default_color) for label in labels]
        return _palette_colors(len(labels), palette)

    @staticmethod
    def _sanitize_label(label: str) -> str:
        """Make an enum label safe for a Neuroglancer #uicontrol variable name.

        Neuroglancer's parser requires control names to match
        ``[a-z][a-zA-Z0-9_]*`` — must start with a lowercase ASCII letter.
        Non-alphanumeric characters are mapped to underscores; an uppercase
        leading letter is lowercased; anything else invalid is prefixed
        with ``c_``.
        """
        cleaned = re.sub(r"[^A-Za-z0-9_]", "_", str(label))
        if not cleaned:
            return "c_"
        first = cleaned[0]
        if first.isascii() and first.isupper():
            cleaned = first.lower() + cleaned[1:]
        elif not (first.isascii() and first.islower()):
            cleaned = "c_" + cleaned
        return cleaned

    @staticmethod
    def _sanitize_show_label(label: str) -> str:
        """Make a label safe to use after a ``show_`` prefix.

        Less aggressive than :meth:`_sanitize_label`: since ``show_`` already
        supplies the leading lowercase letter required by NG, the rest of
        the identifier may keep its original casing. Only non-alphanumeric
        characters are replaced with underscores.
        """
        cleaned = re.sub(r"[^A-Za-z0-9_]", "_", str(label))
        return cleaned or "_"


def auto_annotation_shader(
    info_or_url: Union[dict, str],
    *,
    color_prop: Optional[str] = None,
    size_prop: Optional[str] = None,
    size_scale: Optional[float] = None,
    size_slider_min: float = 0.0,
    size_slider_max: float = 20.0,
    size_slider_mode: str = "invlerp",
    size_invlerp_range: tuple[float, float] = (0.0, 1.0),
    opacity: float = 0.8,
    palette: Union[str, dict[str, str]] = "Tableau_10",
    default_color: Optional[str] = None,
    colormap: str = "cubehelix",
    show_checkboxes: bool = True,
) -> str:
    """Build a default annotation shader from an info file or its URL.

    Convenience wrapper around :meth:`AnnotationShaderBuilder.from_info` that
    also accepts a precomputed annotation source URL — in which case the info
    file is fetched via :func:`nglui.parser.info.get_annotation_info`.

    Parameters
    ----------
    info_or_url : dict or str
        Either a parsed annotation info dict or a string URL pointing at the
        precomputed annotation source (without the trailing ``/info``).
    color_prop, size_prop, opacity, palette, default_color, colormap
        See :meth:`AnnotationShaderBuilder.from_info`.

    Returns
    -------
    str
        Built GLSL shader code.
    """
    if isinstance(info_or_url, str):
        # Local import to avoid a circular dependency at module load time.
        from ..parser.info import get_annotation_info

        info = get_annotation_info(info_or_url)
    else:
        info = info_or_url

    return AnnotationShaderBuilder.from_info(
        info,
        color_prop=color_prop,
        size_prop=size_prop,
        size_scale=size_scale,
        size_slider_min=size_slider_min,
        size_slider_max=size_slider_max,
        size_slider_mode=size_slider_mode,
        size_invlerp_range=size_invlerp_range,
        opacity=opacity,
        palette=palette,
        default_color=default_color,
        colormap=colormap,
        show_checkboxes=show_checkboxes,
    ).build()


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

        # Single namespace for all #uicontrol identifiers — see
        # AnnotationShaderBuilder._claim_name.
        self._used_names: set[str] = set()

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
            The ``vCustomN`` index — must be ``>= 1`` because Neuroglancer's
            vertex-attribute slots are 1-based (``vCustom1``, ``vCustom2``,
            …; ``vCustom0`` does not exist).
        """
        if index < 1:
            raise ValueError(
                f"vertex_attribute index must be >= 1 (vCustomN is 1-based in "
                f"Neuroglancer); got index={index}."
            )
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
        with_show_checkboxes: bool = True,
    ) -> Self:
        """Colour skeleton vertices by discrete float attribute values.

        Generates per-category colour picker and show/hide checkbox UI controls
        and a cascading ``if / else if`` block in ``main()``.

        String-label inputs (``list[str]`` or ``dict[str, str]``) auto-assign
        float values 0.0, 1.0, 2.0, … and populate :attr:`label_map`.

        Labels are sanitized for use as Neuroglancer ``#uicontrol`` identifiers:
        the color control uses a strict-mode name (lowercase first letter,
        non-alphanumerics → underscore), while the ``show_<label>`` checkbox
        preserves the original casing of the label (since the ``show_`` prefix
        already supplies the leading lowercase letter NG requires).

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
        with_show_checkboxes : bool
            If ``True`` (default), per-label ``show_<label>`` visibility
            checkboxes are emitted. Set to ``False`` for properties with
            many categories where the toggles would clutter the side panel.

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

        if not categories:
            raise ValueError(
                "categorical_color() requires a non-empty `categories` argument."
            )

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
                cat_list = []
                for v, info in categories.items():
                    if not isinstance(info, tuple) or len(info) != 2:
                        raise ValueError(
                            f"categorical_color() dict value for key {v!r} must "
                            f"be a 2-tuple of (label, color); got {info!r}."
                        )
                    cat_list.append((float(v), str(info[0]), info[1]))
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
                cat_list = []
                for entry in categories:
                    if not isinstance(entry, tuple) or len(entry) != 3:
                        raise ValueError(
                            "categorical_color() list entries must be 3-tuples "
                            f"of (value, label, color); got {entry!r}."
                        )
                    v, label, color = entry
                    cat_list.append((float(v), str(label), color))

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

        # Sanitize each label twice (see AnnotationShaderBuilder.categorical_color):
        #   color_id — strict NG name for the vec3 control and emitRGB().
        #   show_id  — preserves casing; safe after the literal `show_` prefix.
        color_id_for: dict[str, str] = {}
        show_id_for: dict[str, str] = {}
        for label in label_order:
            color_id_for[label] = AnnotationShaderBuilder._sanitize_label(label)
            show_id_for[label] = AnnotationShaderBuilder._sanitize_show_label(label)

        # Detect collisions before mutating any state — see the matching
        # block in AnnotationShaderBuilder.categorical_color.
        color_names = [color_id_for[label] for label in label_order]
        checkbox_names = (
            [f"show_{show_id_for[label]}" for label in label_order]
            if with_show_checkboxes
            else []
        )
        new_names = color_names + checkbox_names
        if len(set(new_names)) != len(new_names):
            dupes = sorted({n for n in new_names if new_names.count(n) > 1})
            raise ValueError(
                "categorical_color() produced duplicate control names "
                f"{dupes!r} — distinct labels sanitized to the same identifier. "
                "Rename one of the colliding labels."
            )
        for name in new_names:
            self._claim_name(name, kind="categorical")
        for label in label_order:
            self._color_controls.append((color_id_for[label], label_color[label]))
        if with_show_checkboxes:
            for label in label_order:
                self._checkbox_configs.append(
                    (f"show_{show_id_for[label]}", default_visible)
                )

        self._categorical_config = {
            "attr": attr,
            "groups": [
                (
                    color_id_for[label],
                    show_id_for[label],
                    label_color[label],
                    label_values[label],
                )
                for label in label_order
            ],
            "with_show_checkboxes": with_show_checkboxes,
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
        if self._continuous_config is not None:
            raise ValueError(
                "continuous_color() can only be called once per "
                "SkeletonShaderBuilder. Create a new builder for a different "
                "continuous-color rule."
            )
        if self._categorical_config is not None:
            raise ValueError(
                "continuous_color() and categorical_color() cannot both be set."
            )
        if range_min == range_max:
            raise ValueError(
                f"continuous_color() requires range_min != range_max; got both "
                f"= {range_min!r}. The colormap-domain transform would divide "
                "by zero."
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
            self._register_slider(
                name="rangeMin",
                type="float",
                min=slider_min,
                max=slider_max,
                default=range_min,
            )
            self._register_slider(
                name="rangeMax",
                type="float",
                min=slider_min,
                max=slider_max,
                default=range_max,
            )
        return self

    def _claim_name(self, name: str, *, kind: str) -> None:
        """Register a control identifier or raise if it's already in use."""
        if name in self._used_names:
            raise ValueError(
                f"Control name {name!r} ({kind}) collides with an existing "
                "#uicontrol declaration on this builder."
            )
        self._used_names.add(name)

    def _release_name(self, name: str) -> None:
        """Drop a name from the used-set."""
        self._used_names.discard(name)

    def _register_slider(
        self, *, name: str, type: str, min: float, max: float, default: float
    ) -> None:
        """Validate and append a slider config (see AnnotationShaderBuilder)."""
        if min > max:
            raise ValueError(f"Slider {name!r}: min ({min}) cannot exceed max ({max}).")
        if not (min <= default <= max):
            raise ValueError(
                f"Slider {name!r}: default ({default}) is outside "
                f"[min, max] = [{min}, {max}]."
            )
        self._claim_name(name, kind="slider")
        self._slider_configs.append(
            {"name": name, "type": type, "min": min, "max": max, "default": default}
        )

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
        with_checkboxes = self._categorical_config.get("with_show_checkboxes", True)

        lines: list[str] = []
        for i, (color_id, show_id, _color, values) in enumerate(groups):
            conds = [f"{attr} == {_format_float(v)}" for v in sorted(values)]
            condition = " || ".join(conds)
            keyword = "if" if i == 0 else "} else if"
            lines.append(f"  {keyword} ({condition}) {{")
            if with_checkboxes:
                lines.append(f"    if (!show_{show_id}) {{ discard; }}")
            lines.append(f"    emitRGB({color_id});")
        lines.append("  }")
        return "\n".join(lines)

    def _continuous_block(self) -> Optional[str]:
        if self._continuous_config is None:
            return None
        cfg = self._continuous_config
        if cfg["range_slider"]:
            t_expr = (
                f"clamp(({cfg['attr']} - rangeMin) / (rangeMax - rangeMin), 0.0, 1.0)"
            )
        else:
            mn = _format_float(cfg["range_min"])
            mx = _format_float(cfg["range_max"])
            t_expr = f"clamp(({cfg['attr']} - {mn}) / ({mx} - {mn}), 0.0, 1.0)"
        return f"  float t = {t_expr};\n  emitRGB({cfg['glsl_fn']}(t));"

    def _needs_hsl(self) -> bool:
        return self._desaturate_config is not None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, to_clipboard: bool = False) -> str:
        """Generate the complete GLSL skeleton shader string.

        Every shader emits a base colour first — either the segment-colour
        block (when :meth:`use_segment_color` was called, optionally with
        :meth:`desaturate`) or a plain ``emitRGB(segmentColor().rgb)``. This
        guarantees every vertex receives a defined colour: vertices that don't
        match any branch in the categorical or continuous block fall back to
        the base, rather than producing undefined output.

        Calling ``build()`` on a freshly constructed builder — or one with
        only vertex attributes declared but no colour rule — produces the
        minimal valid shader: just the base segment-colour emit.

        Parameters
        ----------
        to_clipboard : bool, optional
            If ``True``, also copy the shader source to the system clipboard
            via ``pyperclip``. Default is ``False``.

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

        # Base colour: always emit something first, so vertices that don't
        # match any categorical/continuous branch have a defined colour.
        seg_block = self._segment_color_block()
        if seg_block:
            parts.append(seg_block)
        else:
            parts.append("  emitRGB(segmentColor().rgb);")

        cat_block = self._categorical_block()
        if cat_block:
            parts.append(cat_block)

        cont_block = self._continuous_block()
        if cont_block:
            parts.append(cont_block)

        parts.append("}")
        shader = "\n".join(parts)
        if to_clipboard:
            pyperclip.copy(shader)
        return shader

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
