import re
from collections import namedtuple

import webcolors


def rgb_to_triple(rgb: webcolors.IntegerRGB) -> list:
    return [rgb.red / 255.0, rgb.green / 255.0, rgb.blue / 255.0]


def parse_color_rgb(clr):
    if isinstance(clr, str):
        hex_match = r"\#[0123456789abcdef]{6}"
        if re.match(hex_match, clr.lower()):
            return rgb_to_triple(webcolors.hex_to_rgb(clr))
        else:
            return rgb_to_triple(webcolors.name_to_rgb(clr))
    else:
        return rgb_to_triple(webcolors.IntegerRGB(*[int(255 * x) for x in clr]))


def color_to_vec3(clr) -> str:
    """
    Convert a color to a string representation of an RGB triple.

    :param clr: Color in various formats (string, tuple, list).
    :return: String representation of the RGB triple.
    """
    rgb = parse_color_rgb(clr)
    return f"vec3({rgb[0]}, {rgb[1]}, {rgb[2]})"


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
        r = g = b = l; // Achromatic
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
    hsl.y *= 0.5; // Reduce saturation by 50%
    vec3 desaturatedColor = {hsl_to_rgb.name}(hsl);
    emitRGB(desaturatedColor);
  }}
}}
"""


class PointShader:
    def __init__(
        self,
        colors=None,
        colormap=None,
        n_colors=None,
        many_label_color="white",
        no_label_color="grey",
        markersize=5,
        markerslider=True,
    ):
        """
        Create a point shader with customizable colors and properties.
        See https://jiffyclub.github.io/palettable/ for available colormaps.

        Parameters
        ----------
        colors : list, optional
            List of colors (strings, hex, or RGB tuples/lists)
        colormap : str, optional
            Name of a palettable colormap (ignoring category) (e.g., 'Set1')
            Not used if colors is specified.
        n_colors : int, optional
            Number of colors to extract from colormap. Cannot be more than the available numbers in the colormap.
            Only used if a colormap is specified.
        many_label_color : str, optional
            Color for points with multiple labels, by default 'white'
        no_label_color : str, optional
            Color for points with no labels, by default 'grey'
        markersize : int, optional
            Default marker size, by default 5
        markerslider : bool, optional
            Whether to include marker size slider, by default True
        """
        self.many_label_color = many_label_color
        self.no_label_color = no_label_color
        self.markersize = markersize
        self.markerslider = markerslider
        self.colormap = colormap

        # Determine colors
        if colors is not None:
            if isinstance(colors, str):
                # Handle single colormap string
                raise ValueError("Use 'colormap' parameter for colormap names")
            self.colors = [parse_color_rgb(c) for c in colors]
        elif colormap is not None and n_colors is not None:
            self.colors = self._get_colormap_colors(colormap, n_colors)
        else:
            raise ValueError(
                "Must provide either 'colors' or both 'colormap' and 'n_colors'"
            )

        self.n_properties = len(self.colors)
        self.code = self._generate_shader_code()

    def _get_colormap_colors(self, colormap_name, n_colors):
        """Get colors from a palettable colormap."""
        import palettable

        # Define all module/subcategory combinations
        module_paths = [
            # ColorBrewer
            ("colorbrewer", "diverging"),
            ("colorbrewer", "qualitative"),
            ("colorbrewer", "sequential"),
            # CartoColors
            ("cartocolors", "diverging"),
            ("cartocolors", "qualitative"),
            ("cartocolors", "sequential"),
            # cmocean
            ("cmocean", "diverging"),
            ("cmocean", "sequential"),
            # Scientific
            ("scientific", "diverging"),
            ("scientific", "sequential"),
            # Light & Bartlein
            ("lightbartlein", "diverging"),
            ("lightbartlein", "sequential"),
            # Top-level modules
            ("matplotlib", None),
            ("mycarta", None),
            ("tableau", None),
            ("wesanderson", None),
        ]

        found_colormap = None
        found_base_name = None
        available_sizes = []

        # First pass: check if base colormap name exists and collect available sizes
        for module_name, subcat_name in module_paths:
            try:
                module = getattr(palettable, module_name)

                if subcat_name is not None:
                    target_module = getattr(module, subcat_name)
                else:
                    target_module = module

                # Look for any colormap that starts with colormap_name + "_"
                for attr_name in dir(target_module):
                    if attr_name.startswith(
                        colormap_name + "_"
                    ) and not attr_name.startswith("_"):
                        found_base_name = colormap_name
                        size = attr_name.split("_")[-1]
                        if size.isdigit():
                            available_sizes.append(int(size))

            except AttributeError:
                continue

        # Check if we found the base colormap name
        if found_base_name is None:
            raise ValueError(
                f"Colormap '{colormap_name}' not found in palettable. "
                f"Check available colormaps at https://jiffyclub.github.io/palettable/"
            )

        # Find the smallest available size >= n_colors
        available_sizes = sorted(set(available_sizes))
        suitable_sizes = [size for size in available_sizes if size >= n_colors]

        if not suitable_sizes:
            raise ValueError(
                f"Colormap '{colormap_name}' does not support {n_colors} colors. "
                f"Maximum available size: {max(available_sizes)}, Available sizes: {available_sizes}"
            )

        chosen_size = min(suitable_sizes)

        # Second pass: find the colormap with the chosen size
        for module_name, subcat_name in module_paths:
            try:
                module = getattr(palettable, module_name)

                if subcat_name is not None:
                    target_module = getattr(module, subcat_name)
                else:
                    target_module = module

                constructed_name = f"{colormap_name}_{chosen_size}"
                if hasattr(target_module, constructed_name):
                    found_colormap = getattr(target_module, constructed_name)
                    break

            except AttributeError:
                continue

        if found_colormap is None:
            raise ValueError(f"Unable to load colormap {colormap_name}_{chosen_size}")

        # Extract colors - return only the requested number
        return self._extract_colors(found_colormap, n_colors)

    def _extract_colors(self, colormap, n_colors):
        """Extract colors from a palettable colormap object."""
        if hasattr(colormap, "mpl_colors"):
            available_colors = colormap.mpl_colors
        elif hasattr(colormap, "colors"):
            colors = colormap.colors
            # Convert from 0-255 to 0-1 range if needed
            if colors and len(colors[0]) >= 3:
                first_color = colors[0]
                if any(c > 1 for c in first_color[:3]):
                    available_colors = [
                        [c[0] / 255.0, c[1] / 255.0, c[2] / 255.0] for c in colors
                    ]
                else:
                    available_colors = colors
        else:
            raise ValueError(f"Colormap object has no recognized color attribute")

        # Check if we have enough colors
        if len(available_colors) < n_colors:
            raise ValueError(
                f"Colormap only has {len(available_colors)} colors, "
                f"but {n_colors} were requested"
            )

        return available_colors[:n_colors]

    def _generate_shader_code(self):
        """Generate the complete shader code."""
        ui_controls = []

        # Color controls for each property using color names or hex
        for i, color in enumerate(self.colors):
            # Convert RGB to hex or use color name
            color_name = self._rgb_to_color_name(color)
            ui_controls.append(
                f'#uicontrol vec3 color{i} color(default="{color_name}")'
            )

        # Special colors
        ui_controls.append(
            f'#uicontrol vec3 manyLabelColor color(default="{self.many_label_color}")'
        )
        ui_controls.append(
            f'#uicontrol vec3 noLabelColor color(default="{self.no_label_color}")'
        )

        # Marker size control
        if self.markerslider:
            ui_controls.append(
                f"#uicontrol float markersize slider(min=0, max=20, default={self.markersize})"
            )

        # Generate main shader logic
        marker_line = (
            "setPointMarkerSize(markersize);"
            if self.markerslider
            else f"setPointMarkerSize({self.markersize}.0);"
        )

        # Generate property checks with counting logic
        property_checks = []
        for i in range(self.n_properties):
            property_checks.append(f"  if (prop_tag{i}()==uint(1)) {{")
            property_checks.append(f"    activeCount = activeCount + 1;")
            property_checks.append(f"    lastActiveColor = color{i};")
            property_checks.append(f"  }}")

        shader_main = f"""void main() {{
    {marker_line}
    setColor(noLabelColor);

    int activeCount = 0;
    vec3 lastActiveColor = noLabelColor;

    {chr(10).join(property_checks)}

    if (activeCount == 1) {{
        setColor(lastActiveColor);
    }} else if (activeCount > 1) {{
        setColor(manyLabelColor);
    }}
}}"""

        return "\n".join(ui_controls) + "\n" + shader_main

    def _rgb_to_color_name(self, rgb):
        """Convert RGB values back to color name or hex."""
        # This is a simple implementation - you might want to use webcolors
        # to convert RGB back to nearest color name or hex
        r, g, b = [int(c * 255) for c in rgb]
        return f"#{r:02x}{g:02x}{b:02x}"

    def make_properties(self):
        """Return property configuration for neuroglancer."""
        properties = []
        for i in range(self.n_properties):
            properties.append(
                {"id": f"tag{i}", "type": "number", "description": f"Tag {i}"}
            )
        return properties

    def __repr__(self):
        return f"PointShader(colormap={self.colors}, colormap={self.colormap})"


simple_point_shader = """
#uicontrol vec3 markerColor color(default="lightblue")
#uicontrol float markerSize slider(min=0, max=20, default=5)
void main() {{
    setPointMarkerSize(markerSize);
    setColor(markerColor);
}}
"""

DEFAULT_SHADER_MAP = {
    "skeleton_compartments": simple_compartment_skeleton_shader,
    "points": simple_point_shader,
    "tags": PointShader(colormap="Set1", n_colors=9).code,
}
