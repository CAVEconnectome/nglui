from collections import namedtuple

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


DEFAULT_SHADER_MAP = {
    "skeleton_compartments": simple_compartment_skeleton_shader,
}
