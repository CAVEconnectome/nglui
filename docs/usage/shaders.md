---
title: Shaders
---

Neuroglancer uses [GLSL](https://www.khronos.org/opengl/wiki/OpenGL_Shading_Language) shaders to control how each layer is rendered.
NGLui provides two builder classes — `AnnotationShaderBuilder` and `SkeletonShaderBuilder` — that generate these shaders programmatically, similar in spirit to how seaborn's `scatterplot` lets you declare visual encodings instead of writing plotting code by hand.

Both builders produce a GLSL string that can be passed directly to a layer's `add_shader()` method.

## Annotation shaders

`AnnotationShaderBuilder` targets annotation layers (points, lines, etc.).
All methods return `self` so calls can be chained; call `.build()` at the end to get the GLSL string.

```python
from nglui.statebuilder.shaders import AnnotationShaderBuilder
```

### Categorical colour

The most common pattern: map a `uint` annotation property to a named category, with one **colour picker** and one **show/hide checkbox** per category exposed in the Neuroglancer UI.

```python
shader = (
    AnnotationShaderBuilder()
    .categorical_color(
        prop="cell_type",
        categories={
            0: ("excitatory", "tomato"),
            1: ("inhibitory", "cyan"),
            2: ("unknown", "grey"),
        },
    )
    .build()
)
```

Multiple property values can share the same label — they share a single colour picker and checkbox, and their GLSL condition uses `||`:

```python
shader = (
    AnnotationShaderBuilder()
    .categorical_color(
        prop="tag_detailed",
        categories={
            0: ("null",       "white"),
            1: ("spine",      "magenta"),
            2: ("spine",      "magenta"),   # same label as value 1
            3: ("multi_spine","purple"),
            4: ("shaft",      "yellow"),
            5: ("soma",       "cyan"),
        },
    )
    .build()
)
```

#### Colours from a string column

If your data has string labels rather than pre-assigned integers, pass a `list[str]` or `dict[str, str]` and the builder assigns integer IDs automatically.
The resulting `label_map` tells you how to encode your DataFrame column before storing it as an annotation property.

```python
builder = (
    AnnotationShaderBuilder()
    .categorical_color(
        prop="cell_type",
        categories=["excitatory", "inhibitory", "unknown"],  # auto-assigns 0, 1, 2
    )
)

# Encode the string column before adding annotations
df["cell_type_int"] = df["cell_type"].map(builder.label_map)
shader = builder.build()
```

To supply explicit colours per label use `dict[str, str]`:

```python
builder = (
    AnnotationShaderBuilder()
    .categorical_color(
        prop="cell_type",
        categories={"excitatory": "tomato", "inhibitory": "cyan"},
    )
)
```

A `palette` argument (any [palettable](https://jiffyclub.github.io/palettable/) colormap name) controls auto-colour assignment when using `list[str]` input.
The default palette is `CartoCOlors Bold_10`, chosen for visibility on Neuroglancer's black background.

### Continuous colour

Map a numeric annotation property through one of Neuroglancer's built-in GLSL colormaps.
Interactive range sliders let you adjust the mapping in the viewer.

```python
shader = (
    AnnotationShaderBuilder()
    .continuous_color(
        prop="synapse_size",
        colormap="cubehelix",   # or "jet"
        range_min=0,
        range_max=500,
    )
    .build()
)
```

Available colormaps (from `neuroglancer/src/webgl/colormaps.ts`):

| Name | Description |
|---|---|
| `cubehelix` | Perceptually uniform, works well on dark backgrounds *(default)* |
| `jet` | Classic rainbow ramp (blue → cyan → green → yellow → red) |

Set `range_slider=False` to hardcode the range into the shader instead of exposing sliders.

### Opacity

Add a global opacity slider that controls annotation transparency:

```python
shader = (
    AnnotationShaderBuilder()
    .opacity(default=0.8)
    .categorical_color(prop="cell_type", categories=["exc", "inh"])
    .build()
)
```

### Highlight

Keep a subset of annotations fully opaque while fading the rest via the opacity slider.
Useful for emphasising annotations that match a selection property:

```python
shader = (
    AnnotationShaderBuilder()
    .opacity(default=0.3)
    .highlight(prop="pre_in_selection", value=1)  # selected annotations → alpha 1.0, others → 0.3
    .categorical_color(...)
    .build()
)
```

### Point size

Control marker size from a fixed value, a UI slider, or an annotation property:

```python
# Fixed size
AnnotationShaderBuilder().point_size(size=8.0)

# Interactive slider
AnnotationShaderBuilder().point_size(size=5.0, slider=True)

# Driven by an annotation property (e.g. synapse volume)
AnnotationShaderBuilder().point_size(prop="volume", scale=0.0002)
```

### Complete example

```python
shader = (
    AnnotationShaderBuilder()
    .point_size(prop="size", scale=0.0002)
    .opacity(default=1.0)
    .highlight(prop="pre_in_selection", value=1)
    .categorical_color(
        prop="tag_detailed",
        categories={
            0: ("null",        "white"),
            1: ("spine",       "magenta"),
            2: ("spine",       "magenta"),
            3: ("multi_spine", "purple"),
            4: ("shaft",       "yellow"),
            5: ("soma",        "cyan"),
        },
    )
    .border_width(0.0)
    .build()
)

layer = AnnotationLayer("synapses", shader=shader)
```

---

## Skeleton shaders

`SkeletonShaderBuilder` targets the `skeleton_shader` of a `SegmentationLayer`.
Skeleton shaders differ from annotation shaders in a few key ways:

- Per-vertex data is stored as `float32` attributes accessed in GLSL as `vCustom1`, `vCustom2`, … in the order they were declared in the skeleton info file.
- Colour is emitted with `emitRGB(vec3)` rather than `setColor(vec4)`.
- Each segment has an assigned colour available via `segmentColor()`.

Pass the ordered vertex attribute names at construction — the same list as `SkeletonManager.vertex_attribute_names`:

```python
from nglui.statebuilder.shaders import SkeletonShaderBuilder
```

### Segment colour with compartment desaturation

The most common pattern in connectomics: use each segment's assigned colour for one compartment (e.g. axon) and desaturate the others so they visually recede.

```python
shader = (
    SkeletonShaderBuilder(["compartment"])
    .use_segment_color()
    .desaturate(attr="compartment", reference_value=2.0, saturation_scale=0.5)
    .build()
)
```

`saturation_scale=0.5` halves the HSL saturation of non-reference vertices.
Use `0.0` for greyscale or `1.0` for no change.

### Categorical colour

Fixed colours per compartment, with per-category colour pickers and show/hide checkboxes:

```python
shader = (
    SkeletonShaderBuilder(["compartment"])
    .categorical_color(
        attr="compartment",
        categories={
            1: ("axon",      "white"),
            2: ("dendrite",  "cyan"),
            3: ("soma",      "yellow"),
        },
    )
    .build()
)
```

String-label inputs and `label_map` work identically to `AnnotationShaderBuilder`:

```python
builder = (
    SkeletonShaderBuilder(["compartment"])
    .categorical_color(
        attr="compartment",
        categories=["axon", "dendrite", "soma"],
    )
)
# builder.label_map == {"axon": 0, "dendrite": 1, "soma": 2}
```

### Continuous colour

Map a float vertex attribute through a Neuroglancer colormap:

```python
shader = (
    SkeletonShaderBuilder(["distance_from_soma"])
    .continuous_color(
        attr="distance_from_soma",
        colormap="cubehelix",
        range_min=0,
        range_max=500,
    )
    .build()
)
```

### With SkeletonManager

`SkeletonManager.make_shader_builder()` returns a `SkeletonShaderBuilder` pre-configured with the manager's vertex attributes.
Assign the result back to `manager.shader`:

```python
manager.shader = (
    manager.make_shader_builder()
    .use_segment_color()
    .desaturate(attr="compartment", reference_value=2.0)
    .build()
)

layer = manager.to_segmentation_layer()
```

### Multiple vertex attributes

When a skeleton has more than one attribute, list them in order at construction.
Each name maps to `vCustom1`, `vCustom2`, … by position:

```python
# compartment → vCustom1, distance_from_soma → vCustom2
builder = SkeletonShaderBuilder(["compartment", "distance_from_soma"])
```

You can also add or override individual attributes explicitly:

```python
builder = SkeletonShaderBuilder()
builder.vertex_attribute("compartment", index=1)
builder.vertex_attribute("distance_from_soma", index=2)
```
