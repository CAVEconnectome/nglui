# Tools and Key Bindings

Neuroglancer tools are actions a user triggers from the keyboard -- **shift plus the bound letter** -- or by clicking a button in a tool palette.
nglui gives every bindable tool a class in `nglui.statebuilder.tools`, so your editor can autocomplete them and flag a misspelled name before the code runs.

``` py
from nglui.statebuilder import tools
```

## Binding tools

There are three ways to bind a tool, and they can be mixed freely.

**On a layer**, mapping keys to tools.
This keeps a layer's controls next to its other options:

``` py
SegmentationLayer(
    source=seg_source,
    tools={"H": tools.MeshSilhouette, "S": tools.SelectSegments},
)
```

**Across the viewer**, with `add_tools`.
Here layer tools say which layer they act on, by name or layer object, and can also be shown in a palette:

``` py
viewerstate.add_tools(
    tools.MeshSilhouette(layer="seg", key="H"),
    tools.Dimension(dimension="z"),                      # key assigned for you
    palette=tools.ToolPalette("Review", side="right"),
)
```

**As raw JSON**, through a layer's `extra` argument: `extra={"toolBindings": {"H": "meshSilhouetteRendering"}}`.
This skips nglui's checks, so prefer the forms above.

In a `tools={...}` mapping, a value can be a tool class (`tools.MeshSilhouette`), a tool object (needed for tools with options, like `tools.ShaderControlTool(control="gain")`), or Neuroglancer's name for the tool (`"meshSilhouetteRendering"`).
Classes are best: a typo in a class name is caught by your editor or type checker, while a typo in a string is only caught when the layer is created (with a suggestion of what you meant).

### How keys are assigned

Neuroglancer has **one set of keys for the whole viewer**: if two tools share a key, it silently drops one when the state loads.
So nglui assigns keys when the state is built, once it knows every key in use:

1. Bindings that came from a raw layer or a base state keep their keys.
2. A tool with an explicit `key` must use a free one, or you get an error saying what holds it.
3. Annotation tag tools keep their usual keys (`Q`, `W`, `E`, ...) when free, and otherwise move to the next free letter.
4. A tool with no `key` gets the next free letter (`Z`, `X`, `C`, ...). Use `key=False` for a tool that should appear only in a palette.

A tool that does not suit its layer -- `MeshSilhouette` on an image layer, say -- raises an error when the layer is created.

## Segmentation layers

### Selection and proofreading

| Tool | Does | Options | Neuroglancer name |
|---|---|---|---|
| `SelectSegments` | Select or deselect segments by painting over them. | -- | `selectSegments` |
| `MergeSegments` | Merge two segments. Needs a proofreading-enabled (graphene) segmentation. | -- | `mergeSegments` |
| `SplitSegments` | Split a segment. Needs a proofreading-enabled (graphene) segmentation. | -- | `splitSegments` |

### Display settings

These correspond to controls in the layer's **Render** tab.
Activating one shows that control at the bottom of the viewer.

| Tool | Neuroglancer label | Adjusts | Using it | Neuroglancer name |
|---|---|---|---|---|
| `SelectedAlpha` | Opacity (on) | 2d opacity of selected segments. | Hold the key and scroll to adjust. | `selectedAlpha` |
| `NotSelectedAlpha` | Opacity (off) | 2d opacity of unselected segments. | Hold the key and scroll to adjust. | `notSelectedAlpha` |
| `Alpha3d` | Opacity (3d) | Opacity of meshes in the 3d view. | Hold the key and scroll to adjust. | `objectAlpha` |
| `MeshSilhouette` | Silhouette (3d) | Silhouette shading of meshes in the 3d view. | Hold the key and scroll to adjust. | `meshSilhouetteRendering` |
| `MeshRenderScale` | Resolution (mesh) | Mesh level of detail. | Hold the key and scroll to adjust. | `meshRenderScale` |
| `Saturation` | Saturation | Saturation of segment colors. | Hold the key and scroll to adjust. | `saturation` |
| `HideSegmentZero` | Hide segment ID 0 | Whether segment 0 is hidden. | Press to toggle. | `hideSegmentZero` |
| `HoverHighlight` | Highlight on hover | Whether the segment under the mouse is highlighted. | Press to toggle. | `hoverHighlight` |
| `BaseSegmentColoring` | Base segment coloring | Whether supervoxels are colored individually. | Press to toggle. | `baseSegmentColoring` |
| `IgnoreNullVisibleSet` | Show all by default | Whether an empty selection shows every segment. | Press to toggle. | `ignoreNullVisibleSet` |
| `ColorSeed` | Color seed | The random segment color palette. | Press to pick new random colors. | `colorSeed` |
| `SegmentDefaultColor` | Fixed color | One color for all segments without an explicit color. | Press to switch to a fixed color, then scroll to adjust it. | `segmentDefaultColor` |

### Skeleton settings

| Tool | Neuroglancer label | Adjusts | Using it | Neuroglancer name |
|---|---|---|---|---|
| `SkeletonMode2d` | Skeleton mode (2d) | Skeletons as lines, or lines and points, in the 2d views. | Hold the key and scroll to cycle through the options. | `skeletonRendering.mode2d` |
| `SkeletonMode3d` | Skeleton mode (3d) | Skeletons as lines, or lines and points, in the 3d view. | Hold the key and scroll to cycle through the options. | `skeletonRendering.mode3d` |
| `SkeletonLineWidth2d` | Line width (2d) | Skeleton line width in the 2d views. | Hold the key and scroll to adjust. | `skeletonRendering.lineWidth2d` |
| `SkeletonLineWidth3d` | Line width (3d) | Skeleton line width in the 3d view. | Hold the key and scroll to adjust. | `skeletonRendering.lineWidth3d` |

## Image layers

| Tool | Neuroglancer label | Adjusts | Using it | Neuroglancer name |
|---|---|---|---|---|
| `Opacity` | Opacity (slice) | Image opacity in the 2d views. | Hold the key and scroll to adjust. | `opacity` |
| `Blend` | Blending (slice) | How the image combines with the layers beneath it. | Hold the key and scroll to cycle through the options. | `blend` |
| `VolumeRendering` | Volume rendering (experimental) | Volume rendering mode in the 3d view. | Hold the key and scroll to cycle through the options. | `volumeRendering` |
| `VolumeRenderingGain` | Gain (3D) | Brightness of the volume rendering. | Hold the key and scroll to adjust. | `volumeRenderingGain` |
| `VolumeRenderingDepthSamples` | Resolution (3D) | Depth samples for the volume rendering. | Hold the key and scroll to adjust. | `volumeRenderingDepthSamples` |

## Image and segmentation layers

| Tool | Neuroglancer label | Adjusts | Using it | Neuroglancer name |
|---|---|---|---|---|
| `CrossSectionRenderScale` | Resolution (slice) | Resolution of the 2d rendering. | Hold the key and scroll to adjust. | `crossSectionRenderScale` |

## Any layer with a shader

| Tool | Does | Options | Neuroglancer name |
|---|---|---|---|
| `ShaderControlTool` | Adjust one of the shader's `#uicontrol` controls, such as an image's contrast. | `control`: the control's name. Default `"normalized"`, the default image shader's contrast control. | `shaderControl` |

## Viewer

| Tool | Does | Options | Neuroglancer name |
|---|---|---|---|
| `Dimension` | Step through one dimension of the viewer position: hold the key and scroll. | `dimension`: the dimension's name, e.g. `"z"`. | `dimension` |

## Annotation tools

Tools that place annotations -- points, lines, boxes, ellipsoids, polylines -- **cannot be bound to keys**.
Neuroglancer only restores them as a layer's active tool, and a key binding for one makes it drop every binding after it on the same layer.
Instead, make one the layer's active tool, so it is ready to use as soon as the state loads:

``` py
AnnotationLayer(name="synapses", ..., active_tool="point")   # or "line", "box", "ellipsoid", "polyline"
```

Annotation tags have their own tools, which nglui binds automatically; see [Tags](statebuilder.md#tags).

## Tool palettes

A palette is a panel of tool buttons.
Pass `palette=` to `add_tools` to show those tools in it as well as binding them, or build one directly:

``` py
viewerstate.add_tool_palette(
    tools.ToolPalette(
        "Display",
        tools=[tools.Alpha3d(layer="seg"), tools.MeshSilhouette(layer="seg")],
        side="right",
    )
)
```

Tools listed in a palette this way are shown but not bound to keys.
