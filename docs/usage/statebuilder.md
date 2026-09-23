---
title: StateBuilder
---

Statebuilder is a submodule that to produce Neuroglancer links and the JSON states that define them.
It is built on top of [neuroglancer-python](https://github.com/google/neuroglancer/tree/master/python), and helps simplify producing more complex and data-driven states, while automating much of the boilerplate code and integrating with CAVEclient for CAVE-backed datasets.

## Neuroglancer Key Concepts

To understand how to use StateBuilder, it's helpful to understand the basic Neuroglancer concepts of layers, states, and annotations.

Neuroglancer is a web-based viewer for large 3d datasets.
Notably, Neuroglancer is designed to be a bit of complex Javascript that runs in your web browser and, generally, looks for data in various cloud-hosted locations.
There are a number of different **deployments** of Neuroglancer, which can be based on slightly different versions of the code or slightly different configurations available.
The two deployments that are built into NGLui right now are ["Mainline" or the default deployment](https://neuroglancer-demo.appspot.com) built from the [main branch of Neuroglancer from Google](https://github.com/google/neuroglancer) and the [Spelunker](https://spelunker.cave-explorer.org) deployment based a [bleeding-edge branch](https://github.com/seung-lab/neuroglancer/tree/spelunker) focused on CAVE-related features.
Additional Neuroglancer deployments can be added using [`site_utils`](config.md).

Virtually every aspect of Neuroglancer is defined by this **state**, which is represented by a collection of keys and values in a JSON object that you can see if you click the `{}` button in the upper left of the Neuroglancer interface.
Everything from selecting new objects or adding annotations to zooming or rotating your view is defined in this state, which can be imported or exported effectively as a readable `json` file.
The complete description of the state is typically encoded in the URL, which is how you can easily share most Neuroglancer URLs.
Building your own Neuroglancer view is thus simply the process of defining the state and passing it to a Neuroglancer viewer.

The state has two main aspects, **viewer options** that define global properties like the position, the layout of views, and what tabs are open, and **layers** that define different datas sources and how they are visualized.
Layers can be different types with different behaviors: Image layers show 3d imagery, Segmentation layers have "segments" with unique ids that can be selected and visualized in 3d with meshes or skeletons, and Annotation layers can show points, lines, or other objects associated with data.
Each layer has a name, a source (typically a path to a cloud-hosted dataset), and various configuration options.
Layers can actually have multiple complementary data sources of the same type, for example a Segmentation layer can have one source for segmentations and meshes while another source provides skeletons or segment property data.
Unlike Image or Segmentation layers, Annotation layers can also have a "local" source, where the annotations are directly defined in the Neuroglancer state.

The Statebuilder module follows this viewer-and-layer pattern, where you define a viewer with a set of options and then add layers of different types to it.
Where possible, Statebuilder tries to tie components together as simply as possible and with reasonble defaults in order to make a Neuroglancer state with as little code as possible.

!!! note

    Common viewer options -- title, camera, background colors, side panels -- are set directly (see [Viewer Options](#viewer-options)).
    Anything nglui does not wrap can still be set with the `extra` argument, which takes raw Neuroglancer JSON, or by manipulating the neuroglancer state from `vs.to_neuroglancer_state()`.

## NGLui ViewerState

The central object in StateBuilder is the `ViewerState`, which is used to initialize a Neuroglancer state, set the viewer options, add layers, and finally return the state in various formats.
A minimal state joins together a ViewerState with layers.
In general, all functions to _add_ data to a state or layer start with *add_*, functions to _set_ options or parameters start with *set_*, and functions to _export_ the data to another form (e.g. a URL) start with *to_*.

For example, to add a single image layer using the [Microns dataset](https://www.microns-explorer.org/cortical-mm3):

``` py
from nglui.statebuilder import ViewerState, ImageLayer, SegmentationLayer

viewerstate = ViewerState()
viewerstate.add_layer(ImageLayer(source='precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie35/em'))
viewerstate.add_layer(SegmentationLayer(source='precomputed://gs://iarpa_microns/minnie/minnie35/seg_m1300'))
```

You can see the layers in the state with in the `viewerstate.layers` attribute.

``` pycon
>>> viewerstate.layers
[ImageLayer(name='img', source='precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie35/em'),
 SegmentationLayer(name='seg', source='precomputed://gs://iarpa_microns/minnie/minnie35/seg')]
```

And then you can turn these options into a Neuroglancer state with functions like `to_link`:

``` pycon
>>> viewerstate.to_link()
```

Which returns a [Neuroglancer Link](https://spelunker.cave-explorer.org/#!%7B%22position%22:%5B112500.0,100000.0,11395.5%5D,%22layout%22:%22xy-3d%22,%22dimensions%22:%7B%22x%22:%5B8e-09,%22m%22%5D,%22y%22:%5B8e-09,%22m%22%5D,%22z%22:%5B4e-08,%22m%22%5D%7D,%22crossSectionScale%22:1.0,%22projectionScale%22:50000.0,%22showSlices%22:false,%22layers%22:%5B%7B%22type%22:%22image%22,%22source%22:%5B%7B%22url%22:%22precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie35/em%22,%22subsources%22:%7B%7D,%22enableDefaultSubsources%22:true%7D%5D,%22shader%22:%22None%22,%22name%22:%22img%22%7D,%7B%22type%22:%22segmentation%22,%22source%22:%5B%7B%22url%22:%22precomputed://gs://iarpa_microns/minnie/minnie35/seg%22,%22subsources%22:%7B%7D,%22enableDefaultSubsources%22:true%7D%5D,%22segments%22:%5B%5D,%22selectedAlpha%22:0.2,%22notSelectedAlpha%22:0.0,%22objectAlpha%22:0.9,%22segmentColors%22:%7B%7D,%22meshSilhouetteRendering%22:0.0,%22skeletonRendering%22:%7B%7D,%22name%22:%22seg%22%7D%5D%7D) with the configuration requested.

!!! danger

    It is highly recommended to install `cloud-volume` (which comes with `pip install nglui[full]`) in order to infer information like the resolution and size of datasets.
    If you do not want or have `cloud-volume` installed, you should always set the `dimensions` property when making a ViewerState (e.g. `ViewerState(dimensions=[4,4,40])`) to set the voxel resolution in nm/voxel.
    You can turn off automated resolution suggestions by setting `infer_dimensions=False` in the ViewerState constructor or explicitly setting the dimensions like above.
    Note that Neuroglancer does not always behave well if the dimensions are not set ahead of time, for example by setting the initial location or zoom level to be extremely far from the data.

Each function like [add_layer](../reference/statebuilder.md#src.nglui.statebuilder.base.ViewerState.add_layer) returns the layer object, so you can also initialize the layers in a pipeline.
This pipeline pattern is the one that we will typically use in this documentation.

``` py
viewerstate = (
    ViewerState()
    .add_layer(ImageLayer(source='precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em'))
    .add_layer(SegmentationLayer(source='precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300'))
)
```

Or, you could use convenience functions that a ViewerState object has to add layers of different types directly:

``` py
viewerstate = (
    ViewerState()
    .add_image_layer(source='precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em')
    .add_segmentation_layer(source='precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300')
)
```

There are many such convenience functions, with the goal of making the most typical use cases as simple as possible while allowing for more complex configurations by using the underlying layer classes directly.

### Viewer Options

Simple, single-value options are keyword arguments on `ViewerState` (and on `set_viewer_properties`, to change them later).
Options left unset are left out of the state, so Neuroglancer's own defaults apply.

``` py
viewerstate = ViewerState(
    title="Pyramidal cell 864691135",   # shown in the browser tab
    show_axis_lines=False,
    show_scale_bar=False,
    projection_background_color="white",
    layout="3d",
)
```

The full list is `title`, `show_axis_lines`, `show_scale_bar`, `show_default_annotations`, `cross_section_background_color`, `projection_background_color`, `hide_cross_section_background_3d`, and `wire_frame`.
Colors can be names, hex strings, or RGB tuples in [0, 1].

#### Camera

Neuroglancer keeps separate zoom, orientation, and depth for the 2d cross-section views and the 3d projection view.
Set them with `set_camera`, which only changes the values you pass:

``` py
viewerstate.set_camera(
    projection_orientation="xz",   # look at the 3d view the way the xz panel does
    projection_scale=12000,        # smaller is closer
)
```

Orientations are either a plane name (`"xy"`, `"xz"`, `"yz"`) or an `[x, y, z, w]` quaternion, such as one copied from an existing state.
Because a `Camera` is an object, you can define a view once and reuse it across many states:

``` py
from nglui.statebuilder import Camera

side_view = Camera(projection_orientation=[0.5, 0.5, 0.5, 0.5], projection_scale=8000)

for root_id in root_ids:
    vs = ViewerState(camera=side_view).add_segmentation_layer(...)
```

`scale_imagery` and `scale_3d` are shorthand for the camera's `cross_section_scale` and `projection_scale`.

#### Multi-panel Layouts

A preset `layout` such as `"xy-3d"` or `"4panel"` shows every layer in every view.
For side-by-side comparisons, build the layout from `Panel`s, each showing its own layers, arranged with `row` and `column`:

``` py
from nglui.statebuilder import Panel, row, column, Camera

viewerstate.set_layout(
    column(
        row(
            Panel(["cell_a"], layout="3d"),
            Panel(["cell_b"], layout="3d", camera=Camera(projection_orientation="xz")),
            flex=2,                                   # the top row takes 2/3 of the height
        ),
        Panel(["img", "cell_a", "cell_b"], layout="xy"),
    )
)
```

Each panel's `layout` is one of the presets, and its layers can be given by name or as layer objects; a panel with no layers shows all of them.
Camera fields set on a panel are decoupled from the rest of the viewer, while unset fields follow the global camera; pass `camera_link="relative"` to make them an offset from the global view instead.

#### Side Panels

`set_panels` opens, closes, and places the layer list, statistics, help, and selected-layer panels.
Pass `True`/`False` to open or close a panel, or a `SidePanel` to also choose its side and size:

``` py
from nglui.statebuilder import SidePanel

viewerstate.set_panels(
    layer_list=True,
    selected_layer=SidePanel(visible=True, side="right", size=500),
)
```

Which layer the selected-layer panel shows is set with `add_layer(..., selected=True)` or `set_selected_layer`.

#### Tools

Neuroglancer tools are actions bound to a key -- activated with shift plus that key -- or shown as buttons in a tool palette: selecting segments, adjusting a shader control or layer setting by dragging, or stepping through a dimension.
Add them with `add_tools`, from the `tools` module:

``` py
from nglui.statebuilder import tools

viewerstate.add_tools(
    tools.SelectSegments(layer="seg", key="S"),
    tools.MergeSegments(layer="seg"),                            # key assigned for you
    tools.ShaderControl(layer="img", control="normalized"),      # image contrast
    tools.LayerSetting(layer="seg", setting="objectAlpha"),      # mesh transparency
    tools.Dimension(dimension="z"),                              # step through z
    palette=tools.ToolPalette("Review", side="right"),           # also show as buttons
)
```

Neuroglancer has one set of keys for the whole viewer, and a key bound twice silently loses one of its tools when the state loads.
So nglui assigns keys when the state is built, once it knows every key in use:

1. Bindings from raw layers or a base state stay where they are.
2. A tool with an explicit `key` must use a free one; otherwise you get an error naming what holds it.
3. Annotation tag tools keep their usual keys (`Q`, `W`, `E`, ...) when free, and otherwise move to the next free letter -- so two tagged layers no longer fight over `Q`.
4. A tool with `key=None` gets the next free letter, and `key=False` binds no key, for a tool that should only appear in a palette.

Layer tools accept the layer as a name or as the layer object, and are checked against the layer type (you cannot bind `SelectSegments` to an image layer).
A layer can also carry its own tools, which need no `layer` argument: `SegmentationLayer(..., tools=[tools.SelectSegments()])`.
These are bound only when the layer is part of a ViewerState.

The available tools are `SelectSegments`, `MergeSegments`, `SplitSegments`, `ShaderControl`, `LayerSetting` (any setting in `tools.LAYER_SETTINGS`), and `Dimension`.

Annotation-placing tools are the exception: Neuroglancer can't restore them from a key binding or palette.
Worse, it drops every binding after them on the same layer.
Instead, make one the layer's active tool, so it is ready to use as soon as the state loads:

``` py
AnnotationLayer(name="synapses", ..., active_tool="point")   # or "line", "box", "ellipsoid", "polyline"
```

#### Anything Else: `extra`

For options nglui does not wrap, pass raw Neuroglancer state JSON as `extra`.
It is merged into the finished state last, so it overrides anything nglui set; nested dictionaries are merged and a value of `None` removes a key.
Every layer takes an `extra` argument as well, merged into that layer's JSON.

``` py
viewerstate = ViewerState(extra={"gpuMemoryLimit": 4_000_000_000, "concurrentDownloads": 64})
seg = SegmentationLayer(source=..., extra={"meshRenderScale": 1})
```

Keys are Neuroglancer's JSON names (camelCase), exactly as they appear in a state's JSON.
Neuroglancer does not validate them when building the state, so a misspelled key is silently ignored by the viewer.

### Exporting States...

You can export the ViewerState to a Neuroglancer state in a number of formats that are useful for different purposes.

#### ...to URLs

The most common way to export a state is to produce a Neuroglancer link or URL, which can be directly opened in a web browser.
The basic `viewerstate.to_url()` function returns the URL as a string formated as a link to a specific Neuroglancer deployment.

``` pycon
>>> viewerstate = ViewerState().add_image_layer('precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em')
>>> viewerstate.to_url()
'https://spelunker.cave-explorer.org/#!%7B%22position%22:%5B120320.0,103936.0,21360.0%5D,%22layout%22:%22xy-3d%22,%22dimensions%22:%7B%22x%22:%5B8e-09,%22m%22%5D,%22y%22:%5B8e-09,%22m%22%5D,%22z%22:%5B4e-08,%22m%22%5D%7D,%22crossSectionScale%22:1.0,%22projectionScale%22:50000.0,%22showSlices%22:false,%22layers%22:%5B%7B%22type%22:%22image%22,%22source%22:%5B%7B%22url%22:%22precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em%22,%22subsources%22:%7B%7D,%22enableDefaultSubsources%22:true%7D%5D,%22shader%22:%22None%22,%22name%22:%22imagery%22%7D%5D%7D'
```


In a notebook context, it is often convenient to return the URL as a formatted HTML link (like we used above), which can be done with `viewerstate.to_link()`.

In addition, CAVE offers a [link shortener](https://caveconnectome.github.io/CAVEclient/tutorials/state/
) that can be used to store JSON states and return a shortened URL that can be used to access the state.
We can use this link shortener directly using [to_link_shortener](../reference/statebuilder.md#src.nglui.statebuilder.base.ViewerState.to_link_shortener) and passing an appropriate CAVEclient client object.

``` py
from caveclient import CAVEclient
client = CAVEclient('minnie65_public')

viewerstate.to_link_shortener(client)
```

will upload the state and return a short link with a form like `'https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/4690769064493056'`.

You can also use the link shortener in the [to_url](../reference/statebuilder.md#src.nglui.statebuilder.base.ViewerState.to_url) and [to_link](../reference/statebuilder.md#src.nglui.statebuilder.base.ViewerState.to_link) methods by setting the `shorten` argument to `True` or `if_long` and passing a CAVEclient object.
The `if_long` option will only shorten the url if it gets long enough to start breaking the URL length limits of most browsers, approximately 1.75 million characters.

There are also convenience functions for copying the URL to the clipboard or opening it in a web browser, both of which have similar paramaters as the `to_url` method.

The [to_clipboard](../reference/statebuilder.md#src.nglui.statebuilder.base.ViewerState.to_clipboard) method will copy the URL to your system clipboard, after passing through the link shortener:

```py
viewerstate.to_clipboard(shorten=True, client=client)
```

And the [to_browser](../reference/statebuilder.md#src.nglui.statebuilder.base.ViewerState.to_browser) method will open the URL in your the web browser of your choosing, again after passing through the link shortener:

```py
viewerstate.to_browser(shorten=True, client=client, browser='firefox')
```

To change the Neuroglancer web deployment used in all of the url and link functions, you can set a different URL in the `target_url` parameter of the `to_url` function.
For example, to use the default Neuroglancer deployment, you can do:

``` pycon
>>> viewerstate.to_url(target_url='https://neuroglancer-demo.appspot.com')
'https://neuroglancer-demo.appspot.com#!%7B%22position%22:%5B120320.0,103936.0,21360.0%5D,%22layout%22:%22xy-3d%22,%22dimensions%22:%7B%22x%22:%5B8e-09,%22m%22%5D,%22y%22:%5B8e-09,%22m%22%5D,%22z%22:%5B4e-08,%22m%22%5D%7D,%22crossSectionScale%22:1.0,%22projectionScale%22:50000.0,%22showSlices%22:false,%22layers%22:%5B%7B%22type%22:%22image%22,%22source%22:%5B%7B%22url%22:%22precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em%22,%22subsources%22:%7B%7D,%22enableDefaultSubsources%22:true%7D%5D,%22shader%22:%22None%22,%22name%22:%22imagery%22%7D%5D%7D'
```

There is also a `target_site` dictionary that can be used to set shortcuts for different Neuroglancer deployments, which is used by default in the `to_url` function.
For example, you can set the `target_site` to `'spelunker'` to use the Spelunker deployment, or `'google'` to use the default Neuroglancer deployment.
Additional deployments can be added to the `target_site` dictionary using `site_utils`.
For example, if you want to add a new Neuroglancer deployment hosted at `'https://neuroglancer.my-deployment.com'` with a  `target_site` shortcut name `'my_new_shortcut'`, you could do the following:

``` py
from nglui.statebuilder import site_utils

site_utils.add_neuroglancer_site(
    site_name: 'my_new_shortcut',
    site_url: 'https://neuroglancer.my-deployment.com',
)
```

Note that this configuration is global and will affect all Neuroglancer states generated in the current Python session.

#### ...to JSON

The ViewerState can also be exported to a JSON object, which can be used to export to a file or to pass to other functions.

The core function is `to_dict` which returns the state as a dictionary, with contents equivalent to what you see in the Neuroglancer `{}` tab.

``` pycon
>> viewerstate.to_dict()
{'position': [120320.0, 103936.0, 21360.0],
 'layout': 'xy-3d',
 'dimensions': {'x': [np.float64(8e-09), 'm'],
  'y': [np.float64(8e-09), 'm'],
  'z': [np.float64(4e-08), 'm']},
 'crossSectionScale': 1.0,
 'projectionScale': 50000.0,
 'showSlices': False,
 'layers': [{'type': 'image',
   'source': [{'url': 'precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em',
     'subsources': {},
     'enableDefaultSubsources': True}],
   'shader': 'None',
   'name': 'imagery'}]}
```

A string-formatted JSON state can be generated with `to_json_string`, which can be directed pasted into the Neuroglancer `{}` tab.
The main difference is that the string version is formatted as a JSON string, with numpy types converted to standard Python types and uses `"` characters as required by JSON.
Note that it also contains `\n` newline characters and (optional) indents and thus is amenable to `print` or writing to a file.

#### ...to Neuroglancer python

The ViewerState object can also be converted to a `neuroglancer.Viewer` object, which is the underlying object used by the Neuroglancer python library. This can be done with the `to_neuroglancer_state` function, which returns a `neuroglancer.Viewer` object that can be used to interact with the Neuroglancer state in Python for advanced functionality.

!!! important
    
    The default `neuroglancer.Viewer` object used here differs from the one used in the `neuroglancer` python library by not launching a web server on creation.
    If you want an interactive Neuroglancer viewer that can be run from Python, set `interactive=True` in the ViewerState initialiation.

## Layers

All layers have a certain common set of properties:

* `source`: One or more cloudpaths pointing Neuroglancer at a data source.
* `name`: A name for the layer, displayed in the tabs.
* `visible`: A boolean value indicating if the layer is actively visible in the Neuroglancer interface.
* `archived`: A boolean value indicating if the layer is archived, which removes it from the tab interface but allows it to be re-enabled from the Layers tab.
* `shader`: A GL shader that can customize how data is rendered in the layer.

In general, the only required property for a simple state is the `source`, with other properties having reasonable defaults for simple states.
However, if you want to use multiple layers of the same type, you will need to set the `name` property for each layer to avoid conflicts.

### Image Layers

Image layers are the simplest type of layer to specify, and are used to display 3d imagery.
The most basic image layer is specified with just a source:

``` py
img_layer = ImageLayer(
    source='precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em',
)
```

which produces a layer with name `img`.
``` pycon
>>> img_layer.name
'img'
```

You can also add **multiple sources** to an image layer, which will be displayed as a single layer in Neuroglancer by combining them in a list.

``` py
img_layer_multi = ImageLayer(
    source=[
      'precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie35/em',
      'precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em',
    ]
)
```

Neuroglancer also lets you specify linear transformations for sources, such as translations or rotations.
You can apply these transformations with the more complex `Source` class, which can specify not only a cloudpath source, but a CoordSpaceTransform that captures an affine transformation matrix.
It requires more information, but also allows more complexity.
For example, to add a source with a translation:

``` py
img_layer_transformed = ImageLayer(
    source=Source(
        url='precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em',
        transform=CoordSpaceTransform(
            output_dimensions=[4,4,40],
            matrix=[[1, 0, 0, 1000],
                    [0, 1, 0, 2000.0],
                    [0, 0, 1, -2000]],
        )
    )
)
```

The first three columns of the CoordSpaceTransform specify a linear transform matrix, while the last column is a translation vector.
The 4th row is implicit and always `[0, 0, 0, 1]`, so it is not specified.

#### Display Options

Image layers take `opacity`, `blend`, `cross_section_render_scale`, and the volume rendering options `volume_rendering_mode`, `volume_rendering_gain`, and `volume_rendering_depth_samples`.
Options you do not set are left to Neuroglancer's defaults.

By default, nglui leaves the image shader to Neuroglancer, whose default maps intensity through an `invlerp` control named `normalized` -- the histogram and contrast slider in the layer's rendering tab.
That default is almost always what you want, so leave it alone unless you need to pin a contrast range.

To pin one, `shader_controls` sets the values of any `#uicontrol` a shader declares, keyed by name:

``` py
img_layer = ImageLayer(
    source=em_source,
    shader=my_shader,
    shader_controls={"normalized": {"range": [40, 210]}, "brightness": 0.2},
)
```

### Segmentation Layers

Segmentation layers are also volumetric data, but have objects with segment ids that can be selected, hidden, and visualized in 3d using meshes or skeletons.
The basic segmentation layer is just like an Image layer:

``` py
seg_source = SegmentationLayer(
    name='my_segmentation',
    source='precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300'
)
```

However, now we can also select objects by segment id. Let's use the pipeline pattern to build up a segmentation layer with a source, two selected ids, and custom colors for them.

``` py
seg_layer = (
    SegmentationLayer()
    .add_source('precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300')
    .add_segments([864691135356428751, 864691136032617659])
    .add_segment_colors(
        {
            864691135356428751: '#ff0000',
            864691136032617659: '#00ff00'
        }
    )
)
```

Packaging that together with an image layer from above,

``` py
(
    ViewerState()
    .add_layer(img_layer)
    .add_layer(seg_layer)
).to_link()
```

would produce this [Neuroglancer link]('https://spelunker.cave-explorer.org/#!%7B%22position%22:%5B218809.0,161359.0,13929.0%5D,%22layout%22:%22xy-3d%22,%22dimensions%22:%7B%22x%22:%5B4e-09,%22m%22%5D,%22y%22:%5B4e-09,%22m%22%5D,%22z%22:%5B4e-08,%22m%22%5D%7D,%22crossSectionScale%22:1.0,%22projectionScale%22:50000.0,%22showSlices%22:false,%22layers%22:%5B%7B%22type%22:%22image%22,%22source%22:%5B%7B%22url%22:%22precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em%22,%22subsources%22:%7B%7D,%22enableDefaultSubsources%22:true%7D%5D,%22name%22:%22img%22%7D,%7B%22type%22:%22segmentation%22,%22source%22:%5B%7B%22url%22:%22precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300%22,%22subsources%22:%7B%7D,%22enableDefaultSubsources%22:true%7D%5D,%22segments%22:%5B%22864691135356428751%22,%22864691136032617659%22%5D,%22selectedAlpha%22:0.2,%22notSelectedAlpha%22:0.0,%22objectAlpha%22:0.9,%22segmentColors%22:%7B%22864691135356428751%22:%22#ff0000%22,%22864691136032617659%22:%22#00ff00%22%7D,%22meshSilhouetteRendering%22:0.0,%22name%22:%22seg%22%7D%5D%7D')

There are also various parameters to set the appearance of the selected meshes and 2d overlays under the `set_view_options` method.

You can also add selections from the data in a dataframe using `add_segments_from_data`, which will select all the segments in a column of the dataframe.

``` py
seg_layer = (
    SegmentationLayer()
    .add_source('precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300')
    .add_segments_from_data(
        data = my_dataframe,
        segment_column = 'pt_root_id,
        visible_column = 'is_visible',
        color_column = 'color_value',
    )
)
```

As in images, any specification of sources can be either a string URL or a list of URLs.

#### Display Options

Beyond `set_view_options`, segmentation layers take these options as keyword arguments:

| Option | Effect |
|---|---|
| `segment_query` | Text in the layer's segment search box |
| `saturation` | Saturation of segment colors, from 0 (gray) to 1 |
| `color_seed` | Seed for random segment colors; change it for a different palette |
| `segment_default_color` | One color for every segment without an explicit color |
| `mesh_render_scale` | Mesh level of detail; smaller loads finer meshes |
| `cross_section_render_scale` | Resolution of the 2d rendering; larger is coarser |
| `hover_highlight` | Whether to highlight the segment under the mouse |
| `base_segment_coloring` | Color supervoxels individually rather than by root |
| `hide_segment_zero` | Whether segment 0 is hidden |
| `ignore_null_visible_set` | Whether an empty selection shows nothing (True) or everything (False) |
| `linked_segmentation_group` | Share segment visibility with another segmentation layer |
| `linked_segmentation_color_group` | Share segment colors with another layer, or `False` to keep them separate |
| `equivalences` | Groups of segment ids to treat as a single object |

Linking is useful for showing the same cells two ways, for example a mesh layer and a skeleton-only layer that follows its selection:

``` py
vs = (
    ViewerState()
    .add_layer(SegmentationLayer(name="seg", source=seg_source, segments=root_ids))
    .add_layer(SegmentationLayer(name="skel", source=skeleton_source, linked_segmentation_group="seg"))
)
```

Would select all segment ids in `my_dataframe['pt_root_id']` to the segmentation layer, toggle their visibility by the boolean values in `my_dataframe['is_visible']`, and set their colors to the values in `my_dataframe['color_value']`.
Colors can be hex values or web-readable color names, such as `'red'`, `'blue'`, or `'green'`.

#### Skeleton Sources

You can add a custom skeleton source to a segmentation layer simply by adding it via `add_source` or including it on the initial source list.

#### Segment Properties

Just like Image layers can have multiple sources, Segmentation layers can also have multiple segmentation sources.
In addition to the actual source of segmentation, you can also add sources to represent other aspects of the objects.

Segment properties can be treated as an additional source and can be added directly to the list of sources (`add_source`) if you have an existing or static cloudpath.
However, if you want to generate segment properties dynamically from a dataframe, you can use the `add_segment_properties` method, which will generate the segment properties file, upload it to a CAVE state server, and attach the resulting URL to the segmentation layer.
Note that `add_segment_properties` requires a CAVEclient object and also has a `dry_run` option to avoid many duplicative uploads while developing your code.

Segment properties can be added inline to the ViewerState and do not require a specified segmentation layer if only one segmentation layer is present.
There is also a `random_columns` parameter which can be used to add a random number column to make it easier to subsample sets of cells without using a smaller table.
For example, to download a CAVE table and then add it to a viewerstate with one random column, you could do:

``` pycon
from caveclient import CAVEclient
client = CAVEclient('minnie65_public')
ct_df = client.materialize.tables.allen_column_mtypes_v2.get_all(split_positions=True)
vs = (
    ViewerState(client=client)
    .add_layers_from_client(segmentation='seg')
    .add_segment_properties(
        data=ct_df,
        id_column='pt_supervoxel_id',
        label_column='target_id',
        tag_value_columns=['cell_type'],
        random_columns=1,
    )
)
```

See the [Segment Properties documentation](segmentprops.md) for more information on how to generate segment properties in Neuroglancer and what different options mean.

#### Skeleton Shader

The `shader` field of a segmentation layer currently specifies how skeletons are rendered in Neuroglancer.
The `statebuilder.shaders` module has some examples and tooling to help generate these shaders, but GL shaders like this are effectively a new language.
Once you have a shader you want to use, you can set it with the `add_shader` method of the segmentation layer.

Other skeleton options -- line widths, whether to draw vertices, and values for the shader's controls -- go in a `SkeletonRendering`:

``` py
from nglui.statebuilder import SkeletonRendering

seg_layer = SegmentationLayer(
    source=seg_source,
    skeleton_rendering=SkeletonRendering(
        line_width_3d=3,
        mode_3d="lines_and_points",
        shader_controls={"axon_saturation": 0.5},
    ),
)
```

A shader can be given either as `shader` or inside `SkeletonRendering`, but not both.

### Annotation Layers

Annotation layers let a user define various types of annotations like points, lines, bounding boxes, ellipses, and polylines (a series of points, each connected to the next by lines).
Annotations can also be associated with segmentations, allowing you to filter annotations by the data that's being selected.

Annotation layers come in two types, **local** annotation layers that store their annotations directly in the Neuroglancer state and **cloud** annotation layers that get their annotations from a cloud-hosted source.
While they can look and are created with the same functions similar, these behave differently in Neuroglancer and not all functions are available for both types.

#### Local Annotations

The simplest annotation layer is a local annotation layer, which can be created with just a name.

``` py
from nglui.statebuilder import AnnotationLayer
annotation_layer = AnnotationLayer(name='my_annotations')

viewer_state.add_layer(annotation_layer)
```

The simplest way to add annotations is through the `add_points`, `add_lines`, `add_boxes`, `add_ellipses`, and `add_polylines` methods.
These methods work similarly, taking a dataframe where each row represents an annotation and the columns are specified by parameters.

For example, to add points to the annotation layer, you can do:

``` py
annotation_layer.add_points(
    data=my_dataframe,
    point_column='location', # Column with point locations as x,y,z coordinates
    segment_column='linked_segment', # Column with linked segment ids
    description_column='description', # Column with annotation descriptions
    data_resolution=[1, 1, 1], # Resolution of the point data in nm/voxel. Will default to the layer's default resolution if not specified.
)
```

Put together into a pipeline, you could generate a Neuroglancer link with an annotation layer from a dataframe like so:

``` py
from nglui.statebuilder import ViewerState, AnnotationLayer
(
    ViewerState()
    .add_layers_from_client(client, segmentation='seg') # Sets the name of the segmentation layer to 'seg'
    .add_layer(
        AnnotationLayer(
            name='my_annotations',
            linked_segmentations='seg', # Uses the `seg` name to link annotations to the segmentation layer
        )
        .add_points(
            data=my_dataframe,
            point_column='location',
            segment_column='linked_segment',
            description_column='description',
            data_resolution=[1, 1, 1],
        )
    )
).to_link()
```

However, while this gives you the most control over details, there's still a lot of boilerplate code to set up an annotation layer, link it to an existing segmentation layer, and then add points to it.
If you just want to create a local annotation layer that will automatically link to the first existing segmentation layer in the ViewerState, you can use the `add_annotation_layer` method of the ViewerState object.
Even better, if you want to create a local annotation layer with points, you can use `add_points` directly on the ViewerState object, which will create a local annotation layer with the specified points and link it to the first segmentation layer in the ViewerState, and do the same annotation creation.
For example, the above code can be simplified to:

``` py 
(
    ViewerState()
    .add_layers_from_client(client)
    .add_points(
        data=my_dataframe,
        name='my_annotations',
        point_column='location',
        segment_column='linked_segment',
        description_column='description',
        data_resolution=[1, 1, 1],
    )
).to_link()
```

This will add a local annotation layer with points defined by the `my_dataframe` dataframe, where each row has a point location in the `location` column, a linked segment id in the `linked_segment` column, and a description in the `description` column.

!!! note Multi-column point locations

    The `point_column` field can also refer to a situation where the x,y, and z coordinates of the point are stored in separate columns if they are formatted according to `{point_column}_x`, `{point_column}_y`, and `{point_column}_z`.

There are also direct class to produce annotations in `statebuilder.ngl_annotations`, which can be added directly to an annotation layer via `add_annotations` for even more control.

##### Tags

Local annotations can have **tags**, which is a way to categorize annotations with shortcuts in Neuroglancer.
When you make an annotation layer, you can specify a list of tags that will be used to categorize the annotations.
The shortcuts for adding these tags run ++q++ to ++t++ for the first five tags, then ++a++ to ++g++ for the next five.

You can specify which tags the annotations have in two ways.
A `tag_column` specifies one or more column names, where each column has a single string per row that will be used as the tag for the annotation.
With this approach, you can only use one tag per annotation per tag column.
Alternatively, you can use `tag_bools`, which is a list of columns where each column name is taken to be a tag and the value in the column is a boolean indicating if the tag is applied to the annotation.
In both cases, the layer will automatically generate the list of required tags based on the columns in the dataframe, added alphabetically if not already present in the specified tag list.

###### Choosing a tag encoding

Neuroglancer deployments encode tags in one of two incompatible ways, and nglui has to
pick the one your target understands.

* **Main Neuroglancer** has no tag concept at all. A tag is an annotation property of
  `type: "bool"`, and hotkeys bind to a generic `toggleBoolProperty` tool.
* **seung-lab forks**, including Spelunker, use `uint8` properties carrying the label in
  a non-standard `tag` key, bound to `tagTool_*` tool types. These are limited to ten
  tags per layer, because there are only ten such tools.

The `capabilities` argument chooses between them, on the viewer state or on an
individual layer:

```python
# Let nglui work it out from the target site (the default).
vs = ViewerState(target_site="google")

# Or say so explicitly, and skip the lookup.
vs = ViewerState(capabilities="main")     # bool annotation properties
vs = ViewerState(capabilities="legacy")   # seung-lab tag properties
```

The two deployments nglui ships with — `spelunker` and `google` — have their
capabilities declared in code, so targeting either costs no network request at all.

For anything else, nglui asks the deployment directly: it fetches the viewer's own client
bundle and looks for the features in it. That is the only answer that survives a fork.
A fork's `git describe` string names the last tag in *its own* repository, so Spelunker
reported `v2.37` both before and after gaining the entire bool-property system, and a
backport puts a new feature on an old release. Reading the bundle needs no list of known
deployments and cannot be fooled by either.

That lookup is only attempted when a local annotation layer actually has tags, its result
is cached, and a failure never raises -- so building states offline works, it just falls
back to the default.

**If the bundle cannot be read**, nothing raises. An unreachable host, a page with no
recognizable bundle, or anything that is not a Neuroglancer deployment all resolve the
same way: nglui falls back to the default, warns once naming the argument to set, and
builds the state. The lookup costs about 0.4s per deployment per process and is cached, failures
included — building states offline pays one failed connection and then nothing. A
deployment that accepts a connection and then stalls is bounded by the read timeouts
rather than hanging indefinitely.

Results are cached per deployment origin for an hour, so every URL pointing at the same
viewer shares one entry, and a long-running session eventually notices a redeployment —
which is not hypothetical, since Spelunker gained bool properties mid-session while this
was being written. To pick up a change immediately:

```python
from nglui.statebuilder.capabilities import clear_capability_cache, prefetch
clear_capability_cache()
prefetch("https://spelunker.cave-explorer.org")   # optional: warm it deliberately
```

If you target one deployment regularly, declare it once and skip the lookup entirely:

```python
from nglui.statebuilder.capabilities import declare_capabilities
declare_capabilities("https://ngl.my-lab.org", "main")
declare_capabilities("https://ngl.my-lab.org", None)   # undo, go back to probing
```

The same call corrects a shipped declaration that has gone stale, without waiting for
an nglui release.

For a deployment nglui cannot identify, say so once and skip the lookup entirely:

```python
vs = ViewerState(capabilities="main", target_url="https://my-lab-neuroglancer.org")

# or, if every state you build targets it:
from nglui.statebuilder import set_default_capabilities
set_default_capabilities("main")
```

Note that a `version.json` missing its `tag` is treated as unidentifiable even when it
carries a repository and a build date. The date records when a build was cut rather than
what is in it, so a fresh rebuild of an old branch would look capable when it is not --
and claiming capability wrongly is the direction that costs the whole annotation layer.

**When nglui cannot identify a deployment it assumes `main`**, since deployments
overwhelmingly descend from it and the ones that do not — Spelunker among them — are
identified by the lookup rather than left to the default.

Know what that costs if it is wrong, because the two formats fail very differently. A
`bool` property sent to a viewer predating the feature makes Neuroglancer **drop the
annotation layer from the state entirely** and rewrite its URL without it, so the
annotations are gone rather than merely unrendered. The reverse is only lossy: a Spelunker
tag property in a current viewer still loads, just without its label or shortcut.

So if you target a deployment that serves no usable `version.json` and predates bool
annotation properties, say so once:

```python
set_default_capabilities("legacy")   # or capabilities="legacy" per state
```

`NGLUI_DISABLE_CAPABILITY_PROBE=1` suppresses the lookup entirely and uses the default.

###### Tag names as property ids

Under the modern encoding a tag becomes a property whose **identifier is what the viewer
displays**, and Neuroglancer requires identifiers to match `^[a-z][a-zA-Z0-9_]*$`. Since
tags usually come from dataframe column names, nglui sanitizes them the same way
Neuroglancer's own annotation tab does -- `"Cell Body"` becomes `cell_body` -- keeping the
original as the property's description, and warns once per layer listing everything it
renamed.

Non-ASCII labels are folded toward ASCII first, so accents survive (`café` → `cafe`)
and Greek letters become their names (`β-cell` → `beta_cell`) rather than being dropped —
without which `α-cell` and `β-cell` would both reduce to `cell`. Scripts with no Latin
reading, such as CJK or Cyrillic, cannot be rendered into a legal identifier and fall
back to a generated one like `tag_0`; the original is kept in the property description,
and `tag_ids` gives exact control.

If you want exact control, name the ids yourself, or make renaming an error:

```python
vs.add_annotation_layer(
    name="annos",
    tags=["Cell Body", "post-synaptic"],
    tag_ids={"Cell Body": "soma", "post-synaptic": "post_syn"},
)

# Or refuse to guess:
vs.add_annotation_layer(name="annos", tags=[...], strict_property_ids=True)
```


#### Annotation Display Options

Local and cloud annotation layers both take `shader_controls`, which sets values for the `#uicontrol` controls an annotation shader declares, and `ignore_null_segment_filter`.
With `filter_by_segmentation`, annotations are shown only if linked to a visible segment, and `ignore_null_segment_filter=False` also hides annotations with no linked segment.
`filter_by_segmentation` takes `True` (every linked relationship), a relationship name, or a list of them.

#### Cloud Annotations

Cloud annotations are similar to local annotations, but they are stored in a cloud-hosted source.
Cloud annotations can be used simply by passing a `source` url to the `AnnotationLayer` or `add_annotation_layer` method.
The AnnotationLayer object will detect if the `source` value is not `None` and will create a cloud annotation layer instead of a local one.

Note that cloud annotations don't mix with local annotations, and if you have an explicit source defined then the local annotations will not be created even if they were specified.
In addition, cloud annotations cannot currently have tags.

### Raw Layers

Raw layers are a way to add existing layers from Neuroglancer states to the ViewerState without additional processing or linking.
You can add a raw layer by calling the `add_raw_layer` method on the ViewerState object, passing the dictionary representation of the layer which you can get from the state JSON in Neuroglancer or downloaded.
Raw layers have few configuration options, becuase the data is intended to be used as-is.
However, there are a few changes you can make when bringing them into your ViewerState.

1. You can specify a new `name` for the layer, which will replace the existing name, otherwise the name will be inherited from the data in the layer JSON. Similarly, you can adjust visibility and archived settings.
2. You can also provide a `source_remap` dictionary to remap any source URLs for the layer. For example, if a cloudpath has changed locations or you want to migrate a state from an authenticated production endpoint to a public endpoint, you can provide a mapping of old paths to new paths as a dictionary.

If you're using raw layers, you might find it useful to be using a base state as well.
You can provide a `base_state` state dictionary to the `ViewerState` class on creation.
You can strip out inconvenient parts with functions `strip_layers` (which strips just the layer definitions and active layer) and `strip_state_properties`, which offers more selective control.
This might be useful if you want to preserve things that cannot be easily set in the python interface such as complex tool panels.
Values from a base state are kept unless you set them explicitly: for example, a base state's layout survives unless you also pass `layout`.
For individual options, the `extra` argument (see [Viewer Options](#anything-else-extra)) is usually simpler than a base state.

### CAVEclient Integration

NGLui integrates with the CAVEclient to make it easy to work with Neuroglancer states that are hosted on CAVE.
You can use an initialized CAVEclient object to configure the resolution, image, and segmentation layers of a ViewerState:

``` pycon
from caveclient import CAVEclient
client = CAVEclient('minnie65_public')
viewer_state = (
    ViewerState()
    .add_layers_from_client(client)
)
```

This will use the info in the CAVEclient to find any relevent information (including skeleton sources for segmentation layers) and add it to the ViewerState.

``` pycon
>>> viewer_state.layers
[ImageLayer(name='imagery', source='precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em'),
 SegmentationLayer(name='segmentation', source=['graphene://https://minnie.microns-daf.com/segmentation/table/minnie65_public', 'precomputed://middleauth+https://minnie.microns-daf.com/skeletoncache/api/v1/minnie65_public/precomputed/skeleton/'])]
```

There are a variety of parameters to control layer properties here, as well.
In all cases, the image layer will be added first (if used) and then the segmentation layer.

### Quick links from root ids

If all you want is a link showing one or more root ids, `helpers.make_segment_link` does the whole thing in one line:

``` pycon
from nglui import statebuilder

statebuilder.helpers.make_segment_link(client, root_ids=[864691135474648896])
```

By default this returns an HTML link that renders in a notebook.
Use `return_as` to get something else: `"url"` for the URL string, `"clipboard"` to copy the URL to your system clipboard, `"browser"` to open it, `"dict"` or `"json"` for the raw state, or `"viewer"` for the `ViewerState` itself so you can keep building on it.

``` pycon
url = statebuilder.helpers.make_segment_link(
    client,
    root_ids=[864691135474648896],
    return_as="url",
    shorten="if_long",
)
```

You can also attach [segment properties](../usage/segmentprops.md) at the same time.
Pass either a `SegmentProperties` object or the property JSON it produces, and it will be uploaded via the client's state service and added as a source on the segmentation layer:

``` pycon
statebuilder.helpers.make_segment_link(
    client,
    root_ids=root_ids,
    segment_properties=seg_prop,
)
```

If you want the `ViewerState` rather than a rendered link, `helpers.make_segment_state` takes the same state-building arguments and returns the viewer directly.

## Mapping Data

In some situations, it can make sense to separate data from the Neuroglancer state creation rules.
This effectively allows you to build one set of rules and then apply them to different data with the same code.

This is handled now through a special `DataMap` class that can be used to replace certain arguments in functions relating to data sources and annotation creation.
For example, instead of making a Image and Segmentation layers with pre-specified sources, you can add the source as a 'DataMap` object.
Each DataMap has a `key` attribute that is used to map the data you will provide later to the correct role in state creation.

``` pycon
from nglui.statebuilder import ViewerState, SegmentationLayer, DataMap

viewer_state = (
    ViewerState()
    .add_image_layer(source=DataMap('img_source'))
    .add_segmentation_layer(source=DataMap('seg_source'))
)
```

If you tried to run `viewer_state.to_link()` now, you would get an `UnmappedDataError` indicating that these values have yet to be replaced with actual data.

To actually map data, you can use the `map` method of the ViewerState object, which takes a dictionary of key-value pairs where the keys are the DataMap keys and the values are the actual data to be used.
For example, to replicate the previous example with the CAVEclient info, you could do:

``` pycon
viewer_state.map(
    {
        'img_source': 'precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em',
        'seg_source': [
                'graphene://https://minnie.microns-daf.com/segmentation/table/minnie65_public',
            ],
    }
).to_link()
```

This will replace the DataMap keys with the actual data and produce a Neuroglancer link with the specified sources.
Applying this across a list of data sources can easily generate a large collection of neuroglancer states.

The other principle use of DataMaps is to support annotation creation by replacing the `data` argument in the `add_points`, `add_lines`, `add_boxes`, `add_ellipses`, and `add_polylines` methods.
For example, you can create a DataMap for the annotation data and then use it to add points to the annotation layer with the following pattern:

``` pycon
from nglui.statebuilder import ViewerState, AnnotationLayer, DataMap
viewer_state = (
    ViewerState()
    .add_annotation_layer(name='my_annotations', linked_segmentations='seg')
    .add_points(
        data=DataMap('annotation_data'),
        point_column='location',
        segment_column='linked_segment',
        description_column='description',
        data_resolution=[1, 1, 1],
    )
)

viewer_state.map(
    {
        'annotation_data': my_dataframe,
    }
).to_link()
```

Note that the `map` method returns the ViewerState object itself, so you could in principle chain maps together to sequentially replace DataMap values.
Only when all DataMaps are resolv$$ed will the `to_link` or other export functions work without error.

A DataMap with an empty argument is treated as an implicit "None" value and a `map` call that gets anything other than a dictionary will be treated as a dictionary with a single key of `None` and the value being the data to be mapped.
