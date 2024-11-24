---
title: StateBuilder
---

The Statebuilder is a submodule that helps produce custom Neuroglancer states based on data in the form of Pandas dataframes. Neuroglancer organizes data sources into layers. Layers come in three types, image, segmentation, and annotation. Each has different properties and functions. The state builder lets the user define a set of rules for initializing layers and how to map data columns to selections and annotations. Let's see the simplest example in use now.

```python
from nglui.statebuilder import *
```

### Image Layers

Image layers in Neuroglancer are pretty simple. An image has a name (by default 'img' here) and a source, which is typically a path to a cloud hosted 3d image volume. We define an image layer with an ImageLayerConfig that lets the user set these key parameters. We are going to use the public Layer 2/3 EM dataset at [Microns Explorer](https://layer23.microns-explorer.org) as an example.

```python
img_source = 'precomputed://gs://neuroglancer/pinky100_v0/son_of_alignment_v15_rechunked'
img_layer = ImageLayerConfig(name='layer23',
                             source=img_source,
                             )
```

### Segmentation Layers

Segmentation layers in Neuroglancer are more complex, since each object has a unique id and can be selected, loading the object and its mesh into the neuroglancer state. Like images, a segmentation layer has a name and a source. Neuroglancer supports two types of segmentation volume, 'precomputed' and 'graphene'. Precomputed is what we call a "flat" segmentation, in that the segmentation is frozen into an unchanging state. Flat segmentations are fast, but cannot be edited to fix mistakes. Graphene segmentations are dynamic, using an efficient graph data representation to allow edits to the segmentation state. For the most part, users should not need to care about the difference.

Segmentation layers are configured by a SegmentationLayerConfig. At a minimum, a SegmentionLayerConfig has the same setup as an ImageLayerConfig.

```python
seg_source = 'precomputed://gs://microns_public_datasets/pinky100_v185/seg'
seg_layer = SegmentationLayerConfig(name = 'seg',
                                    source = seg_source)
```

### Annotation Layers

Annotation layers let a user define various spatial data annotations. We can define annotation layers with AnnotationLayerConfig objects. While annotation layers have a name, they don't have a source. We'll discuss how to map data to annotations later, but for now we will add a blank annotation layer.

```python
anno_layer = AnnotationLayerConfig(name='annos')
```

### StateBuilders and state rendering

At it's most basic, the StateBuilder is initialized with layer configs for each layer the user wants. A state isn't actually generated until the user calls `render_state`. While the function can also take a dataframe, it works with no argument to generate a default state. The default output is a url string that specifies the neuroglancer state, however other options are available with the optional `return_as` parameter.

```python
sb = StateBuilder(layers=[img_layer, seg_layer, anno_layer])
sb.render_state()
```

Using `return_as="html"` provides a link, useful for interactive notebooks.

```python
sb.render_state(return_as='html')
```

Using `sb.render_state(return_as='json')` returns the JSON state as a string.
This can be pasted directly into the neuroglancer JSON state and `sb.render_state(return_as='dict')` returns the state as a dictionary, useful for inspection and debugging.
Finally, using `sb.render_state(return_as='viewer')` returns an EasyViewer object, which can be further manipulated (see documentation).

!!!warning

    The URL at which you want to view your state matters! While there are reasonable defaults, please look at the [configuration documentation](config.md) to make sure you are using the right settings for what you want.

## Data-responsive state generation

The key feature of the StateBuilder is to use data to add selected objects or annotations to a Neuroglancer state. Each layer has rules for how to map dataframe columns to its state. This is designed to be useful for consistent mapping of queries or analysis directly into data exploration.

### Selected segmentations

Segmentation layers principally control what what objects are selected. Objects are specified by root id and `SegmentationLayerConfig` can hold rules for how to select data.

The `soma_df` example dataframe describes all excitatory neurons in the layer23 dataset. Each row is a different cell and it is described by a number of columns, of which 'pt_root_id' has the root id for each excitatory neuron.

```python
import pandas as pd

soma_df = pd.read_hdf('soma_data.h5', 'soma')
soma_df.head()
```

One common task is to select all the ids in a column of the dataframe, perhaps as a result of a query or some bit of analysis. We can tell the `SegmentationLayerConfig` which layer (or list of layers) to use with the `selected_ids_column` argument.

```python
seg_layer = SegmentationLayerConfig(name = 'seg',
                                    source = seg_source,
                                    selected_ids_column='pt_root_id')

sb = StateBuilder(layers=[img_layer, seg_layer])
```

Now our statebuilder object also needs data. The `render_state` method can always take a dataframe. To avoid selecting hundreds of neurons, let's just take the first five rows of the dataframe. 

```python
sb.render_state(soma_df.head(), return_as='html')
```

For some purposes, it can also be useful to force certain objects to be selected no matter the data, for example if you want to show various points in space relative to a specific neuron.
For that, we can use the `fixed_ids` argument. Here, we use it to load neuron `648518346349537042` no matter the data. Both data-driven and fixed ids can be used together.

```python
seg_layer = SegmentationLayerConfig(name = 'seg',
                                    source = seg_source,
                                    selected_ids_column='pt_root_id',
                                    fixed_ids=[648518346349537042])

sb = StateBuilder(layers=[img_layer, seg_layer, anno_layer])
```

### Assigning colors to cells

You can also provide a column that species the color for each root id with the `color_column` argument. Colors in Neuroglancer are specified as an RGB hex string, for example a pure red would be `#ff0000`.

In the example below, we specify the color list directly. Data visualization packages typically have methods to convert numeric RGB vectors to hex, for instance `matplotlib.colors.to_hex` for `matplotlib`.

```python
# First we have to add a column with legitimate colors
reds = ['#fdd4c2', '#fca082', '#fb694a', '#e32f27', '#b11218']
soma_colored_df = soma_df.head(5).copy()
soma_colored_df['color'] = reds

# Next we specify the color column when defining the segmentation layer.
seg_layer = SegmentationLayerConfig(name = 'seg', source=seg_source, selected_ids_column='pt_root_id', color_column='color')

sb = StateBuilder(layers=[img_layer, seg_layer])
sb.render_state(soma_colored_df, return_as='html', link_text='State with color')
```

### Data-driven annotations

Neuroglancer offers Annotation layers, which can put different kinds of markers in the volume. There are three main marker types:

* Points
* Lines
* Spheres

Each annotation layer can hold any collection of types, however colors and other organizational properties are shared among all annotations in one layer. To make a new annotation layer, we use an `AnnotationLayerConfig`. Unlike segmentation and image layers, there is no data source, but the data mapping options are more rich. Each annotation type has a mapper class (PointMapper, LineMapper, SphereMapper) to designate the rules. Each AnnotationLayerConfig works for a single annotation layer, but can take an arbitrary number of mappers.

#### Point annotations

Point annotations are a simple point in 3d space. Let's use the positions from the soma_df above. The only thing you need to set a point mapper is the column name, which should reference a column of 3-element locations in units of voxels.

```python
points = PointMapper(point_column='pt_position')
anno_layer = AnnotationLayerConfig(name='annos',
                                   mapping_rules=points )


# Make a basic segmentation source
seg_layer = SegmentationLayerConfig(seg_source)

sb = StateBuilder([img_layer, seg_layer, anno_layer])
sb.render_state(soma_df, return_as='html')
```

#### Line annotations

Line differ from point annotations only by requiring two columns, not just one. The two columns set the beginning and end points of the line and are thus both columns should contain three element points. Let's use the example synapse dataframe as an example. We're going to use the `pre_pt_root_id` field to select the neuron ids to show and use the lines to indicate their outgoing synapses, which go from `ct_pt_position`, a point on the synaptic cleft itself, to `post_pt_position`, a point somewhat inside the target neuron.

```python
lines = LineMapper(point_column_a='ctr_pt_position', point_column_b='post_pt_position')
anno_layer = AnnotationLayerConfig(name='synapses',
                                   mapping_rules=lines)

# Make a segmentation source with a selected ids column
seg_layer = SegmentationLayerConfig(seg_source, selected_ids_column='pre_pt_root_id')

sb = StateBuilder([img_layer, seg_layer, anno_layer])
sb.render_state(pre_syn_df, return_as='html')
```

#### Sphere annotations

Like line annotations, sphere annotations take two columns. The first is the center point (and thus a point in space) while the second is the radius in voxels (x/y only), and thus numeric. We're going to use the soma positions for the example, using the `pt_position` for the centers. Since we don't have radius data in the frame, we will add a column with a random radius.

```python
import numpy as np
soma_df['radius'] = np.random.normal(1500, 250, len(soma_df)).astype(int)


spheres = SphereMapper(center_column='pt_position', radius_column='radius')
anno_layer = AnnotationLayerConfig(name='soma', mapping_rules=spheres)

# Make a basic segmentation layer
seg_layer = SegmentationLayerConfig(seg_source)

sb = StateBuilder([img_layer, seg_layer, anno_layer])
sb.render_state(soma_df, return_as='html')
```

### Enriching annotations

Annotations can be enriched with metadata. There are three main types:
1. Descriptions — Each annotation has a free text field.
2. Linked Segmentations — Each annotation can have one or more linked segmentation ids. These can be made to automatically load on annotation selection.
3. Tags — Neuroglancer states can have discrete tags that are useful for quickly categorizing annotations. Each annotation layer has a list of tags, and each annotation can have any number of tags.

#### Descriptions
Descriptions need a simple free text field (or None). Any annotation mapper can take a description column with strings that is then displayed with each annotation in Neuroglancer. After running the following code, right click on the annotation layer and you can see that each of the point annotions in the list has an 'e' or 'i' letter underneath it.

```python
points = PointMapper(point_column='pt_position', description_column='cell_type')

anno_layer = AnnotationLayerConfig(name='annos', 
                                   mapping_rules=points)

# Basic segmentation layer
seg_layer = SegmentationLayerConfig(seg_source)

sb = StateBuilder([img_layer, seg_layer, anno_layer])
sb.render_state(soma_df.head(), return_as='html')
```

#### Linked Segmentations

An annotation can also be linked to an underlying segmentation, for example a synapse can be linked to its neurons. On the Neuroglancer side, the annotation layer has to know the name of the segmentation layer to use, while the annotation needs to know what root id or ids to look up. To make data-driven annotations with linked segmentations, we both
1. Add a linked segmentation column name to the annotation Mapper class that will be one or more column names in the dataframe
2. Pass a segmentation layer name to the AnnotationLayerConfig. This defaults to `None`. Note that while the default segmentation layer name is `seg`, no segmentation layer name is set by default. Thus, if you plan to use a segmentation layer that was not given an explicit name, use `seg` as the argument here.

```python
points = PointMapper('pt_position', linked_segmentation_column='pt_root_id')
anno_layer = AnnotationLayerConfig(mapping_rules=points, linked_segmentation_layer='seg')

# Basic segmentation layer
seg_layer = SegmentationLayerConfig(seg_source)

sb = StateBuilder([img_layer, seg_layer, anno_layer])
sb.render_state(soma_df.head(), return_as='html')
```

#### Tags (Seung-lab neuroglancer only)

Tags are categorical labels on annotations. Each annotation layer can have a defined set of up to ten tags (seen under the "shortcuts" tab). Each annotation can have any number of tags, toggled on and off with the key command from the shortcut tab. To pre-assign tags to annotations based on the data, you can assign a `tag_column`. For each row in the data, if the element in the tag column is in the annotation layer's tag list, it will assign it to the resulting annotation. Elements of the tag column can also be collections of values, if you want multiple tags assigned. Note that any values that are not in the layer's tag list are ignored.

As before, there are two steps:
1. If you want to pre-assign tags, add a `tag_column` argument to the annotation Mapper. This isn't needed if you just want to set tags for the layer.
2. Pass a list of tags to the AnnotationLayerConfig. The order of the list matters for determining the exact shortcuts that are used for each tag; the first tag has the shortcut `shift-q`, the second tag `shift-w`, etc.

```python
points = PointMapper('pt_position', tag_column='cell_type')
anno_layer = AnnotationLayerConfig(mapping_rules=points, tags=['e'])

sb = StateBuilder([img_layer, seg_layer, anno_layer])
sb.render_state(soma_df, return_as='html')
```

## Setting the View

The StateBuilder offers some control over the initial position of the view and how data is visualized, while offering some fairly sensible defaults that look okay in most situations.

#### Position, layout, and zoom options

View options that do not affect individual layers can be set with a dict passed to the `view_kws` argument in StateBuilder, which are passed to `viewer.set_view_options`.

* *show_slices* : Boolean, sets if slices are shown in the 3d view. Defaults to False.
* *layout* : `xy-3d`/`xz-3d`/`yz-3d` (sections plus 3d pane), `xy`/`yz`/`xz`/`3d` (only one pane), or `4panel` (all panes). Default is `xy-3d`.
* *show_axis_lines* : Boolean, determines if the axis lines are shown in the middle of each view.
* *show_scale_bar* : Boolean, toggles showing the scale bar.
* *orthographic* : Boolean, toggles orthographic view in the 3d pane.
* *position* : 3-element vector, determines the centered location.
* *zoom_image* : Zoom level for the imagery in units of nm per voxel. Defaults to 8.
* *zoom_3d* : Zoom level for the 3d pane. Defaults to 2000. Smaller numbers are more zoomed in.
* *background_color* : Sets the background color of the 3d mode. Defaults to black.

Here's an example of setting some of these rules. Note that only providing default values for some parameters does not override the default values of others.

```python
view_options = {'layout': '4panel',
                'show_slices': True,
                'zoom_3d': 500,
                'position': [71832, 54120, 1089],
                'background_color': 'white',
                }

seg_layer = SegmentationLayerConfig(seg_source)

sb = StateBuilder([img_layer, seg_layer], view_kws=view_options)
sb.render_state(return_as='html')
```

#### Data-driven centering

It can also be convenient to center the user on an annotation by default.
Because this is tied to a specific spatial point column, this option is associated with a particular AnnotationMapper.
Data-driven view centering takes precidence over the global view set in `view_kws`.
For any of the mappers, setting `set_position` to `True` will center the view on the first annotation in the list.
If multiple Mapper objects have `set_position=True`, then the view will follow the end-most one in the end-most annotation layer.

This option is now set to True by default, but can be turned off if desired with `set_position=False`.

```python
points = PointMapper('pt_position', set_position=True)
anno_layer = AnnotationLayerConfig(mapping_rules=points)

sb = StateBuilder([img_layer, seg_layer, anno_layer])
sb.render_state(soma_df, return_as='html')
```

#### Segmentation transparency options

Each segmentation layer can control the transparency of selected, unselected, and 3d meshes. As with the global view, these can be passed as keyword arguments to `view_kws` in the SegmentationLayerConfig.

* *alpha_selected* : Transparency (0–1) of selected segmentations in the imagery pane. Defaults to 0.3. 
* *alpha_3d* : Transparency (0–1) of selected meshes in the 3d view. Defaults to 1.
* *alpha_unselected* : Transparency (0–1) of unselected segmentations in the imagery pane. Defaults to 0.

## Applying multple rules with multiple dataframes

The default use of a Statebuilder only takes one dataframe, but there are many cases when multiple dataframes are useful to pass different kinds of points, for example pre and postsyanptic points of the same neuron, or synapses and soma.

### Mapping Sets

The simplest option is to handle this with `mapping_sets`, which let you name dataframesa and mapping rules.
For each annotation or segmentation layer mapping, you can optionally set a `mapping_set` argument.
This changes the way data is processed, and the statebuilder now expects to get the dataframe as a dictionary with keys that are the mapping sets and values that are the dataframes.
Multiple mapping rules can use the same `mapping_set` value (and pull info from the same dataframe).
Also, if you use mapping sets for any mapping rule, they must be used for all mapping rules and the data must be passed as a dictionary.

Let's use mapping rules to make a statebuilder that displays presynaptic and postsynaptic points for the same neuron.

```python
seg_layer = SegmentationLayerConfig(seg_source, selected_ids_column='post_pt_root_id')

postsyn_mapper = LineMapper(
    point_column_a='pre_pt_position',
    point_column_b='ctr_pt_position',
    mapping_set='post',     # This tells the LineMapper to use the dataframe passed with the dictionary key `post`
)
postsyn_annos = AnnotationLayerConfig('post', color='#00CCCC', mapping_rules=postsyn_mapper)

presyn_mapper = LineMapper(
    point_column_a='ctr_pt_position',
    point_column_b='post_pt_position',
    mapping_set='pre',      # This tells the LineMapper to use the dataframe passed with the dictionary key `pre`
)
presyn_annos = AnnotationLayerConfig('pre', color='#CC1111', mapping_rules=presyn_mapper)

sb = StateBuilder([seg_layer, postsyn_annos, presyn_annos], client=client)

# Note that the data is passed as a dictionary with the same keys as the mapping rules above.
sb.render_state(
    {
        'post': post_syn_df,
        'pre': pre_syn_df,
    },
    return_as='html'
)
```