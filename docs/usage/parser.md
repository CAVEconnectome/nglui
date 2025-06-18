---
title: Parser
heading_level: 2
---

The `parser` module offers a number of tools to get information about neuroglancer states out of the JSON format that neuroglancer uses.
The recommended approach here is to pass a dictionary representation of the JSON object the `StateParser` class and build various kinds of dataframes from it.

## Importing a State

We will use the `CAVEclient` class to get a state JSON based on its ID from the "share" button, but you could also use the text you can download from the `{}` button in the viewer or a JSON file you have saved locally.

```python
from caveclient import CAVEclient
from nglui import parser

client = CAVEclient('minnie65_public')
state_json = client.state.get_state_json(6107390807113728)
state_parser = parser.StateParser(state_json)
```

You can now access different aspects of the state through this `state_parser` object.

## Layer Data 

For example, to get a list of all layers and their core info, you can use the `layer_dataframe` method.

```python
state_parser.layer_dataframe()
```

will give you a table with a row for each layer and columns for layer name, type, source, and whether the layer is archived (i.e. visible) or not.

You can also get a list of all selected segments with the `selection_dataframe` method.

```python
state_parser.selection_dataframe()
```

## Annotation Data

will give you a dataframe where each row is a selected segment, and columns show layer name, segment id, and whether or not the segment is visible.

Finally, you can get a list of all annotations with the `annotation_dataframe` method.

```python
state_parser.annotation_dataframe()
```

will give you a dataframe where each row is an annotation, and columns show layer name, points locations, annotation type, annotation id, linked segmentations, tags, etc. If you are using tags, the `expand_tags=True` argument will create a column for every tag and assign a boolean value to the row based on whether the tag is present in the annotation.
Another option that is sometimes useful is `split_points=True`, which will create a separate column for each x, y, or z coordinate in the annotation.

