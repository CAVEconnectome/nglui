# NGLui

NGLui is a library for programmatically generating shareable Neuroglancer states in order to explore and visualize large, 3d image datasets.
It is designed to be used in conjunction with the [Neuroglancer](https://github.com/google/neuroglancer) web viewer and the [CAVE analysis ecosystem](https://caveconnectome.github.io/CAVEclient/).
It aims to separate the *rules* you want to use to generate states from the *data* you want to visualize, allowing you make reusable code to generate states from different datasets and analyses.

## Installation

NGLui is available on PyPI and can be installed with pip:

```bash
pip install nglui
```

If you want to clone the repository and develop on NGLui, note that it uses [Hatch](https://hatch.pypa.io/latest/) for development and packaging.

## Quick Usage

!!! note

    Using the CAVEclient with the MICrONs dataset is required for the following examples.

Here's a quick example of how to use NGLui to generate a Neuroglancer state from the [Microns cortical dataset](https://www.microns-explorer.org).

```python
import caveclient
from nglui import statebuilder

client = caveclient.CAVEclient('minnie65_public')

# Get a root id of a specific neuron
root_id = client.materialize.query_table(
    'nucleus_detection_v0',
    filter_equal_dict={'id': 255258}
)['pt_root_id']

statebuilder.helpers.make_neuron_neuroglancer_link(
    client,
    root_id,
    show_inputs=True,
    show_outputs=True,
    ngl_url='https://spelunker.cave-explorer.org',
)
```

This code will generate a link to produce a Neuroglancer state showing a neuron and its synapses.