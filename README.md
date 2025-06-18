
# NGLui

## Documentation

Please find [documentation here](https://caveconnectome.github.io/nglui/).

## Installation

To get the most out of NGLui (interacting with source info, uploading skeletons, and more), we suggest installing the full version of NGLui, which includes the [`cloud-volume`](https://github.com/seung-lab/cloud-volume/) dependency:

``` bash
pip install nglui[full]
```

You can also install a more minimal version of NGLui without the cloud-volume dependency:

``` bash
pip install nglui
```

However, note that cloud-volume is required for some features such as uploading skeletons and getting information about sources during state generation.

## Quick Usage

### Building a Neuroglancer state directly

Here, let's use the Hemibrain dataset information to build a Neuroglancer state.

``` py
from nglui import statebuilder

viewer_state = (
    statebuilder.ViewerState(dimensions=[8,8,8])
    .add_image_layer(
        source='precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg',
        name='emdata'
    )
    .add_segmentation_layer(
        source='precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation',
        name='seg',
        segments=[5813034571],
    )
    .add_annotation_layer(
        source='precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/synapses',
        linked_segmentation={'pre_synaptic_cell': 'seg'},
        filter_by_segmentation=True,
        color='tomato',
    )
)
viewer_state.to_link(target_url='https://hemibrain-dot-neuroglancer-demo.appspot.com')
```

This will return the link: [Neuroglancer link](https://hemibrain-dot-neuroglancer-demo.appspot.com#!%7B%22position%22:%5B17216.0,19776.0,20704.0%5D,%22layout%22:%22xy-3d%22,%22dimensions%22:%7B%22x%22:%5B8e-09,%22m%22%5D,%22y%22:%5B8e-09,%22m%22%5D,%22z%22:%5B8e-09,%22m%22%5D%7D,%22crossSectionScale%22:1.0,%22projectionScale%22:50000.0,%22showSlices%22:false,%22layers%22:%5B%7B%22type%22:%22image%22,%22source%22:%5B%7B%22url%22:%22precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg%22,%22subsources%22:%7B%7D,%22enableDefaultSubsources%22:true%7D%5D,%22shader%22:%22None%22,%22name%22:%22emdata%22%7D,%7B%22type%22:%22segmentation%22,%22source%22:%5B%7B%22url%22:%22precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.0/segmentation%22,%22subsources%22:%7B%7D,%22enableDefaultSubsources%22:true%7D%5D,%22segments%22:%5B%5D,%22selectedAlpha%22:0.2,%22notSelectedAlpha%22:0.0,%22objectAlpha%22:0.9,%22segmentColors%22:%7B%7D,%22meshSilhouetteRendering%22:0.0,%22skeletonRendering%22:%7B%7D,%22name%22:%22seg%22%7D,%7B%22type%22:%22annotation%22,%22source%22:%5B%7B%22url%22:%22precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.0/synapses%22,%22subsources%22:%7B%7D,%22enableDefaultSubsources%22:true%7D%5D,%22linkedSegmentationLayer%22:%7B%22pre_synaptic_cell%22:%22seg%22,%22post_synaptic_cell%22:%22seg%22%7D,%22filterBySegmentation%22:%5B%22pre_synaptic_cell%22,%22post_synaptic_cell%22%5D,%22shader%22:%22%5Cnvoid%20main()%20%7B%5Cn%20%20setColor(defaultColor());%5Cn%7D%5Cn%22,%22name%22:%22annotation%22%7D%5D%7D).

### Additional features

NGLui also has additional features such as:

- **CAVE**: Broad integrations with existing CAVE tooling.
- **Parser**: Parse neuroglancer states to extract information about layers and annotations.
- **SegmentProperties**: Easily build segment property lists from data to make segmentation views more discoverable.
- **SkeletonManager**: Upload skeletons to cloud buckets and push quickly into neuroglancer (requires cloud-volume, see [Installation](#installation)).
- **Shaders**: Support for better default shaders for neuroglancer layers.

## Development

If you want to clone the repository and develop on NGLui, note that it uses [uv](https://astral.sh/blog/uv) for development and packaging, [material for mkdocs](https://squidfunk.github.io/mkdocs-material/) for documentation, and [pre-commit](https://pre-commit.com/) with [ruff](https://docs.astral.sh/ruff/) for code quality checks.
[Poe-the-poet](https://poethepoet.natn.io/index.html) is used to simplify repetitive tasks, and you can run `poe help` to see the available tasks.

## Migration from older versions

If you are migrating from `nglui` v3.x to v4.0.0+, you will need to dramatically update your code.

First and foremost, `nglui` now only works with contemporary versions of neuroglancer, *not* the older Seung-lab version.
If you still need to support the older deployment, do not upgrade.

Please read the new usage documentation!
The main change is that it is now recommended to create states directly where possible, and there are now many more convenience functions.
Instead of making a bunch of layer configs, now you make a `ViewerState` object and directly add layers and their information with functions like `add_image_layer`, `add_segmentation_layer`, and `add_annotation_layer`.
Instead of always mapping annotation rules and data separately, you can now directly add annotation data through functions like `add_points` and then export with functions like `to_url`.
You can still use the old pattern of rendering a state and mapping data with [DataMap](https://caveconnectome.github.io/nglui/usage/statebuilder/#mapping-data) objects.
A new "pipeline" pattern makes it more efficient to build complex states in a smaller number of lines of code.
