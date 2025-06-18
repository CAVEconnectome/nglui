---
title: Skeletons
---

Neuroglancer treats skeletons as a collection of vertices and edges and an arbitrary collection of properties that can be mapped to the scene vizualization using GL shaders.
Because of that, skeletons — which can even just be point clouds! — offer a way to visualize complex neuronal structures in an interactive mannger.

However, to use your own skeletons in Neuroglancer you need to create a readable cloud storage bucket with an appropriate structure and info file to host skeletons.
The new `nglui.skeletons` module helps ease this process.

!!! important "Cloud-volume required!"

    The `nglui.skeletons` module requires the `cloud-volume` package to be installed.
    You can install it with `pip install cloud-volume` or `pip install nglui[full]`.

## Creating a SkeletonManager

The `SkeletonManager` class is the main entry point for working with skeletons in NGLui.
In order to make a new skeleton bucket, you need to know the following information:

    - 'segmentation_source`: This is an existing segmentation source whose ids are associated with the skeletons you want to make. This can also be a caveclient, in which case the segmentation source is used.
    - `cloudpath`: This is the cloud storage bucket where the skeletons will be stored. It must be a path that you have access to and have configured `cloud-volume` to write to. Please see the [cloud-volume documentation](https://github.com/seung-lab/cloud-volume?tab=readme-ov-file#credentials) on setting up credentials for more information.
    - `vertex_attributes`: This is a list of attribute names that store properties for skeleton vertices. Because these must be specified in the info file for all skeletons, they must be set ahead of time.

From there, you can upload skeletons to the bucket using the `SkeletonManager` methods, which will automatically create the appropriate structure and info file for you.
If you already have an existing skeleton bucket, you can provide just a `segmentation_source` and `cloudpath` and the `SkeletonManager` will read the info file and use it to manage skeletons in that bucket.
The attributes will be automatically loaded from the info file, so you do not need to specify them again.

For example, let's build a skeleton manager that uses the `caveclient` as the segmentation source, stores skeletons in a Google Cloud Storage bucket, and has a vertex attribute called `radius`.
Assume we have a skeleton `sk` with segment id `12345` and a structure represented with a dictionary with `'vertices'` as a numpy array of shape `(n, 3)`, `edges` as a numpy array of shape `(n,2)` and `'radius'` as a numpy array of shape `(n,)`:

``` py
from nglui import skeletons

skm = skeletons.SkeletonManager(
    segmentation_source=client,
    cloudpath="gs://my-bucket/ngl-skeletons",
    vertex_attributes=['radius'],
    initialize_info=True,
)

skm.upload_skeleton(
    root_id=12345),
    vertices=sk['vertices'],
    edges=sk['edges'],
    vertex_attributes={'radius': sk['radius']},
)
```

This will create the appropriate structure in the bucket, including an `info` file that describes the skeletons and their attributes, and then upload this skeleton.
You will be able to see the skeleton in Neuroglance if this location is added as a source for a segmentation layer.
The source path needed for Neuroglancer can be found at the `skm.skeleton_source` property, which will be something like `precomputed://gs://my-bucket/ngl-skeletons`.

You can also make a segmentation layer directly using the `to_segmentation_layer` method, which will return a `SegmentationLayer` object that you can use in a Neuroglancer state.
In addition, it will automatically select the segment ids you have uploaded in this session for quick visulization.

For example, to visualize the skeletons we just uploaded in a complete Neuroglancer state, we could just do:

``` py
(
    ViewerState()
    .add_image_layer(client.info.image_source())
    .add_layer(skm.to_segmentation_layer())
).to_link()
```
