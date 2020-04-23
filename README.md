*Due to evolving needs of the connectomics projects here and at Google, the current form of nglite works only on the [Seung Lab branch of Neuroglancer](https://github.com/seung-lab/neuroglancer) and is currently incompatible with the Google branch.*

This package offers a set of tools designed to ease the programmatic generation of Neuroglancer states and close the loop between analysis and data exploration. The key tool is the notion of a StateBuilder that is easily configured to map dataframes into neuroglancer states. See the [example notebook](https://github.com/seung-lab/NeuroglancerAnnotationUI/blob/master/examples/statebuilder_examples.ipynb) for what it can do an how it can simplify the process of making neuroglancer states, including setting annotations, segmentations, colors, tags, and more. It's also designed to operate easily with [DashDataFrame](https://github.com/AllenInstitute/DashDataFrame) to interactively explore complex data.

Install from PyPi with 
```
pip install nglui
```

All code in the 'nglite' submodule is stripped down from v1.1.6 of the 'neuroglancer' python module from the fantastic [Neuroglancer](https://github.com/google/neuroglancer) suite by Jeremy Maitin-Shepard from the Google connectomics team. Anything that works is their fault, anything that doesn't is ours. Please do not use this as a replacement for neuroglancer on pypi, which has more capabilities, albeit more sensitive installation procedures, than are needed for this work.
