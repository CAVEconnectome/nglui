# Changelog

This project attempts to follow [Semantic Versioning](https://semver.org) and uses [Keep-a-Changelog formatting](https://keepachangelog.com/en/1.0.0/). But I make mistakes sometimes.

## [3.0.1] – 2023-12-20

### Fixed

- Fixed an unnecessary import that caused an error.
- Added python package neuroglancer to the requirements so versions with errors or incompatibilities can be avoided.

## [3.0.0] – 2023-12-20

### Added

- **StateBuilder** and **EasyViewer** Substantial refactor, adding basic support for the main Google Neuroglancer branch and the Cave-Explorer/Spelunker deployments coming online with CAVE.
Most functionality should be seamless, with the client checking for version.
Currently, only the [Cave-Explorer](https://ngl.cave-explorer.org/) and [Spelunker]((https://ngl.cave-explorer.org/)) deployments are fully supported for this automated version check, as the response depends on a build-specific version.json file.
If no automated version is set or one wants to override any settings, the `target_site` argument can be set to `cave-explorer` or `mainline` for the main Google branch or `seunglab` for the older Seung-lab branch.


## [2.14.1] - 2023-05-14

### Fixed

- **StateBuilder** Recast all layer names as strings to avoid issues with numeric.

## [2.14.0] - 2023-04-12
### Fixed

- **StateBuilder** Fixed various issues with stacking multiple points into a single row.

## [2.13.0] - 2023-02-21

### Added

- **StateBuilder** Adding a helper function for basic line annotation states
- **StateBuilder** Helper functions can take `split_position` argument.

## [2.12.1] - 2023-01-30

### Fixed

- **StateBuilder** Handle None data correctly with mapping sets.

## [2.12.0] - 2023-01-30

### Added

- **EasyViewer** At long last, you can set the background color in `set_view_options` with the argument `background_color`. As elsewhere, this uses [webcolors](https://webcolors.readthedocs.io/) to deal with color parsing, so it can take names or rgb or hex.
- **StateBuilder** StateBuilder can now set the background color in in `view_kws` with the key `background_color`.
- **StateBuilder** *Mapping sets* offer a new and simple iapproach to having multiple dataframes with different mapping rules.
- **StateBuilder** Mapping rules can handle dataframes where the point position components are split across different columns, assuming that the suffices are always `_x`, `_y`, and `_z`. Set `split_positions=True` in the mapping rule to use this.

### Changed

- **StateBuilder** Mapping rules now default to `set_position = True`.

## [2.10.0] - 2022-08-11

### Added

- **StateBuilder** A number of helper functions to produce common states have been added to statebuilder.
You can use the function `make_neuron_neuroglancer_link` to generate a link with one or more root ids and, optionally, their synapses.
Further, you can use the function `make_synapse_neuroglancer_link` to generate a state from a synapse dataframe.
- **StateBuilder** Linked segmentations and annotation groups now support multiple columns.
For linked segmentations, this adds multiple root ids per annotation.
For groups, each unique combinations of values gets its own group.
- **StateBuilder** and **EasyViewer** Colors for layers or segment ids can now be in RGB tuples with values between 0--1 or CSS3 named colors, not just hex.
- **StateBuilder** When using the `client=` argument, viewer resolution is inferred from the client info and does not need to be passed as an additional argument.

### Fixed

- **StateBuilder** For SegmentationLayerConfigs, adding selection maps via `add_selection_map` is more robust.

## [2.7.2] - 2021-06-13

### Fixed

- **StateBuilder** Multipoint now works with `set_position`.

## [2.7.1] - 2021-06-13

### Fixed

- **StateBuilder** Multipoint performance has improved.

## [2.7.0] - 2021-06-13

### Added

- **StateBuilder** In all Mappers, setting `multipoint=True` will allow the point columns to contain multiple points per row, with the other columns assigned to all points in the row. Note that for Mappers with multiple point columns (e.g. LineMappers, SphereMappers, and BoundingBoxMappers), the number of values must be the same in both columns.

## [2.6.0] - 2021-06-11

### Added

- **StateBuilder** You can now set a timestamp in SegmentationLayerConfig. Either datetime or unix epoch are allowed.

## [2.5.0]

- **StateBuilder** You can now set split points using a SplitPointMapper

## [2.4.0] - 2021-01-22

### Added

- **Parser**: `get_selected_ids` does what it says for a layer.

### Fixed

- **Parser**: Getting annotations with tags/descriptions/etc works.
- **StateBuilder**: `return_as` parameter works with ChainedStateBuilder now

## [2.3.1] - 2021-01-15

### Fixed

- Remove pytables from requirements, because the pip install does not work on OS X (at least). If you want to run tests, you will need to install either with `pip install tables` if you're on a system where that works or `conda install pytables` otherwise.

## [2.3.0] - 2021-01-14

### Added

- **StateBuilder**: At long last, a BoundingBoxMapper
- **StateBuilder**: A `from_client` function to generate simple image & segmentation states from a FrameworkClient instance.
- **StateBuilder**: Statebuilder can take a client to configure certain default parameters.
- **StateBuilder**: SegmentationLayerConfig explicitly takes some view keyword arguments: `alpha_selected`, `alpha_3d`, and `alpha_unselected`.
- **Parser**: New options to get annotation groups and group ids for all annotations.
- **Parser**: New function to extract multicut information from a state.

### Fixed

- **StateBuilder**: Fixed a scenario where int64s get altered by a conversion through floats.

### Changed

- **Statebuilder**: Behind the scenes refactoring that should not affect use.

## [2.2.1] — 2020-10-21

### Fixed

- **StateBuilder**: Setting view options for Graphene segmentation layers now works
- **Parser**: Now imported as a property of nglui if you import as `import nglui`.
- Small bug fixes.

## [2.2.0] — 2020-09-04

### Added

- Parser submodule `nglui.parser` for quickly extracting data from neuroglancer states.
This should remove some of the boilerplate one writes every time you want to get data out of a state.

- AnnotationLayerConfig now can take arguments about user interactions such as filtering by segmentation and bracket shortcuts showing segmentations.

### Changed

- Default behavior for annotation layers now has filtering by segmentation turned off and bracket shortcuts showing segmentations turned on.

## [2.1.1] — 2020-08-17

### Changed

- Switched from numpy `isnan` to pandas `isnull` in StateBuilder. This allows nullable pandas Int64 dtypes to work as columns

## [2.1.0] — 2020-07-14

### Added

- Grouped annotations in nglite base and EasyViewer's annotation module.
This feature lets you create annotation groups in the Seung-lab branches of Neuroglancer, where multiple annotations are related to one another.
It works by passing a list of already-created annotations and getting a new "CollectionAnnotation" that groups them together.

- Grouped annotations in StateBuilder. PointMapper, SphereMapper, and LineMapper objects can take a 'group_column' value.
Data in this column is intended to be numeric or NaN, and rows that share the same value are grouped together.
At the moment, this feature only works within individual mapper objects.

- Two options for linked segmentation ids on grouped annotations:
`gather_linked_segmentations` assigns all linked ids of objects within the group to the group annotation itself (True by default).
`share_linked_segmentations` will add all linked objects within the group to all annotations within the group (False by default).

- `array_data` option on AnnotationLayerConfig for simple cases where you just want to map Nx3 arrays to points, a pair of Nx3 arrays to lines, or an Nx3 + N array to spheres (centers+radii).

### Fixed

- Bug in GL Shader that caused StateBuilder to fail when `constrast_controls` was set to True.

## [2.0.2] - 2020-06-18

### Added

- Custom GL Shaders (see [documentation](https://github.com/google/neuroglancer/blob/a12552d03844fb6092cf300171d2f2077b3960e2/src/neuroglancer/sliceview/image_layer_rendering.md)) for image and segmentation layers can be set in EasyViewer.

- JSON State Server can be set in EasyViewer.

- Added `contrast_controls` argument to `statebuilder.ImageLayerConfig` to provide brightness/contrast controls to layer through GL Shader.

- Added `state_server` argument to `statebuilder.StateBuilder` to pre-set state server endpoints.

## [2.0.1] - 2020-04-27

### Fixed

- Removed unnecessary parts of the included neuroglancer module (`nglite`) for cleanliness.

## [2.0.0] - 2020-04-25

This update significantly changed the underlying nature and goals of the Neuroglancer Annotation UI project.

### Added

- StateBuilder module for rule-based method of generating Neuroglancer states from Pandas DataFrames.

- Added various features to make use of specific aspects of the Seung-lab Neuroglancer fork.

### Changed

- Removed dependency on neuroglancer python package, which was not designed for the Seung-Lab Neuroglancer fork and had some difficult install requirements for features that were unnecessary for this project demanded.

### Removed

- Dynamic state management via python server.
