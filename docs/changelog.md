# Changelog

This project attempts to follow [Semantic Versioning](https://semver.org) and uses [Keep-a-Changelog formatting](https://keepachangelog.com/en/1.0.0/). But I make mistakes sometimes.

## [4.1.0] - 2025-06-24

## Added

- **StateBuilder**: Added the `pick` option to layers that enable you to turn off new ID selection (`pick=False`).
- **StateBuilder**: Added `color` options to `add_points`/`add_lines`/etc methods to set the layer color.
- **StateBuilder**: Converted `viewer_state.layers` to a modified list class with names. Only objects with a `.name` attribute can be added to the list, and objects can be accessed by either name or index.
- **StateBuilder**: Added a `client` argument to `ViewerState` that will be used as a default for any viewer_state functions that take a client.

## [4.0.0]

This is a **major breaking release** that fully changes the way most of the library works.
Code from previous verions will not work without significant changes.
Parser and SegmentationProperties work as before, but the StateBuilder and EasyViewer modules have been merged with a different pattern and approach.
Only the current main and Spelunker branches of Neuroglancer are supported, and the older Seung-lab branch is no longer supported.
The following is a small summary of the changes.

### Added

- **StateBuilder/EasyViewer**: Complete rewrite joining StateBuilder and Easyviewer into a single module under the statebuilder namespace. Only supports spelunker.
- **StateBuilder**: A new SkeletonManager to upload skeletons for use in Neuroglancer.
- **StateBuilder**: Support for building some shaders.
- **StateBuilder**: Added inference of properties from source info files.
- **StateBuilder**: Added DataMaps to several places to allow for more flexible data handling and rendering.

#### Removed

- **StateBuilder/EasyViewer**: Fully removed support for the Seung-lab branch of Neuroglancer. This is no longer supported and will not be updated.
- **StateBuilder/EasyViewer**: Removed both main classes and their functionality. Functionality is now merged into the ViewerState class.

## [3.8.2] - 2025-03-27

### Fixed

- **Parser** Fix issues with dimensions in the parser.

## [3.8.1] - 2025-01-30

### Fixed

- **StateBuilder** SegmentationLayerConfig with a color column will ignore `None` values for data.

## [3.8.0] - 2025-01-29

### Added

- **StateBuilder** Segmentation layers can be set to be non-interactive by default using the `selectable` parameter in `SegmentationLayerConfig`. Only works for `spelunker` branch.

### Fixed

- **StateBuilder** Adjusted sphere radius scaling to be correct.


## [3.7.2] - 2024-12-18

### Fixed

- **Parser**: Fixed a bug in annotation parsing if some layers have tags and some do not.

## [3.7.1] - 2024-12-11

### Fixed

- **StateBuilder**:  Fixed a bug where the use of "helpers.make_state_url" (which includes most of the helpers) was not using information about urls to infer target sites.

## [3.7.0] - 2024-12-10

### Added

- **StateBuilder** and **SegmentProps**: A new parameter for segment properties will create columns with a random number for you in a segment property, useful for sampling large lists of ids in neuroglancer.
- **StateBuilder**: Made it so that `site_utils` configuration can be accessed from `statebuilder.site_utils` instead of needing a separate import.

### Fixed

- **StateBuilder**: Updated helpers to be compatible with new site_utils functionality.

## [3.6.2] - 2024-11-22

### Fixed

- Fixed a bug where default configurations were overriding user configurations for target sites.

## [3.6.1] - 2024-11-20

### Added

- **StateBuilder**: Added default configurations to make it much easier to configure links where you want them to go.

## [3.5.2] - 2024-11-11

### Fixed

- Fixed behavior of `statebuilder.helpers.make_neuron_neuroglancer_link` to actually render the state as it used to.

## [3.5.1] - 2024-11-10

### Fixed

- Removed inadvertent debug statement in the last release.
- Fixed a bug in the `expand_tags` option in the `annotation_dataframe` function.

## [3.5.0] - 2024-11-10

### Added

- **StateBuilder**: Added tags to Spelunker neuroglancer states! Note that this might not be available in all mainline-style deployments yet.
- **Parser**: Added a number of new features to parse classic and Spelunker-style states, including layer and segment information, filtering archived layers, and a new class-level interface to make life easier.

### Fixed

- Reformatted use of `target_site` to be more consistent when using "spelunker" as a name.


## [3.4.0] - 2024-08-26

### Added

- **StateBuilder** : Added `use_skeleton_source` as an optional argument for `from_client` to use a skeleton service advertised in the info service.
- **SegmentProperties** New option `prepend_col_name` in `SegmentProperties.from_dataframe` will prepend the column name to the tag values.

## [3.3.7] - 2024-07-24

### Added

- **StateBuilder** New option on `StateBuilder.render_as`: `return_as="short"` will upload the state and return a short URL to the state.
- **StateBuilder** New helper `statebuilder.helpers.segment_property_link` will quickly generate a basic link from a segment property.

### Changed

- Default "spelunker" neuroglancer deployment was changed.

### Fixed

- **StateBuilder** Fixed a bug when setting the background color in Spelunker sites.
- **StateBuilder** Fixed a bug in using segment property maps.


## [3.3.6] - 2024-07-23

### Added

- **SegmentProperties** New method "label_format_map" allows you to build arbitrary label formats from dataframe columns using 

### Fixed

- **SegmentProperties** Nulls in label columns are ignored as intended.

## [3.3.5] - 2024-07-19

### Added

- **SegmentProperties** The `label_col` argument in `SegmentProperties.from_dataframe` can now take a list of column names. Labels are concatenated with a seperator set with `label_separator` that defaults to an underscore (`_`).
- **StateBuilder** Recent segment property arguments included in the segment property maps for SegmentationLayerConfigs.

### Changed

- **SegmentProperties** All property names (the `id` field) are coerced to strings, as required.

## [3.3.4] - 2024-07-19

### Fixed

- **SegmentProperties**: Fixed a bug in building tags from dataframes.


## [3.3.3] - 2024-07-19

### Added

- **SegmentProperties**: Tag columns will now automatically disambuguate tags if they are duplicated in different columns of the dataframe.
For example, if you had "my_column" with value "type_a" and "their_column" also with value "type_a", the tag would become "my_column:type_a" and "their_column:type_a". This can be turned off by setting the `allow_disambiguation` argument to False.
- **SegmentProperties**: The `SegmentProperties.from_dataframe` method now has a `allow_disambiguation` argument to control whether the disambiguation above is performed.

### Changed

- **SegmentProperties**: Tag property generation performance for long dataframes is improved.


### Fixed

- **SegmentProperties**: Fixed a bug in handling tag columns with categorical dtype.

## [3.3.2] - 2024-07-18

### Changed

- **StateBuilder**: When providing a URL but not a client, you will now get a warning that target-site cannot be inferred.

### Fixed

- Small changes to work in python 3.8, because the only issue was type hints.

## [3.3.1] - 2024-07-12

### Fixed

- Removed debug statements left in after testing.

## [3.3.0] - 2024-07-12

### Added

- **StateBuilder** : Added a `skeleton_source` and `skeleton_shader` parameter to SegmentationLayerConfig to put skeleton info into the state.
- **EasyViewer** : Added `set_skeleton_source` and `set_skeleton_shader` methods to EasyViewer.

### Fixed

- **SegmentProperties** : Better handling of null values (nan, None, and empty strings) in tag columns, all of which are ignored.
- Various requirements updates that were necessary.

## [3.2.1] - 2024-07-11

### Added

- **SegmentProperties** : Added a `.to_dataframe()` method for segment properties to convert them back to a dataframe.

## [3.2.0] - 2024-07-10

### Added

- **SegmentProperties** : A new module has been created to build Segment Properties, which are a method of organizing information about segment IDs in a Neuroglancer segmentation layer.
Segment Properties only work with the newer Google/Spelunker branches, and offer a way to add browsable and searchable metadata to segments that can be viewed in the viewer.
- **StateBuilder** : Various changes were made to support segment properties from both explicit URLs and from data-driven mapping. See documentation for details.

## Changed

- **StateBuilder** : Image layers for spelunker now use native contrast controls.

### Fixed

- **StateBuilder** : Improvements to the use of `target_site` and `url_prefix` in general when both creating statebuilder and rendering states. In particular, the values in `render_state` should now correctly take precendence over the values in `StateBuilder` when both are set.


## [3.1.0] - 2024-05-29

### Added

- The **Parser** function `annotation_dataframe` now has a parameter `expand_tags` that will create a boolean column for each tag in each annotation layer indicating if the annotation has that tag or not.

### Fixed

- **Parser** : Fixed bugs in getting bounding box annotations and in getting linked segmentations for Spelunker-style states.

## [3.0.3] – 2024-4-30

### Fixed

- **StateBuilder** : Fixed a bug preventing tags from being set.

## [3.0.2] – 2024-1-19

### Fixed

- Improve consistency of target_site argument in StateBuilder.


## [3.0.1] – 2023-12-20

### Fixed

- Fixed an unnecessary import that caused an error.
- Added python package neuroglancer to the requirements so versions with errors or incompatibilities can be avoided.

## [3.0.0] – 2023-12-20

### Added

- **StateBuilder** and **EasyViewer** Substantial refactor, adding basic support for the main Google Neuroglancer branch and the Cave-Explorer/Spelunker deployments coming online with CAVE.
Most functionality should be seamless, with the client checking for version.
Currently, only the [Cave-Explorer](https://ngl.cave-explorer.org/) and [Spelunker](https://spelunker.cave-explorer.org/) deployments are fully supported for this automated version check, as the response depends on a build-specific version.json file.
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
