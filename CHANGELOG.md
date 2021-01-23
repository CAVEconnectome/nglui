# Changelog

This project attempts to follow [Semantic Versioning](https://semver.org) and uses [Keep-a-Changelog formatting](https://keepachangelog.com/en/1.0.0/). But I make mistakes sometimes.

<!-- ## Unreleased -->
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
