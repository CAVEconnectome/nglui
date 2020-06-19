# Changelog

This project attempts to follow [Semantic Versioning](https://semver.org).

## Unreleased

### Added
- Changelog created

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