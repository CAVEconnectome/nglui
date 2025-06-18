from typing import Optional, Union

import caveclient
import numpy as np
import pandas as pd

from ..statebuilder.ngl_components import SegmentationLayer
from ..statebuilder.shaders import shader_base

try:
    import cloudvolume

    use_cloudvolume = True
except ImportError:
    use_cloudvolume = False


class InfoMismatchException(Exception):
    pass


__all__ = ["SkeletonManager", "InfoMismatchException"]


class SkeletonManager:
    def __init__(
        self,
        segmentation_source: Union[str, caveclient.CAVEclient],
        cloudpath: str,
        vertex_attributes: Optional[list[str]] = None,
        initialize_info: bool = False,
        shader: Optional[str] = None,
    ):
        """
        Create a manager to format and upload skeletons to a cloud bucket.

        Parameters
        ----------
        segmentation_source : Union[str, caveclient.CAVEclient]
            The source of the segmentation data, either a string URL or a CAVEclient instance.
            If a CAVEclient instance is provided, the segmentation_source will be derived from it.
        cloudpath : str
            The cloudpath where the skeletons will be stored. This is typically a gs:// or s3:// URL.
        vertex_attributes : Optional[list[str]], optional
            An ordered list of vertex attribute names to be included in the skeletons, by default None.
            If None, no vertex attributes will be included.
        initialize_info : bool, optional
            Whether to initialize the skeleton info in the cloud bucket if it does not exist, by default False.
            If False, an exception will be raised if the info file does not exist.
        shader : Optional[str], optional
            A shader string to be used for rendering the skeletons in Neuroglancer, by default None.
        """
        if use_cloudvolume is False:
            raise ImportError(
                "CloudVolume is required for handling skeleton data but is not available. Please install using `pip install cloud-volume`."
            )
        self.cv = None
        self.skel_info = None
        self.initialize_info = initialize_info
        self.segmentation_source = segmentation_source
        self._shader = shader
        if isinstance(segmentation_source, caveclient.frameworkclient.CAVEclientFull):
            self.seg_cv = segmentation_source.info.segmentation_cloudvolume()
        else:
            self.seg_cv = cloudvolume.CloudVolume(segmentation_source)
        self.info = self.seg_cv.info.copy()
        self.info["skeletons"] = "skeletons"

        if "precomputed://" not in cloudpath:
            cloudpath = "precomputed://" + cloudpath
        self.cloudpath = cloudpath

        self._vertex_attributes = vertex_attributes

        self.initialize_skeleton_info()

        self._uploaded_root_ids = set()

    def _format_attribute_info(self):
        attribute_info = []
        attributes = self._vertex_attributes or []
        for attr in attributes:
            attribute_info.append(
                {
                    "id": attr,
                    "data_type": "float32",
                    "num_components": 1,
                }
            )
        return attribute_info

    @property
    def vertex_attributes(self) -> dict:
        """
        The neuroglancer-formated vertex attribute dictionary for the skeletons.
        """
        return self._format_attribute_info()

    @property
    def vertex_attribute_names(self) -> list[str]:
        """
        The names of the vertex attributes in the skeletons.
        """
        return [attr["id"] for attr in self._format_attribute_info()]

    def _generate_skeleton_metadata(self):
        "Build expectation of vertex attributes from class values"
        if self.cv is None:
            raise ValueError("CloudVolume not initialized.")
        sk_info = self.cv.skeleton.meta.default_info()
        sk_info["vertex_attributes"] = self.vertex_attributes
        return sk_info

    def _get_attributes_from_info(self):
        "retrieve vertex attributes from existing info file"
        skel_info = self.cv.skeleton.meta.info
        self._vertex_attributes = [
            attr["id"] for attr in skel_info.get("vertex_attributes", [])
        ]

    def initialize_skeleton_info(self):
        """
        Initialize the skeleton info with the vertex attributes.
        """
        try:
            self.cv = cloudvolume.CloudVolume(
                self.cloudpath,
            )
            cv_info = self.cv.info.copy()
            cv_info.pop("vertex_attributes")
            if cv_info != self.info:
                raise InfoMismatchException(
                    "CloudVolume bucket exists with different info than requested."
                )
            if self._vertex_attributes is None:
                # If no vertex attributes are provided, load the ones already present
                self._get_attributes_from_info()
            if (
                self.cv.skeleton.meta.info["vertex_attributes"]
                != self._generate_skeleton_metadata()["vertex_attributes"]
            ):
                raise InfoMismatchException(
                    f"CloudVolume bucket exists with different skeleton attributes than requested. Expected: {self.vertex_attribute_names}, Found: {[a['name'] for a in self.cv.skeleton.meta.info['vertex_attributes']]}"
                )
        except cloudvolume.exceptions.InfoUnavailableError:
            if self.initialize_info:
                self.cv = cloudvolume.CloudVolume(
                    self.cloudpath, info=self.info, compress=False
                )

                self.cv.commit_info()

                self.cv.info["vertex_attributes"] = self.vertex_attributes
                self.cv.commit_info()

                self.skel_info = self._generate_skeleton_metadata()
                self.cv.skeleton.meta.info = self.skel_info
                self.cv.skeleton.meta.commit_info()
            else:
                raise ValueError(
                    "CloudVolume bucket with an info file does not yet exist. Set initialize_info to True to create needed files."
                )

    def _create_skeleton(
        self,
        vertices: np.ndarray,
        edges: np.ndarray,
        root_id: int,
        vertex_attribute_data: Optional[Union[dict, pd.DataFrame]] = None,
    ) -> "cloudvolume.Skeleton":
        """
        Create a skeleton object with the given vertices, edges, and root ID.
        """
        if edges is None:
            edges = np.zeros((0, 2), dtype=np.uint32)
        skel = cloudvolume.Skeleton(
            vertices=vertices.astype(np.float32),
            edges=edges,
            radii=np.ones(len(vertices), dtype=np.float32),
            segid=root_id,
            vertex_types=None,
            extra_attributes=self.vertex_attributes,
        )
        if len(self.vertex_attributes) > 0:
            for attribute in self.vertex_attributes:
                skel.__setattr__(
                    attribute["id"],
                    np.array(
                        vertex_attribute_data.get(
                            attribute["id"], np.zeros(len(vertices))
                        )
                    ),
                )
        return skel

    def upload_skeleton(
        self,
        root_id: int,
        vertices: np.ndarray,
        edges: Optional[np.ndarray] = None,
        vertex_attribute_data: Optional[Union[dict, pd.DataFrame]] = None,
    ):
        """
        Upload a skeleton to the cloud bucket.
        Note that uploading the skeleton will add the root ID to the set of uploaded root IDs property.

        Parameters
        ----------
        root_id : int
            The root ID of the skeleton.
        vertices : np.ndarray
            The vertices of the skeleton as a numpy array of shape (N, 3).
        edges : Optional[np.ndarray], optional
            The edges of the skeleton as a numpy array of shape (M, 2), by default None.
            If None, an empty edge list will be created and the "skeleton" will be treated as a set of disconnected vertices.
        vertex_attribute_data : Optional[Union[dict, pd.DataFrame]], optional
            A dictionary or DataFrame containing vertex attribute data, where keys are attribute names and values are arrays of the same length as vertices.
            If any of the vertex attributes are not provided, they will default to zero arrays of the same length as vertices.

        """
        skel = self._create_skeleton(
            vertices=vertices,
            edges=edges,
            root_id=root_id,
            vertex_attribute_data=vertex_attribute_data,
        )
        self.cv.skeleton.upload(skel)
        self._uploaded_root_ids.add(root_id)

    @property
    def uploaded_root_ids(self) -> list[int]:
        """
        Get the set of root IDs that have been uploaded since the SkeletonManager was initialized.
        """
        return list(self._uploaded_root_ids)

    @property
    def skeleton_source(self) -> str:
        """
        Get the skeleton source URL for neuroglancer.
        """
        return "precomputed://" + self.cv.cloudpath

    def make_shader(
        self,
        checkbox_controls: Optional[Union[dict, list]] = None,
        sliders: Optional[Union[dict, list]] = None,
        defined_colors: Optional[Union[dict, list]] = None,
        body: Optional[str] = None,
    ) -> None:
        """
        Create a shader for the skeletons based on the vertex attributes and other parameters.
        The value will be stored in the shader property of the SkeletonManager instance.

        Parameters
        ----------
        checkbox_controls : Optional[dict], optional
            A dictionary of checkbox controls for the shader, by default None.
            Keys are the control names and values are booleans indicating whether the control is enabled by default.
            A pure list can also be provided, in which case all controls will be enabled by default.
        sliders : Optional[dict], optional
            A dictionary or list of slider controls for the shader, by default None.
            Keys are the control names and values are tuples with the type (float or int), min, max, and default value.
            If provided as al list, all sliders will be set to float type with default range of 0 to 1 and value of 0.5.
        defined_colors : Optional[dict], optional
            A dictionary of defined colors for the shader, by default None.
            Keys are the color variable names and values are tuples with hex or web color names.
            If provided as a list, sequential colors from the Tableau 10 colormap will be used.
        body : Optional[str], optional
            Additional body code for the shader, by default None.
            This can be used to add custom shader logic or functions and should contain the emitRGB function.
            If None, a default body will be generated.
        """
        self._shader = shader_base(
            vertex_attributes=self.vertex_attribute_names,
            checkbox_controls=checkbox_controls,
            sliders=sliders,
            defined_colors=defined_colors,
            body=body,
        )

    @property
    def shader(self) -> str:
        """
        Return a skeleton shader with appropriate vertex properties and other attributes as set in `make_shader`.
        If no shader is set, return a default shader that uses the vertex attributes defined in the SkeletonManager.
        """
        if self._shader is None:
            return shader_base(vertex_attributes=self.vertex_attribute_names)
        return self._shader

    def to_segmentation_layer(
        self,
        name: str = "seg",
        uploaded_segments: bool = True,
        segments_visible: bool = True,
        shader: Optional[Union[bool, str]] = None,
    ) -> SegmentationLayer:
        """
        Generate a SegmentationLayer with the segmentation source and skeleton source, as well as uploaded segments and shader specified

        Parameters
        ----------
        name : str, optional
            The name of the segmentation layer, by default "seg".
        uploaded_segments : bool, optional
            Whether to include the uploaded segments in the layer, by default True.
            If True, the segments will be set to the uploaded root IDs with visibility set to segments_visible.
        segments_visible : bool, optional
            Whether the uploaded segments should be visible in the layer, by default True.
        shader : Optional[Union[bool, str]], optional
            A shader string to be used for the segmentation layer, by default None.
            If False, no shader will be applied. If None, the shader from the SkeletonManager will be used.

        Returns
        -------
        SegmentationLayer
            A SegmentationLayer object with the specified parameters.
            This can be used in a ViewerState.add_layer method.
        """
        seg_source = []
        if isinstance(self.segmentation_source, str):
            seg_source.append(self.segmentation_source)
        else:
            # Assume is caveclient
            seg_source.append(self.segmentation_source.info.segmentation_source())
        seg_source.append(self.skeleton_source)
        if uploaded_segments:
            segments = {seg: segments_visible for seg in self.uploaded_root_ids}
        else:
            segments = None
        if shader is None:
            shader = self.shader
        elif shader is False:
            shader = None
        return SegmentationLayer(
            name=name,
            source=seg_source,
            segments=segments,
            shader=shader,
        )
