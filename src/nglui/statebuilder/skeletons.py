from typing import Optional, Union

import caveclient
import numpy as np
import pandas as pd

from .ngl_components import SegmentationLayer
from .shaders import skeleton_shader_base, skeleton_shader_default

try:
    import cloudvolume

    use_cloudvolume = True
except ImportError:
    use_cloudvolume = False


class InfoMismatchException(Exception):
    pass


class SkeletonManager:
    def __init__(
        self,
        segmentation_source: Union[str, caveclient.CAVEclient],
        cloudpath: str,
        vertex_attributes: Optional[list[str]] = None,
        initialize_info: bool = False,
        shader: Optional[str] = None,
    ):
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
    def vertex_attributes(self):
        """
        Get the vertex attributes for the skeletons.
        """
        return self._format_attribute_info()

    @property
    def vertex_attribute_names(self) -> list[str]:
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
        vertices: np.ndarray,
        edges: np.ndarray,
        root_id: int,
        vertex_attribute_data: Optional[Union[dict, pd.DataFrame]] = None,
    ):
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
        Get the set of root IDs that have been uploaded.
        """
        return list(self._uploaded_root_ids)

    @property
    def skeleton_source(self) -> str:
        """
        Get the skeleton source URL.
        """
        return self.cv.cloudpath

    def make_shader(
        self,
        checkbox_controls: Optional[dict] = None,
        sliders: Optional[dict] = None,
        defined_colors: Optional[dict] = None,
        body: Optional[str] = None,
    ) -> None:
        """
        Create a shader for the skeletons.
        """
        self._shader = skeleton_shader_base(
            vertex_attributes=self.vertex_attribute_names,
            checkbox_controls=checkbox_controls,
            sliders=sliders,
            defined_colors=defined_colors,
            body=body,
        )

    @property
    def shader(self) -> str:
        """
        Get the shader for the skeletons.
        If no shader is set, return the default shader.
        """
        if self._shader is None:
            return skeleton_shader_default(self.vertex_attribute_names)
        return self._shader

    def to_segmentation_layer(
        self,
        name: str = "seg",
        uploaded_segments: bool = True,
        segments_visible: bool = True,
        shader: Optional[Union[bool, str]] = None,
    ) -> SegmentationLayer:
        """
        Convert the skeleton manager to a SegmentationLayer.
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
