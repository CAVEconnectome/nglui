import neuroglancer
import copy
import numpy as np
from neuroglancer.viewer_state import (Layer, _AnnotationLayerOptions, interpolate_linear,
                                      volume_source, uint64_equivalence_map, layer_types)
from neuroglancer.json_wrappers import (JsonObjectWrapper, array_wrapper, optional, text_type, typed_list,
                                        typed_set, typed_string_map, wrapped_property)

default_static_content_source='https://neuromancer-seung-import.appspot.com/'

def set_static_content_source(source=default_static_content_source):
    neuroglancer.set_static_content_source(url=source)

def stop_ngl_server():
    """
    Shuts down the neuroglancer tornado server
    """
    neuroglancer.server.stop()

## Build ChunkedGraph vs Precomputed Segmentation Layers
class SegmentationLayerBase(Layer, _AnnotationLayerOptions):
    __slots__ = ()
    def __init__(self, *args, type=None, **kwargs):
        super(SegmentationLayerBase, self).__init__(*args, type=type, **kwargs)
    source = wrapped_property('source', optional(volume_source))
    mesh = wrapped_property('mesh', optional(text_type))
    skeletons = wrapped_property('skeletons', optional(text_type))
    segments = wrapped_property('segments', typed_set(np.uint64))
    equivalences = wrapped_property('equivalences', uint64_equivalence_map)
    hide_segment_zero = hideSegmentZero = wrapped_property('hideSegmentZero', optional(bool, True))
    selected_alpha = selectedAlpha = wrapped_property('selectedAlpha', optional(float, 0.5))
    not_selected_alpha = notSelectedAlpha = wrapped_property('notSelectedAlpha', optional(float, 0))
    object_alpha = objectAlpha = wrapped_property('objectAlpha', optional(float, 1.0))
    skeleton_shader = skeletonShader = wrapped_property('skeletonShader', text_type)
    color_seed = colorSeed = wrapped_property('colorSeed', optional(int, 0))
    cross_section_render_scale = crossSectionRenderScale = wrapped_property(
        'crossSectionRenderScale', optional(float, 1))
    mesh_render_scale = meshRenderScale = wrapped_property('meshRenderScale', optional(float, 10))

    @staticmethod
    def interpolate(a, b, t):
        c = copy.deepcopy(a)
        for k in ['selected_alpha', 'not_selected_alpha', 'object_alpha']:
            setattr(c, k, interpolate_linear(getattr(a, k), getattr(b, k), t))
        return c

class ChunkedgraphSegmentationLayer(SegmentationLayerBase):
    __slots__ = ()
    def __init__(self, *args, **kwargs):
        super(ChunkedgraphSegmentationLayer, self).__init__(*args, type='segmentation_with_graph', **kwargs)
