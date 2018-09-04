from annotation_tools import interface
import neuroglancer
import pytest


def test_action(action):
    assert(isinstance(action, neuroglancer.viewer_config_state.ActionState))


def print_test(action):
    print("Test")


if __name__ == '__main__':
    # create dict of actions that bind to interface
    actions = {
        "print_screen": ['shift+keyp', test_action],
        "test_action": ['shift+keyt', print_test]
    }
    # invoke interface
    test_interface = interface.Interface()

    # bind actions to interface
    for action, bindings in actions.items():
        print(action, bindings[0], bindings[1])
        test_interface.add_action(action, bindings[0], bindings[1])

    # add a fake segmentation layer, fake annotation layer, then show
    segment_source = 'precomputed://gs://neuroglancer/pinky40_v11/watershed_mst_trimmed_sem_remap'
    image_source = 'precomputed://gs://neuroglancer/pinky40_v11/image_rechunked'
    test_interface.add_segmentation_layer('Segmentation', segment_source)
    test_interface.add_image_layer('Image', image_source)
    test_interface.add_annotation_layer('Fake Points',  'PointAnnotationLayer')

    # add many annotation layers
    layers = ['potato', 'basmati', 'pancake', 'sst',
              'vertical', 'I2I_VIP', ' ivy', 'clutch', 'unknown']
    for layer in layers:
        test_interface.add_annotation_layer(layer, 'PointAnnotationLayer')
    test_interface.show()
