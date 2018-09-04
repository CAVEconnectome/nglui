import neuroglancer
import random
import webbrowser
import pytest


def test_bulk_annotations(samples):
    data = []
    for i in range(samples):
        data.append(neuroglancer.LineAnnotation(
            point_a=random.sample(range(1, 100), 3),
            point_b=random.sample(range(1, 100), 3),
            id='test{}'.format(i)))
    return data


def test_visualize_data(data):
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        s.layers.append(
            name='Test',
            layer=neuroglancer.AnnotationLayer(
                annotations=data,
                annotation_color='#ff006c',
            ))
    return viewer


if __name__ == '__main__':
    test_bulk = test_bulk_annotations(10000)
    test_viewer = test_visualize_data(test_bulk)
    webbrowser.open_new_tab(test_viewer.get_viewer_url())
