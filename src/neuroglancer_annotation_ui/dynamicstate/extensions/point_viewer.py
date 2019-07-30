from neuroglancer_annotation_ui.extension_core import AnnotationExtensionBase
from neuroglancer_annotation_ui.annotation import point_annotation
from analysisdatalink.datalink_ext import AnalysisDataLinkExt as AnalysisDataLink
from collections import defaultdict


def PointViewerFactory(table_name, config, layer_name=None, layer_color='#ffffff'):
    '''
    Simple renderer for point data from a single column
    '''
    if layer_name is None:
        layer_name=table_name

    database_uri = config.get('sqlalchemy_database_uri', None)
    materialization_version = config.get('materialization_version', None)
    dataset_name = config.get('dataset_name', None)

    class PointViewerExtension(AnnotationExtensionBase):
        def __init__(self, easy_viewer, annotation_client=None):
            super(PointViewerExtension, self).__init__(easy_viewer, None)
            self.layer_color = layer_color
            self.table_name = table_name
            self.db_tables = {}
            self.allowed_layers = [layer_name]
            self.dl = AnalysisDataLink(dataset_name=dataset_name,
                                       sqlalchemy_database_uri=database_uri,
                                       materialization_version=materialization_version,
                                       verbose=False)
            self._create_layers()

        @staticmethod
        def _defined_layers():
            return [layer_name]

        def _create_layers(self):
            for ln in self._defined_layers():
                self.viewer.add_annotation_layer(ln, self.layer_color)

        def load_annotations(self, query_in_filter={}, query_notin_filter={}):
            anno_map = self._query_annotations(query_in_filter, query_notin_filter)
            self.viewer.add_annotation_one_shot(anno_map)

        def _query_annotations(self, query_in_filter={}, query_notin_filter={}):
            anno_df = self.dl.specific_query(tables=[self.table_name],
                                             filter_in_dict = {self.table_name: query_in_filter},
                                             filter_notin_dict = {self.table_name: query_notin_filter})

            anno_map = self._annotations_from_df(anno_df)
            return anno_map

        def _annotations_from_df(self, anno_df, position_column='pt_position'):
            anno_map = defaultdict(list)
            for _, row in anno_df.iterrows():
                anno_map[self.table_name].append(point_annotation(row[position_column].tolist()))
            return anno_map

    return PointViewerExtension
