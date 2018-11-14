from neuroglancer_annotation_ui.extension_core import ExtensionBase
from neuroglancer_annotation_ui.annotation import point_annotation, sphere_annotation
from analysisdatalink.datalink import DataLink
import numpy as np
from collections import defaultdict
from matplotlib.pyplot import get_cmap
from matplotlib.colors import to_hex
from itertools import cycle

def cell_type_extension_factory(table_name,
                                schema_name,
                                db_config,
                                colorset='Dark2',
                                use_points = True,
                                radius = 1000):
    
    database_uri = db_config['uri']
    dataset = db_config['dataset']
    data_version = db_config['data_version']
    dl = DataLink(dataset=dataset, version=data_version, database_uri=database_uri)
    dl.add_annotation_model('cell_type', table_name=table_name, schema_name=schema_name)
    annos = dl.query_cell_type('cell_type')
    defined_layer_names = list(np.unique(annos.cell_type))

    clrs = cycle( get_cmap(colorset).colors )
    color_map = {ln: to_hex(next(clrs)).upper() for ln in defined_layer_names}

    class CellTypeGetterExtension(ExtensionBase):
        def __init__(self, easy_viewer, annotation_client=None):
            super(CellTypeGetterExtension, self).__init__(easy_viewer, None)
            self.dl = dl
            self.use_points = use_points
            self.allowed_layers = self._defined_layers()
            self.radius = radius
            self.color_map = color_map
            self.anno_df = None
            for ln in self._defined_layers():
                self.viewer.add_annotation_layer(ln, color=self.color_map[ln])

            self._reload_all_annotations()

        @staticmethod
        def _defined_layers():
            return defined_layer_names

        @staticmethod
        def _default_key_bindings():
            bindings = {'toggle_sphere_point': 'shift+keyp'}
            return bindings

        def _reload_all_annotations(self):
            self.viewer.clear_annotation_layers(self._defined_layers())
            if self.anno_df is None:
                anno_df = self.dl.query_cell_type('cell_type')
            anno_layer_map = self.annotations_from_df(anno_df)
            self.viewer.add_annotation_one_shot(anno_layer_map)

        def annotations_from_df(self, anno_df):
            anno_layer_map = defaultdict(list)
            for _, row in anno_df.iterrows():
                if self.use_points:
                    new_anno = point_annotation(row['pt_position'].tolist())
                else:
                    new_anno = sphere_annotation(center=row['pt_position'].tolist(),
                                                 radius=self.radius,
                                                 z_multiplier=0.1)
                anno_layer_map[row['cell_type']].append(new_anno)

            return anno_layer_map

        def toggle_sphere_point(self, s):
            self.use_points = not self.use_points
            self._reload_all_annotations()

    return CellTypeGetterExtension