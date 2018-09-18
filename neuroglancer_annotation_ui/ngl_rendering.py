from collections import defaultdict
from neuroglancer_annotation_ui import annotation
from marshmallow.fields import Nested
from emannotationschemas.base import SpatialPoint

annotation_function_map = {'point': annotation.point_annotation,
                           'line': annotation.line_annotation,
                           'ellipsoid': annotation.ellipsoid_annotation,
                           'bounding_box': annotation.bounding_box_annotation}

class SchemaRenderer():
    def __init__(self, EMSchema, render_rule=None):
        if render_rule is None:
            self.render_rule = RenderRule(EMSchema)
            # Todo: Introduce a default point render rule
        else:
            self.render_rule = RenderRule(EMSchema, render_rule=render_rule)
        self.annotations = {}
        self.reset_annotations()
        self.render_functions = self.render_rule.generate_processors()

    def __call__(self, 
                 viewer,
                 data,
                 anno_id=None,
                 layermap=None,
                 colormap=None,
                 replace_annotations=None):
        viewer_ids = self.render_data(viewer, data, anno_id=anno_id, layermap=layermap, colormap=colormap, replace_annotations=replace_annotations)
        return viewer_ids

    def render_data(self, viewer, data, anno_id=None, layermap=None, colormap=None, replace_annotations=None ):
        """
        Takes a formatted data point and returns annotation layers based on the schema's RenderRule
        """
        if layermap is None:
            layermap = {layer:layer for layer in self.all_layers() }

        self.apply(data, anno_id=anno_id)
        viewer_ids = self.send_annotations_to_viewer(viewer, layermap=layermap, colormap=colormap)

        if replace_annotations is not None:
            for layer, ngl_id in replace_annotations.items():
                viewer.remove_annotation(layer, ngl_id)

        self.reset_annotations()
        return viewer_ids

    def apply(self, data, anno_id=None):
        for func in self.render_functions:
            func(self, data, anno_id=anno_id)

    def send_annotations_to_viewer(self, viewer, layermap=None, colormap=None):
        if colormap is None:
            colormap={layermap[layer]:None for layer in self.annotations}
        viewer_ids = defaultdict(list)
        for layer, anno_list in self.annotations.items():
            nl = layermap[layer]
            for anno in anno_list:
                viewer.add_annotation(nl,anno,color=colormap[nl])
                viewer_ids[layer].append(anno.id)
        return viewer_ids

    def all_fields(self):
        return self.render_rule.fields

    def all_layers(self):
        return self.render_rule.layers

    def reset_annotations(self):
        self.annotations = {layer:[] for layer in self.all_layers()}


class RenderRule():
    def __init__(self, EMSchema, render_rule=None):
        # Should improve the validation here
        self.schema_fields = EMSchema().fields
        if render_rule is None:
            try:
                self.render_rule = EMSchema.render_rule()
            except:
                raise Exception('No render rule defined for {}!'.format(EMSchema))
        else:
            self.render_rule=render_rule

    @property
    def layers( self ):
        all_layers = set()
        for anno_type, type_rule in self.render_rule.items():
            for layer in type_rule.keys():
                all_layers.add(layer)
        return list(all_layers)

    @property
    def fields( self ):
        all_fields = set()
        for anno_type, type_rule in self.render_rule.items():
            for _, rule_list in type_rule.items():
                for rule in rule_list:
                    for f in [*rule]:
                        all_fields.add(f)
        return list(all_fields)

    def generate_processors( self ):
        annotation_processor_list = []
        for anno_type in self.render_rule.keys():
            annotation_processor_list.append(
                self._annotation_processor_factory(anno_type, annotation_function_map[anno_type]))
        return annotation_processor_list

    def _annotation_processor_factory(self, anno_type, annotation_function):
        if anno_type in self.render_rule:
            rule_category = self.render_rule[anno_type]
            def annotation_processor(ngr, data, anno_id=None):
                for layer, rule_list in rule_category.items():
                    for rule in rule_list:
                        if isinstance(rule,str):
                            rule_fields = [rule]
                        else:
                            rule_fields = [*rule]
                        anno_args = []
                        for field in rule_fields:
                            if isinstance(self.schema_fields[field], Nested):
                                anno_args.append( data[field]['position'] )
                            else:
                                anno_args.append( data[field] )
                        ngr.annotations[layer].append(
                            annotation_function(*anno_args, description=anno_id))
        else:
            def annotation_processor(ngr, data, anno_id=None):
                return
        return annotation_processor

    @classmethod
    def default_render_rule(EMSchema):
        render_rule = {'point':{'annotations':[]}}
        schema_fields = EMSchema().fields
        for field_name, field in schema_fields.items():
            if issubclass(type(field), SpatialPoint):
                render_rule['point']['annotations'].append(field_name)
            elif isinstance(type(field), Nested):
                is_spatial = [issubtype(subfield) for _x, subfield in field.nested._declared_fields.items()]
                if any(is_spatial):
                    render_rule['point']['annotations'].append(field_name)
        return cls(render_rule)
