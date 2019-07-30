from collections import defaultdict
from neuroglancer_annotation_ui import annotation

annotation_function_map = {'point': annotation.point_annotation,
                           'line': annotation.line_annotation,
                           'ellipsoid': annotation.ellipsoid_annotation,
                           'sphere': annotation.sphere_annotation,
                           'bounding_box': annotation.bounding_box_annotation}

class SchemaRenderer():
    def __init__(self, EMSchema, render_rule=None):
        if render_rule is None:
            self.render_rule = RenderRule(EMSchema)
        else:
            self.render_rule = RenderRule(EMSchema, render_rule=render_rule)

        self.apply_description_rule = self.render_rule.make_description_rule()

        self.annotations = {}
        self.render_functions = self.render_rule.generate_processors()

        self.reset_annotations()


    def __call__(self, 
                 viewer,
                 data,
                 anno_id=None,
                 layermap=None,
                 colormap=None,
                 replace_annotations=None):
        viewer_ids = self.render_data(viewer, data, anno_id=anno_id, layermap=layermap, colormap=colormap, replace_annotations=replace_annotations)
        return viewer_ids

    def render_data(self,
                    viewer,
                    data,
                    anno_id=None,
                    layermap=None,
                    colormap=None,
                    replace_annotations=None ):
        """
        Takes a formatted data point and returns annotation layers based on the schema's RenderRule
        """
        if layermap is None:
            layermap = {layer:layer for layer in self.all_layers() }

        self.apply_render_rules(data, anno_id=anno_id)
        viewer_ids = self.send_annotations_to_viewer(viewer, layermap=layermap, colormap=colormap)
        self.apply_description_rule(data, viewer_ids, viewer)

        if replace_annotations is not None:
            for layer, ngl_id in replace_annotations.items():
                viewer.remove_annotation(layer, ngl_id)
        self.reset_annotations()
        return viewer_ids

    def apply_render_rules(self, data, anno_id=None):
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
                viewer_ids[nl].append(anno.id)

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

    def make_description_rule(self, spacing_character=':'):
        description_keys = self.render_rule.get('description_field', [])
        if len(description_keys) > 0:
            def dr(data, viewer_ids, viewer):
                added_description = spacing_character.join(data[f] for f in description_keys)
                viewer.update_description(viewer_ids, added_description)
        else:
            def dr(data, viewer_ids, viewer):
                pass
        return dr

    @property
    def layers( self ):
        all_layers = set()
        for anno_type, type_rule in self.render_rule.items():
            if anno_type == 'description_field':
                continue
            for layer in type_rule.keys():
                all_layers.add(layer)
        return list(all_layers)

    @property
    def fields( self ):
        all_fields = set()
        for anno_type, type_rule in self.render_rule.items():
            if anno_type == 'description_field':
                continue
            for _, rule_list in type_rule.items():
                for rule in rule_list:
                    for f in [*rule]:
                        all_fields.add(f)
        return list(all_fields)

    def generate_processors( self ):
        annotation_processor_list = []
        for anno_type in self.render_rule.keys():
            if anno_type == 'description_field':
                continue
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
                            if field in self.schema_fields:
                                if isinstance(self.schema_fields[field], Nested):
                                    anno_args.append( data[field]['position'] )
                                else:
                                    anno_args.append( data[field] )
                            else:
                                anno_args.append(field)
                        ngr.annotations[layer].append(
                            annotation_function(*anno_args, description=anno_id))
        else:
            def annotation_processor(ngr, data, anno_id=None):
                return
        return annotation_processor