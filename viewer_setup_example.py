import neuroglancer_annotation_ui.interface as anno_ui

syn_anno_viewer = anno_ui.Connector()

actions = {
    'select_pre': ['shift+keyq', syn_anno_viewer.select_pre_process],
    'add_synapse': ['shift+keyw', syn_anno_viewer.add_synapse],
    'select_post': ['shift+keye', syn_anno_viewer.select_post_process],
    'save_json': ['shift+keys', syn_anno_viewer.save_json],
    'clear_all': ['shift+keyc', syn_anno_viewer.clear_all],
    'undo_last_point': ['shift+keyz', syn_anno_viewer.undo_last_point],
    'delete_synapse': ['delete', syn_anno_viewer.delete_synapse],
    'clear_segment': ['shift+keyv', syn_anno_viewer.clear_segment],
}
for action, bindings in actions.items():
    syn_anno_viewer.add_action(action, bindings[0], bindings[1])

print(syn_anno_viewer)
