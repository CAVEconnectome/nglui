from neuroglancer_annotation_ui.dynamicstate.extension_core import ExtensionBase
import numpy as np
import json
from collections import defaultdict

YES_LAYER = 'Yes'
NOT_LAYER = 'No'
UNCERTAIN_LAYER = 'Uncertain'

def StateListBrowserFactory(state_ids, json_client, file_name, backup_state=None, backup_name=None, save_on_switch=True):
    class StateListBrowser(StateListBrowserGeneric):
        def __init__(self, easy_viewer, annotation_client=None):
            super(StateListBrowser, self).__init__(easy_viewer, annotation_client)
            self.state_ids = state_ids
            self.json_client = json_client
            self.file_name = file_name
            self.backup_name = backup_name
            self.save_on_switch = save_on_switch
            if backup_state is not None:
                self.state_labels = backup_state
            self.get_state(self.state_ids[self.ind_state])
    return StateListBrowser


class StateListBrowserGeneric(ExtensionBase):
    def __init__(self, easy_viewer, annotation_client=None):
        super(StateListBrowserGeneric, self).__init__(easy_viewer, None)
        self.ind_state = 0
        self.state_ids = []
        self.json_client = None
        self.file_name = None
        self.state_labels = dict()
        self.save_on_switch = False
        self.backup_name = None

    @staticmethod
    def _defined_layers():
        return []

    @staticmethod
    def _default_key_bindings():
        bindings = {'mark_yes': 'shift+keyy',
                    'mark_no': 'shift+keyn',
                    'mark_uncertain': 'shift+keyu',
                    'next_state': 'shift+keyq',
                    'prev_state': 'shift+keyw',
                    'save_to_file': 'shift+control+keys'}
        return bindings

    def mark_state(self, label):
        curr_state = self.state_ids[self.ind_state]
        self.state_labels[curr_state] = label
        with self.viewer.txn() as s:
            if YES_LAYER in self.viewer.layer_names:
                del s.layers[YES_LAYER]
            if NOT_LAYER in self.viewer.layer_names:
                del s.layers[NOT_LAYER]
            if UNCERTAIN_LAYER in self.viewer.layer_names:
                del s.layers[UNCERTAIN_LAYER]

        self.viewer.add_annotation_layer(label)
        if len( set(self.state_ids).difference(set(self.state_labels.keys())) ) == 0:
            self.viewer.update_message('All states have labels')

    def mark_yes(self, s):
        self.mark_state(YES_LAYER)

    def mark_no(self, s):
        self.mark_state(NOT_LAYER)

    def mark_uncertain(self, s):
        self.mark_state(UNCERTAIN_LAYER)

    def next_state(self, s):
        self.ind_state = (self.ind_state + 1) % len(self.state_ids)
        self.get_state(self.state_ids[self.ind_state])

    def prev_state(self, s):
        self.ind_state = (self.ind_state - 1) % len(self.state_ids)
        self.get_state(self.state_ids[self.ind_state])

    def get_state(self, state_id):
        try:
            json_data = self.json_client.get_state_json(state_id)
            self.viewer.set_state(json_data)
            label = self.state_labels.get(state_id, None)
            if label is not None:
                self.viewer.add_annotation_layer(label)
            self.viewer.update_message('Loading state #{}/{}'.format(self.ind_state, len(self.state_ids)))
            if self.save_on_switch:
                self.save_backup()

        except Exception as e:
            print(e)
            self.viewer.update_message('Could not load state {}'.format(state_id))

    def save_to_file(self, s):
        with open(self.file_name, 'w') as f:
            json.dump(self.state_labels, f)
        self.viewer.update_message('Saved file to {}'.format(self.file_name))

    def save_backup(self):
        if self.backup_name is not None:
            with open(self.backup_name, 'w') as f:
                json.dump(self.state_labels, f)
