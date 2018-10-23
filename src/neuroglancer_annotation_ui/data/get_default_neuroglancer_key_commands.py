import requests
import re
import json

event_bindings_url = "https://github.com/google/neuroglancer/blob/master/src/neuroglancer/ui/default_input_event_bindings.ts"
event_bindings_text = requests.get(event_bindings_url).text

modifiers = ["shift\\+",
             "control\\+",
             "alt\\+",
             "meta\\+"]

potential_keys = ['key[a-z]',
                  'digit[0-9]',
                  'arrowleft',
                  'arrowright',
                  'arrowup',
                  'arrowdown',
                  'minus',
                  'equal',
                  'comma',
                  'period',
                  'space',
                  'wheel',
                  'dblclick0',
                  'dblclick2',
                  'mousedown0',
                  'mousedown2']

re_query = '(({mods})*|\')({keys})'.format(mods='|'.join(modifiers), keys='|'.join(potential_keys))
action_key_binding_list = re.findall(re_query, event_bindings_text)
action_bindings = [''.join(x) for x in action_key_binding_list]

layer_bindings = ['digit{}'.format(i) for i in range(0,10)]
ctrl_layer_bindings = ['control+digit{}'.format(i) for i in range(0,10)]

key_bindings = sorted(list(set(action_bindings + layer_bindings + ctrl_layer_bindings)))
with open('data/default_key_bindings.json','w') as fid:
    json.dump(key_bindings, fid)