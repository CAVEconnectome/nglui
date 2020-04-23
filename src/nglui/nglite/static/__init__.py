# @license
# Copyright 2016 Google Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import posixpath

static_content_filenames = set([
    'main.bundle.js',
    'chunk_worker.bundle.js',
    'tfjs-library.bundle.js',
    'async_computation.bundle.js',
    'index.html',
    'draco.bundle.js',
])

mime_type_map = {
    '.css': 'text/css',
    '.js': 'application/javascript',
    '.html': 'text/html',
    '.map': 'application/json'
}

default_ngl_url = 'https://neuromancer-seung-import.appspot.com'

def guess_mime_type_from_path(path):
    return mime_type_map.get(posixpath.splitext(path)[1], 'application/octet-stream')


class StaticContentSource(object):
    def get(self, name):
        if name == '':
            name = 'index.html'
        return self.get_content(name), guess_mime_type_from_path(name)

    def get_content(self, name):
        raise NotImplementedError

class HttpSource(StaticContentSource):
    def __init__(self, url):
        self.url = url

    def get_content(self, name):
        import requests
        full_url = posixpath.join(self.url, name)
        r = requests.get(full_url)
        if r.status_code >= 200 and r.status_code < 300:
            return r.content
        raise ValueError('Failed to retrieve %r: %s' % (full_url, r.reason))

def get_default_static_content_source():
    return HttpSource(default_ngl_url)

def get_static_content_source(url=None):
    if url is not None:
        return HttpSource(url)
    else:
        return get_default_static_content_source()
