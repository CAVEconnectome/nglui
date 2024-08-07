# @license
# Copyright 2017 Google Inc.
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


from . import viewer_base


class _ViewerHelper:
    """Mixin for implementing viewers based on the built-in server [kept for compatibility]."""

    def __init__(self):
        super(_ViewerHelper, self).__init__()

    def defer_callback(self, callback, *args, **kwargs):
        pass

    def __repr__(self):
        return self.get_viewer_url()

    def _repr_html_(self):
        return '<a href="%s" target="_blank">Viewer</a>' % self.get_viewer_url()

    def get_viewer_url(self):
        raise NotImplementedError


class Viewer(viewer_base.ViewerBase, _ViewerHelper):
    def __init__(self):
        super(Viewer, self).__init__()

    def get_viewer_url(self):
        return "Not implemented"


class UnsynchronizedViewer(viewer_base.UnsynchronizedViewerBase, _ViewerHelper):
    def __init__(self):
        super(UnsynchronizedViewer, self).__init__()

    def get_viewer_url(self):
        return "Not implemented"
