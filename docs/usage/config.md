---
title: Site Configuration
---

## Configuring Neuroglancer Sites

NGLui defaults to using the `spelunker` deployment of Neuroglancer, which is the most aligned with CAVE priorities.
However, most if not all of the features in NGLui are compatibile across any contemporary deployment of Neuroglancer and you can use any deployment URL you would like.
To help manage different Neuroglancer URLs, there are a small number of functions to set and get the default configuration for Neuroglancer sites.

### `target_site`

As a convenience, NGLui keeps a dictionary of Neuroglancer URLs (`target_url`) with readable names (`target_site`).
You can find the current configuration by running `get_neuroglancer_sites()`.
For example, the default values are:

``` pycon
>>> from nglui.statebuilder import *
>>> get_neuroglancer_sites()
{'spelunker': 'https://spelunker.cave-explorer.org/',
 'google': 'https://neuroglancer-demo.appspot.com/'}
```

This means that setting `target_site='spelunker'` in Statebuilder export functions will use the URL `https://spelunker.cave-explorer.org/` as the default Neuroglancer site, and likewise for `google`.

You can add your own sites to this dictionary by using the `add_neuroglancer_site` function, which takes a `target_site` and `target_url` as arguments.
You can further set the new site as the default by setting `set_default=True`.

``` py
add_neuroglancer_site(
    'my_neuroglancer',
    'https://my-neuroglancer.example.com',
    set_default=True,
)
```

And you can always see the current default site by running `get_default_neuroglancer_site()`, which will return the `target_site` and `target_url` of the current default site.
For example, by default this will show:

``` pycon
>>> get_default_neuroglancer_site()
{'spelunker': 'https://spelunker.cave-explorer.org/'}
```

### Environment Variables

If you find yourself always using a different Neuroglancer site, you can set the environment variables `NGLUI_TARGET_SITE` and `NGLUI_TARGET_URL` to the shortcut name and URL of your preferred Neuroglancer.
This will automatically add these values to the site list and set it the default site when you import NGLui, so you don't have to manually set it every time.
Note that both variables must be set.

