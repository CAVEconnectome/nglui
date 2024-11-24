---
title: Site Configuration
---

!!!abstract

    TLDR: Use `site_config` to set default configurations for your `StateBuilder` objects.
    After your imports, the recommended default code is: `site_config.set_default_config(target_site='spelunker')`

## Using `site_config` to configure your StateBuilder

The `site_config` module is intended to access and change the baseline configuration of StateBuilder sites.
This will let you configure a target site, url, and caveclient to use as defaults throughout your analysis without further specification.

The first thing to note is the options within the default configuration.
You can see this via

```python
from nglui import site_config, statebuilder

site_config.get_default_config()
```

This will return a dictionary with a number of variables:

```python
{
    'target_site': 'seunglab',
    'target_url': 'https://neuromancer-seung-import.appspot.com/',
    'seunglab_fallback_url': 'https://neuromancer-seung-import.appspot.com/',
    'mainline_fallback_url': 'https://spelunker.cave-explorer.org/',
    'caveclient': None,
    'datastack_name': None
}
```

Each of these has a specific purpose:

* `target_site`: The Neuroglancer version to use. This can be either `seunglab` or `mainline`/`spelunker`/`cave-explorer` (all refer to the same state). This establishes the broad category of Neuroglancer to use, and states for one type do not work with the other.
* `target_url`: A URL to the deployment of Neuroglancer to use. The `target_url` should be a deployment of the same version as `target_site`.
* `seunglab_fallback_url`: A URL to the deployment of Neuroglancer to use if the target site is `seunglab` and no target URL is specified.
* `mainline_fallback_url`: A URL to the deployment of Neuroglancer to use if the target site is `mainline` and no target URL is specified.
* `caveclient`: A `CAVEclient` object to use for specifying layers and metadata. This is optional, but if you have a `CAVEclient` object, you can specify it here and it will be used as the default for all `StateBuilder` objects. Will be set automatically if a `datastack_name` is specified.
* `datastack_name`: The name of the datastack to use. This is optional and will be set if a `CAVEclient` is specified.

If you build a `StateBuilder` object without specifying things like `target_site` or `target_url`, these are the values that will be used.
For example, if you run the following code using a small dataset from the Kasthuri et al:

```python
img = statebuilder.ImageLayerConfig("precomputed://gs://neuroglancer-public-data/kasthuri2011/image")
seg = statebuilder.SegmentationLayerConfig("precomputed://gs://neuroglancer-public-data/kasthuri2011/image_color_corrected")
sb = statebuilder.StateBuilder([img, seg])
sb.render_state()
```

the result will be a `url` that is based on the value in `target_url` and is constructed for a `seunglab` deployment of Neuroglancer.

You can change any of these values through the `set_default_config` method. For example, to change the target site to `spelunker`, you can use

```python
site_config.set_default_config(target_site='spelunker')
```

If you run this and then check the default config again, you will see that the `target_site` has changed to `spelunker` and the `target_url` has changed to `https://spelunker.cave-explorer.org/` the value that was shown in `mainline_fallback_url`.
Now if you run the same code as above, the `url` will be constructed for a `spelunker` Neuroglancer and be based on the value in the `mainline_fallback_url`, which is the default for `spelunker`.
This change occured because of the change in the default configuration, and will continue to apply for any statebuilder runs until the configuration is changed again.

You can also set a `caveclient` object here directly or via a `datastack_name`.
For example, to set the default datastack to `minnie65_public` (the public MICrONs dataset), you can use

```python
site_utils.set_default_config(datastack=CAVEclient('minnie65_public'))
```

This is particularly useful when working with `spelunker` datasets in particular, as they require a resolution to be specified to work properly and the CAVEclient object can provide this value.

While `nglui` will attempt to look up which version is associated with a given url, that that you will need to include a caveclient to do so.
You may see a warning if you do not include a caveclient object, but the code will still run.
If the resulting state does not work, you may need to specify the version of Neuroglancer you are using or add a caveclient.


## Why these different types of Neuroglancer?

Neuroglancer is a webapp for viewing large scale 3d image datasets that has been developed over many years.
At one point, there was a major fork by the lab of Sebastian Seung to allow proofreading of connectomics data, which led to the development of the `seunglab` version of Neuroglancer.
More recently, many of the features that were developed for the `seunglab` version have been integrated into the main version of Neuroglancer, which is sometimes referred to as `mainline`.
Unfortunately the states generated by the two versions are not compatible, so it is important to specify which version you are using when generating states.

### What about `spelunker`/`cave-explorer`?

Neuroglancer is a webapp where the same code can be deployed to many different URLs with different configurations.
Each configuration sets things like key commands and state loading options.
Meanwhile, we continue to add features to the mainline version of Neuroglancer, and the `spelunker` version is a deployment of the mainline version as well as the bleeding edge features that are almost ready to be added to the mainline version.
The states generated by the `spelunker` version are compatible with the mainline version, but the `spelunker` version might have additional features that are not yet available in the mainline version.
The default URL for all of `mainline`, `spelunker`, and `cave-explorer` is a deployment of the spelunker code, but you can specify a different URL if you want to use a different deployment.

### What version should I use?

If you are not sure, you should use the `spelunker` version. This will give you the most features and the most up-to-date version of Neuroglancer. It will also be the default in an upcoming release.

The simplest way to do this is to set the default configuration to `spelunker` as shown above:

```python
site_config.set_default_config(target_site='spelunker')
```
