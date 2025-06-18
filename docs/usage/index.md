---
title: Introduction
---

NGLui is split into three main modules, each with its own purpose:

- **statebuilder**: This module provides a framework for building Neuroglancer states programmatically. It allows you to create and manipulate layers, selections, and annotations in a structured way, making it easier to manage complex Neuroglancer states.
- **segmentprops**: This module is designed to generate segment properties for Neuroglancer states, either explicitly or through dataframes. It integrates closely with `statebuilder`, but creates a distinct file outside of the state that must be hosted online.
- **parser**: This module provides tools to easily extract information from Neuroglancer states without needing to parse the JSON manually. It allows you to create dataframes for layers, selections, and annotations.

You can find documentation for each module in the sidebar.