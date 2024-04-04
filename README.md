# Auto Vertical Stitch

This program is used to stitch together tomographic images. It was created to automate common workflow tasks when processing images after image acquisition.

Vertical stitching is necessary as overlapping image sets are created when the sample stage is moved up or down during image acquisition.

![Screenshot of the application](screenshot.png)

Created by Iain Emslie in 2022 for Jarvis Stobbs at the Canadian Light Source

# Setup

```sh
$ python -m venv stitchenv
$ source stitchenv/Scripts/activate
$ pip install requirements.txt
$ python auto_vertical_stitch.funcs.py
```
