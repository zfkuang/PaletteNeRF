# Efficient palette-based decomposition and recoloring of images via RGBXY-space geometry

This code implements the pipeline described in the SIGGRAPH Asia 2018 paper ["Efficient palette-based decomposition and recoloring of images via RGBXY-space geometry"](https://cragl.cs.gmu.edu/fastlayers/) Jianchao Tan, Jose Echevarria, and Yotam Gingold.

A different and simpler prototype implementation can be found in [this link](https://cragl.cs.gmu.edu/fastlayers/RGBXY_weights.py)

## Running

### Launching the GUI:

The GUI runs in the browser at: http://localhost:8000/
For that to work, you need to run a server.

You can run the server via Docker (no need to install any dependencies on your machine). You won't get an OpenCL implementation of the layer updating, but it is still quite fast.

    docker pull cragl/fastlayers
    docker run -p 8000:8000 -p 9988:9988 cragl/fastlayers

If you install all the dependencies (see below), you can run without Docker:

    cd image-layer-updating-GUI
    ./runboth.sh

If you are on Windows (untested), the `runboth.sh` script probably won't work. Instead, run the two Python server commands manually in two separate command lines:

    cd image-layer-updating-GUI
    python3 server.py

and

    cd image-layer-updating-GUI
    python3 -m http.server

Some videos of GUI usage can be found in [this link](https://cragl.cs.gmu.edu/fastlayers/)

The `turquoise.png` image is copyright [Michelle Lee](https://cargocollective.com/michellelee).

### Testing

To test the whole pipeline without launching the GUI server, run `Our_preprocessing_pipeline.ipynb` as a Jupyter notebook.

You can test if your installation is working by comparing your output to the `test/turquoise groundtruth results/` directory.

## Image Recoloring GUI:

You can perform global recoloring using the resulting layers via our [web GUI](https://yig.github.io/image-rgb-in-3D/).
First load the original image, then drag-and-drop the saved palette `.js` file, and finally drag-and-drop the saved mixing weights `.js` file. Then you can click and move the palette vertices in the GUI to perform image recoloring.
This image recoloring web GUI is also used in our previous project, [Decomposing Images into Layers via RGB-space Geometry](https://github.com/JianchaoTan/Decompose-Single-Image-Into-Layers).

## Dependencies

You can install all of the following via: `pip install -r requirements.txt`.

* Python3.6
* NumPy
* SciPy
* Cython
* [GLPK](https://www.gnu.org/software/glpk/) (`brew install glpk`)
* cvxopt, built with the [GLPK](https://www.gnu.org/software/glpk/) linear programming solver interface (`CVXOPT_BUILD_GLPK=1 pip install cvxopt`)
* PIL or Pillow (Python Image Library) (`pip install Pillow`)
* pyopencl
* websockets (`pip install websockets`)
