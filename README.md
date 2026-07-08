# OBB Vision — 3D Oriented Bounding Box Calculator

Upload a raw 3D mesh (`.obj`, `.ply`, `.stl`) and see the script read it, compute the
**tightest-fitting Oriented Bounding Box (OBB)**, and render the box directly around the
object in an interactive 3D viewer — right in the browser.

Built for the Cube / Cylinder / Teapot measurement task: it reads the file, calculates the
OBB (rotated to fit tightly, not an axis-aligned box), and reports Volume and Dimensions
(L × W × H).

---

## How it works

1. **Backend (`app.py`)** — Flask server.
   - Loads the mesh with `trimesh`.
   - Computes the OBB using **PCA** (Principal Component Analysis):
     centers the vertices, finds the eigenvectors of the covariance matrix (the object's
     natural axes), projects the mesh onto those axes, and measures the tight extents.
   - Sends the mesh geometry and OBB box corners to the browser as JSON.

2. **Frontend (`templates/index.html`)** — Three.js viewer.
   - Renders the uploaded mesh in 3D.
   - Draws the OBB as a cyan wireframe box around it.
   - Fully orbit/zoom/pan-able so you can visually confirm the tight fit.
   - Shows a live console log of what the script is doing (reading file → computing OBB → done).

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv env
env\Scripts\activate        # Windows
source env/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
python app.py
```

Open your browser to **http://127.0.0.1:5000**

---

## Usage

- **Upload your own file:** drag and drop `CUBE.obj`, `CYLINDER.obj`, `TEAPOT.obj` (or any
  `.obj`/`.ply`/`.stl`) onto the upload zone, or click it to browse.
- **Or try a sample:** click **Cube**, **Cylinder**, or **Teapot** in the sidebar — these are
  auto-generated on first use if you don't have the assignment's files handy.
- The 3D viewer updates instantly showing the mesh with its tight OBB wrapped around it.
- The sidebar shows Volume, Length, Width, Height, vertex/face counts, and how much smaller
  the OBB is compared to a plain axis-aligned box.

---


## File structure

```
app.py                 Flask backend + OBB algorithm
templates/index.html   3D viewer + upload UI (Three.js)
requirements.txt        Python dependencies
```

## Requirements

- Python 3.8+
- A modern browser (Chrome, Firefox, Edge, Safari) with WebGL support
