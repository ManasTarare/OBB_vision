"""
3D Object Oriented Bounding Box (OBB) Calculator
-------------------------------------------------
Reads a raw 3D mesh (.obj / .ply / .stl), computes the tightest-fitting
Oriented Bounding Box using PCA, and returns the mesh + box geometry
so the browser can render both together.
"""

import os
import logging
import numpy as np
import trimesh
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"obj", "ply", "stl"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def compute_obb(mesh):
    """
    Compute the Oriented Bounding Box (OBB) of a mesh using PCA.

    Steps:
      1. Center the vertices at the mesh centroid.
      2. Build the covariance matrix of the centered vertices.
      3. Eigen-decompose it -> eigenvectors are the OBB's principal axes.
      4. Project vertices onto those axes to get the tight extents.
      5. Rebuild the 8 box corners in world space for visualization.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)

    if len(vertices) < 4:
        raise ValueError("Mesh has too few vertices for OBB calculation")

    centroid = vertices.mean(axis=0)
    centered = vertices - centroid

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort axes by eigenvalue, largest spread first
    order = np.argsort(-eigvals)
    eigvecs = eigvecs[:, order]

    # Ensure a right-handed rotation matrix (avoids mirrored boxes)
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, -1] *= -1

    local = centered @ eigvecs
    min_b, max_b = local.min(axis=0), local.max(axis=0)
    dimensions = max_b - min_b
    volume = float(np.prod(dimensions))

    local_center = (min_b + max_b) / 2
    world_center = local_center @ eigvecs.T + centroid

    half = dimensions / 2
    corners_local = np.array([
        [-half[0], -half[1], -half[2]],
        [ half[0], -half[1], -half[2]],
        [ half[0],  half[1], -half[2]],
        [-half[0],  half[1], -half[2]],
        [-half[0], -half[1],  half[2]],
        [ half[0], -half[1],  half[2]],
        [ half[0],  half[1],  half[2]],
        [-half[0],  half[1],  half[2]],
    ])
    corners_world = corners_local @ eigvecs.T + world_center

    # Sort dimensions descending for reporting L x W x H
    sorted_dims = np.sort(dimensions)[::-1]

    return {
        "volume": volume,
        "dimensions": {
            "length": float(sorted_dims[0]),
            "width": float(sorted_dims[1]),
            "height": float(sorted_dims[2]),
        },
        "center": world_center.tolist(),
        "corners": corners_world.tolist(),
    }


def compute_aabb(mesh):
    """Axis-Aligned Bounding Box, kept only as a reference comparison."""
    vertices = np.asarray(mesh.vertices)
    dims = vertices.max(axis=0) - vertices.min(axis=0)
    return {
        "volume": float(np.prod(dims)),
        "dimensions": {
            "length": float(dims[0]),
            "width": float(dims[1]),
            "height": float(dims[2]),
        },
    }


def load_mesh(filepath):
    """
    Load a mesh's geometry only (vertices/faces). Materials and textures are
    skipped on purpose -- OBB math only needs geometry, and skipping materials
    avoids a hard dependency on Pillow for image-based textures some OBJ/MTL
    files reference.
    """
    try:
        mesh = trimesh.load(filepath, force="mesh", skip_materials=True, process=False)
    except TypeError:
        # Older trimesh versions don't accept skip_materials as a kwarg here;
        # fall back to a plain load.
        mesh = trimesh.load(filepath, force="mesh", process=False)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.geometry.values())
    return mesh


def mesh_to_response(filename, mesh):
    obb = compute_obb(mesh)
    aabb = compute_aabb(mesh)
    improvement = (
        (aabb["volume"] - obb["volume"]) / aabb["volume"] * 100
        if aabb["volume"] > 0 else 0
    )

    return {
        "status": "success",
        "filename": filename,
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        # Geometry sent to the browser for rendering
        "mesh": {
            "vertices": np.asarray(mesh.vertices, dtype=np.float32).flatten().tolist(),
            "faces": np.asarray(mesh.faces, dtype=np.int32).flatten().tolist(),
        },
        "obb": obb,
        "aabb": aabb,
        "improvement": float(improvement),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    filename = file.filename
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        logger.info("Reading mesh file: %s", filename)
        mesh = load_mesh(filepath)
        response = mesh_to_response(filename, mesh)
        logger.info("Computed OBB for %s -> volume=%.2f", filename, response["obb"]["volume"])
        return jsonify(response)
    except Exception as exc:
        logger.exception("Failed to process %s", filename)
        return jsonify({"error": f"Error processing file: {exc}"}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
