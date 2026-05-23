import os
import io
import base64
import heapq

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from flask import Flask, render_template, request, jsonify
from src.model import UNet
from src.ndvi import compute_ndvi

app = Flask(__name__)
app.secret_key = os.urandom(24)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- A* PATHFINDING LOGIC ---
directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

def heuristic(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def astar(cost_map, start, goal):
    HEIGHT, WIDTH = cost_map.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    while open_set:
        current = heapq.heappop(open_set)[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        for dx, dy in directions:
            nx, ny = current[0]+dx, current[1]+dy
            neighbor = (nx, ny)
            if nx < 0 or ny < 0 or nx >= HEIGHT or ny >= WIDTH:
                continue
            terrain_cost = cost_map[nx, ny] * 8
            movement_cost = 1.4 if dx != 0 and dy != 0 else 1.0
            tentative_g = g_score[current] + terrain_cost + movement_cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + 0.7 * heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None

# --- MODEL LOADING ---
_model = None
def load_model():
    global _model
    if _model is None:
        _model = UNet(in_channels=3, out_channels=1)
        model_path = "models/eco_rgb_model.pth"
        if os.path.exists(model_path):
            _model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        _model.to(DEVICE)
        _model.eval()
    return _model

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor='#0a0f0a', edgecolor='none', dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded

def normalize_img(img):
    """
    Robustly extract an RGB array from a .npy file regardless of shape.
    Supports shapes: (C, H, W) with C>=3  OR  (H, W, C) with C>=3  OR  (H, W) grayscale.
    Returns float32 array of shape (H, W, 3) clipped to [0,1],
    and the original (C, H, W) form for model input.
    """
    img = img.astype(np.float32)

    # Normalise value range to [0,1] if it looks like uint8/uint16
    if img.max() > 1.0:
        img = img / (img.max() + 1e-8)

    if img.ndim == 2:
        # Grayscale → replicate to 3 channels in (C,H,W)
        img_chw = np.stack([img, img, img], axis=0)
    elif img.ndim == 3:
        if img.shape[0] <= 8 and img.shape[0] < img.shape[1]:
            # (C, H, W) — already channel-first; good for model
            img_chw = img
        else:
            # (H, W, C) — transpose to (C, H, W)
            img_chw = np.transpose(img, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported array shape: {img.shape}")

    C = img_chw.shape[0]
    # Pick R,G,B bands safely (fall back to band 0 if fewer than 3)
    r = img_chw[min(2, C-1)]
    g = img_chw[min(1, C-1)]
    b = img_chw[0]
    rgb = np.stack([r, g, b], axis=-1)  # (H,W,3)
    rgb = np.clip(rgb, 0, 1)
    return rgb, img_chw   # rgb for display, img_chw for model / ndvi

def make_rgb_figure(rgb):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#0a0f0a')
    ax.imshow(rgb)
    ax.axis('off')
    return fig

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/device')
def device():
    return jsonify({'device': 'cuda' if torch.cuda.is_available() else 'cpu'})

@app.route('/api/load_sample', methods=['POST'])
def load_sample():
    os.makedirs("outputs", exist_ok=True)

    # Try both sample filenames
    candidates = [
        "data/sample/img_0010.npy",
        "data/sample/img_0001.npy",
    ]
    img_raw = None
    for path in candidates:
        if os.path.exists(path):
            img_raw = np.load(path)
            break

    if img_raw is None:
        # Generate a synthetic sample so the UI is always usable
        rng = np.random.default_rng(42)
        H, W = 128, 128
        img_raw = rng.random((4, H, W)).astype(np.float32)
        # Add some structure: simulate NDVI-like band variation
        xx, yy = np.meshgrid(np.linspace(0,1,W), np.linspace(0,1,H))
        img_raw[0] = np.clip(0.2 + 0.5*np.sin(xx*8)*np.cos(yy*8) + rng.random((H,W))*0.1, 0, 1)
        img_raw[1] = np.clip(0.3 + 0.4*np.cos(xx*5)*np.sin(yy*5) + rng.random((H,W))*0.1, 0, 1)
        img_raw[2] = np.clip(0.15 + 0.3*np.sin((xx+yy)*6) + rng.random((H,W))*0.1, 0, 1)
        img_raw[3] = np.clip(0.6 + 0.3*np.cos((xx-yy)*4) + rng.random((H,W))*0.05, 0, 1)

    np.save("outputs/current_img.npy", img_raw)
    rgb, _ = normalize_img(img_raw)
    fig = make_rgb_figure(rgb)
    img_b64 = fig_to_b64(fig)
    H, W = rgb.shape[:2]
    return jsonify({'success': True, 'rgb_image': img_b64, 'height': H, 'width': W})

@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    try:
        img_bytes = f.read()
        img_raw = np.load(io.BytesIO(img_bytes))
    except Exception as e:
        return jsonify({'error': f'Failed to read .npy file: {str(e)}'}), 400

    os.makedirs("outputs", exist_ok=True)
    np.save("outputs/current_img.npy", img_raw)
    try:
        rgb, _ = normalize_img(img_raw)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    fig = make_rgb_figure(rgb)
    img_b64 = fig_to_b64(fig)
    H, W = rgb.shape[:2]
    return jsonify({'success': True, 'rgb_image': img_b64, 'height': H, 'width': W})

@app.route('/api/inference', methods=['POST'])
def inference():
    if not os.path.exists("outputs/current_img.npy"):
        return jsonify({'error': 'No image loaded. Load data first.'}), 400
    img_raw = np.load("outputs/current_img.npy")
    try:
        _, img_chw = normalize_img(img_raw)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    model = load_model()
    # Use first 3 channels for RGB input
    input_rgb = img_chw[:3].astype(np.float32)
    input_rgb = np.clip(input_rgb, 0, 1)
    tensor = torch.tensor(input_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(tensor).squeeze().cpu().numpy()

    os.makedirs("outputs", exist_ok=True)
    np.save("outputs/probability_map.npy", pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#0a0f0a')
    cax = ax.imshow(pred, cmap="viridis")
    fig.colorbar(cax, ax=ax)
    ax.axis('off')
    pred_b64 = fig_to_b64(fig)
    return jsonify({'success': True, 'pred_image': pred_b64})

@app.route('/api/cost_surface', methods=['POST'])
def cost_surface():
    data = request.get_json()
    w_ndvi = float(data.get('w_ndvi', 0.45))
    w_seg  = float(data.get('w_seg', 0.45))

    if not os.path.exists("outputs/probability_map.npy"):
        return jsonify({'error': 'Run Model Inference first (Tab 1).'}), 400
    if not os.path.exists("outputs/current_img.npy"):
        return jsonify({'error': 'No image loaded.'}), 400

    img_raw = np.load("outputs/current_img.npy")
    _, img_chw = normalize_img(img_raw)
    segmentation = np.load("outputs/probability_map.npy")

    try:
        ndvi = compute_ndvi(img_chw)
    except Exception:
        # Fallback: use band difference if compute_ndvi fails
        C = img_chw.shape[0]
        nir = img_chw[min(3, C-1)]
        red = img_chw[min(2, C-1)]
        ndvi = (nir - red) / (nir + red + 1e-8)

    ndvi_norm = (ndvi + 1) / 2
    cost = (w_ndvi * ndvi_norm) + (w_seg * segmentation)
    cost = (cost - cost.min()) / (cost.max() - cost.min() + 1e-8)
    np.save("outputs/cost_surface.npy", cost)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#0a0f0a')
    cax = ax.imshow(cost, cmap="inferno")
    fig.colorbar(cax, ax=ax, label="Traversal Cost")
    ax.axis('off')
    cost_b64 = fig_to_b64(fig)
    return jsonify({'success': True, 'cost_image': cost_b64})

@app.route('/api/find_path', methods=['POST'])
def find_path():
    data = request.get_json()
    start_y = int(data.get('start_y', 0))
    start_x = int(data.get('start_x', 0))
    goal_y  = int(data.get('goal_y', 100))
    goal_x  = int(data.get('goal_x', 100))

    if not os.path.exists("outputs/cost_surface.npy"):
        return jsonify({'error': 'Generate Cost Surface first (Tab 2).'}), 400
    if not os.path.exists("outputs/current_img.npy"):
        return jsonify({'error': 'No image loaded.'}), 400

    cost_surface = np.load("outputs/cost_surface.npy")
    img_raw = np.load("outputs/current_img.npy")
    rgb, _ = normalize_img(img_raw)

    HEIGHT, WIDTH = cost_surface.shape
    # Clamp coords to valid range
    start_y = max(0, min(start_y, HEIGHT-1))
    start_x = max(0, min(start_x, WIDTH-1))
    goal_y  = max(0, min(goal_y,  HEIGHT-1))
    goal_x  = max(0, min(goal_x,  WIDTH-1))

    path = astar(cost_surface, (start_y, start_x), (goal_y, goal_x))
    if path is None:
        return jsonify({'error': 'No valid path found between these points.'}), 400

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#0a0f0a')
    ax.imshow(rgb)
    ax.plot([p[1] for p in path], [p[0] for p in path],
            color="#00ffaa", linewidth=2.5, alpha=0.9)
    ax.scatter(start_x, start_y, color="#39ff14", s=120,
               edgecolors="white", linewidths=1.5, label="Start", zorder=5)
    ax.scatter(goal_x, goal_y, color="#ff4444", s=120,
               edgecolors="white", linewidths=1.5, label="Goal", zorder=5)
    ax.legend(facecolor='#0a0f0a', edgecolor='#2a4a2a', labelcolor='#c8e6c9')
    ax.axis('off')
    path_b64 = fig_to_b64(fig)
    return jsonify({'success': True, 'path_image': path_b64, 'path_length': len(path)})

if __name__ == '__main__':
    os.makedirs("outputs", exist_ok=True)
    app.run(debug=True, port=5000)
