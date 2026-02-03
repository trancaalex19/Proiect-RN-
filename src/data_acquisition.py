# data_acquisition.py
import cv2
import numpy as np
try:
    import fitz  # PyMuPDF
except ImportError:
    pass 
from config import IMG_SIZE

# --- Funcții Geometrice & Preprocesare ---

def center_image_by_mass(img):
    if img is None: return None
    M = cv2.moments(img)
    if M["m00"] == 0: return img
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    h, w = img.shape
    centerY, centerX = h // 2, w // 2
    dX = centerX - cX
    dY = centerY - cY
    M_trans = np.float32([[1, 0, dX], [0, 1, dY]])
    centered = cv2.warpAffine(img, M_trans, (w, h))
    return centered

def get_projections(roi_img):
    if roi_img is None: return None, None
    binary = roi_img / 255.0
    proj_x = np.sum(binary, axis=0)
    proj_y = np.sum(binary, axis=1)
    target_len = 64 
    def resize_vector(vec):
        if len(vec) == 0: return np.zeros(target_len)
        indices = np.linspace(0, len(vec)-1, target_len)
        return np.interp(indices, np.arange(len(vec)), vec)
    vec_x = resize_vector(proj_x)
    vec_y = resize_vector(proj_y)
    norm_x = np.linalg.norm(vec_x)
    norm_y = np.linalg.norm(vec_y)
    if norm_x > 0: vec_x = vec_x / norm_x
    if norm_y > 0: vec_y = vec_y / norm_y
    return vec_x, vec_y

def get_binary_roi(img):
    if img is None: return None, 0
    if len(img.shape) == 3: gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: gray = img
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    if coords is None: return None, 0
    x, y, w, h = cv2.boundingRect(coords)
    roi = thresh[y:y+h, x:x+w]
    aspect_ratio = float(w) / h if h > 0 else 0
    canvas_h, canvas_w = int(h * 1.5), int(w * 1.5)
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    y_off = (canvas_h - h) // 2
    x_off = (canvas_w - w) // 2
    canvas[y_off:y_off+h, x_off:x_off+w] = roi
    final_roi = center_image_by_mass(canvas)
    coords2 = cv2.findNonZero(final_roi)
    if coords2 is not None:
        x2, y2, w2, h2 = cv2.boundingRect(coords2)
        final_roi = final_roi[y2:y2+h2, x2:x2+w2]
    return final_roi, aspect_ratio

def get_density_grid(roi_img, grid_rows=4, grid_cols=5):
    if roi_img is None: return np.zeros((grid_rows, grid_cols))
    target_h, target_w = grid_rows * 20, grid_cols * 20
    resized = cv2.resize(roi_img, (target_w, target_h))
    h, w = resized.shape
    step_h = h // grid_rows
    step_w = w // grid_cols
    density_map = np.zeros((grid_rows, grid_cols))
    for r in range(grid_rows):
        for c in range(grid_cols):
            y1, y2 = r * step_h, (r+1) * step_h
            x1, x2 = c * step_w, (c+1) * step_w
            cell = resized[y1:y2, x1:x2]
            total = cell.size
            ink = cv2.countNonZero(cell)
            density_map[r, c] = ink / total if total > 0 else 0
    return density_map

# --- Funcții Achiziție (PDF -> Img) ---

def pdf_to_hd_image(content):
    doc = fitz.open(stream=content, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if pix.n >= 3 else img

def finalize_crop(img):
    if img is None or img.size == 0: return np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
    if len(img.shape) == 3: gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: gray = img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    if coords is None: return np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(coords)
    pad = 5
    h_img, w_img = thresh.shape
    x = max(0, x - pad); y = max(0, y - pad)
    w = min(w_img - x, w + 2*pad); h = min(h_img - y, h + 2*pad)
    roi = thresh[y:y+h, x:x+w]
    h_roi, w_roi = roi.shape
    scale = (IMG_SIZE * 0.8) / max(h_roi, w_roi)
    nw = int(w_roi * scale); nh = int(h_roi * scale)
    resized = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    dx = (IMG_SIZE - nw) // 2; dy = (IMG_SIZE - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = resized
    return cv2.bitwise_not(canvas)

def extract_signatures_blob_mode(img, is_grid=False, expected_count=20):
    h_img, w_img = img.shape[:2]
    safe_height = int(h_img * 0.90) 
    work_img_full = img[:safe_height, :] 
    
    if not is_grid:
        x_start = int(w_img * 0.55)
        work_img = work_img_full[:, x_start:]
    else:
        work_img = work_img_full
        x_start = 0

    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 8))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 20 and h > 15: 
            blobs.append((x + x_start, y, w, h))
    blobs.sort(key=lambda b: (b[1] // 80, b[0]) if is_grid else b[1])
    visuals = []
    raw_crops = []
    for (x, y, w, h) in blobs:
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(safe_height, y + h + pad) 
        crop = img[y1:y2, x1:x2] 
        visuals.append(finalize_crop(crop))
        raw_crops.append(crop)
    while len(visuals) < expected_count:
        visuals.append(np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8))
        raw_crops.append(None)
    return visuals[:expected_count], raw_crops[:expected_count]

def compare_structures_balanced(roi1, ar1, proj1, roi2, ar2, proj2, ai_confidence):
    if roi1 is None or roi2 is None: return 0.0, 0.0, 0.0, 0.0
    diff_ar = abs(ar1 - ar2) / max(ar1, ar2)
    penalty_ar = max(0, diff_ar - 0.20)
    ar_score = max(0, 1.0 - (penalty_ar * 5.0)) 
    grid1 = get_density_grid(roi1)
    grid2 = get_density_grid(roi2)
    diff_grid = np.abs(grid1 - grid2)
    diff_grid[diff_grid < 0.15] = 0 
    mse = np.mean(diff_grid ** 2)
    density_score = max(0, 1.0 - (mse * 5.0))
    vec_x1, vec_y1 = proj1
    vec_x2, vec_y2 = proj2
    corr_x = np.corrcoef(vec_x1, vec_x2)[0, 1] if np.std(vec_x1) > 0 and np.std(vec_x2) > 0 else 0
    corr_y = np.corrcoef(vec_y1, vec_y2)[0, 1] if np.std(vec_y1) > 0 and np.std(vec_y2) > 0 else 0
    corr_x = max(0, corr_x)
    corr_y = max(0, corr_y)
    proj_score = (corr_x * 0.5) + (corr_y * 0.5)
    return ar_score, density_score, proj_score, diff_ar