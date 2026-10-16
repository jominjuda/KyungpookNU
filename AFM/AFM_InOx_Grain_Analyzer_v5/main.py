
import subprocess
import sys
import os
import warnings
import json
import re
warnings.filterwarnings('ignore')


def install(pkg):
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', pkg],
        stdout=subprocess.DEVNULL
    )


required = {
    'numpy': 'numpy',
    'matplotlib': 'matplotlib',
    'scipy': 'scipy',
    'scikit-image': 'skimage',
    'Pillow': 'PIL',
}

for pip_name, import_name in required.items():
    try:
        __import__(import_name)
    except ImportError:
        print(f'Installing {pip_name}...')
        install(pip_name)


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.ndimage import (
    gaussian_filter,
    median_filter,
    label as ndi_label,
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
)
from scipy.stats import rankdata
from scipy.spatial import cKDTree

from skimage.feature import canny
from skimage.filters import sobel
from skimage.segmentation import watershed, clear_border
from skimage.morphology import (h_maxima, h_minima, binary_closing, disk, remove_small_objects)
from skimage.measure import regionprops
from skimage.transform import resize

from PIL import Image
from datetime import datetime


# ══════════════════════════════════════════════════════════════
#  I/O
# ══════════════════════════════════════════════════════════════

def _read_text_robust(filepath):
    """여러 인코딩을 순서대로 시도하여 AFM 텍스트 파일을 읽는다."""
    for enc in ('utf-8-sig', 'utf-8', 'cp949', 'latin-1'):
        try:
            with open(filepath, 'r', encoding=enc) as f:
                lines = f.readlines()
            if any(line.startswith('Data Size') for line in lines):
                return lines, enc
        except (UnicodeDecodeError, UnicodeError):
            continue

    with open(filepath, 'r', encoding='latin-1') as f:
        return f.readlines(), 'latin-1(fallback)'


def load_afm(filepath):
    """AFM 텍스트 데이터를 읽고 X/Y 공간 단위를 내부적으로 nm로 통일한다."""
    meta = {}
    data = []
    lines, used_enc = _read_text_robust(filepath)
    data_start = 0

    for i, line in enumerate(lines):
        if line.startswith('Data Size'):
            p = line.split('\t')[1].strip().split(' x ')
            meta['nx'], meta['ny'] = int(p[0]), int(p[1])

        elif line.startswith('Surface Size'):
            p = line.split('\t')[1].strip().split(' x ')
            meta['sx'], meta['sy'] = float(p[0]), float(p[1])

        elif line.startswith('X Unit'):
            meta['x_unit'] = line.split('\t')[1].strip()

        elif line.startswith('Z Unit'):
            meta['z_unit'] = line.split('\t')[1].strip()

        elif line.strip() == 'X\tY\tZ':
            data_start = i + 1
            break

    for line in lines[data_start:]:
        p = line.strip().split('\t')
        if len(p) >= 3:
            try:
                data.append(float(p[2]))
            except ValueError:
                pass

    if 'nx' not in meta or 'ny' not in meta:
        raise ValueError(
            f'"{filepath}" 파일에서 "Data Size" 헤더를 찾지 못했습니다 '
            f'(최종 사용 인코딩: {used_enc}).'
        )

    expected = meta['nx'] * meta['ny']
    if len(data) != expected:
        raise ValueError(
            f'"{filepath}" 데이터 개수가 맞지 않습니다. '
            f'expected={expected}, actual={len(data)}'
        )

    z = np.asarray(data, dtype=float).reshape(meta['ny'], meta['nx'])

    unit_raw = meta.get('x_unit', '')
    unit = unit_raw.lower()
    print(f'  [Unit] Raw X Unit field: "{unit_raw}"')

    is_um = any(
        tag in unit
        for tag in ['um', '\u00b5m', '\u03bcm', 'micron', 'micro']
    )
    if not is_um and meta['sx'] < 100:
        is_um = True
        print(
            f'  [Unit] sx={meta["sx"]:.3f} is too small for nm-scale scan; '
            f'treating as µm.'
        )

    if is_um:
        meta['sx'] *= 1000.0
        meta['sy'] *= 1000.0
        meta['x_unit'] = 'nm'
        print(
            f'  [Unit] Converted to nm: '
            f'{meta["sx"]:.1f} x {meta["sy"]:.1f} nm'
        )
    else:
        meta['x_unit'] = 'nm'

    return z, meta


def load_reference_image(filepath, target_shape):
    """PNG/JPG 이미지를 읽고 target_shape에 맞춘다."""
    if filepath is None or not os.path.isfile(filepath):
        return None

    img = np.array(Image.open(filepath).convert('RGB'))
    if img.shape[:2] != target_shape:
        img = resize(
            img,
            (*target_shape, 3),
            preserve_range=True,
            anti_aliasing=True
        ).astype(np.uint8)
    return img


def sanitize_label(label):
    if not label:
        return ''
    return re.sub(r'[<>:"/\\|?*]', '', label).strip().strip('.')


def extract_condition_label(phase_fp):
    parts = [p for p in re.split(r'[\\/]+', phase_fp) if p]

    if len(parts) >= 3:
        grandparent, parent = parts[-3], parts[-2]
        if parent.lower().startswith(grandparent.lower()):
            label = sanitize_label(grandparent)
            if label:
                return label

    if len(parts) >= 2:
        parent = parts[-2]
        cleaned = re.sub(
            r'_(?:Phase|Amplitude|Amp|Height|Topography)?_?'
            r'(?:Forward|Backward)$',
            '',
            parent,
            flags=re.IGNORECASE
        )
        label = sanitize_label(cleaned)
        if label:
            return label

    fname = os.path.splitext(parts[-1])[0] if parts else 'sample'
    for kw in ['_Forward', '_Backward', '_forward', '_backward']:
        fname = fname.replace(kw, '')
    return sanitize_label(fname) or 'sample'


def make_output_dir(phase_fp):
    condition = extract_condition_label(phase_fp)
    timestamp = datetime.now().strftime('%y%m%d_%H%M')
    dirname = f'{timestamp}_{condition}' if condition else timestamp

    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, dirname)

    candidate = out_dir
    suffix = 1
    while os.path.exists(candidate):
        candidate = f'{out_dir}_{suffix}'
        suffix += 1

    os.makedirs(candidate, exist_ok=True)
    print(f'  Output folder: {candidate}')
    return candidate


def basename_clean(filepath):
    b = os.path.splitext(os.path.basename(filepath))[0]
    for kw in ['_Forward', '_Backward', '_forward', '_backward']:
        b = b.replace(kw, '')
    return b


def save_fig(fig, ref_path, suffix, out_dir):
    out = os.path.join(
        out_dir,
        f'{basename_clean(ref_path)}_{suffix}.png'
    )
    fig.savefig(
        out,
        dpi=150,
        bbox_inches='tight',
        facecolor=fig.get_facecolor()
    )
    print(f'  Saved: {os.path.basename(out)}')


def save_independent(
    draw_fn,
    ref_path,
    panel_name,
    out_dir,
    figsize=(6, 5.5),
    facecolor='white'
):
    fig, ax = plt.subplots(
        1,
        1,
        figsize=figsize,
        facecolor=facecolor,
        constrained_layout=True
    )
    draw_fn(fig, ax)

    out = os.path.join(
        out_dir,
        f'{basename_clean(ref_path)}_{panel_name}.png'
    )
    fig.savefig(
        out,
        dpi=150,
        bbox_inches='tight',
        facecolor=facecolor
    )
    print(f'  Saved: {os.path.basename(out)}')
    plt.close(fig)


def style_ax(ax, title, xu='X (µm)', yu='Y (µm)'):
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel(xu, fontsize=8)
    ax.set_ylabel(yu, fontsize=8)
    ax.tick_params(labelsize=8)


def make_extent(meta):
    return [
        0,
        meta['sx'] / 1000.0,
        0,
        meta['sy'] / 1000.0
    ]


# ══════════════════════════════════════════════════════════════
#  전처리
# ══════════════════════════════════════════════════════════════

def remove_line_artifacts(z, method='median'):
    z_out = z.copy().astype(float)

    if method == 'median':
        for i in range(z_out.shape[0]):
            z_out[i] -= np.median(z_out[i])

    elif method == 'mean':
        for i in range(z_out.shape[0]):
            z_out[i] -= np.mean(z_out[i])

    elif method == 'poly1':
        x = np.arange(z_out.shape[1])
        for i in range(z_out.shape[0]):
            coeffs = np.polyfit(x, z_out[i], 1)
            z_out[i] -= np.polyval(coeffs, x)

    else:
        raise ValueError(f'Unknown line leveling method: {method}')

    return z_out


def remove_spikes(z, threshold_sigma=6.0, kernel=7, max_cluster_px=5):
    z_out = z.copy().astype(float)
    z_mean = float(np.mean(z_out))
    z_std = float(np.std(z_out))

    if z_std <= 1e-12:
        return z_out, np.zeros_like(z_out, dtype=bool)

    spike_candidate = np.abs(z_out - z_mean) > threshold_sigma * z_std
    labeled, n_clusters = ndi_label(spike_candidate)

    spike_mask = np.zeros_like(z_out, dtype=bool)
    for i in range(1, n_clusters + 1):
        cluster = labeled == i
        if cluster.sum() <= max_cluster_px:
            spike_mask |= cluster

    if spike_mask.sum() == 0:
        return z_out, spike_mask

    z_filtered = median_filter(z_out, size=kernel)
    z_out[spike_mask] = z_filtered[spike_mask]
    return z_out, spike_mask


def preprocess(
    z,
    line_level=True,
    line_method='median',
    spike_remove=True,
    spike_sigma=6.0,
    spike_kernel=7,
    max_cluster_px=5,
    verbose=True
):
    z_out = z.copy().astype(float)
    spike_mask = np.zeros_like(z_out, dtype=bool)

    if line_level:
        z_out = remove_line_artifacts(z_out, method=line_method)
        if verbose:
            print(f'  [Preprocess] Line leveling ({line_method}) applied')

    if spike_remove:
        z_out, spike_mask = remove_spikes(
            z_out,
            threshold_sigma=spike_sigma,
            kernel=spike_kernel,
            max_cluster_px=max_cluster_px
        )
        if verbose:
            ratio = spike_mask.sum() / spike_mask.size * 100.0
            print(
                f'  [Preprocess] Spike removal: '
                f'{spike_mask.sum()} px ({ratio:.2f}%)'
            )

    return z_out, spike_mask


# ══════════════════════════════════════════════════════════════
#  Local Phase feature 계산
# ══════════════════════════════════════════════════════════════

def _window_bounds(shape, r, c, radius):
    r = int(r)
    c = int(c)
    radius = max(1, int(radius))

    r0 = max(0, r - radius)
    r1 = min(shape[0], r + radius + 1)
    c0 = max(0, c - radius)
    c1 = min(shape[1], c + radius + 1)
    return r0, r1, c0, c1


def _make_core_ring_masks(shape, r, c, core_radius=1, ring_radius=3):
    ring_radius = max(int(ring_radius), int(core_radius) + 1)
    r0, r1, c0, c1 = _window_bounds(shape, r, c, ring_radius)

    yy, xx = np.ogrid[r0:r1, c0:c1]
    d2 = (yy - int(r)) ** 2 + (xx - int(c)) ** 2

    core = d2 <= core_radius ** 2
    ring = (d2 > core_radius ** 2) & (d2 <= ring_radius ** 2)
    return (r0, r1, c0, c1), core, ring


def compute_seed_feature(
    z_seed,
    gradient_map,
    r,
    c,
    seed_type,
    core_radius_px=1,
    ring_radius_px=3
):
    """Seed 주변의 3개 Phase 특성을 계산한다.

    1) local_gradient:
       seed 주변 ring의 Sobel gradient 90 percentile

    2) local_contrast:
       core median과 ring median의 절댓값 차이

    3) prominence:
       maximum이면 center - ring median,
       minimum이면 ring median - center
    """
    bounds, core_mask, ring_mask = _make_core_ring_masks(
        z_seed.shape,
        r,
        c,
        core_radius=core_radius_px,
        ring_radius=ring_radius_px
    )
    r0, r1, c0, c1 = bounds

    z_win = z_seed[r0:r1, c0:c1]
    g_win = gradient_map[r0:r1, c0:c1]

    core_vals = z_win[core_mask]
    ring_vals = z_win[ring_mask]
    ring_grads = g_win[ring_mask]

    if core_vals.size == 0:
        core_vals = np.array([z_seed[int(r), int(c)]], dtype=float)

    if ring_vals.size == 0:
        ring_vals = z_win.ravel()

    if ring_grads.size == 0:
        ring_grads = g_win.ravel()

    core_med = float(np.median(core_vals))
    ring_med = float(np.median(ring_vals))

    local_gradient = float(np.percentile(ring_grads, 90))
    local_contrast = abs(core_med - ring_med)

    if seed_type == 'max':
        prominence = max(0.0, core_med - ring_med)
    elif seed_type == 'min':
        prominence = max(0.0, ring_med - core_med)
    else:
        prominence = abs(core_med - ring_med)

    return {
        'local_gradient': local_gradient,
        'local_contrast': float(local_contrast),
        'prominence': float(prominence),
    }


def robust_percentile(values, percentile, fallback=1e-6):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float(fallback)

    return float(max(np.percentile(values, percentile), fallback))


# ══════════════════════════════════════════════════════════════
#  Wafer patch 기반 gradient threshold
# ══════════════════════════════════════════════════════════════

def compute_patchwise_gradient_thresholds(
    elevation,
    patch_rows=4,
    patch_cols=4,
    weak_percentile=95.0,
    strong_percentile=99.0
):
    """Wafer gradient map을 patch로 나누고 percentile의 중앙값을 계산한다.

    Tweak  = median(patch별 P95)
    Tstrong = median(patch별 P99)

    단일 wafer 전체의 한 지점에 존재하는 scratch/국소 artifact가
    threshold를 과도하게 끌어올리는 영향을 줄이기 위한 robust calibration이다.
    """
    patch_rows = max(1, int(patch_rows))
    patch_cols = max(1, int(patch_cols))
    weak_percentile = float(np.clip(weak_percentile, 0.0, 100.0))
    strong_percentile = float(np.clip(strong_percentile, weak_percentile, 100.0))

    row_chunks = np.array_split(np.arange(elevation.shape[0]), patch_rows)
    col_chunks = np.array_split(np.arange(elevation.shape[1]), patch_cols)

    weak_values = []
    strong_values = []
    patch_records = []

    for ri, rows in enumerate(row_chunks):
        for ci, cols in enumerate(col_chunks):
            if rows.size == 0 or cols.size == 0:
                continue
            patch = elevation[np.ix_(rows, cols)]
            vals = patch[np.isfinite(patch)]
            if vals.size == 0:
                continue

            w = float(np.percentile(vals, weak_percentile))
            s = float(np.percentile(vals, strong_percentile))
            weak_values.append(w)
            strong_values.append(s)
            patch_records.append({
                'patch_row': int(ri),
                'patch_col': int(ci),
                'weak_threshold': w,
                'strong_threshold': s,
                'n_pixels': int(vals.size),
            })

    if not weak_values:
        raise ValueError('No valid wafer gradient patches were available.')

    return {
        'weak_threshold': float(np.median(weak_values)),
        'strong_threshold': float(np.median(strong_values)),
        'weak_percentile': weak_percentile,
        'strong_percentile': strong_percentile,
        'patch_rows': patch_rows,
        'patch_cols': patch_cols,
        'patch_records': patch_records,
    }

# ══════════════════════════════════════════════════════════════
#  Seed 검출 + Watershed
# ══════════════════════════════════════════════════════════════


def detect_boundaries_qc(
    z_clean,
    canny_sigma_px,
    low_threshold=0.10,
    high_threshold=0.30
):
    return canny(
        z_clean,
        sigma=canny_sigma_px,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        use_quantiles=True
    )


def build_gradient_boundary_band(
    elevation,
    strong_percentile=99.0,
    weak_percentile=95.0,
    strong_threshold=None,
    weak_threshold=None,
    closing_radius_px=1,
    min_component_px=2
):
    """Strong/weak gradient threshold로 hysteresis-style boundary band 생성.

    권장 모드에서는 percentile을 sample에서 다시 계산하지 않고,
    grain-free wafer에서 저장한 absolute threshold를 그대로 사용한다.
    """
    strong_percentile = float(np.clip(strong_percentile, 0, 100))
    weak_percentile = float(np.clip(weak_percentile, 0, strong_percentile))

    if strong_threshold is None:
        strong_thr = float(np.percentile(elevation, strong_percentile))
        threshold_source = 'current-image percentile (fallback)'
    else:
        strong_thr = float(strong_threshold)
        threshold_source = 'wafer absolute gradient threshold'

    if weak_threshold is None:
        weak_thr = float(np.percentile(elevation, weak_percentile))
    else:
        weak_thr = float(weak_threshold)

    if weak_thr > strong_thr:
        raise ValueError('weak_threshold must be <= strong_threshold.')

    strong = elevation >= strong_thr
    weak = elevation >= weak_thr

    weak_labels, n = ndi_label(weak)
    connected = np.zeros_like(weak, dtype=bool)
    for lab_id in range(1, n + 1):
        comp = weak_labels == lab_id
        if np.any(comp & strong):
            connected |= comp

    if int(closing_radius_px) > 0:
        connected = binary_closing(
            connected,
            footprint=disk(int(closing_radius_px))
        )

    if int(min_component_px) > 1:
        connected = remove_small_objects(
            connected,
            min_size=int(min_component_px)
        )

    return {
        'strong': strong,
        'weak': weak,
        'boundary_band': connected,
        'strong_threshold': strong_thr,
        'weak_threshold': weak_thr,
        'strong_percentile': strong_percentile,
        'weak_percentile': weak_percentile,
        'threshold_source': threshold_source,
    }

def _region_seed_records(mask, seed_type):
    lab, n = ndi_label(mask)
    if n == 0:
        return []

    records = []
    for p in regionprops(lab):
        r, c = np.round(p.centroid).astype(int)
        records.append({
            'r': int(r),
            'c': int(c),
            'seed_type': seed_type,
        })
    return records


def deduplicate_seed_records(
    seed_records,
    min_distance_px,
    z_seed,
    elevation,
    boundary_band,
    core_radius_px=1,
    ring_radius_px=3,
    prominence_tie_fraction=0.05
):
    """가까운 seed를 cluster로 묶고 대표 seed 하나를 선택한다.

    우선순위:
      1) prominence가 큰 seed
      2) prominence 차이가 최고값의 5% 이내이면 local contrast
      3) 그래도 유사하면 boundary band에서 더 먼 seed

    기존처럼 입력 순서상 뒤 seed를 무조건 삭제하지 않는다.
    """
    if len(seed_records) <= 1:
        records = list(seed_records)
    else:
        records = list(seed_records)

    distance_to_boundary = distance_transform_edt(~boundary_band)
    for s in records:
        feat = compute_seed_feature(
            z_seed,
            elevation,
            s['r'],
            s['c'],
            s['seed_type'],
            core_radius_px=core_radius_px,
            ring_radius_px=ring_radius_px
        )
        s.update(feat)
        s['boundary_distance_px'] = float(
            distance_to_boundary[s['r'], s['c']]
        )

    if len(records) <= 1:
        return records

    coords = np.asarray([[s['r'], s['c']] for s in records], dtype=float)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=float(min_distance_px))

    adjacency = [set() for _ in records]
    for i, j in pairs:
        adjacency[i].add(j)
        adjacency[j].add(i)

    visited = set()
    kept = []
    for root in range(len(records)):
        if root in visited:
            continue
        stack = [root]
        cluster = []
        visited.add(root)
        while stack:
            i = stack.pop()
            cluster.append(i)
            for j in adjacency[i]:
                if j not in visited:
                    visited.add(j)
                    stack.append(j)

        members = [records[i] for i in cluster]
        max_prom = max(s['prominence'] for s in members)
        tol = max(abs(max_prom) * float(prominence_tie_fraction), 1e-12)
        finalists = [s for s in members if max_prom - s['prominence'] <= tol]
        winner = max(
            finalists,
            key=lambda s: (
                s['local_contrast'],
                s['boundary_distance_px'],
                s['prominence']
            )
        )
        winner['seed_cluster_size'] = len(members)
        kept.append(winner)

    return kept

def find_seeds(
    z_clean,
    seed_sigma_px,
    min_distance_px,
    elevation,
    candidate_h,
    boundary_band,
    boundary_exclusion_px=1,
    core_radius_px=1,
    ring_radius_px=3
):
    """Local extrema seed를 boundary band 밖의 plateau에서만 유지한다.

    기존:
      local maxima/minima 생성 → gradient percentile로 점 단위 제거

    변경:
      gradient boundary band 전체를 먼저 생성
      → boundary band와 그 주변을 seed 금지 영역으로 설정
      → plateau 내부 extrema만 seed로 사용
    """
    z_seed = gaussian_filter(z_clean, seed_sigma_px)

    candidate_h = max(float(candidate_h), 1e-6)
    print(f'  Candidate h threshold: {candidate_h:.6f} deg')

    max_mask = h_maxima(z_seed, candidate_h)
    min_mask = h_minima(z_seed, candidate_h)

    # boundary band 주변은 seed 금지
    exclusion = boundary_band.copy()
    if int(boundary_exclusion_px) > 0:
        exclusion = binary_dilation(
            exclusion,
            iterations=int(boundary_exclusion_px)
        )

    max_mask &= ~exclusion
    min_mask &= ~exclusion

    records = (
        _region_seed_records(max_mask, 'max')
        + _region_seed_records(min_mask, 'min')
    )

    records = deduplicate_seed_records(
        records,
        min_distance_px,
        z_seed=z_seed,
        elevation=elevation,
        boundary_band=boundary_band,
        core_radius_px=core_radius_px,
        ring_radius_px=ring_radius_px
    )

    markers = np.zeros_like(z_clean, dtype=int)
    final_records = []

    for i, s in enumerate(records, start=1):
        r, c = s['r'], s['c']
        markers[r, c] = i
        final_records.append({
            **s,
            'label': i,
        })

    return markers, final_records, z_seed, exclusion


def segment_grains_watershed(
    z_clean,
    canny_sigma_px,
    seed_sigma_px,
    min_distance_px,
    candidate_h,
    low_threshold=0.10,
    high_threshold=0.30,
    boundary_strong_percentile=99.0,
    boundary_weak_percentile=95.0,
    boundary_strong_threshold=None,
    boundary_weak_threshold=None,
    boundary_closing_radius_px=1,
    boundary_min_component_px=2,
    boundary_exclusion_px=1,
    core_radius_px=1,
    ring_radius_px=3
):
    """Gradient 후보 → 연결 boundary band → plateau seed → watershed."""
    z_smooth = gaussian_filter(
        z_clean,
        canny_sigma_px
    )
    elevation = sobel(z_smooth)

    boundary_info = build_gradient_boundary_band(
        elevation,
        strong_percentile=boundary_strong_percentile,
        weak_percentile=boundary_weak_percentile,
        strong_threshold=boundary_strong_threshold,
        weak_threshold=boundary_weak_threshold,
        closing_radius_px=boundary_closing_radius_px,
        min_component_px=boundary_min_component_px
    )

    markers, seed_records, z_seed, seed_exclusion = find_seeds(
        z_clean,
        seed_sigma_px,
        min_distance_px,
        elevation,
        candidate_h=candidate_h,
        boundary_band=boundary_info['boundary_band'],
        boundary_exclusion_px=boundary_exclusion_px,
        core_radius_px=core_radius_px,
        ring_radius_px=ring_radius_px
    )

    print(
        f'  Boundary band: strong={boundary_info["strong_threshold"]:.6f}, '
        f'weak={boundary_info["weak_threshold"]:.6f}, '
        f'pixels={boundary_info["boundary_band"].sum()} '
        f'({boundary_info["boundary_band"].mean()*100:.2f}%)'
    )
    print(
        f'  Seeds placed in plateau regions: '
        f'{len(seed_records)}'
    )

    labs = watershed(
        elevation,
        markers
    )

    edges_qc = detect_boundaries_qc(
        z_smooth,
        canny_sigma_px,
        low_threshold=low_threshold,
        high_threshold=high_threshold
    )

    return (
        labs,
        edges_qc,
        markers,
        seed_records,
        z_seed,
        elevation,
        boundary_info,
        seed_exclusion,
    )


# ══════════════════════════════════════════════════════════════
#  Wafer baseline calibration
# ══════════════════════════════════════════════════════════════

def collect_wafer_seed_features(
    z_clean,
    canny_sigma_px,
    seed_sigma_px,
    seed_spacing_px,
    initial_h,
    core_radius_px,
    ring_radius_px,
    boundary_strong_percentile=99.0,
    boundary_weak_percentile=95.0,
    boundary_strong_threshold=None,
    boundary_weak_threshold=None,
    boundary_closing_radius_px=1,
    boundary_min_component_px=2,
    boundary_exclusion_px=1
):
    """웨이퍼에서 느슨한 seed 후보를 만든 뒤 각 후보의 3특성을 수집한다."""
    labs, edges_qc, markers, seed_records, z_seed, elevation, boundary_info, seed_exclusion = (
        segment_grains_watershed(
            z_clean,
            canny_sigma_px,
            seed_sigma_px,
            seed_spacing_px,
            candidate_h=initial_h,
            boundary_strong_percentile=boundary_strong_percentile,
            boundary_weak_percentile=boundary_weak_percentile,
            boundary_strong_threshold=boundary_strong_threshold,
            boundary_weak_threshold=boundary_weak_threshold,
            boundary_closing_radius_px=boundary_closing_radius_px,
            boundary_min_component_px=boundary_min_component_px,
            boundary_exclusion_px=boundary_exclusion_px,
            core_radius_px=core_radius_px,
            ring_radius_px=ring_radius_px
        )
    )

    labs = clear_border(labs)

    features = []
    valid_records = []

    valid_labels = set(np.unique(labs[labs > 0]))

    for s in seed_records:
        if s['label'] not in valid_labels:
            continue

        feat = compute_seed_feature(
            z_seed,
            elevation,
            s['r'],
            s['c'],
            s['seed_type'],
            core_radius_px=core_radius_px,
            ring_radius_px=ring_radius_px
        )
        record = {**s, **feat}
        features.append(record)
        valid_records.append(record)

    return {
        'labs': labs,
        'edges_qc': edges_qc,
        'markers': markers,
        'seed_records': valid_records,
        'z_seed': z_seed,
        'elevation': elevation,
        'boundary_info': boundary_info,
        'seed_exclusion': seed_exclusion,
        'features': features,
    }


def apply_three_feature_posthoc_filter(
    labs,
    markers,
    seed_records,
    z_seed,
    gradient_map,
    thresholds,
    vote_required=2,
    core_radius_px=1,
    ring_radius_px=3
):
    """모든 seed로 watershed 후, seed 주변 3특성을 이용해 grain 전체를 삭제한다.

    통과 조건:
      local_gradient > wafer gradient threshold
      local_contrast > wafer contrast threshold
      prominence > wafer prominence threshold

    vote_required=2이면 3개 중 2개 이상 초과해야 유지한다.
    """
    labs_out = labs.copy()
    markers_out = markers.copy()

    valid_labels = set(np.unique(labs_out[labs_out > 0]))

    kept_records = []
    removed_records = []

    for s in seed_records:
        label_id = int(s['label'])

        if label_id not in valid_labels:
            markers_out[markers_out == label_id] = 0
            continue

        feat = compute_seed_feature(
            z_seed,
            gradient_map,
            s['r'],
            s['c'],
            s['seed_type'],
            core_radius_px=core_radius_px,
            ring_radius_px=ring_radius_px
        )

        passes = {
            'gradient_pass':
                feat['local_gradient'] > thresholds['gradient_threshold'],
            'contrast_pass':
                feat['local_contrast'] > thresholds['contrast_threshold'],
            'prominence_pass':
                feat['prominence'] > thresholds['prominence_threshold'],
        }

        vote_count = int(sum(bool(v) for v in passes.values()))

        record = {
            **s,
            **feat,
            **passes,
            'vote_count': vote_count,
            'keep': vote_count >= int(vote_required),
        }

        if record['keep']:
            kept_records.append(record)
        else:
            labs_out[labs_out == label_id] = 0
            markers_out[markers_out == label_id] = 0
            removed_records.append(record)

    after_labels = set(np.unique(labs_out[labs_out > 0]))

    info = {
        'before_count': len(valid_labels),
        'after_count': len(after_labels),
        'removed_count': len(removed_records),
        'kept_count': len(after_labels),
        'vote_required': int(vote_required),
        'thresholds': thresholds,
        'kept_records': kept_records,
        'removed_records': removed_records,
    }

    return labs_out, markers_out, info


def calibrate_baseline(
    wafer_phase_fp,
    min_grain_nm=8.0,
    max_grain_nm=50.0,
    canny_sigma_nm=3.0,
    seed_sigma_nm=5.0,
    seed_spacing_nm=8.0,
    feature_percentile=95.0,
    candidate_h_fraction=0.50,
    vote_required=2,
    core_radius_px=1,
    ring_radius_px=3,
    boundary_strong_percentile=99.0,
    boundary_weak_percentile=95.0,
    boundary_patch_rows=4,
    boundary_patch_cols=4,
    boundary_threshold_preset='wafer_patch_P95_P99',
    boundary_closing_radius_px=1,
    boundary_min_component_px=2,
    boundary_exclusion_px=1,
    absent_margin_count=1.5,
    present_margin_count=3.0,
    line_level=True,
    spike_remove=True,
    spike_sigma=6.0,
    save_path=None
):
    """한 장의 grain-free wafer Phase에서 baseline을 생성한다.

    Boundary threshold는 wafer Sobel gradient map을 patch로 나눠
    Tweak=median(patchwise P95), Tstrong=median(patchwise P99) 방식으로
    계산한다. Percentile 값은 CLI에서 변경할 수 있지만 권장 기본값은
    각각 95와 99이다.
    """
    print('=' * 65)
    print('  [Calibration] Grain-free wafer Phase baseline')
    print('=' * 65)

    z_raw, meta = load_afm(wafer_phase_fp)
    px_nm = meta['sx'] / meta['nx']
    z_clean, _ = preprocess(
        z_raw, line_level=line_level, spike_remove=spike_remove,
        spike_sigma=spike_sigma, verbose=True
    )

    canny_sigma_px = max(0.3, canny_sigma_nm / px_nm)
    seed_sigma_px = max(0.5, seed_sigma_nm / px_nm)
    seed_spacing_px = max(1.0, seed_spacing_nm / px_nm)

    wafer_elevation = sobel(gaussian_filter(z_clean, canny_sigma_px))
    boundary_cal = compute_patchwise_gradient_thresholds(
        wafer_elevation,
        patch_rows=boundary_patch_rows,
        patch_cols=boundary_patch_cols,
        weak_percentile=boundary_weak_percentile,
        strong_percentile=boundary_strong_percentile
    )
    boundary_weak_threshold = boundary_cal['weak_threshold']
    boundary_strong_threshold = boundary_cal['strong_threshold']

    print('\n  Wafer patchwise boundary thresholds')
    print(f'    Tweak  = median(patchwise P{boundary_weak_percentile:.1f}) '
          f'= {boundary_weak_threshold:.6f}')
    print(f'    Tstrong = median(patchwise P{boundary_strong_percentile:.1f}) '
          f'= {boundary_strong_threshold:.6f}')
    print(f'    Patch grid = {boundary_patch_rows} x {boundary_patch_cols}')

    z_seed_initial = gaussian_filter(z_clean, seed_sigma_px)
    residual = z_clean - z_seed_initial
    residual_std = float(np.std(residual))
    initial_h = max(residual_std * 0.5, 1e-6)

    wafer_run = collect_wafer_seed_features(
        z_clean, canny_sigma_px, seed_sigma_px, seed_spacing_px,
        initial_h=initial_h,
        core_radius_px=core_radius_px,
        ring_radius_px=ring_radius_px,
        boundary_strong_percentile=boundary_strong_percentile,
        boundary_weak_percentile=boundary_weak_percentile,
        boundary_strong_threshold=boundary_strong_threshold,
        boundary_weak_threshold=boundary_weak_threshold,
        boundary_closing_radius_px=boundary_closing_radius_px,
        boundary_min_component_px=boundary_min_component_px,
        boundary_exclusion_px=boundary_exclusion_px
    )

    features = wafer_run['features']
    gradients = [x['local_gradient'] for x in features]
    contrasts = [x['local_contrast'] for x in features]
    prominences = [x['prominence'] for x in features]

    gradient_threshold = robust_percentile(gradients, feature_percentile)
    contrast_threshold = robust_percentile(contrasts, feature_percentile)
    prominence_threshold = robust_percentile(prominences, feature_percentile)
    candidate_h = max(prominence_threshold * float(candidate_h_fraction), 1e-6)

    thresholds = {
        'gradient_threshold': gradient_threshold,
        'contrast_threshold': contrast_threshold,
        'prominence_threshold': prominence_threshold,
    }

    print('\n  Wafer-derived seed/post-hoc feature thresholds')
    print(f'    local gradient P{feature_percentile:.0f}: {gradient_threshold:.6f}')
    print(f'    local contrast P{feature_percentile:.0f}: {contrast_threshold:.6f} deg')
    print(f'    prominence P{feature_percentile:.0f}: {prominence_threshold:.6f} deg')
    print(f'    candidate h = prominence threshold x {candidate_h_fraction:.2f} '
          f'= {candidate_h:.6f} deg')

    labs, edges_qc, markers, seed_records, z_seed, elevation, boundary_info, seed_exclusion = (
        segment_grains_watershed(
            z_clean, canny_sigma_px, seed_sigma_px, seed_spacing_px,
            candidate_h=candidate_h,
            boundary_strong_percentile=boundary_strong_percentile,
            boundary_weak_percentile=boundary_weak_percentile,
            boundary_strong_threshold=boundary_strong_threshold,
            boundary_weak_threshold=boundary_weak_threshold,
            boundary_closing_radius_px=boundary_closing_radius_px,
            boundary_min_component_px=boundary_min_component_px,
            boundary_exclusion_px=boundary_exclusion_px,
            core_radius_px=core_radius_px,
            ring_radius_px=ring_radius_px
        )
    )
    labs = clear_border(labs)
    labs_filtered, markers_filtered, filter_info = apply_three_feature_posthoc_filter(
        labs, markers, seed_records, z_seed, elevation,
        thresholds=thresholds, vote_required=vote_required,
        core_radius_px=core_radius_px, ring_radius_px=ring_radius_px
    )

    wafer_n_before = filter_info['before_count']
    wafer_n_after = filter_info['after_count']
    count_reference = max(wafer_n_after, 1)
    detected_grain_absent = count_reference * float(absent_margin_count)
    detected_grain_present = count_reference * float(present_margin_count)

    print('\n  Wafer detected-count baseline')
    print(f'    Before post-hoc filter: {wafer_n_before}')
    print(f'    After post-hoc filter:  {wafer_n_after}')
    print(f'    Grain-absent threshold <= {detected_grain_absent:.1f}')
    print(f'    Grain-present threshold >= {detected_grain_present:.1f}')

    baseline = {
        'version': 'v19_wafer_absolute_boundary_preset_fb_consistency',
        'single_wafer_baseline': True,
        'wafer_phase_fp': wafer_phase_fp,
        'wafer_n_detected_before_posthoc': wafer_n_before,
        'wafer_n_detected_after_posthoc': wafer_n_after,
        'boundary_thresholds': {
            **boundary_cal,
            'method': 'median_of_patchwise_percentiles',
            'preset': str(boundary_threshold_preset),
        },
        'phase_feature_thresholds': {
            'percentile': float(feature_percentile),
            'gradient_threshold': gradient_threshold,
            'contrast_threshold': contrast_threshold,
            'prominence_threshold': prominence_threshold,
            'candidate_h_fraction': float(candidate_h_fraction),
            'candidate_h': candidate_h,
            'vote_required': int(vote_required),
            'core_radius_px': int(core_radius_px),
            'ring_radius_px': int(ring_radius_px),
        },
        'grain_absent': {'detected': float(detected_grain_absent)},
        'grain_present': {'detected': float(detected_grain_present)},
        'pipeline_params': {
            'min_grain_nm': float(min_grain_nm),
            'max_grain_nm': float(max_grain_nm),
            'canny_sigma_nm': float(canny_sigma_nm),
            'seed_sigma_nm': float(seed_sigma_nm),
            'seed_spacing_nm': float(seed_spacing_nm),
            'line_level': bool(line_level),
            'spike_remove': bool(spike_remove),
            'spike_sigma': float(spike_sigma),
            'boundary_strong_percentile': float(boundary_strong_percentile),
            'boundary_weak_percentile': float(boundary_weak_percentile),
            'boundary_patch_rows': int(boundary_patch_rows),
            'boundary_patch_cols': int(boundary_patch_cols),
            'boundary_threshold_preset': str(boundary_threshold_preset),
            'boundary_closing_radius_px': int(boundary_closing_radius_px),
            'boundary_min_component_px': int(boundary_min_component_px),
            'boundary_exclusion_px': int(boundary_exclusion_px),
        },
        'margins': {
            'absent_margin_count': float(absent_margin_count),
            'present_margin_count': float(present_margin_count),
        },
        'method_description': (
            'Boundary thresholds are absolute wafer-derived Sobel-gradient '
            'thresholds. Nearby extrema are clustered and represented by the '
            'highest-prominence seed, with local contrast and distance from '
            'the boundary as tie-breakers. Count gate remains exploratory.'
        ),
    }

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(baseline, f, indent=2, ensure_ascii=False)
        print(f'\n  Baseline saved to: {save_path}')
    return baseline

def load_baseline(path):
    with open(path, 'r', encoding='utf-8') as f:
        baseline = json.load(f)

    version = baseline.get('version', '')
    if version not in ('v13_phase_3feature_posthoc', 'v13_1_phase_3feature_posthoc_inverted_phase_qc', 'v13_2_phase_3feature_posthoc_inverted_phase_boundary_only', 'v13_4_phase_gradient_candidates_first', 'v14_gradient_unified_pipeline', 'v15_reference_aligned_boundary_support', 'v17_kept_population_boundary_confidence', 'v18_patchwise_boundary_fb_consistency', 'v19_wafer_absolute_boundary_preset_fb_consistency'):
        print(
            '  [Warning] Loaded baseline was not created by v13. '
            'Old Height-based JSON is incompatible and may fail.'
        )

    return baseline


# ══════════════════════════════════════════════════════════════
#  Grain 통계 + size + boundary visibility
# ══════════════════════════════════════════════════════════════

def compute_grain_stats(labs, z_stats_source, meta):
    px_area_nm2 = (
        (meta['sx'] / meta['nx'])
        * (meta['sy'] / meta['ny'])
    )

    props = regionprops(labs, intensity_image=z_stats_source)
    stats = []

    for p in props:
        area_nm2 = float(p.area * px_area_nm2)
        diameter_nm = float(2.0 * np.sqrt(area_nm2 / np.pi))
        vals = z_stats_source[labs == p.label]

        stats.append({
            'label': int(p.label),
            'area_nm2': area_nm2,
            'diameter_nm': diameter_nm,
            'phase_std': float(np.std(vals)),
            'phase_mean': float(np.mean(vals)),
            'centroid': (
                float(p.centroid[0]),
                float(p.centroid[1])
            ),
        })

    return stats


def filter_by_size(grain_stats, min_nm, max_nm):
    kept = []
    too_small = []
    too_large = []

    for g in grain_stats:
        d = g['diameter_nm']

        if d < min_nm:
            too_small.append(g)

        elif d > max_nm:
            too_large.append(g)

        else:
            kept.append(g)

    return kept, too_small, too_large


def compute_boundary_consistency(
    forward_boundary,
    backward_boundary,
    tolerance_px=1
):
    """Forward/Backward boundary의 거리 허용 대칭 일치도를 계산한다."""
    tolerance_px = max(0, int(tolerance_px))
    if not np.any(forward_boundary) or not np.any(backward_boundary):
        return {
            'forward_supported_ratio': 0.0,
            'backward_supported_ratio': 0.0,
            'symmetric_consistency': 0.0,
        }

    d_to_backward = distance_transform_edt(~backward_boundary)
    d_to_forward = distance_transform_edt(~forward_boundary)
    f_support = float(np.mean(d_to_backward[forward_boundary] <= tolerance_px))
    b_support = float(np.mean(d_to_forward[backward_boundary] <= tolerance_px))
    return {
        'forward_supported_ratio': f_support,
        'backward_supported_ratio': b_support,
        'symmetric_consistency': float(0.5 * (f_support + b_support)),
    }


def add_forward_backward_consistency(
    grain_stats,
    labs_forward,
    backward_boundary,
    tolerance_px=1
):
    """각 Forward grain 외곽이 Backward boundary에서 재현되는 비율 계산."""
    d_to_backward = distance_transform_edt(~backward_boundary)
    for g in grain_stats:
        mask = labs_forward == g['label']
        eroded = binary_erosion(mask, iterations=1)
        boundary = mask & ~eroded
        if np.any(boundary) and np.any(backward_boundary):
            score = float(np.mean(d_to_backward[boundary] <= int(tolerance_px)))
        else:
            score = 0.0
        g['fb_boundary_consistency'] = score
        g['boundary_confidence'] = score
        g['score'] = score
    return grain_stats


def score_and_select(grain_stats, n_select):
    """Forward-Backward boundary consistency 상위 N개를 QC subset으로 표시."""
    if len(grain_stats) == 0:
        return [], []
    ranked = sorted(
        grain_stats,
        key=lambda g: g.get('fb_boundary_consistency', 0.0),
        reverse=True
    )
    for i, g in enumerate(ranked, start=1):
        g['boundary_confidence_rank'] = i
    n_select = min(int(n_select), len(ranked))
    return ranked[:n_select], grain_stats


# ══════════════════════════════════════════════════════════════
#  최종 label boundary + Amplitude 공간 일치도
# ══════════════════════════════════════════════════════════════


def make_gradient_boundary_candidates(
    elevation,
    percentile=90.0,
    min_component_px=1
):
    """Sobel gradient가 높은 전이 구간 전체를 boundary 후보로 만든다.

    예: 20 20 20 | 25 30 35 | 40 40 40
    여기서 25-30-35처럼 Phase가 빠르게 변하는 ramp 구간이
    boundary candidate로 표시된다.

    이 결과는 Watershed 및 post-hoc grain 제거와 무관하다.
    """
    percentile = float(np.clip(percentile, 0.0, 100.0))
    threshold = float(np.percentile(elevation, percentile))
    candidate_mask = elevation >= threshold

    if int(min_component_px) > 1:
        labeled, n = ndi_label(candidate_mask)
        cleaned = np.zeros_like(candidate_mask, dtype=bool)

        for label_id in range(1, n + 1):
            component = labeled == label_id
            if component.sum() >= int(min_component_px):
                cleaned |= component

        candidate_mask = cleaned

    return candidate_mask, threshold


def labels_to_boundary(labs):
    """시각화용으로 남아 있는 각 label의 전체 내부 외곽선을 반환한다.

    label-background(0) 경계도 표시하지만, 이는 새 물리적 boundary를
    생성하는 것이 아니라 최종 segmentation mask의 외곽선을 보여주는 것이다.
    Grain 면적과 size 계산에는 이 표시용 외곽선을 사용하지 않는다.
    """
    boundary = np.zeros_like(labs, dtype=bool)

    for label_id in np.unique(labs):
        if label_id == 0:
            continue

        mask = labs == label_id
        if not np.any(mask):
            continue

        eroded = binary_erosion(mask, iterations=1)
        boundary |= mask & ~eroded

    return boundary


def filter_label_image(labs, valid_labels):
    """지정된 label만 남기고 나머지를 background(0)로 바꾼다."""
    valid_labels = list(valid_labels)
    if len(valid_labels) == 0:
        return np.zeros_like(labs)
    return np.where(np.isin(labs, valid_labels), labs, 0)


def filter_markers(markers, valid_labels):
    """지정된 grain label에 대응하는 seed marker만 남긴다."""
    valid_labels = list(valid_labels)
    if len(valid_labels) == 0:
        return np.zeros_like(markers)
    return np.where(np.isin(markers, valid_labels), markers, 0)


# ══════════════════════════════════════════════════════════════
#  통계 저장
# ══════════════════════════════════════════════════════════════

def summarize_grain_population(
    grain_stats,
    kept,
    selected,
    meta,
    edges_qc=None,
    posthoc_info=None
):
    area_um2 = (meta['sx'] * meta['sy']) / 1e6

    def mean_std(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return np.nan, np.nan
        return float(np.mean(arr)), float(np.std(arr))

    raw_d = [g['diameter_nm'] for g in grain_stats]
    kept_d = [g['diameter_nm'] for g in kept]
    selected_d = [g['diameter_nm'] for g in selected]

    raw_m, raw_s = mean_std(raw_d)
    kept_m, kept_s = mean_std(kept_d)
    selected_m, selected_s = mean_std(selected_d)

    edge_density_pct = (
        float(np.mean(edges_qc) * 100.0)
        if edges_qc is not None
        else np.nan
    )

    kept_support = [g.get('boundary_support_ratio', np.nan) for g in kept]
    selected_support = [g.get('boundary_support_ratio', np.nan) for g in selected]
    kept_support_m, kept_support_s = mean_std(kept_support)
    selected_support_m, selected_support_s = mean_std(selected_support)

    summary = {
        'area_um2': area_um2,
        'raw_count_after_posthoc': len(grain_stats),
        'kept_count_after_size_filter': len(kept),
        'selected_count': len(selected),

        'raw_mean_nm': raw_m,
        'raw_std_nm': raw_s,
        'kept_mean_nm': kept_m,
        'kept_std_nm': kept_s,
        'selected_mean_nm': selected_m,
        'selected_std_nm': selected_s,

        'kept_boundary_support_mean': kept_support_m,
        'kept_boundary_support_std': kept_support_s,
        'selected_boundary_support_mean': selected_support_m,
        'selected_boundary_support_std': selected_support_s,

        'edge_density_pct': edge_density_pct,
    }

    if posthoc_info is not None:
        summary.update({
            'watershed_count_before_posthoc':
                posthoc_info.get('before_count', np.nan),
            'watershed_count_after_posthoc':
                posthoc_info.get('after_count', np.nan),
            'posthoc_removed_count':
                posthoc_info.get('removed_count', np.nan),
            'posthoc_vote_required':
                posthoc_info.get('vote_required', np.nan),
        })


    return summary


def print_population_summary(summary):
    print('  [Population summary]')
    print(f'    Scan area = {summary["area_um2"]:.3f} µm²')

    if 'watershed_count_before_posthoc' in summary:
        print(
            f'    Watershed before post-hoc = '
            f'{summary["watershed_count_before_posthoc"]}'
        )
        print(
            f'    After 3-feature post-hoc = '
            f'{summary["watershed_count_after_posthoc"]}'
        )

    print(
        f'    Raw grains after post-hoc = '
        f'{summary["raw_count_after_posthoc"]}, '
        f'mean={summary["raw_mean_nm"]:.1f}'
        f'±{summary["raw_std_nm"]:.1f} nm'
    )
    print(
        f'    Size-filter kept grains = '
        f'{summary["kept_count_after_size_filter"]}, '
        f'mean={summary["kept_mean_nm"]:.1f}'
        f'±{summary["kept_std_nm"]:.1f} nm'
    )
    print(
        f'    Selected grains = '
        f'{summary["selected_count"]}, '
        f'mean={summary["selected_mean_nm"]:.1f}'
        f'±{summary["selected_std_nm"]:.1f} nm'
    )



def save_summary_csv(
    summary,
    kept,
    selected,
    posthoc_info,
    ref_path,
    out_dir
):
    csv_path = os.path.join(
        out_dir,
        f'{basename_clean(ref_path)}_grain_population_summary.csv'
    )

    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('metric,value\n')
        for k, v in summary.items():
            f.write(f'{k},{v}\n')

        f.write('\nset,N,mean_nm,std_nm\n')
        f.write(
            f'kept,{len(kept)},'
            f'{summary["kept_mean_nm"]},'
            f'{summary["kept_std_nm"]}\n'
        )
        f.write(
            f'selected,{len(selected)},'
            f'{summary["selected_mean_nm"]},'
            f'{summary["selected_std_nm"]}\n'
        )

        f.write(
            '\nlabel,diameter_nm,area_nm2,boundary_support_ratio,'
            'edge_overlap_ratio,boundary_gradient,boundary_gradient_norm,'
            'region_phase_contrast,boundary_confidence,'
            'boundary_confidence_rank,is_top_n\n'
        )
        selected_ids = {g['label'] for g in selected}
        for g in sorted(kept, key=lambda x: x.get('boundary_confidence_rank', 10**9)):
            f.write(
                f'{g["label"]},{g["diameter_nm"]},{g["area_nm2"]},'
                f'{g.get("boundary_support_ratio", 0.0)},'
                f'{g.get("edge_overlap_ratio", 0.0)},'
                f'{g.get("boundary_gradient", 0.0)},'
                f'{g.get("boundary_gradient_norm", 0.0)},'
                f'{g.get("region_phase_contrast", 0.0)},'
                f'{g.get("boundary_confidence", 0.0)},'
                f'{g.get("boundary_confidence_rank", "")},'
                f'{g["label"] in selected_ids}\n'
            )

        f.write(
            '\nlabel,row,col,seed_type,local_gradient,'
            'local_contrast,prominence,gradient_pass,'
            'contrast_pass,prominence_pass,vote_count,keep\n'
        )

        all_records = (
            posthoc_info.get('kept_records', [])
            + posthoc_info.get('removed_records', [])
        )

        for x in all_records:
            f.write(
                f'{x["label"]},{x["r"]},{x["c"]},'
                f'{x["seed_type"]},{x["local_gradient"]},'
                f'{x["local_contrast"]},{x["prominence"]},'
                f'{x["gradient_pass"]},{x["contrast_pass"]},'
                f'{x["prominence_pass"]},{x["vote_count"]},'
                f'{x["keep"]}\n'
            )

    print(f'  Saved: {os.path.basename(csv_path)}')


# ══════════════════════════════════════════════════════════════
#  시각화
# ══════════════════════════════════════════════════════════════

def draw_boundary_seed_overlay(
    phase_ref_img,
    phase_boundary,
    markers,
    meta,
    title
):
    def fn(fig, ax):
        extent = make_extent(meta)

        if phase_ref_img is not None:
            ax.imshow(
                phase_ref_img,
                extent=extent,
                origin='lower',
                aspect='equal',
                interpolation='nearest'
            )
        else:
            ax.imshow(
                np.zeros((meta['ny'], meta['nx'])),
                extent=extent,
                origin='lower',
                cmap='gray',
                aspect='equal'
            )

        boundary_ma = np.ma.masked_where(
            ~phase_boundary,
            phase_boundary
        )
        ax.imshow(
            boundary_ma,
            extent=extent,
            origin='lower',
            cmap='autumn',
            alpha=0.90,
            aspect='equal'
        )

        rs, cs = np.nonzero(markers)
        px_x_um = (
            meta['sx'] / meta['nx']
        ) / 1000.0
        px_y_um = (
            meta['sy'] / meta['ny']
        ) / 1000.0

        ax.scatter(
            cs * px_x_um,
            rs * px_y_um,
            s=6,
            c='cyan',
            edgecolors='black',
            linewidths=0.3,
            label='Kept seed'
        )

        ax.legend(fontsize=7, loc='upper right')
        style_ax(ax, title)

    return fn


def draw_confidence_scatter(
    all_stats,
    selected_labels,
    min_nm,
    max_nm,
    title
):
    def fn(fig, ax):
        if len(all_stats) == 0:
            ax.text(
                0.5,
                0.5,
                'No size-filtered grains',
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=10
            )
            ax.set_xlim(min_nm, max_nm)
            ax.set_ylim(0, 1)

        else:
            diams = np.array(
                [g['diameter_nm'] for g in all_stats],
                dtype=float
            )
            labels_arr = np.array(
                [g['label'] for g in all_stats]
            )
            is_sel = np.isin(
                labels_arr,
                list(selected_labels)
            )
            scores = np.array(
                [
                    g.get('boundary_confidence', np.nan)
                    for g in all_stats
                ],
                dtype=float
            )

            ax.scatter(
                diams[~is_sel],
                scores[~is_sel],
                s=10,
                alpha=0.35,
                color='gray',
                label='Not selected'
            )
            ax.scatter(
                diams[is_sel],
                scores[is_sel],
                s=16,
                alpha=0.90,
                color='#f2a154',
                label='Selected'
            )
            ax.set_ylim(-0.03, 1.03)

        ax.axvline(
            min_nm,
            color='red',
            ls='--',
            lw=1,
            label=f'min={min_nm:.0f} nm'
        )
        ax.axvline(
            max_nm,
            color='red',
            ls='--',
            lw=1,
            label=f'max={max_nm:.0f} nm'
        )

        ax.set_xlabel('Grain diameter (nm)', fontsize=9)
        ax.set_ylabel('Forward–Backward consistency', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
        ax.set_facecolor('#f8f8ff')

    return fn


def draw_all_kept_histogram(
    kept,
    selected,
    min_nm,
    max_nm,
    summary,
    title
):
    def fn(fig, ax):
        kept_d = np.array(
            [g['diameter_nm'] for g in kept],
            dtype=float
        )
        selected_d = np.array(
            [g['diameter_nm'] for g in selected],
            dtype=float
        )

        # 두 집단을 공정하게 비교하려면 동일한 bin 경계를 사용해야 한다.
        # 특히 len(selected) == len(kept)인 경우 두 히스토그램은 정확히 겹쳐야 한다.
        common_bins = np.linspace(min_nm, max_nm, 21)

        if kept_d.size:
            ax.hist(
                kept_d,
                bins=common_bins,
                color='lightgray',
                edgecolor='white',
                alpha=0.90,
                label=f'All kept (N={len(kept)})'
            )

        if selected_d.size:
            ax.hist(
                selected_d,
                bins=common_bins,
                histtype='step',
                color='#f2a154',
                linewidth=2.0,
                label=f'Selected (N={len(selected)})'
            )

        ax.axvline(min_nm, color='red', ls='--', lw=1)
        ax.axvline(max_nm, color='red', ls='--', lw=1)

        ax.set_xlabel('Grain size (nm)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)

        txt = (
            f'Kept: {summary["kept_mean_nm"]:.1f}'
            f'±{summary["kept_std_nm"]:.1f} nm (N={len(kept)})\n'
            f'Selected: {summary["selected_mean_nm"]:.1f}'
            f'±{summary["selected_std_nm"]:.1f} nm (N={len(selected)})'
        )

        ax.text(
            0.97,
            0.95,
            txt,
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=8,
            color='dimgray'
        )
        ax.set_facecolor('#f8f8ff')

    return fn



def draw_inverted_phase_gradient_candidates(
    inverted_phase_img,
    boundary_candidates,
    boundary_threshold,
    boundary_percentile,
    meta,
    title
):
    """Phase 반전 이미지 위에 Sobel gradient boundary 후보 전체를 표시한다."""
    def fn(fig, ax):
        extent = make_extent(meta)

        if inverted_phase_img is None:
            ax.imshow(
                np.zeros((meta['ny'], meta['nx'])),
                extent=extent,
                origin='lower',
                cmap='gray',
                aspect='equal'
            )
            ax.text(
                0.5, 0.5,
                'No inverted Phase image provided',
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=10,
                color='white'
            )
            style_ax(ax, title)
            return

        ax.imshow(
            inverted_phase_img,
            extent=extent,
            origin='lower',
            aspect='equal',
            interpolation='nearest'
        )

        overlay = np.zeros((*boundary_candidates.shape, 4), dtype=float)
        overlay[..., 0] = 1.0
        overlay[..., 3] = boundary_candidates.astype(float) * 0.82

        ax.imshow(
            overlay,
            extent=extent,
            origin='lower',
            aspect='equal',
            interpolation='nearest'
        )

        ax.text(
            0.03,
            0.97,
            (
                f'Sobel gradient >= P{boundary_percentile:.1f}\n'
                f'Threshold: {boundary_threshold:.5f}'
            ),
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=8,
            color='white',
            bbox={
                'facecolor': 'black',
                'alpha': 0.65,
                'edgecolor': 'none',
                'pad': 4
            }
        )

        style_ax(ax, title)

    return fn


# ══════════════════════════════════════════════════════════════
#  Sample analysis
# ══════════════════════════════════════════════════════════════

def analyze(
    phase_fp,
    backward_phase_fp,
    ref_image_fp=None,
    inverted_phase_fp=None,
    n_select=200,
    line_level=True,
    spike_remove=True,
    spike_sigma=6.0,
    canny_sigma_nm=3.0,
    low_threshold=0.10,
    high_threshold=0.30,
    min_grain_nm=8.0,
    max_grain_nm=50.0,
    seed_spacing_nm=None,
    seed_sigma_nm=None,
    baseline=None,
    lock_pipeline_params=True
):
    if baseline is None:
        raise ValueError(
            'baseline is required. Run calibrate_baseline() first.'
        )

    out_dir = make_output_dir(phase_fp)

    detected_grain_absent = float(
        baseline['grain_absent']['detected']
    )
    detected_grain_present = float(
        baseline['grain_present']['detected']
    )

    pp = baseline.get('pipeline_params', {})
    pf = baseline.get('phase_feature_thresholds', {})
    bt = baseline.get('boundary_thresholds', {})
    boundary_weak_threshold = float(bt['weak_threshold'])
    boundary_strong_threshold = float(bt['strong_threshold'])

    if lock_pipeline_params:
        min_grain_nm = float(
            pp.get('min_grain_nm', min_grain_nm)
        )
        max_grain_nm = float(
            pp.get('max_grain_nm', max_grain_nm)
        )
        canny_sigma_nm = float(
            pp.get('canny_sigma_nm', canny_sigma_nm)
        )
        seed_sigma_nm = float(
            pp.get('seed_sigma_nm', seed_sigma_nm or 5.0)
        )
        seed_spacing_nm = float(
            pp.get('seed_spacing_nm', seed_spacing_nm or 8.0)
        )
    else:
        seed_sigma_nm = seed_sigma_nm or 5.0
        seed_spacing_nm = seed_spacing_nm or 8.0
        print(
            '  [Warning] Custom pipeline parameters may reduce '
            'comparability with wafer detected-count baseline.'
        )

    thresholds = {
        'gradient_threshold':
            float(pf['gradient_threshold']),
        'contrast_threshold':
            float(pf['contrast_threshold']),
        'prominence_threshold':
            float(pf['prominence_threshold']),
    }

    candidate_h = float(pf['candidate_h'])
    vote_required = int(pf.get('vote_required', 2))
    core_radius_px = int(pf.get('core_radius_px', 1))
    ring_radius_px = int(pf.get('ring_radius_px', 3))

    boundary_strong_percentile = float(
        pp.get('boundary_strong_percentile', 99.0)
    )
    boundary_weak_percentile = float(
        pp.get('boundary_weak_percentile', 95.0)
    )
    boundary_closing_radius_px = int(
        pp.get('boundary_closing_radius_px', 1)
    )
    boundary_min_component_px = int(
        pp.get('boundary_min_component_px', 2)
    )
    boundary_exclusion_px = int(
        pp.get('boundary_exclusion_px', 1)
    )

    print('=' * 65)
    print('  [Sample] Phase-only grain analysis')
    print('=' * 65)
    print(
        f'  Wafer thresholds: '
        f'gradient={thresholds["gradient_threshold"]:.6f}, '
        f'contrast={thresholds["contrast_threshold"]:.6f}, '
        f'prominence={thresholds["prominence_threshold"]:.6f}'
    )
    print(
        f'  Post-hoc vote: {vote_required}/3, '
        f'candidate_h={candidate_h:.6f}'
    )

    z_raw, meta = load_afm(phase_fp)
    px_nm = meta['sx'] / meta['nx']

    print(
        f'  Loaded Phase data: '
        f'{meta["nx"]}x{meta["ny"]} px, '
        f'{meta["sx"]:.1f}x{meta["sy"]:.1f} nm'
    )
    print(f'  Pixel size: {px_nm:.3f} nm/px')

    for d_nm in (min_grain_nm, max_grain_nm):
        d_px = d_nm / px_nm
        print(
            f'  Grain diameter {d_nm:.0f} nm '
            f'-> {d_px:.2f} px'
        )
        if d_px < 3:
            print(
                f'  [Warning] {d_nm:.0f} nm grain is '
                f'only {d_px:.1f} px across.'
            )

    canny_sigma_px = max(
        0.3,
        canny_sigma_nm / px_nm
    )
    seed_sigma_px = max(
        0.5,
        seed_sigma_nm / px_nm
    )
    seed_spacing_px = max(
        1.0,
        seed_spacing_nm / px_nm
    )

    print(
        f'  Canny sigma: {canny_sigma_nm:.1f} nm '
        f'-> {canny_sigma_px:.2f} px'
    )
    print(
        f'  Seed sigma: {seed_sigma_nm:.1f} nm '
        f'-> {seed_sigma_px:.2f} px'
    )
    print(
        f'  Seed spacing: {seed_spacing_nm:.1f} nm '
        f'-> {seed_spacing_px:.2f} px'
    )

    # 1) Phase 전처리
    z_clean, spike_mask = preprocess(
        z_raw,
        line_level=line_level,
        spike_remove=spike_remove,
        spike_sigma=spike_sigma
    )

    # 2) 모든 seed 후보로 watershed
    labs, edges_qc, markers, seed_records, z_seed, elevation, boundary_info, seed_exclusion = (
        segment_grains_watershed(
            z_clean,
            canny_sigma_px,
            seed_sigma_px,
            seed_spacing_px,
            candidate_h=candidate_h,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            boundary_strong_percentile=boundary_strong_percentile,
            boundary_weak_percentile=boundary_weak_percentile,
            boundary_strong_threshold=boundary_strong_threshold,
            boundary_weak_threshold=boundary_weak_threshold,
            boundary_closing_radius_px=boundary_closing_radius_px,
            boundary_min_component_px=boundary_min_component_px,
            boundary_exclusion_px=boundary_exclusion_px,
            core_radius_px=core_radius_px,
            ring_radius_px=ring_radius_px
        )
    )

    # 이미지 경계에 닿은 영역 제거
    labs = clear_border(labs)

    # 3) 3특성 wafer 기준 post-hoc 제거
    labs, markers, posthoc_info = (
        apply_three_feature_posthoc_filter(
            labs,
            markers,
            seed_records,
            z_seed,
            elevation,
            thresholds=thresholds,
            vote_required=vote_required,
            core_radius_px=core_radius_px,
            ring_radius_px=ring_radius_px
        )
    )

    print(
        f'  [Post-hoc 3-feature filter] '
        f'{posthoc_info["before_count"]} '
        f'-> {posthoc_info["after_count"]} grains '
        f'(removed={posthoc_info["removed_count"]})'
    )

    n_after_posthoc = int(posthoc_info['after_count'])

    # 4) Phase detected-count gate
    if n_after_posthoc <= detected_grain_absent:
        classification = 'grain-absent'
    elif n_after_posthoc >= detected_grain_present:
        classification = 'grain-present'
    else:
        classification = 'ambiguous'

    print(
        f'  [Detected-count gate] count={n_after_posthoc}, '
        f'absent<={detected_grain_absent:.1f}, '
        f'present>={detected_grain_present:.1f}'
    )
    print(f'  [Detected-count gate] classification={classification}')

    # grain-absent여도 결과를 삭제하지 않고 저장한다.
    # 한 장의 wafer baseline만 사용하므로, 분석 결과를 투명하게 남긴다.
    if classification == 'grain-absent':
        print(
            '  [Notice] Sample is wafer-like by detected count. '
            'Results will be saved, but interpret as grain-absent.'
        )
    elif classification == 'ambiguous':
        print(
            '  [Notice] Sample is between absent/present thresholds. '
            'Results will be saved as ambiguous.'
        )

    phase_ref_img = load_reference_image(
        ref_image_fp,
        (meta['ny'], meta['nx'])
    )
    inverted_phase_img = load_reference_image(
        inverted_phase_fp,
        (meta['ny'], meta['nx'])
    )

    # 5) Grain 통계
    grain_stats = compute_grain_stats(
        labs,
        z_clean,
        meta
    )

    # 6) size filter
    kept, too_small, too_large = filter_by_size(
        grain_stats,
        min_grain_nm,
        max_grain_nm
    )

    print(
        f'  Size filter: kept={len(kept)}, '
        f'too_small(<{min_grain_nm:.0f} nm)={len(too_small)}, '
        f'too_large(>{max_grain_nm:.0f} nm)={len(too_large)}'
    )

    # 7) Phase Forward-Backward consistency validation
    z_back_raw, meta_back = load_afm(backward_phase_fp)
    if z_back_raw.shape != z_raw.shape:
        raise ValueError(
            f'Forward/Backward shape mismatch: {z_raw.shape} vs {z_back_raw.shape}'
        )
    z_back_clean, _ = preprocess(
        z_back_raw,
        line_level=line_level,
        spike_remove=spike_remove,
        spike_sigma=spike_sigma,
        verbose=False
    )
    back_labs, _, back_markers, back_seed_records, back_z_seed, back_elevation, back_boundary_info, _ = (
        segment_grains_watershed(
            z_back_clean,
            canny_sigma_px,
            seed_sigma_px,
            seed_spacing_px,
            candidate_h=candidate_h,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            boundary_strong_percentile=boundary_strong_percentile,
            boundary_weak_percentile=boundary_weak_percentile,
            boundary_strong_threshold=boundary_strong_threshold,
            boundary_weak_threshold=boundary_weak_threshold,
            boundary_closing_radius_px=boundary_closing_radius_px,
            boundary_min_component_px=boundary_min_component_px,
            boundary_exclusion_px=boundary_exclusion_px,
            core_radius_px=core_radius_px,
            ring_radius_px=ring_radius_px
        )
    )
    back_labs = clear_border(back_labs)
    back_labs, back_markers, back_posthoc = apply_three_feature_posthoc_filter(
        back_labs, back_markers, back_seed_records, back_z_seed, back_elevation,
        thresholds=thresholds, vote_required=vote_required,
        core_radius_px=core_radius_px, ring_radius_px=ring_radius_px
    )
    back_stats = compute_grain_stats(back_labs, z_back_clean, meta_back)
    back_kept, _, _ = filter_by_size(back_stats, min_grain_nm, max_grain_nm)
    back_kept_labels = {g['label'] for g in back_kept}
    back_labs_kept = filter_label_image(back_labs, back_kept_labels)
    backward_boundary = labels_to_boundary(back_labs_kept)

    fb_tolerance_px = max(1, int(round(2.0 / px_nm)))
    forward_all_boundary = labels_to_boundary(filter_label_image(
        labs, {g['label'] for g in kept}
    ))
    fb_global = compute_boundary_consistency(
        forward_all_boundary,
        backward_boundary,
        tolerance_px=fb_tolerance_px
    )
    kept = add_forward_backward_consistency(
        kept,
        labs,
        backward_boundary,
        tolerance_px=fb_tolerance_px
    )
    selected, all_kept_stats = score_and_select(kept, n_select)
    kept_labels = {g['label'] for g in kept}
    selected_labels = {g['label'] for g in selected}

    print(
        f'  Forward-Backward boundary consistency: '
        f'{fb_global["symmetric_consistency"]:.3f} '
        f'(tolerance={fb_tolerance_px} px)'
    )
    print(
        f'  Selected top {len(selected)} / {len(kept)} '
        f'by Forward-Backward boundary consistency'
    )
    # 8) (b) 패널과 size 통계의 population을 일치시킨다.
    # size filter를 통과한 전체 kept label만 남기고, 각 label의
    # background 경계까지 포함한 완전한 외곽선을 추출한다.
    labs_kept = filter_label_image(labs, kept_labels)
    markers_kept = filter_markers(markers, kept_labels)
    phase_boundary = labels_to_boundary(labs_kept)

    # 계산용 boundary는 1 pixel 그대로 유지한다.
    # 표시용으로만 1 pixel dilation을 적용하여 가독성을 높인다.
    phase_boundary_display = binary_dilation(
        phase_boundary,
        iterations=1
    )

    # 8.5) 전체 파이프라인에서 실제 사용된 gradient boundary band
    boundary_candidates = boundary_info['boundary_band']
    boundary_candidate_threshold = boundary_info['strong_threshold']
    boundary_candidate_percentile = boundary_info['strong_percentile']

    print(
        f'  Wafer-absolute gradient boundary band: '
        f'strong={boundary_info["strong_threshold"]:.6f}, '
        f'weak={boundary_info["weak_threshold"]:.6f}, '
        f'pixels={boundary_candidates.sum()} '
        f'({boundary_candidates.mean()*100:.2f}%)'
    )

    # 9) Summary
    population_summary = summarize_grain_population(
        grain_stats,
        kept,
        selected,
        meta,
        edges_qc=edges_qc,
        posthoc_info=posthoc_info
    )
    population_summary['detected_count_classification'] = classification
    population_summary['fb_forward_supported_ratio'] = fb_global['forward_supported_ratio']
    population_summary['fb_backward_supported_ratio'] = fb_global['backward_supported_ratio']
    population_summary['fb_symmetric_consistency'] = fb_global['symmetric_consistency']
    population_summary['fb_tolerance_px'] = fb_tolerance_px
    population_summary['detected_grain_absent_threshold'] = (
        detected_grain_absent
    )
    population_summary['detected_grain_present_threshold'] = (
        detected_grain_present
    )

    print_population_summary(population_summary)

    save_summary_csv(
        population_summary,
        kept,
        selected,
        posthoc_info,
        phase_fp,
        out_dir
    )

    # 10) 4패널
    panels = [
        (
            'a_inverted_phase_gradient_boundary_candidates',
            draw_inverted_phase_gradient_candidates(
                inverted_phase_img,
                boundary_candidates,
                boundary_candidate_threshold,
                boundary_candidate_percentile,
                meta,
                '(a) Inverted Phase + Unified Gradient Boundary Band'
            )
        ),
        (
            'b_phase_boundary_seed_overlay',
            draw_boundary_seed_overlay(
                phase_ref_img,
                phase_boundary_display,
                markers_kept,
                meta,
                '(b) Size-filtered Grain Outlines + Kept Seeds'
            )
        ),
        (
            'c_boundary_validation_scatter',
            draw_confidence_scatter(
                all_kept_stats,
                selected_labels,
                min_grain_nm,
                max_grain_nm,
                '(c) Forward–Backward Consistency'
            )
        ),
        (
            'd_grain_size_distribution',
            draw_all_kept_histogram(
                kept,
                selected,
                min_grain_nm,
                max_grain_nm,
                population_summary,
                '(d) Grain Size Distribution'
            )
        ),

    ]

    fig = plt.figure(
        figsize=(18, 5.5),
        facecolor='white'
    )
    gs = gridspec.GridSpec(
        1,
        4,
        fig,
        left=0.04,
        right=0.98,
        top=0.86,
        bottom=0.12,
        wspace=0.34
    )

    for i, (_, draw_fn) in enumerate(panels):
        draw_fn(fig, fig.add_subplot(gs[0, i]))

    fig.suptitle(
        f'AFM Phase Grain Analysis — '
        f'{os.path.basename(phase_fp)} '
        f'[{classification}]',
        fontsize=13,
        fontweight='bold'
    )

    save_fig(
        fig,
        phase_fp,
        'grain_boundary_overview',
        out_dir
    )
    plt.close(fig)

    for panel_name, draw_fn in panels:
        save_independent(
            draw_fn,
            phase_fp,
            panel_name,
            out_dir
        )

    return {
        'labs': labs,
        'phase_boundary': phase_boundary,
        'phase_boundary_display': phase_boundary_display,
        'labs_kept': labs_kept,
        'markers_kept': markers_kept,
        'boundary_candidates': boundary_candidates,
        'boundary_candidate_threshold': boundary_candidate_threshold,
        'boundary_info': boundary_info,
        'seed_exclusion': seed_exclusion,
        'edges_qc': edges_qc,
        'all_stats': grain_stats,
        'kept': kept,
        'selected': selected,
        'posthoc_info': posthoc_info,
        'classification': classification,
        'out_dir': out_dir,
        'population_summary': population_summary,
        'backward_boundary': backward_boundary,
        'fb_global': fb_global,
    }


# ══════════════════════════════════════════════════════════════
#  Boundary threshold preset
# ══════════════════════════════════════════════════════════════

def ask_boundary_threshold_preset():
    """웨이퍼 patch 기반 절대 gradient threshold 조합을 선택한다.

    1) 기본 권장:
       Tweak  = median(patchwise P95)
       Tstrong = median(patchwise P99)

    2) 경계 후보를 조금 더 넓게 잡는 대안:
       Tweak  = median(patchwise P90)
       Tstrong = median(patchwise P99)

    두 모드 모두 sample 자체 percentile을 다시 계산하지 않고,
    grain-free wafer에서 계산한 absolute threshold를 sample의
    Forward/Backward 분석에 동일하게 적용한다.
    """
    print(
        '  Wafer patch absolute boundary threshold preset:\n'
        '    [1] P95 / P99  (default, stricter weak boundary)\n'
        '    [2] P90 / P99  (wider weak boundary band)\n'
    )
    choice = input('  Select preset [1]: ').strip()

    if choice == '2':
        weak_percentile = 90.0
        strong_percentile = 99.0
        preset_name = 'wafer_patch_P90_P99'
    else:
        weak_percentile = 95.0
        strong_percentile = 99.0
        preset_name = 'wafer_patch_P95_P99'

    print(
        f'  Selected: Tweak=median(patchwise P{weak_percentile:.0f}), '
        f'Tstrong=median(patchwise P{strong_percentile:.0f})'
    )
    return weak_percentile, strong_percentile, preset_name

# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def ask_file(prompt, required=True):
    while True:
        path = input(prompt).strip().strip('"').strip("'")

        if path == '' and not required:
            return None

        if os.path.isfile(path):
            return path

        print(f'  [Error] File not found: {path}')


def ask_float(prompt, default):
    value = input(prompt).strip()
    return float(value) if value else default


def ask_int(prompt, default):
    value = input(prompt).strip()
    return int(value) if value else default


if __name__ == '__main__':
    print('=' * 70)
    print(
        ' AFM Grain Boundary Analyzer v19 '
        '— wafer absolute P95/P99 or P90/P99 + Phase F/B consistency'
    )
    print('=' * 70)

    # ── Step 0: baseline ────────────────────────────────────────
    print('\n[Step 0] Grain-free wafer Phase baseline calibration')

    baseline_path = input(
        '  Load existing v19/v18 baseline JSON? '
        '(path, Enter to calibrate new):\n  > '
    ).strip()

    baseline = None

    if baseline_path and os.path.isfile(baseline_path):
        baseline = load_baseline(baseline_path)
        print(f'  Loaded baseline from: {baseline_path}')

    else:
        wafer_phase_fp = ask_file(
            '  Wafer (grain-free) Phase text file:\n  > '
        )

        min_grain_nm = ask_float(
            '  Min grain size, nm (default 8):\n  > ',
            8.0
        )
        max_grain_nm = ask_float(
            '  Max grain size, nm (default 50):\n  > ',
            50.0
        )
        canny_sigma_nm = ask_float(
            '  Canny sigma, nm (default 3.0):\n  > ',
            3.0
        )
        seed_sigma_nm = ask_float(
            '  Seed smoothing sigma, nm (default 5.0):\n  > ',
            5.0
        )
        seed_spacing_nm = ask_float(
            '  Seed minimum spacing, nm (default 8.0):\n  > ',
            8.0
        )
        feature_percentile = ask_float(
            '  Wafer feature percentile, % (default 95):\n  > ',
            95.0
        )
        candidate_h_fraction = ask_float(
            '  Candidate h / wafer prominence threshold '
            '(default 0.5):\n  > ',
            0.5
        )
        vote_required = ask_int(
            '  Required feature votes out of 3 '
            '(default 2):\n  > ',
            2
        )

        (
            boundary_weak_percentile,
            boundary_strong_percentile,
            boundary_threshold_preset
        ) = ask_boundary_threshold_preset()

        boundary_patch_rows = ask_int(
            '  Number of wafer patch rows (default 4):\n  > ',
            4
        )
        boundary_patch_cols = ask_int(
            '  Number of wafer patch columns (default 4):\n  > ',
            4
        )

        boundary_closing_radius_px = ask_int(

            '  Boundary gap-closing radius, px '
            '(default 1):\n  > ',
            1
        )
        boundary_min_component_px = ask_int(
            '  Minimum boundary component size, px '
            '(default 2):\n  > ',
            2
        )
        boundary_exclusion_px = ask_int(
            '  Seed exclusion distance from boundary, px '
            '(default 1):\n  > ',
            1
        )

        print('  Baseline preprocessing:')
        ll = input(
            '    Line leveling? (Y/n): '
        ).strip().lower()
        sp = input(
            '    Spike removal? (Y/n): '
        ).strip().lower()

        line_level = ll != 'n'
        spike_remove = sp != 'n'

        save_path = input(
            '  Save baseline JSON path '
            '(Enter to skip saving):\n  > '
        ).strip()

        baseline = calibrate_baseline(
            wafer_phase_fp=wafer_phase_fp,
            min_grain_nm=min_grain_nm,
            max_grain_nm=max_grain_nm,
            canny_sigma_nm=canny_sigma_nm,
            seed_sigma_nm=seed_sigma_nm,
            seed_spacing_nm=seed_spacing_nm,
            feature_percentile=feature_percentile,
            candidate_h_fraction=candidate_h_fraction,
            vote_required=vote_required,
            boundary_strong_percentile=boundary_strong_percentile,
            boundary_weak_percentile=boundary_weak_percentile,
            boundary_patch_rows=boundary_patch_rows,
            boundary_patch_cols=boundary_patch_cols,
            boundary_threshold_preset=boundary_threshold_preset,
            boundary_closing_radius_px=boundary_closing_radius_px,
            boundary_min_component_px=boundary_min_component_px,
            boundary_exclusion_px=boundary_exclusion_px,
            line_level=line_level,
            spike_remove=spike_remove,
            save_path=save_path if save_path else None
        )

    # ── Step 1: samples ─────────────────────────────────────────
    print('\n[Step 1] Analyze samples using the Phase baseline')
    print('  Enter blank Phase file path to finish.')

    while True:
        print('\n' + '-' * 70)

        phase_fp = ask_file(
            'Sample Phase text file '
            '(Enter to finish):\n> ',
            required=False
        )
        if not phase_fp:
            break

        backward_phase_fp = ask_file(
            '  Matching Backward Phase text file (required for validation):\n  > '
        )

        ref_fp = ask_file(
            '  Original Phase image for panel (b) '
            '(Enter to skip):\n  > ',
            required=False
        )

        inverted_phase_fp = ask_file(
            '  Inverted Phase image for panel (a) '
            '(Enter to skip):\n  > ',
            required=False
        )

        print(
            f'  Locked baseline params: '
            f'min={baseline["pipeline_params"]["min_grain_nm"]} nm, '
            f'max={baseline["pipeline_params"]["max_grain_nm"]} nm, '
            f'canny_sigma='
            f'{baseline["pipeline_params"]["canny_sigma_nm"]} nm, '
            f'seed_sigma='
            f'{baseline["pipeline_params"]["seed_sigma_nm"]} nm, '
            f'seed_spacing='
            f'{baseline["pipeline_params"]["seed_spacing_nm"]} nm'
        )

        override = input(
            '  Override pipeline parameters? (y/N): '
        ).strip().lower()

        lock_pipeline_params = True
        min_grain_nm = float(
            baseline['pipeline_params']['min_grain_nm']
        )
        max_grain_nm = float(
            baseline['pipeline_params']['max_grain_nm']
        )
        canny_sigma_nm = float(
            baseline['pipeline_params']['canny_sigma_nm']
        )
        seed_sigma_nm = float(
            baseline['pipeline_params']['seed_sigma_nm']
        )
        seed_spacing_nm = float(
            baseline['pipeline_params']['seed_spacing_nm']
        )

        if override == 'y':
            print(
                '  [Warning] Override reduces direct comparability '
                'with the wafer count baseline.'
            )

            min_grain_nm = ask_float(
                '    Min grain size, nm:\n    > ',
                min_grain_nm
            )
            max_grain_nm = ask_float(
                '    Max grain size, nm:\n    > ',
                max_grain_nm
            )
            canny_sigma_nm = ask_float(
                '    Canny sigma, nm:\n    > ',
                canny_sigma_nm
            )
            seed_sigma_nm = ask_float(
                '    Seed sigma, nm:\n    > ',
                seed_sigma_nm
            )
            seed_spacing_nm = ask_float(
                '    Seed spacing, nm:\n    > ',
                seed_spacing_nm
            )
            lock_pipeline_params = False

        n_select = ask_int(
            '  Number of grains for top-N statistics '
            '(default 200):\n  > ',
            200
        )

        print('  Sample preprocessing:')
        ll = input(
            '    Line leveling? (Y/n): '
        ).strip().lower()
        sp = input(
            '    Spike removal? (Y/n): '
        ).strip().lower()

        line_level = ll != 'n'
        spike_remove = sp != 'n'

        analyze(
            phase_fp=phase_fp,
            backward_phase_fp=backward_phase_fp,
            ref_image_fp=ref_fp,
            inverted_phase_fp=inverted_phase_fp,
            n_select=n_select,
            line_level=line_level,
            spike_remove=spike_remove,
            min_grain_nm=min_grain_nm,
            max_grain_nm=max_grain_nm,
            canny_sigma_nm=canny_sigma_nm,
            seed_sigma_nm=seed_sigma_nm,
            seed_spacing_nm=seed_spacing_nm,
            baseline=baseline,
            lock_pipeline_params=lock_pipeline_params
        )

    print('\nDone!')
