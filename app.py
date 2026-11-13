"""
SafeStep: Smartphone Fall Detection, Activity Recognition and Caregiver Alerts

Project       Daffodil Robo Tech Olympiad
Author        Dip Khastagir
Institution   American International University Bangladesh

Folder layout expected next to this file

    Daffodil Robo Tech Olympiad/
        app.py              this application (single file)
        setup.bat           one time environment setup
        run_app.bat         starts the application
        SafeStep/
            app_bundle/     models, data, results and demo phone files
            SafeStep_bundle.zip   extracted into app_bundle by setup.bat when needed
            data/           optional: features_345.parquet for the full training set

Command line helpers (used by setup.bat and run_app.bat)

    python app.py --check      verifies that the model bundle can be found
    python app.py --prepare    extracts the bundle, builds the training set and verifies the models
"""
from __future__ import annotations

import sys

_LAUNCHED_BY_STREAMLIT = 'streamlit' in sys.modules      # must be read before any other import

import argparse
import io
import json
import os
import re
import tempfile
import time
import urllib.request
import warnings
import wave
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

APP_NAME = 'SafeStep'
APP_VERSION = '3'
AUTHOR = 'Dip Khastagir'
INSTITUTE = 'American International University-Bangladesh'


# ==================================================================================================
# 1  Project Paths
# ==================================================================================================
ROOT = Path(__file__).resolve().parent
PROJECT_DIR = Path(os.environ.get('SAFESTEP_PROJECT', ROOT / 'SafeStep'))
BUNDLE_CANDIDATES = [Path(os.environ['SAFESTEP_BUNDLE'])] if os.environ.get('SAFESTEP_BUNDLE') else []
BUNDLE_CANDIDATES += [PROJECT_DIR / 'app_bundle', PROJECT_DIR / 'bundle', ROOT / 'app_bundle', ROOT / 'bundle']
ZIP_CANDIDATES = [PROJECT_DIR / 'SafeStep_bundle.zip', ROOT / 'SafeStep_bundle.zip']
DEFAULT_BUNDLE = PROJECT_DIR / 'app_bundle'


def locate_bundle() -> Path | None:
    for p in BUNDLE_CANDIDATES:
        if (p / 'models' / 'deploy_config.json').exists() and (p / 'code' / 'safestep_core.py').exists():
            return p
    return None


def locate_bundle_zip() -> Path | None:
    return next((z for z in ZIP_CANDIDATES if z.exists()), None)


BUNDLE = locate_bundle()
core = None                                                     # safestep_core, loaded from the bundle


def load_core(bundle: Path):
    """Imports the exact feature code that was used for training."""
    global core
    code_dir = str(bundle / 'code')
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    import safestep_core as _core
    core = _core
    return _core


def bpath(*parts) -> str:
    return str(BUNDLE.joinpath(*parts))


def has(*parts) -> bool:
    return BUNDLE is not None and BUNDLE.joinpath(*parts).exists()


# ==================================================================================================
# 2  Model and Data Loading (pure functions, cached later by the interface)
# ==================================================================================================
def load_models_raw() -> dict:
    import joblib
    cfg = json.load(open(bpath('models', 'deploy_config.json')))
    m = dict(cfg=cfg, fall=joblib.load(bpath('models', 'fall_model.joblib')),
             activity=joblib.load(bpath('models', 'activity_model.joblib')))
    if has('models', 'extra_config.json'):
        m['extra_cfg'] = json.load(open(bpath('models', 'extra_config.json')))
        for k in ['direction', 'context', 'preimpact']:
            if has('models', f'{k}_model.joblib'):
                m[k] = joblib.load(bpath('models', f'{k}_model.joblib'))
    if has('models', 'explain_model.joblib'):
        m['explain'] = joblib.load(bpath('models', 'explain_model.joblib'))
        m['explain_cfg'] = json.load(open(bpath('models', 'explain_config.json')))
    return m


def alarm_indices(p, thr, k, t_end, refractory=10.0):
    """Alarm rule: k windows in a row above the threshold, then a refractory pause."""
    hit = p >= thr
    if k > 1:
        hit = np.convolve(hit.astype(int), np.ones(k, int), 'full')[:len(hit)] >= k
    out, last = [], -1e9
    for i in np.where(hit)[0]:
        if t_end[i] - last >= refractory:
            out.append(int(i)); last = t_end[i]
    return out


def analyze_raw(M, acc_g, gyro_dps, thr=None, k=None):
    """One recording through the whole system: signals, windows, 345 features, every model and the alarm rule."""
    cfg = M['cfg']
    acc_g, gyro_dps = np.asarray(acc_g, float), np.asarray(gyro_dps, float)
    n = len(acc_g)
    if n < core.WIN:
        raise ValueError(f'Recording Too Short: {n / core.FS:.1f} s. Minimum Length {core.WIN / core.FS:.0f} s.')
    starts = np.arange(0, n - core.WIN + 1, core.STRIDE)
    sig = core.derive_signals(acc_g, gyro_dps)
    S = core.window_signals(sig, starts)
    X = pd.DataFrame(core.extract_features(S), columns=core.FEATURE_NAMES)
    feats = cfg['features']
    p_fall = M['fall'].predict_proba(X[feats].values)[:, 1]
    p_act = M['activity'].predict_proba(X[feats].values)
    classes = cfg['activity_classes']
    t_end = (starts + core.WIN) / core.FS
    thr = cfg['threshold'] if thr is None else thr
    k = cfg['consecutive_windows'] if k is None else k
    alarms = alarm_indices(p_fall, thr, k, t_end, cfg.get('refractory_s', 10.0))
    out = dict(starts=starts, t_end=t_end, X=X, p_fall=p_fall, p_act=p_act, classes=classes,
               act=np.array(classes)[p_act.argmax(1)], alarms=alarms, thr=thr, k=k, sig=sig,
               acc=acc_g, gyro=gyro_dps, n=n)
    ec = M.get('extra_cfg')
    if ec:
        if 'preimpact' in M:
            out['p_pre'] = M['preimpact'].predict_proba(X[ec['features']].values)[:, 1]
            out['pre_thr'] = ec['preimpact']['thr']
        for key in ['direction', 'context']:
            if key in M:
                out[f'p_{key}'] = M[key].predict_proba(X[ec['features']].values)
                out[f'{key}_classes'] = ec[key]['classes']
    gu = sig['gu']; g0 = gu[:core.FS].mean(0); g0 /= np.linalg.norm(g0) + 1e-9
    out['posture'] = np.degrees(np.arccos(np.clip(gu @ g0, -1, 1)))
    lying = np.isin(out['act'], ['Lying after fall', 'Lying']) & (X['fall__still_frac'].values > 0.5)
    ll = np.zeros(len(starts)); after = False; run = 0; aset = set(alarms)
    for i in range(len(starts)):
        if i in aset:
            after, run = True, 0
        run = run + 1 if (after and lying[i]) else 0
        ll[i] = run * core.STRIDE / core.FS
    out['long_lie_s'] = ll
    return out


# ==================================================================================================
# 3  Phone Files and Signal Perturbations
# ==================================================================================================
G_CONST = 9.80665


def _find(cols, keys, axis):
    for c in cols:
        lc = c.lower().replace(' ', '')
        if any(k in lc for k in keys) and (lc.endswith(axis) or f'{axis}(' in lc or f'_{axis}' in lc or f'{axis}_' in lc):
            return c
    return None


def _time_seconds(t):
    t = np.asarray(t, float); t = t - t[0]
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.02
    if dt > 1e5:
        t = t / 1e9
    elif dt > 0.5:
        t = t / 1e3
    return t


def auto_calibrate(acc, gyr):
    """A still phone must read exactly 1 g. The accelerometer is rescaled so that it does."""
    mag = np.linalg.norm(acc, axis=1)
    win = 25
    if len(mag) < 4 * win:
        return acc, 1.0, 0
    s = pd.Series(mag).rolling(win, center=True).std().values
    w = pd.Series(np.linalg.norm(gyr, axis=1)).rolling(win, center=True).mean().values
    still = (s < 0.02) & (w < 15)
    if still.sum() < win:
        return acc, 1.0, int(still.sum())
    gain = 1.0 / np.median(mag[still])
    if not 0.7 < gain < 1.4:
        return acc, 1.0, int(still.sum())
    return acc * gain, float(gain), int(still.sum())


def parse_phone(files, calibrate=True):
    """Reads one file with both sensors or two separate files. Detects columns, time unit and sensor units."""
    info, acc, gyr, t_acc, t_gyr, g_names = [], None, None, None, None, []
    for f in files:
        if hasattr(f, 'getvalue'):
            raw = f.getvalue()
        elif hasattr(f, 'read'):
            raw = f.read()
        else:
            raw = Path(f).read_bytes()
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine='python')
        df.columns = [str(c).strip() for c in df.columns]
        cols = list(df.columns)
        tcol = next((c for c in cols if 'time' in c.lower() or c.lower() in ('t', 't(s)', 'timestamp')), cols[0])
        t = _time_seconds(pd.to_numeric(df[tcol], errors='coerce').ffill().values)
        a = [_find(cols, ['gf'], ax) for ax in 'xyz']
        if None in a:
            a = [_find(cols, ['acc'], ax) for ax in 'xyz']
        g = [_find(cols, ['gyr', 'rot', 'w'], ax) for ax in 'xyz']
        if None not in a and acc is None:
            acc = df[a].apply(pd.to_numeric, errors='coerce').ffill().fillna(0).values; t_acc = t
            info.append(('Accelerometer Columns', ', '.join(a)))
        if None not in g and gyr is None and not (set(g) & set(a)):
            gyr = df[g].apply(pd.to_numeric, errors='coerce').ffill().fillna(0).values; t_gyr = t; g_names = g
            info.append(('Gyroscope Columns', ', '.join(g)))
    if acc is None:
        raise ValueError('No Accelerometer x, y and z Columns Found in the File.')
    if np.median(np.linalg.norm(acc, axis=1)) > 4:
        acc = acc / G_CONST; info.append(('Accelerometer Unit', 'm/s² Converted to g'))
    else:
        info.append(('Accelerometer Unit', 'g'))
    if gyr is None:
        gyr = np.zeros_like(acc); t_gyr = t_acc
        info.append(('Gyroscope', 'Not Found, Rotation Features Limited'))
    else:
        name = ' '.join(g_names).lower()
        if 'rad' in name or ('deg' not in name and np.abs(gyr).max() < 40):
            gyr = np.degrees(gyr); info.append(('Gyroscope Unit', 'rad/s Converted to °/s'))
        else:
            info.append(('Gyroscope Unit', '°/s'))
    fs_in = 1.0 / np.median(np.diff(t_acc))
    grid, A = core.resample_uniform(t_acc, acc)
    Gy = np.column_stack([np.interp(grid, t_gyr, gyr[:, i]) for i in range(3)])
    gain = 1.0
    if calibrate:
        A, gain, n_still = auto_calibrate(A, Gy)
        info.append(('Auto Calibration', f'Gain {gain:.3f} from {n_still / core.FS:.1f} s of Stillness'
                     if gain != 1.0 else 'Not Required'))
    info.append(('Sampling', f'{fs_in:.0f} Hz Resampled to {core.FS} Hz'))
    return A, Gy, info, fs_in, gain


def rotation(deg, seed=0):
    r = np.random.default_rng(seed); ax = r.normal(size=3); ax /= np.linalg.norm(ax)
    a = np.radians(deg); K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def perturb(acc, gyr, kind, level, seed=0):
    """The same perturbations as notebook 6."""
    r = np.random.default_rng(seed); acc, gyr = acc.copy(), gyr.copy()
    if kind == 'noise':
        acc += r.normal(0, level, acc.shape); gyr += r.normal(0, level * 100, gyr.shape)
    elif kind == 'rate':
        step = max(int(round(core.FS / level)), 1); t = np.arange(len(acc)); ts = t[::step]
        acc = np.column_stack([np.interp(t, ts, acc[::step, i]) for i in range(3)])
        gyr = np.column_stack([np.interp(t, ts, gyr[::step, i]) for i in range(3)])
    elif kind == 'gyro_off':
        gyr[:] = 0
    elif kind == 'rotation':
        R = rotation(level, seed); acc, gyr = acc @ R.T, gyr @ R.T
    elif kind == 'scale':
        acc *= level; gyr *= level
    return acc, gyr


# ==================================================================================================
# 4  Command Line Tools (setup.bat and run_app.bat)
# ==================================================================================================
def _say(text=''):
    print(f'  {text}', flush=True)


def extract_bundle(zip_path: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        members = z.infolist()
        for i, m in enumerate(members, 1):
            target = (dest / m.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f'Unsafe path inside the archive: {m.filename}')
            z.extract(m, dest)
            if i % 10 == 0 or i == len(members):
                print(f'\r  Extracting {i} of {len(members)} files', end='', flush=True)
    print()


def build_training_set(bundle: Path) -> str:
    """Builds data/train_val_windows.parquet so the Model Lab can train on every training window."""
    dst = bundle / 'data' / 'train_val_windows.parquet'
    if dst.exists():
        return 'Already Present'
    feats = PROJECT_DIR / 'data' / 'features_345.parquet'
    meta = PROJECT_DIR / 'data' / 'windows_meta.parquet'
    if not (feats.exists() and meta.exists()):
        return 'Skipped (features_345.parquet Not Found, the Lab Uses a 20,000 Window Sample)'
    F = pd.read_parquet(feats).set_index('win_id')
    wm = pd.read_parquet(meta).set_index('win_id')
    W, FS = 150, 50
    t = wm.t_end_rel_impact.values
    wm['y_fall_train'] = np.select([wm.phase == 'impact', wm.phase == 'adl', (wm.phase == 'pre_fall') & (t < -1.0),
                                    (wm.phase == 'post_fall') & (t > W / FS + 1.0)], [1, 0, 0, 0], -1)
    keep = wm.split.isin(['train', 'val'])
    cols = ['trial_id', 'start', 'phase', 'y_fall_train', 'y_activity', 'split', 'subject', 'age_group', 'code']
    out = pd.concat([wm.loc[keep, cols], F.loc[keep].astype(np.float32)], axis=1).reset_index()
    out.to_parquet(dst, index=False)
    return f'Built with {len(out):,} Windows'


def cli(argv) -> int:
    global BUNDLE
    p = argparse.ArgumentParser(prog='app.py', description=f'{APP_NAME} command line tools')
    p.add_argument('--check', action='store_true', help='verify that the model bundle can be found')
    p.add_argument('--prepare', action='store_true', help='extract the bundle, build the training set, verify the models')
    a = p.parse_args(argv)

    if a.check:
        if BUNDLE is None:
            _say('Model Bundle Not Found.')
            _say(f'Expected Location: {DEFAULT_BUNDLE}')
            return 1
        _say(f'Model Bundle: {BUNDLE}')
        return 0

    if a.prepare:
        _say('Model Bundle')
        if BUNDLE is None:
            z = locate_bundle_zip()
            if z is None:
                _say('   Not Found. Place SafeStep_bundle.zip or the app_bundle folder inside:')
                _say(f'   {PROJECT_DIR}')
                return 1
            _say(f'   Extracting {z.name} into {DEFAULT_BUNDLE}')
            extract_bundle(z, DEFAULT_BUNDLE)
            BUNDLE = locate_bundle()
            if BUNDLE is None:
                _say('   The Archive Lacks models/deploy_config.json or code/safestep_core.py.')
                return 1
        _say(f'   Ready at {BUNDLE}')

        _say('Full Training Set')
        try:
            _say(f'   {build_training_set(BUNDLE)}')
        except MemoryError:
            _say('   Skipped (Not Enough Memory, the Lab Uses a 20,000 Window Sample)')

        _say('Model Verification')
        load_core(BUNDLE)
        t0 = time.time()
        M = load_models_raw()
        demo = sorted((BUNDLE / 'demo_phone').glob('*.csv'))
        if demo:
            acc, gyr, *_ = parse_phone([str(demo[0])])
            res = analyze_raw(M, acc, gyr)
            _say(f'   {demo[0].name}: {len(res["t_end"])} Windows Analysed, {len(res["alarms"])} Alarm(s)')
        _say(f'   Fall Model {M["cfg"]["fall_model"]}, Activity Model {M["cfg"]["activity_model"]}, '
             f'Loaded and Tested in {time.time() - t0:.1f} s')
        return 0

    p.print_help()
    _say('')
    _say('To Open the Application, Double Click run_app.bat')
    return 0


if not _LAUNCHED_BY_STREAMLIT:
    sys.exit(cli(sys.argv[1:]))


# ==================================================================================================
# 5  Streamlit Application
# ==================================================================================================
import plotly.express as px                                     # noqa: E402
import plotly.graph_objects as go                               # noqa: E402
import streamlit as st                                          # noqa: E402

ASSETS = Path(tempfile.gettempdir()) / 'safestep_assets'
ASSETS.mkdir(exist_ok=True)
LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="240" height="56" viewBox="0 0 240 56">
  <rect x="2" y="8" width="40" height="40" rx="11" fill="#E4572E"/>
  <path d="M12 36 h8 v-6 h8 v-6 h8" fill="none" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
  <text x="54" y="37" font-family="Bahnschrift, Segoe UI, Arial, sans-serif" font-size="26" font-weight="700" fill="#FFFFFF">SafeStep</text>
</svg>"""
MARK_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="56" height="56" viewBox="0 0 56 56">
  <rect x="4" y="4" width="48" height="48" rx="13" fill="#E4572E"/>
  <path d="M15 39 h9 v-7 h9 v-7 h9" fill="none" stroke="#FFFFFF" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""
PHYPHOX_XML = """<phyphox version="1.14" locale="en">
    <title>SafeStep</title>
    <category>SafeStep</category>
    <description>Streams the accelerometer (with gravity) and the gyroscope to the SafeStep app. Clip the phone at the waist, enable remote access and press play.</description>
    <data-containers>
        <container size="0" static="false">acc_time</container>
        <container size="0" static="false">accX</container>
        <container size="0" static="false">accY</container>
        <container size="0" static="false">accZ</container>
        <container size="0" static="false">gyr_time</container>
        <container size="0" static="false">gyrX</container>
        <container size="0" static="false">gyrY</container>
        <container size="0" static="false">gyrZ</container>
    </data-containers>
    <input>
        <sensor type="accelerometer" rate="100">
            <output component="x">accX</output>
            <output component="y">accY</output>
            <output component="z">accZ</output>
            <output component="t">acc_time</output>
        </sensor>
        <sensor type="gyroscope" rate="100">
            <output component="x">gyrX</output>
            <output component="y">gyrY</output>
            <output component="z">gyrZ</output>
            <output component="t">gyr_time</output>
        </sensor>
    </input>
    <views>
        <view label="SafeStep">
            <graph label="Acceleration" labelX="t" unitX="s" labelY="a" unitY="m/s²" partialUpdate="true">
                <input axis="x">acc_time</input>
                <input axis="y">accY</input>
            </graph>
            <graph label="Rotation" labelX="t" unitX="s" labelY="w" unitY="rad/s" partialUpdate="true">
                <input axis="x">gyr_time</input>
                <input axis="y">gyrX</input>
            </graph>
        </view>
    </views>
    <export>
        <set name="Accelerometer">
            <data name="Time (s)">acc_time</data>
            <data name="Acceleration x (m/s^2)">accX</data>
            <data name="Acceleration y (m/s^2)">accY</data>
            <data name="Acceleration z (m/s^2)">accZ</data>
        </set>
        <set name="Gyroscope">
            <data name="Time (s)">gyr_time</data>
            <data name="Gyroscope x (rad/s)">gyrX</data>
            <data name="Gyroscope y (rad/s)">gyrY</data>
            <data name="Gyroscope z (rad/s)">gyrZ</data>
        </set>
    </export>
</phyphox>
"""
for _name, _svg in [('logo.svg', LOGO_SVG), ('mark.svg', MARK_SVG)]:
    try:
        (ASSETS / _name).write_text(_svg, encoding='utf-8')
    except OSError:
        pass

st.set_page_config(page_title='SafeStep  Fall Detection', layout='wide', initial_sidebar_state='expanded',
                   page_icon=str(ASSETS / 'mark.svg'))


# --------------------------------------------------------------------------------------------------
# 5.1  Wording Helpers
# --------------------------------------------------------------------------------------------------
_MINOR = {'a', 'an', 'the', 'and', 'or', 'but', 'nor', 'so', 'yet', 'while', 'if', 'than', 'of', 'in', 'on', 'at',
          'to', 'for', 'by', 'that', 'when', 'whether', 'with', 'from', 'per', 'as', 'into', 'onto', 'via', 'vs', 'after', 'before', 'over',
          'under', 'during', 'without', 'within', 'between', 'through', 'is', 'are', 'was', 'were', 'be', 'been',
          'has', 'have', 'had', 'do', 'does', 'did', 'can', 'will'}


def tc(text) -> str:
    """Title Case: capital first letters for key words, lower case for prepositions, connectors and auxiliaries."""
    words = str(text).split(' ')
    out = []
    for i, w in enumerate(words):
        if not w:
            out.append(w); continue
        force = i == 0 or (out and out[-1].endswith((':', '(')))

        def cap(seg, force_):
            m = re.match(r'^([^A-Za-z]*)([A-Za-z].*)?$', seg)
            pre, body = m.group(1), m.group(2) or ''
            if not body:
                return seg
            if any(c.isupper() for c in body[1:]) or any(c.isdigit() for c in body):
                return seg
            if not (force_ or '(' in pre) and body.lower() in _MINOR:
                return pre + body.lower()
            return pre + body[0].upper() + body[1:]
        parts = re.split(r'([/\-])', w)
        out.append(''.join(s if s in ('/', '-') else cap(s, force or j > 0) for j, s in enumerate(parts)))
    return ' '.join(out)


ACT_DISPLAY = {'Lying after fall': 'Lying after a Fall', 'Near-fall': 'Near Fall', 'Sit/Stand': 'Sit and Stand',
               'Starting': 'Warming Up'}


def act_name(a) -> str:
    return ACT_DISPLAY.get(str(a), tc(a))


SIGNAL_WORDS = {'svm': 'Impact Strength', 'dyn': 'Shaking', 'vert': 'Vertical Movement', 'horiz': 'Sideways Movement',
                'gyro': 'Rotation Speed', 'jerk': 'Sudden Change', 'tilt': 'Body Tilt', 'fall': 'Fall Pattern',
                'xcorr': 'Coordination', 'axcorr': 'Axis Coordination', 'ax': 'Axis X', 'ay': 'Axis Y', 'az': 'Axis Z',
                'gx': 'Rotation X', 'gy': 'Rotation Y', 'gz': 'Rotation Z'}
STAT_WORDS = {'mean': 'Average', 'std': 'Variability', 'min': 'Minimum', 'max': 'Peak', 'range': 'Range',
              'p05': '5th Percentile', 'median': 'Median', 'iqr': 'Spread', 'p95': '95th Percentile', 'skew': 'Skewness',
              'kurt': 'Peakedness', 'rms': 'RMS Level', 'mad': 'Mean Deviation', 'mcr': 'Mean Crossing Rate',
              'npeaks': 'Peak Count', 'argmax': 'Peak Timing', 'argmin': 'Low Point Timing', 'domfreq': 'Dominant Frequency',
              'domratio': 'Dominant Power Share', 'centroid': 'Spectral Centroid', 'spread': 'Spectral Spread',
              'entropy': 'Spectral Entropy', 'rolloff': 'Spectral Rolloff', 'd1_mean': 'Rate of Change',
              'd2_mean': 'Acceleration of Change', 'ac_lag05': 'Rhythm at 0.5 s', 'ac_lag1': 'Rhythm at 1 s',
              'ac_peak': 'Rhythm Strength', 'ac_peak_lag': 'Rhythm Period'}
FALL_WORDS = {'tilt_rate_max': 'How Fast the Body Turned', 'vvel_range': 'How Fast the Body Dropped',
              'tilt_end': 'Tilt at the End', 'n_impacts': 'Number of Impacts', 'highg_time': 'Time above 2 g',
              'peak_drop': 'Free Fall to Impact Jump', 'still_frac': 'Stillness', 'post_std': 'Movement after Impact',
              'freefall_time': 'Free Fall Time', 'impact_ratio': 'Impact Ratio', 'prepeak_min': 'Lowest Point before Impact',
              'post_mean': 'Level after Impact', 'post_gyro_mean': 'Rotation after Impact', 'pre_std': 'Movement before Impact',
              'vvel_min': 'Downward Speed', 'vvel_max': 'Upward Speed', 'jerk_peak': 'Peak Jerk', 'still_longest': 'Longest Stillness',
              'peak_svm': 'Peak Impact Strength', 'gyro_peak': 'Peak Rotation Speed', 'freefall_longest': 'Longest Free Fall',
              'gyro_lag': 'Rotation Delay after Impact', 'hv_energy_ratio': 'Sideways to Vertical Energy', 'jerk_peak_pos': 'Peak Jerk Timing',
              'peak_pos': 'Impact Timing', 'post_pre_std_ratio': 'Movement Change across Impact', 't_min_to_max': 'Free Fall to Impact Time',
              'tilt_max': 'Largest Body Tilt'}


def feature_words(f: str) -> str:
    s, _, stat = f.partition('__')
    if s == 'fall':
        return FALL_WORDS.get(stat, tc(stat.replace('_', ' ')))
    if s in ('xcorr', 'axcorr'):
        return f"{SIGNAL_WORDS[s]} of {tc(stat.replace('_', ' and '))}"
    if stat.startswith('band_'):
        lo, hi = stat.split('_')[1:]
        return f'{SIGNAL_WORDS.get(s, s)}, {lo} to {hi} Hz Energy'
    return f'{SIGNAL_WORDS.get(s, s)}, {STAT_WORDS.get(stat, tc(stat.replace("_", " ")))}'


def model_name(m: str) -> str:
    m = str(m)
    for a, b in [('Deep · cnn_bilstm', 'Deep CNN BiLSTM'), ('Deep · cnn', 'Deep CNN'), ('Deep · hybrid', 'Deep Hybrid'),
                 ('Classic · ', ''), ('SVM-RBF', 'SVM RBF'), ('LogReg', 'Logistic Regression'),
                 ('RandomForest', 'Random Forest'), ('ExtraTrees', 'Extra Trees'),
                 ('SafeStep (phone-ready)', 'SafeStep (Phone Ready Features)'),
                 ('device-axis features', 'Device Axis Features'),
                 ('Threshold 3 g (classic)', 'Classic 3 g Threshold')]:
        m = m.replace(a, b)
    m = re.sub(r'Threshold ([\d.]+) g \(tuned\)', r'Tuned \1 g Threshold', m)
    return m


def pct(x) -> str:
    return f'{x * 100:.1f}%'


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(c - h, 0.0), min(c + h, 1.0)


def poisson_upper(events, hours, conf=0.95):
    from scipy.stats import chi2
    return float(chi2.ppf(conf, 2 * (int(events) + 1)) / 2 / max(hours, 1e-9))


# --------------------------------------------------------------------------------------------------
# 5.2  Design System
# --------------------------------------------------------------------------------------------------
NAVY, TEAL, CORAL, AMBER, SKY, PLUM, GREEN, GREY, INK = (
    '#13294B', '#0E7C86', '#E4572E', '#F2A541', '#3A86FF', '#7B2CBF', '#2A9D8F', '#8A94A6', '#1F2A44')
ACT_COLORS_RAW = {'Walking': '#3A86FF', 'Jogging': '#1D4ED8', 'Stairs': '#7B2CBF', 'Sit/Stand': '#0E7C86',
                  'Lying': '#5B8E7D', 'Bending': '#9C6644', 'Car': '#6C757D', 'Near-fall': '#F2A541',
                  'Fall': '#E4572E', 'Lying after fall': '#B5179E'}
ACT_COLORS = {act_name(k): v for k, v in ACT_COLORS_RAW.items()}
SPLIT_COLORS = {'train': '#0E7C86', 'val': '#F2A541', 'test': '#E4572E'}
FONT = "'Plus Jakarta Sans', 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
DISPLAY = "'Outfit', 'Plus Jakarta Sans', 'Segoe UI', Arial, sans-serif"

# Plus Jakarta Sans and Outfit (SIL Open Font License 1.1), latin variable fonts embedded so the app works offline
FONT_JAKARTA_B64 = "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"
FONT_OUTFIT_B64 = "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"
FONT_FACES = ("@font-face{font-family:'Plus Jakarta Sans';font-style:normal;font-display:swap;font-weight:200 800;"
              "src:url(data:font/woff2;base64," + FONT_JAKARTA_B64 + ") format('woff2');}"
              "@font-face{font-family:'Outfit';font-style:normal;font-display:swap;font-weight:100 900;"
              "src:url(data:font/woff2;base64," + FONT_OUTFIT_B64 + ") format('woff2');}")

SIDEBAR_TOP, SIDEBAR_MID, SIDEBAR_BOTTOM, SIDEBAR_CARD = '#0F4C5C', '#11606E', '#147A85', '#0E4855'

CSS = f"""
<style>
  {FONT_FACES}
  html {{ font-size: 17px; }}
  html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp td, .stApp th,
  [data-testid="stMarkdownContainer"], [data-baseweb="select"] div, [data-testid="stSliderThumbValue"],
  [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"], [data-testid="stCaptionContainer"],
  [data-testid="stSidebarNavLink"] span:not([data-testid="stIconMaterial"]), [data-testid="stNavSectionHeader"],
  [data-testid="stFileUploaderDropzoneInstructions"] span, [data-testid="stFileUploaderDropzoneInstructions"] small {{ font-family: {FONT}; }}
  .stApp h1, .stApp h2, .stApp h3 {{ font-family: {DISPLAY}; }}
  .stApp {{ background: #F4F7FB; }}
  .block-container {{ max-width: 1460px; padding: 2.2rem 3.0rem 4rem 3.0rem; }}
  #MainMenu, footer, [data-testid="stDecoration"] {{ visibility: hidden; height: 0; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  [data-stale="true"], .stale-element {{ opacity: 1 !important; transition: none !important; }}
  [data-testid="stAudio"] {{ height: 0 !important; min-height: 0 !important; overflow: hidden; opacity: 0; margin: 0 !important; }}
  p, li, label, .stMarkdown {{ font-size: 1.0rem; }}
  [data-testid="stWidgetLabel"] p {{ font-size: 1.0rem !important; font-weight: 600; color: #2B3650; }}
  [data-testid="stCaptionContainer"] p {{ font-size: 0.95rem !important; color: #5E6A82; }}

  /* ---------- sidebar */
  section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, {SIDEBAR_TOP} 0%, {SIDEBAR_MID} 55%, {SIDEBAR_BOTTOM} 100%); }}
  section[data-testid="stSidebar"] * {{ color: #F2FBFB; }}
  [data-testid="stNavSectionHeader"] {{ color: #A9E4E2 !important; font-size: 0.9rem !important; font-weight: 700; margin-top: 1.1rem; }}
  [data-testid="stSidebarNavLink"] {{ border-radius: 12px; margin: 3px 8px; padding: 0.55rem 0.9rem; transition: background 0.15s ease; }}
  [data-testid="stSidebarNavLink"]:hover {{ background: rgba(255,255,255,0.10); }}
  [data-testid="stSidebarNavLink"][aria-current="page"] {{ background: rgba(255,255,255,0.18); box-shadow: inset 4px 0 0 {CORAL}; }}
  [data-testid="stSidebarNavLink"] span {{ font-size: 1.04rem !important; font-weight: 500; }}

  @property --ss-angle {{ syntax: '<angle>'; initial-value: 0deg; inherits: false; }}
  .ss-side {{ margin-top: 18px; padding: 16px 18px; border-radius: 16px; border: 2px solid transparent;
             background: linear-gradient({SIDEBAR_CARD}, {SIDEBAR_CARD}) padding-box,
                         conic-gradient(from var(--ss-angle), {CORAL}, {AMBER}, #5EEAD4, #3A86FF, {CORAL}) border-box;
             animation: ss-spin 5s linear infinite; font-size: 0.95rem; line-height: 1.85; }}
  @keyframes ss-spin {{ to {{ --ss-angle: 360deg; }} }}
  .ss-si {{ display: flex; align-items: center; gap: 10px; margin-top: 7px; line-height: 1.3; }}
  .ss-si-i {{ width: 28px; height: 28px; border-radius: 9px; border: 1px solid; display: inline-flex; align-items: center;
             justify-content: center; flex: 0 0 28px; }}
  .ss-si-i svg {{ width: 16px; height: 16px; }}
  .ss-si-live {{ background: rgba(46,204,113,0.15); border-color: rgba(46,204,113,0.45); }}
  .ss-si-live .ss-dot {{ margin: 0; }}
  .ss-side-h {{ font-weight: 700; color: #A9E4E2 !important; font-size: 0.92rem; }}
  .ss-dot {{ width: 10px; height: 10px; border-radius: 50%; background: #2ECC71; display: inline-block; margin-right: 8px;
            box-shadow: 0 0 0 0 rgba(46,204,113,0.7); animation: ss-beat 1.8s infinite; }}
  @keyframes ss-beat {{ 0% {{ box-shadow: 0 0 0 0 rgba(46,204,113,0.65); }} 70% {{ box-shadow: 0 0 0 9px rgba(46,204,113,0); }}
                       100% {{ box-shadow: 0 0 0 0 rgba(46,204,113,0); }} }}
  .ss-side-foot {{ margin-top: 18px; font-size: 0.88rem; color: #CBEDEB !important; line-height: 1.6; }}

  /* ---------- headings */
  .ss-hero {{ background: {NAVY}; border-radius: 22px; padding: 42px 48px; color: white; margin-bottom: 1.6rem; }}
  .ss-hero h1 {{ font-family: {DISPLAY}; color: white; font-size: 3.2rem; margin: 0 0 10px 0; font-weight: 700; }}
  .ss-hero p {{ color: #CFDAEC; font-size: 1.2rem; margin: 0; line-height: 1.55; max-width: 62ch; }}
  .ss-trace {{ height: 46px; margin-top: 22px; }}
  .ss-title {{ font-family: {DISPLAY}; font-size: 2.35rem; font-weight: 700; color: {INK}; margin: 0 0 6px 0; }}
  .ss-sub {{ color: #4F5B73; font-size: 1.12rem; margin: 0 0 1.6rem 0; max-width: 95ch; }}
  .ss-sec {{ display: flex; align-items: baseline; gap: 14px; margin: 2.4rem 0 1.0rem 0; }}
  .ss-num {{ font-family: {DISPLAY}; color: {CORAL}; font-weight: 700; font-size: 1.15rem; min-width: 28px; }}
  .ss-sec-t {{ font-family: {DISPLAY}; font-size: 1.42rem; font-weight: 600; color: {INK}; }}
  .ss-line {{ flex: 1; height: 1px; background: #D9E1EC; }}

  /* ---------- cards and banners */
  .ss-card {{ background: white; border: 1px solid #E1E7F0; border-radius: 16px; padding: 20px 22px; height: 100%; }}
  .ss-k {{ color: #4F5B73; font-size: 1.0rem; font-weight: 600; }}
  .ss-v {{ font-family: {DISPLAY}; color: {INK}; font-size: 2.2rem; font-weight: 700; margin-top: 6px; line-height: 1.15; }}
  .ss-s {{ color: #6B768C; font-size: 0.95rem; margin-top: 6px; line-height: 1.5; }}
  .ss-link-card {{ border-top: 4px solid {SIDEBAR_MID}; min-height: 130px; }}
  .ss-link-card .ss-ct {{ font-family: {DISPLAY}; font-size: 1.3rem; font-weight: 600; color: {NAVY}; }}
  .ss-banner {{ border-radius: 18px; padding: 18px 26px; text-align: center; }}
  .ss-banner-t {{ font-family: {DISPLAY}; font-size: 1.75rem; font-weight: 700; }}
  .ss-banner-s {{ font-size: 1.05rem; margin-top: 4px; opacity: 0.95; }}
  .ss-ok {{ background: #E3F4EF; color: #0B6B5D; border: 1px solid #BDE5DA; }}
  .ss-alarm {{ background: {CORAL}; color: white; animation: ss-pulse 1s infinite; }}
  .ss-warn {{ background: #FFF3DD; color: #8A5300; border: 1px solid #F7D9A6; }}
  @keyframes ss-pulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.75; }} 100% {{ opacity: 1; }} }}
  .ss-note {{ background: white; border: 1px solid #E1E7F0; border-left: 5px solid {AMBER}; border-radius: 14px;
              padding: 16px 20px; color: #2F3A52; line-height: 1.7; font-size: 1.02rem; }}
  .ss-chip {{ display: inline-block; padding: 5px 13px; border-radius: 999px; background: #EEF2F8; color: {NAVY};
              font-size: 0.9rem; margin: 0 6px 6px 0; }}

  /* ---------- feature tiles */
  .ss-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(232px, 1fr)); gap: 10px; }}
  .ss-tile {{ background: white; border: 1px solid #E5EAF2; border-radius: 12px; padding: 10px 12px; }}
  .ss-tn {{ font-size: 0.95rem; font-weight: 700; color: #24304A; line-height: 1.25; }}
  .ss-tid {{ font-size: 0.7rem; color: #8A94A6; margin-top: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .ss-tv {{ font-size: 1.08rem; font-weight: 700; color: {INK}; margin-top: 4px; }}
  .ss-bar {{ height: 5px; border-radius: 3px; background: #EEF1F6; margin-top: 6px; overflow: hidden; }}
  .ss-fam {{ font-family: {DISPLAY}; font-size: 1.15rem; font-weight: 600; color: {NAVY}; margin: 22px 0 10px 2px; }}

  /* ---------- features: meaning first and large, technical name small underneath */
  .ss-fm {{ font-size: 1.04rem; font-weight: 700; color: {INK}; line-height: 1.25; }}
  .ss-k .ss-fm {{ font-size: 1.02rem; }}
  .ss-fid {{ font-size: 0.72rem; color: #8A94A6; font-weight: 500; margin-top: 1px; line-height: 1.2; letter-spacing: 0.01em; }}
  .ss-feat {{ display: inline-block; background: #F4F7FB; border: 1px solid #E3E8F1; border-radius: 12px; padding: 7px 12px; margin: 4px 6px 4px 0; }}
  .ss-feat b {{ color: {INK}; font-size: 1.04rem; display: block; line-height: 1.25; }}

  /* ---------- tables */
  @keyframes ss-slide-in {{ from {{ opacity: 0; transform: translateX(-34px); }} to {{ opacity: 1; transform: none; }} }}
  @keyframes ss-rise {{ from {{ opacity: 0; transform: translateY(16px); }} to {{ opacity: 1; transform: none; }} }}
  .ss-table-wrap {{ overflow-x: auto; border-radius: 16px; border: 1px solid #DCE4EF; background: white;
                    box-shadow: 0 10px 30px rgba(19,41,75,0.08); animation: ss-slide-in 0.55s cubic-bezier(.2,.7,.2,1) both; }}
  .ss-table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: 1.0rem; }}
  .ss-table thead th {{ background: linear-gradient(90deg, {SIDEBAR_TOP}, {SIDEBAR_BOTTOM}); color: white; text-align: left;
                        font-weight: 600; padding: 13px 18px; white-space: nowrap; }}
  .ss-table thead th.n {{ text-align: right; }}
  .ss-table tbody td {{ padding: 12px 18px; border-top: 1px solid #EDF1F7; color: #24304A; }}
  .ss-table tbody td.n {{ text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }}
  .ss-table tbody tr {{ animation: ss-rise 0.45s ease-out both; animation-delay: calc(var(--i) * 35ms + 160ms); }}
  .ss-table tbody tr:nth-child(even) td {{ background: #F7FAFD; }}
  .ss-table tbody tr:hover td {{ background: #FFF1EC; }}
  .ss-table td.ss-good {{ color: #0B6B5D; }}
  .ss-table td.ss-bad {{ color: {CORAL}; }}

  /* ---------- expanders and figures */
  [data-testid="stExpander"] details {{ border-radius: 16px !important; border: 1px solid #DCE4EF !important; background: white; }}
  [data-testid="stExpander"] summary {{ padding: 0.85rem 1.1rem !important; }}
  [data-testid="stExpander"] summary p {{ font-size: 1.06rem !important; font-weight: 600; color: {NAVY}; }}
  [data-testid="stExpander"] summary:hover p {{ color: {CORAL}; }}
  .ss-fig-cap {{ color: #4F5B73; font-size: 1.0rem; margin: 0 0 12px 2px; line-height: 1.55; }}

  /* ---------- widgets */
  div[data-testid="stMetric"] {{ background: white; border: 1px solid #E1E7F0; border-radius: 16px; padding: 14px 18px; }}
  [data-testid="stPlotlyChart"] {{ background: white; border: 1px solid #E1E7F0; border-radius: 16px; padding: 10px; }}
  .stButton button, .stDownloadButton button {{ border-radius: 12px; padding: 0.62rem 1.25rem; font-weight: 600; font-size: 1.0rem; }}
  .stButton button p, .stDownloadButton button p {{ font-size: 1.02rem !important; }}
  [data-testid="stSegmentedControl"] button p {{ font-size: 1.0rem !important; font-weight: 600; }}
  .ss-cta button {{ font-size: 1.15rem !important; padding: 0.9rem 1.6rem !important; }}

  /* ---------- home */
  @keyframes ss-float {{ 0%, 100% {{ transform: translateY(0); box-shadow: 0 14px 34px rgba(19,41,75,0.10); }}
                        50% {{ transform: translateY(-10px); box-shadow: 0 26px 46px rgba(19,41,75,0.16); }} }}
  .ss-intro {{ background: white; border: 1px solid #E1E7F0; border-left: 6px solid {TEAL}; border-radius: 20px; padding: 24px 28px;
               animation: ss-float 6s ease-in-out infinite; }}
  .ss-intro-t {{ font-family: {DISPLAY}; font-size: 1.6rem; font-weight: 700; color: {NAVY}; }}
  .ss-intro-s {{ font-size: 1.08rem; color: #3B465E; margin-top: 8px; line-height: 1.65; }}
  .ss-home-hero {{ background: radial-gradient(1200px 400px at 10% -10%, #1E5A7A 0%, {NAVY} 55%);
                  border-radius: 24px; padding: 48px 52px 40px 52px; color: white; margin-bottom: 1.4rem; }}
  .ss-home-hero h1 {{ font-family: {DISPLAY}; font-size: 3.6rem; margin: 0; color: white; font-weight: 700; }}
  .ss-home-hero .ss-tag {{ font-size: 1.3rem; color: #D5E1F2; margin-top: 10px; max-width: 60ch; line-height: 1.5; }}
  .ss-home-hero .ss-by {{ margin-top: 22px; font-size: 1.0rem; color: #A9BBD6; }}
  .ss-flow {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; position: relative; }}
  .ss-step {{ background: white; border: 1px solid #E1E7F0; border-radius: 16px; padding: 20px 18px 18px 18px; position: relative; }}
  .ss-step::before {{ content: ''; position: absolute; left: 0; right: 0; top: 0; height: 5px; background: #E9EEF5; border-radius: 16px 16px 0 0; }}
  .ss-step::after {{ content: ''; position: absolute; left: 0; right: 0; top: 0; height: 5px; background: {CORAL}; opacity: 0; border-radius: 16px 16px 0 0;
                    animation: ss-spot 6s ease-in-out infinite; animation-delay: calc(var(--i) * 1.2s); }}
  @keyframes ss-spot {{ 0%, 24%, 100% {{ opacity: 0; }} 6%, 18% {{ opacity: 1; }} }}
  .ss-step-arrow {{ position: absolute; right: -12px; top: 50%; width: 22px; height: 22px; margin-top: -11px; z-index: 2; }}
  .ss-step-n {{ font-family: {DISPLAY}; color: {CORAL}; font-weight: 700; font-size: 1.05rem; }}
  .ss-step-t {{ font-family: {DISPLAY}; font-size: 1.2rem; font-weight: 600; color: {NAVY}; margin-top: 4px; }}
  .ss-step-s {{ color: #5E6A82; font-size: 0.95rem; margin-top: 6px; line-height: 1.5; }}

  /* ---------- larger widget text */
  [data-testid="stRadio"] p, [data-testid="stCheckbox"] p, [data-testid="stToggle"] p {{ font-size: 1.05rem !important; }}
  .stTextInput input, .stNumberInput input, textarea {{ font-size: 1.08rem !important; }}
  [data-baseweb="select"] > div {{ font-size: 1.05rem !important; min-height: 2.9rem; }}
  [data-baseweb="popover"] li {{ font-size: 1.02rem !important; }}
  [data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {{ font-size: 1.0rem !important; }}
  [data-testid="stMarkdownContainer"] li {{ font-size: 1.05rem; line-height: 1.75; }}
  [data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploaderDropzone"] small {{ font-size: 1.0rem !important; }}
  [data-testid="stButtonGroup"] button, [data-testid="stSegmentedControl"] button {{ min-height: 3.0rem; padding: 0.55rem 1.3rem; }}
  [data-testid="stButtonGroup"] button p {{ font-size: 1.05rem !important; font-weight: 600; }}

  /* ---------- live motion */
  @keyframes ss-zoom {{ from {{ opacity: 0; transform: scale(0.86); }} }}
  @keyframes ss-swing {{ from {{ opacity: 0; transform: perspective(800px) rotateX(-70deg); }} }}
  .ss-anim {{ animation: ss-zoom 0.6s cubic-bezier(.2,.8,.2,1) both; animation-delay: calc(var(--i, 0) * 110ms); }}
  .ss-swing {{ transform-origin: top center; animation: ss-swing 0.8s cubic-bezier(.2,.8,.2,1) both; animation-delay: calc(var(--i, 0) * 120ms); }}
  .stButton button[kind="primary"]:hover {{ box-shadow: 0 8px 22px rgba(228,87,46,0.35); transform: translateY(-1px); }}


  /* ---------- help panel */
  @keyframes ss-unfold {{ from {{ opacity: 0; clip-path: inset(0 0 100% 0); }} to {{ opacity: 1; clip-path: inset(0 0 0 0); }} }}
  .ss-help {{ background: white; border: 1px solid #F3D9A8; border-left: 6px solid {AMBER}; border-radius: 16px; padding: 20px 24px;
              animation: ss-unfold 0.7s ease-out both; }}
  .ss-help-t {{ font-family: {DISPLAY}; font-size: 1.35rem; font-weight: 700; color: #8A5300; }}
  .ss-help-s {{ font-size: 1.05rem; color: #3B465E; margin-top: 4px; }}
  .ss-help ol {{ margin: 12px 0 0 0; padding-left: 1.4rem; }}
  .ss-help li {{ font-size: 1.06rem; line-height: 1.8; color: #24304A; }}
  .ss-help code {{ background: #FFF3DD; color: #8A5300; padding: 1px 7px; border-radius: 6px; font-size: 0.98rem; }}

  @media (prefers-reduced-motion: reduce) {{
    .ss-anim, .ss-swing, .ss-help, .ss-intro {{ animation: none !important; }}
    .ss-alarm, .ss-side, .ss-dot, .ss-step::after {{ animation: none; }}
    .ss-table-wrap, .ss-table tbody tr {{ animation: none; }}
  }}
  @media (max-width: 900px) {{ .ss-flow {{ grid-template-columns: 1fr; }} .block-container {{ padding: 1.4rem 1rem 3rem 1rem; }} }}
</style>
"""


def style():
    st.markdown(CSS, unsafe_allow_html=True)


def header(title, sub=''):
    st.markdown(f'<div class="ss-title">{title}</div><div class="ss-sub">{sub}</div>', unsafe_allow_html=True)


def section(n, title):
    st.markdown(f'<div class="ss-sec"><div class="ss-num">{n:02d}</div><div class="ss-sec-t">{title}</div>'
                f'<div class="ss-line"></div></div>', unsafe_allow_html=True)


def card(label, value, sub='', color=None, anim=None):
    c = f' style="color:{color}"' if color else ''
    a = f' ss-anim" style="--i:{anim}' if anim is not None else ''
    st.markdown(f'<div class="ss-card{a}"><div class="ss-k">{label}</div><div class="ss-v"{c}>{value}</div>'
                f'<div class="ss-s">{sub}</div></div>', unsafe_allow_html=True)


def cards(items, anim=False):
    cols = st.columns(len(items), gap='medium')
    for i, (col, it) in enumerate(zip(cols, items)):
        with col:
            it = tuple(it) + (None,) * (4 - len(it))
            card(*it[:4], anim=i if anim else None)


def banner(text, kind='ok', sub=''):
    s = f'<div class="ss-banner-s">{sub}</div>' if sub else ''
    st.markdown(f'<div class="ss-banner ss-{kind}"><div class="ss-banner-t">{text}</div>{s}</div>', unsafe_allow_html=True)


def note(html_text):
    st.markdown(f'<div class="ss-note">{html_text}</div>', unsafe_allow_html=True)


def gap(h=1.0):
    st.markdown(f'<div style="height:{h}rem"></div>', unsafe_allow_html=True)


def fancy_table(df, title, fmt=None, expanded=False, icon=':material/table_chart:', tone=None, html_cols=()):
    """A decorated table inside an expander. It slides in from the left and its rows rise when opened."""
    import html as _html
    fmt = fmt or {}; tone = tone or {}
    num_cols = {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])}
    head = ''.join(f'<th class="{"n" if c in num_cols else ""}">{_html.escape(str(c))}</th>' for c in df.columns)
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        cells = []
        for c in df.columns:
            v = r[c]; f = fmt.get(c)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                s = ''
            elif callable(f):
                s = f(v)
            elif f:
                s = f.format(v)
            else:
                s = str(v)
            cls = ['n'] if c in num_cols else []
            if c in tone:
                cls.append(tone[c](v))
            cells.append(f'<td class="{" ".join(cls)}">{s if c in html_cols else _html.escape(s)}</td>')
        rows.append(f'<tr style="--i:{min(i, 30)}">{"".join(cells)}</tr>')
    with st.expander(title, expanded=expanded, icon=icon):
        st.markdown(f'<div class="ss-table-wrap"><table class="ss-table"><thead><tr>{head}</tr></thead>'
                    f'<tbody>{"".join(rows)}</tbody></table></div>', unsafe_allow_html=True)


def layout(fig, h=340, title=None, legend=True, legend_pos='top'):
    """One chart style for the whole app. Zoom and drag are disabled, and title, legend and plot never overlap."""
    names = [d.name for d in fig.data if getattr(d, 'name', None) and getattr(d, 'showlegend', None) is not False]
    if legend:
        for d in fig.data:                       # every legend line is drawn with the same weight
            if getattr(d, 'type', '') == 'scatter' and 'lines' in str(getattr(d, 'mode', None) or 'lines') and getattr(d, 'name', None):
                d.line.width = 2.6
    longest = max((len(n) for n in names), default=8)
    if legend and legend_pos == 'bottom':
        leg = dict(orientation='h', x=0, xanchor='left', y=-0.22, yanchor='top'); mt, mb = (64 if title else 24), 110
    else:
        leg = dict(orientation='h', x=0, xanchor='left', y=1.02, yanchor='bottom')
        mt, mb = (88 if (title and legend and names) else 62 if title else 46 if (legend and names) else 20), 24
    leg.update(font=dict(size=14), entrywidth=int(longest * 8.4 + 56), entrywidthmode='pixels', bgcolor='rgba(0,0,0,0)')
    fig.update_layout(template='plotly_white', height=h, font=dict(family=FONT, color=INK, size=14),
                      margin=dict(l=18, r=18, t=mt, b=mb),
                      title=dict(text=title, x=0.01, xanchor='left', y=1, yref='container', yanchor='top', pad=dict(t=16),
                                 font=dict(size=17, family=DISPLAY)) if title else None,
                      paper_bgcolor='white', plot_bgcolor='white', showlegend=legend, legend=leg, dragmode=False,
                      hoverlabel=dict(font=dict(size=14, family=FONT)),
                      colorway=[SKY, CORAL, TEAL, AMBER, PLUM, GREEN, NAVY, GREY])
    fig.update_xaxes(gridcolor='#EEF1F6', zerolinecolor='#E1E6EF', fixedrange=True, tickfont=dict(size=13), title_font=dict(size=14))
    fig.update_yaxes(gridcolor='#EEF1F6', zerolinecolor='#E1E6EF', fixedrange=True, tickfont=dict(size=13), title_font=dict(size=14))
    return fig


PLOT_CONFIG = {'displayModeBar': False, 'scrollZoom': False, 'doubleClick': False, 'showAxisDragHandles': False,
               'showAxisRangeEntryBoxes': False, 'responsive': True}


def chart(fig, key=None):
    st.plotly_chart(fig, width='stretch', key=key, config=PLOT_CONFIG)


def signal_fig(res, t0, t1, mark=None, h=300, title=None):
    fs = core.FS; a, b = int(max(t0, 0) * fs), int(min(t1 * fs, res['n']))
    t = np.arange(a, b) / fs
    fig = go.Figure()
    fig.add_scatter(x=t, y=res['sig']['svm'][a:b], name='Acceleration (g)', line=dict(color=SKY, width=2.2))
    fig.add_scatter(x=t, y=res['sig']['gyro'][a:b] / 100, name='Rotation (100 °/s)', line=dict(color=PLUM, width=1.3))
    for i in res['alarms']:
        te = res['t_end'][i]
        if t0 <= te <= t1:
            fig.add_vline(x=te, line=dict(color=CORAL, width=3, dash='dot'))
    if mark is not None:
        fig.add_vline(x=mark, line=dict(color=AMBER, width=1.5))
    fig.update_xaxes(title='Time (s)')
    return layout(fig, h, title)


def recording_label(r):
    return f"{r['code']}   {tc(r['name'])}   (Person {r['subject']}, Trial {int(r['rep'])})"


def feature_label(f: str) -> str:
    """Plain text form for tooltips: the meaning, then the technical name on the next line."""
    return f'{feature_words(f)}&#10;{f}'


def feature_html(f: str) -> str:
    """Display form: the meaning is the headline, the technical name sits small right underneath."""
    return f'<div class="ss-fm">{feature_words(f)}</div><div class="ss-fid">{f}</div>'


def feature_tick(f: str) -> str:
    """Chart axis form for Plotly."""
    return f'<b>{feature_words(f)}</b><br><span style="font-size:11px;color:#8A94A6">{f}</span>'


# --------------------------------------------------------------------------------------------------
# 5.4  Cached Data Access (resources are shared, never copied, so pages open fast)
# --------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner='Loading Models')
def models():
    return load_models_raw()


@st.cache_resource(show_spinner=False)
def trials():
    return pd.read_parquet(bpath('data', 'trials_meta.parquet')).set_index('trial_id')


@st.cache_resource(show_spinner='Loading Sensor Recordings')
def _signals():
    return np.load(bpath('data', 'signals_50hz_f16.npz'))['signals']


def trial_signal(tid):
    r = trials().loc[tid]
    return _signals()[r.offset:r.offset + r.n_samples].astype(np.float32)


@st.cache_resource(show_spinner=False)
def table(name):
    for sub in ['results', 'data']:
        p = bpath(sub, name)
        if os.path.exists(p):
            return pd.read_parquet(p) if p.endswith('.parquet') else pd.read_csv(p)
    return None


@st.cache_resource(show_spinner=False)
def jload(name):
    for sub in ['results', 'models', 'data']:
        p = bpath(sub, name)
        if os.path.exists(p):
            return json.load(open(p))
    return None


@st.cache_resource(show_spinner=False)
def catalog():
    return pd.DataFrame(core.feature_catalog(), columns=['feature', 'group', 'signal', 'frame'])


@st.cache_resource(show_spinner=False)
def feature_stats():
    S = table('features_sample.parquet'); FN = core.FEATURE_NAMES
    return S[FN].mean(), S[FN].std().replace(0, 1)


FAMILY_NAMES = {'statistical': 'Statistical', 'spectral': 'Spectral', 'temporal': 'Temporal',
                'fall_dynamics': 'Fall Dynamics', 'cross_signal': 'Cross Signal', 'device_axis': 'Device Axis'}


def analyze(acc_g, gyro_dps, thr=None, k=None):
    return analyze_raw(models(), acc_g, gyro_dps, thr, k)


@st.cache_resource(show_spinner='Analysing the Recording', max_entries=256)
def analyze_trial(tid):
    x = trial_signal(tid)
    return analyze(x[:, 0:3], x[:, 3:6])


def window_at(res, t):
    i = np.searchsorted(res['t_end'], t, side='right') - 1
    return int(np.clip(i, 0, len(res['t_end']) - 1))


def contributions(X_rows):
    """Per feature contribution to the fall score (exact TreeSHAP from the LightGBM twin model)."""
    M = models()
    if 'explain' not in M:
        return None
    feats = M['explain_cfg']['features']
    Xv = X_rows[feats].values
    try:
        contrib = M['explain'].predict(Xv, pred_contrib=True)
    except TypeError:
        imp = getattr(M['explain'], 'feature_importances_', np.ones(len(feats)))
        ref = np.array(M['explain_cfg'].get('feature_mean', np.zeros(len(feats))))
        contrib = np.c_[(Xv - ref) * imp / (np.abs(imp).sum() + 1e-9), np.zeros(len(Xv))]
    return pd.DataFrame(contrib[:, :len(feats)], columns=feats), contrib[:, -1]


@st.cache_resource(show_spinner='Preparing the Test People')
def test_index():
    """Groups every test window by recording once (used by the Final Test)."""
    TW = table('test_windows.parquet').sort_values(['trial_id', 'start']).reset_index(drop=True)
    T = trials(); groups = []
    for tid, g in TW.groupby('trial_id', sort=False):
        r = T.loc[tid]
        groups.append((int(tid), g.index.values, (g.start.values + core.WIN) / core.FS,
                       r.peak_idx / core.FS if r.type == 'Fall' else None, r.type, r.code, r.age_group, r.duration_s / 3600))
    return TW, groups


def stream_eval(p, thr, k=1):
    """Streaming evaluation with the same rules as the notebooks."""
    _, groups = test_index()
    caught = n_fall = fa = 0; hours = 0.0; delays = []
    for tid, idx, te, imp, typ, code, age, h in groups:
        al = [te[i] for i in alarm_indices(p[idx], thr, k, te)]
        if typ == 'Fall':
            n_fall += 1; ok = [a for a in al if imp <= a <= imp + core.WIN / core.FS + 0.5]
            caught += bool(ok)
            if ok:
                delays.append(ok[0] - imp)
        else:
            fa += len(al); hours += h
    return dict(sensitivity=caught / max(n_fall, 1), fa_per_hour=fa / max(hours, 1e-9), false_alarms=fa,
                falls=n_fall, caught=caught, adl_hours=hours, median_delay=float(np.median(delays)) if delays else np.nan)


def trial_picker(key, split='test', default_type='Fall'):
    T = trials()
    pool = T[T.split == split] if split else T
    c1, c2, c3 = st.columns([1, 1, 3], gap='medium')
    typ = c1.selectbox('Recording Type', ['Fall', 'ADL'], index=0 if default_type == 'Fall' else 1, key=f'{key}_type',
                       format_func=lambda v: 'Fall' if v == 'Fall' else 'Daily Activity')
    age = c2.selectbox('Person', ['young', 'elderly'], key=f'{key}_age', format_func=str.capitalize)
    p = pool[(pool.type == typ) & (pool.age_group == age)]
    if p.empty:
        p = pool[pool.type == typ]
    if p.empty:
        p = T[T.type == typ]
    opts = p.sort_values(['code', 'subject', 'rep'])
    i = c3.selectbox('Recording', range(len(opts)), format_func=lambda j: recording_label(opts.iloc[j]), key=f'{key}_rec')
    return int(opts.index[i])


# --------------------------------------------------------------------------------------------------
# 5.5  Live Monitoring Dashboard
# --------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def siren():
    fs = 22050; t = np.arange(int(fs * 2.4)) / fs
    f = 700 + 350 * (np.sin(2 * np.pi * 1.6 * t) > 0)
    y = (0.45 * np.sin(2 * np.pi * np.cumsum(f) / fs) * 32767).astype(np.int16)
    b = io.BytesIO()
    with wave.open(b, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(fs); w.writeframes(y.tobytes())
    return b.getvalue()


def play_siren():
    try:
        st.audio(siren(), format='audio/wav', autoplay=True)
    except TypeError:
        st.audio(siren(), format='audio/wav')


def body(tilt, state='safe', h=340):
    tilt = float(np.clip(tilt, 0, 100)); th = np.radians(tilt); frac = np.clip(tilt / 85, 0, 1)
    col = {'safe': TEAL, 'pre': AMBER, 'fall': CORAL, 'longlie': CORAL}[state]
    hip = np.array([0.0, 1.0 - 0.80 * frac])
    up = np.array([np.sin(th), np.cos(th)])
    shoulder, head = hip + 0.75 * up, hip + 1.02 * up
    side = np.array([up[1], -up[0]])
    if frac < 0.5:
        legs = [hip + np.array([-0.18, -1.0 + 0.8 * frac]), hip + np.array([0.18, -1.0 + 0.8 * frac])]
    else:
        legs = [hip - 0.95 * up + 0.12 * side, hip - 0.95 * up - 0.12 * side]
    legs = [np.array([l[0], max(l[1], 0.02)]) for l in legs]
    arms = [shoulder + 0.45 * (-up * 0.3 + side), shoulder + 0.45 * (-up * 0.3 - side)]
    fig = go.Figure()
    fig.add_shape(type='rect', x0=-2.2, x1=2.2, y0=-0.25, y1=0, fillcolor='#E9EDF4', line=dict(width=0))
    for a, b in [(hip, shoulder)] + [(hip, l) for l in legs] + [(shoulder, a_) for a_ in arms]:
        fig.add_scatter(x=[a[0], b[0]], y=[a[1], b[1]], mode='lines', line=dict(color=col, width=12), hoverinfo='skip')
    fig.add_scatter(x=[head[0]], y=[head[1]], mode='markers', marker=dict(size=42, color=col), hoverinfo='skip')
    fig.update_xaxes(range=[-2.2, 2.2], visible=False)
    fig.update_yaxes(range=[-0.3, 2.35], visible=False, scaleanchor='x')
    layout(fig, h, legend=False)
    # the label sits in its own strip above the drawing, so it can never touch the head
    fig.add_annotation(xref='paper', yref='paper', x=0.5, y=1.0, yanchor='bottom', showarrow=False,
                       text=f'<b>{tilt:.0f}°</b> from Upright', font=dict(size=17, color=INK))
    fig.update_layout(margin=dict(t=52, b=10))
    return fig


def phone(alarm_time=None, long_lie=0.0, title='', tilt=0.0):
    now = time.strftime('%H:%M')
    if alarm_time is None:
        body_html = ('<div style="background:#E6F5F1;border-radius:14px;padding:13px;color:#0B6B5D">'
                     '<div style="font-weight:700;font-size:1.02rem">SafeStep</div>'
                     '<div style="font-size:0.95rem;margin-top:4px">All Clear. Monitoring Active.</div></div>')
    else:
        extra = (f'<div style="margin-top:8px;font-weight:700">Not Getting Up for {long_lie:.0f} s</div>'
                 '<div style="margin-top:10px;background:white;color:#E4572E;border-radius:10px;padding:9px;'
                 'text-align:center;font-weight:700">Call Now</div>') if long_lie >= 3 else ''
        body_html = (f'<div style="background:#E4572E;border-radius:14px;padding:13px;color:white;font-size:0.95rem">'
                     f'<div style="font-weight:700;font-size:1.05rem">SafeStep Alert</div>'
                     f'<div style="margin-top:4px">Fall Detected at {alarm_time:.1f} s</div>'
                     f'<div style="opacity:0.92">{title}</div>'
                     f'<div style="opacity:0.92">Body Tilt {tilt:.0f}°</div>{extra}</div>')
    st.markdown(f"""
    <div style="width:250px;margin:0 auto;background:#0F172A;border-radius:36px;padding:14px;box-shadow:0 10px 30px rgba(0,0,0,0.25)">
      <div style="background:#F8FAFC;border-radius:26px;height:380px;padding:14px">
        <div style="text-align:center;font-size:2.0rem;font-weight:300;color:#1F2A44;margin-top:6px">{now}</div>
        <div style="text-align:center;font-size:0.85rem;color:#8A94A6;margin-bottom:16px">Caregiver Phone</div>{body_html}
      </div></div>""", unsafe_allow_html=True)


def state_at(res, t, title=''):
    i = window_at(res, t)
    s = int(min(t * core.FS, res['n'] - 1))
    fired = [res['t_end'][a] for a in res['alarms'] if res['t_end'][a] <= t]
    pre = 'p_pre' in res and res['p_pre'][i] >= res.get('pre_thr', 1.1) and not fired
    a0 = max(0, int((t - 10) * core.FS))
    return dict(t=t, activity=res['act'][i], classes=res['classes'], p_act=res['p_act'][i], p_fall=float(res['p_fall'][i]),
                thr=res['thr'], tilt=float(res['posture'][s]), impact=float(res['X']['svm__max'].iloc[i]),
                long_lie=float(res['long_lie_s'][i]), alarm_time=fired[0] if fired else None, pre=pre,
                sig_t=np.arange(a0, s + 1) / core.FS, sig=res['sig']['svm'][a0:s + 1], gyro=res['sig']['gyro'][a0:s + 1],
                hist_t=res['t_end'][:i + 1], hist_p=res['p_fall'][:i + 1], title=title)


def dashboard(S, key):
    alarm = S['alarm_time'] is not None
    status = 'longlie' if alarm and S['long_lie'] >= 3 else ('fall' if alarm else ('pre' if S['pre'] else 'safe'))
    act = act_name(S['activity'])
    text, sub = {'longlie': ('Fall Detected', f'No Movement for {S["long_lie"]:.0f} s after the Fall. Call for Help.'),
                 'fall': ('Fall Detected', f'Alarm Raised at {S["alarm_time"]:.1f} s' if alarm else ''),
                 'pre': ('Falling Now', 'Pre Impact Warning'),
                 'safe': ('All Clear', f'Current Activity: {act}')}[status]
    banner(text, {'longlie': 'alarm', 'fall': 'alarm', 'pre': 'warn', 'safe': 'ok'}[status], sub)
    gap(1.0)
    c1, c2, c3 = st.columns([1.15, 1.9, 1.0], gap='large')
    with c1:
        chart(body(S['tilt'], status, h=420), key=f'body_{key}')
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=S['sig_t'], y=S['sig'], name='Acceleration (g)', line=dict(color=SKY, width=2.4))
        fig.add_scatter(x=S['sig_t'], y=np.asarray(S['gyro']) / 100, name='Rotation (100 °/s)', line=dict(color=PLUM, width=1.4))
        if alarm and len(S['sig_t']) and S['sig_t'][0] <= S['alarm_time']:
            fig.add_vline(x=S['alarm_time'], line=dict(color=CORAL, width=3, dash='dot'))
        fig.update_yaxes(range=[0, max(4, float(np.max(S['sig'])) + 0.5) if len(S['sig']) else 4])
        chart(layout(fig, 250, 'Motion in the Last 10 Seconds'), key=f'sig_{key}')
        f2 = go.Figure(go.Scatter(x=S['hist_t'], y=S['hist_p'], fill='tozeroy', line=dict(color=CORAL, width=2),
                                  fillcolor='rgba(228,87,46,0.15)', name='Fall Probability'))
        f2.add_hline(y=S['thr'], line=dict(color=NAVY, dash='dash', width=1)); f2.update_yaxes(range=[0, 1.03])
        chart(layout(f2, 200, 'Fall Probability', legend=False), key=f'pf_{key}')
    with c3:
        phone(S['alarm_time'], S['long_lie'], S.get('title', ''), S['tilt'])
    gap(0.8)
    cards([('Activity', act, f"{float(np.max(S['p_act'])) * 100:.0f}% Confidence"),
           ('Fall Probability', f"{S['p_fall'] * 100:.0f}%", f"Alarm at {S['thr'] * 100:.0f}%",
            CORAL if S['p_fall'] >= S['thr'] else INK),
           ('Impact', f"{S['impact']:.1f} g", 'Peak in the Last 3 s'),
           ('Lying Still', f"{S['long_lie']:.1f} s", 'After the Fall', CORAL if S['long_lie'] >= 3 else INK)])
    return status


def play(res, title, ph, speed=1.0, start=None, key='play'):
    """Replays an analysed recording in real time through the dashboard, with the alarm sound."""
    dur = res['n'] / core.FS; t = core.WIN / core.FS if start is None else start; k = 0; rang = False
    snd = st.empty()
    while t <= dur + 1e-6:
        S = state_at(res, t, title)
        with ph.container():
            dashboard(S, f'{key}{k}')
        if S['alarm_time'] is not None and not rang:
            with snd.container():
                play_siren()
            rang = True
        time.sleep(0.5 / speed); t += 0.5; k += 1
    return rang


# --------------------------------------------------------------------------------------------------
# 5.6  Live Sensor Streaming
# --------------------------------------------------------------------------------------------------
class Phyphox:
    """Talks to phyphox remote access, for example http://192.168.0.105:8080, running the SafeStep experiment."""
    BUF_A, BUF_G = ['accX', 'accY', 'accZ'], ['gyrX', 'gyrY', 'gyrZ']

    def __init__(self, url):
        self.url = url.strip().rstrip('/')
        if not self.url.startswith('http'):
            self.url = 'http://' + self.url
        self.ta = self.tg = -1.0

    def _get(self, path, timeout=1.5):
        with urllib.request.urlopen(f'{self.url}{path}', timeout=timeout) as r:
            return json.loads(r.read().decode())

    def connect(self):
        d = self._get('/get?acc_time&gyr_time', timeout=4.0)
        if 'buffer' not in d:
            raise ConnectionError('phyphox answered, but not with sensor data')
        self._get('/control?cmd=clear'); self._get('/control?cmd=start')
        self.ta = self.tg = -1.0
        return True

    def stop(self):
        try:
            self._get('/control?cmd=stop')
        except Exception:
            pass

    def poll(self):
        qa = '&'.join([f'{b}={self.ta}|acc_time' for b in self.BUF_A] + [f'acc_time={self.ta}'])
        qg = '&'.join([f'{b}={self.tg}|gyr_time' for b in self.BUF_G] + [f'gyr_time={self.tg}'])
        d = self._get(f'/get?{qa}&{qg}')['buffer']

        def arr(names, tname):
            t = np.asarray(d.get(tname, {}).get('buffer', []), float)
            cols = [np.asarray(d.get(n, {}).get('buffer', []), float) for n in names]
            n = min([len(t)] + [len(c) for c in cols])
            return t[:n], (np.column_stack([c[:n] for c in cols]) if n else np.zeros((0, 3)))
        ta, A = arr(self.BUF_A, 'acc_time'); tg, Gy = arr(self.BUF_G, 'gyr_time')
        if len(ta):
            self.ta = float(ta[-1])
        if len(tg):
            self.tg = float(tg[-1])
        return ta, A, tg, Gy                         # m/s² and rad/s, like a phone


class Replay:
    """Feeds a SisFall recording in real time, converted to phone units."""
    def __init__(self, tid, speed=1.0):
        x = trial_signal(tid).astype(float); self.t = np.arange(len(x)) / core.FS
        self.A = x[:, 0:3] * G_CONST; self.Gy = np.radians(x[:, 3:6]); self.speed = speed; self.t0 = None; self.i = 0

    def connect(self):
        self.t0 = time.time(); return True

    def stop(self):
        pass

    def done(self):
        return self.i >= len(self.t)

    def poll(self):
        now = (time.time() - self.t0) * self.speed
        j = int(np.searchsorted(self.t, now, side='right'))
        sl = slice(self.i, j); self.i = j
        return self.t[sl], self.A[sl], self.t[sl], self.Gy[sl]


class Engine:
    """Keeps the incoming samples, runs SafeStep every 0.5 s on the last 12 s and applies the alarm rules."""
    def __init__(self, calibrate=True):
        cfg = models()['cfg']
        self.thr, self.k, self.refr = cfg['threshold'], cfg['consecutive_windows'], cfg.get('refractory_s', 10.0)
        self.ta, self.A, self.tg, self.Gy = [], [], [], []
        self.hist = []; self.alarms = []; self.gain = 1.0; self.calibrate = calibrate; self.calibrated = not calibrate
        self.g0 = None; self.long_lie = 0.0; self.last_eval = -1.0; self.ms = 0.0

    def add(self, ta, A, tg, Gy):
        if len(ta):
            self.ta.append(ta); self.A.append(A)
        if len(tg):
            self.tg.append(tg); self.Gy.append(Gy)

    def _arrays(self, last=12.0):
        if not self.ta:
            return None
        ta = np.concatenate(self.ta); A = np.concatenate(self.A) / G_CONST
        if self.tg:
            tg = np.concatenate(self.tg); Gy = np.degrees(np.concatenate(self.Gy))
        else:
            tg, Gy = ta, np.zeros_like(A)
        t1 = ta[-1]; t0 = max(ta[0], t1 - last)
        if t1 - t0 < 0.5:
            return None
        grid = np.arange(t0, t1, 1.0 / core.FS)
        a = np.column_stack([np.interp(grid, ta, A[:, i]) for i in range(3)])
        g = np.column_stack([np.interp(grid, tg, Gy[:, i]) for i in range(3)]) if len(tg) > 1 else np.zeros_like(a)
        return grid, a * self.gain, g

    def duration(self):
        return float(self.ta[-1][-1] - self.ta[0][0]) if self.ta else 0.0

    def step(self):
        """Returns the dashboard state, or None while there is not enough data yet."""
        arr = self._arrays()
        if arr is None:
            return None
        grid, a, g = arr
        if not self.calibrated and self.duration() >= 3.0:
            a_all = self._arrays(last=3.0)
            if a_all is not None:
                _, gain, _ = auto_calibrate(a_all[1] / self.gain, a_all[2])
                self.gain = gain; self.calibrated = True
                g0 = a_all[1].mean(0); self.g0 = g0 / (np.linalg.norm(g0) + 1e-9)
        if self.g0 is None:
            g0 = a[: core.FS].mean(0); self.g0 = g0 / (np.linalg.norm(g0) + 1e-9)
        t_now = float(grid[-1])
        cur = a[-core.FS // 2:].mean(0); cur /= np.linalg.norm(cur) + 1e-9
        tilt = float(np.degrees(np.arccos(np.clip(cur @ self.g0, -1, 1))))
        base = dict(t=t_now, activity='Starting', classes=['Starting'], p_act=np.array([1.0]), p_fall=0.0, thr=self.thr, tilt=tilt,
                    impact=float(np.linalg.norm(a[-core.WIN:], axis=1).max()), long_lie=self.long_lie,
                    alarm_time=self.alarms[0] if self.alarms else None, pre=False,
                    sig_t=grid[-10 * core.FS:], sig=np.linalg.norm(a[-10 * core.FS:], axis=1),
                    gyro=np.linalg.norm(g[-10 * core.FS:], axis=1),
                    hist_t=np.array([h[0] for h in self.hist]), hist_p=np.array([h[1] for h in self.hist]), title='Live Phone')
        if len(a) < core.WIN or t_now - self.last_eval < 0.45:
            return base
        t0 = time.perf_counter(); res = analyze(a, g); self.ms = (time.perf_counter() - t0) * 1000
        self.last_eval = t_now
        p = float(res['p_fall'][-1]); act = res['act'][-1]
        pre = 'p_pre' in res and res['p_pre'][-1] >= res.get('pre_thr', 1.1)
        self.hist.append((t_now, p, act))
        ps = [h[1] for h in self.hist[-self.k:]]
        if len(ps) == self.k and min(ps) >= self.thr and (not self.alarms or t_now - self.alarms[-1] >= self.refr):
            self.alarms.append(t_now); self.long_lie = 0.0
        if self.alarms and act in ('Lying after fall', 'Lying') and res['X']['fall__still_frac'].iloc[-1] > 0.5:
            self.long_lie += 0.5
        elif self.alarms and act not in ('Lying after fall', 'Lying'):
            self.long_lie = 0.0
        base.update(activity=act, classes=res['classes'], p_act=res['p_act'][-1], p_fall=p, long_lie=self.long_lie,
                    alarm_time=self.alarms[-1] if self.alarms else None, pre=pre and not self.alarms,
                    hist_t=np.array([h[0] for h in self.hist]), hist_p=np.array([h[1] for h in self.hist]),
                    impact=float(res['X']['svm__max'].iloc[-1]))
        return base

    def recording(self):
        """Everything received so far, as a phone style CSV table."""
        if not self.ta:
            return None
        ta = np.concatenate(self.ta); A = np.concatenate(self.A)
        tg = np.concatenate(self.tg) if self.tg else ta
        Gy = np.concatenate(self.Gy) if self.Gy else np.zeros_like(A)
        Gi = np.column_stack([np.interp(ta, tg, Gy[:, i]) for i in range(3)])
        return pd.DataFrame({'time_s': ta - ta[0], 'acc_x_ms2': A[:, 0], 'acc_y_ms2': A[:, 1], 'acc_z_ms2': A[:, 2],
                             'gyro_x_rads': Gi[:, 0], 'gyro_y_rads': Gi[:, 1], 'gyro_z_rads': Gi[:, 2]})


# --------------------------------------------------------------------------------------------------
# 5.7  Model Studio (live training in front of the audience)
# --------------------------------------------------------------------------------------------------
STUDIO_MODELS = ['LightGBM', 'Random Forest', 'Extra Trees', 'Neural Network (MLP)', 'Logistic Regression']
STUDIO_TASKS = {'fall': 'Fall Detection', 'activity': 'Activity Recognition'}
STUDIO_STEPS = {'LightGBM': 300, 'Random Forest': 150, 'Extra Trees': 150, 'Neural Network (MLP)': 20, 'Logistic Regression': 1000}
LIVE_VAL_WINDOWS = 6000                      # the live curve scores a fixed validation subset, the final score uses all


@st.cache_resource(show_spinner='Loading Training Data')
def studio_pool():
    full = has('data', 'train_val_windows.parquet')
    df = table('train_val_windows.parquet' if full else 'features_sample.parquet').copy()
    if 'age_group' not in df:
        df['age_group'] = np.where(df.subject.str.startswith('SE'), 'elderly', 'young')
    return df, full


def studio_labels(df, task, full):
    if task == 'fall':
        if full and 'y_fall_train' in df:
            return df['y_fall_train'].values.astype(int)
        a = df['y_activity'].values
        return np.where(a == 'Fall', 1, np.where(a == 'ignore', -1, 0))
    return df['y_activity'].values


def studio_feature_sets():
    M = models(); sel = jload('feature_selection.json') or {}
    out = {f'Phone Ready, Selected ({len(M["cfg"]["features"])})': M['cfg']['features']}
    if sel.get('clean_phone'):
        out[f'Phone Ready, All ({len(sel["clean_phone"])})'] = sel['clean_phone']
    if sel.get('clean'):
        out[f'Every Unique Feature ({len(sel["clean"])})'] = sel['clean']
    return out


@st.cache_resource(show_spinner='Preparing Training Data', max_entries=12)
def studio_matrices(task, feats: tuple, cap: int):
    """Training and validation matrices, built once per task, feature set and size, then reused instantly."""
    df, full = studio_pool()
    y = studio_labels(df, task, full)
    ok = (y != -1) if task == 'fall' else (y != 'ignore')
    tr = np.where(ok & (df.split.values == 'train'))[0]
    va = np.where(ok & (df.split.values == 'val'))[0]
    rng = np.random.default_rng(42)
    if len(tr) > cap:
        if task == 'fall':
            pos = tr[y[tr] == 1]; neg = tr[y[tr] == 0]
            neg = rng.choice(neg, max(cap - len(pos), 1), replace=False); tr = np.sort(np.r_[pos, neg])
        else:
            tr = np.sort(rng.choice(tr, cap, replace=False))
    cols = list(feats)
    Xtr = df[cols].values[tr].astype(np.float32); Xva = df[cols].values[va].astype(np.float32)
    ytr, yva = y[tr], y[va]
    if len(va) > LIVE_VAL_WINDOWS:
        if task == 'fall':
            pos = np.where(yva == 1)[0]; neg = np.where(yva == 0)[0]
            vsub = np.sort(np.r_[pos, rng.choice(neg, max(LIVE_VAL_WINDOWS - len(pos), 1), replace=False)])
        else:
            vsub = np.sort(rng.choice(len(va), LIVE_VAL_WINDOWS, replace=False))
    else:
        vsub = np.arange(len(va))
    return dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, vsub=vsub, full=full)


def studio_score(task, y_true, P, classes):
    from sklearn.metrics import average_precision_score, f1_score
    if task == 'fall':
        return average_precision_score(y_true, P[:, 1])
    return f1_score(y_true, classes[P.argmax(1)], average='macro')


def studio_predict(S, X):
    Xs = S['scaler'].transform(X) if S.get('scaler') is not None else X
    return S['model'].predict_proba(Xs)


def _balanced_epoch(y, rng, task):
    if task == 'fall':
        pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
        k = min(len(neg), 4 * len(pos))
        return rng.permutation(np.r_[rng.choice(pos, k // 2, replace=True), rng.choice(neg, k, replace=False)])
    cls, inv = np.unique(y, return_inverse=True)
    per = int(np.median(np.bincount(inv)))
    return rng.permutation(np.concatenate([rng.choice(np.where(inv == c)[0], per, replace=True) for c in range(len(cls))]))


def studio_train(name, task, feats, steps, cap, on_step):
    """Trains while reporting every step through on_step(step, total, history). Returns the studio state."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.neural_network import MLPClassifier
    D = studio_matrices(task, tuple(feats), int(cap))
    Xtr, ytr, Xva, yva, vsub = D['Xtr'], D['ytr'], D['Xva'], D['yva'], D['vsub']
    Xv_live, yv_live = Xva[vsub], yva[vsub]
    classes = np.array([0, 1]) if task == 'fall' else np.array(sorted(np.unique(ytr)))
    hist = []; t0 = time.time(); rng = np.random.default_rng(42)
    sub = rng.choice(len(Xtr), min(4000, len(Xtr)), replace=False)
    scaler = None

    def log(step, total, P_tr, P_va, loss=np.nan):
        hist.append(dict(step=step, train=studio_score(task, ytr[sub], P_tr, classes),
                         val=studio_score(task, yv_live, P_va, classes), loss=loss, seconds=time.time() - t0))
        on_step(step, total, pd.DataFrame(hist))

    if name == 'LightGBM':
        import lightgbm as lgb
        yt = np.searchsorted(classes, ytr)
        m = lgb.LGBMClassifier(n_estimators=steps, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, class_weight='balanced', n_jobs=-1, random_state=42, verbose=-1)
        every = max(steps // 20, 1)

        def cb(env):
            it = env.iteration + 1
            if it % every == 0 or it == steps:
                def pp(X):
                    p = env.model.predict(X, num_iteration=it)
                    return np.c_[1 - p, p] if p.ndim == 1 else p
                log(it, steps, pp(Xtr[sub]), pp(Xv_live))
        m.fit(Xtr, yt, callbacks=[cb])
    elif name in ('Random Forest', 'Extra Trees'):
        Cls = RandomForestClassifier if name == 'Random Forest' else ExtraTreesClassifier
        m = Cls(n_estimators=0, warm_start=True, min_samples_leaf=2, class_weight='balanced_subsample', n_jobs=-1, random_state=42)
        chunk = max(steps // 10, 5)
        for n in range(chunk, steps + 1, chunk):
            m.set_params(n_estimators=n); m.fit(Xtr, ytr)
            log(n, steps, m.predict_proba(Xtr[sub]), m.predict_proba(Xv_live))
    elif name.startswith('Neural'):
        scaler = StandardScaler().fit(Xtr); Xt, Xv, Xs = scaler.transform(Xtr), scaler.transform(Xv_live), scaler.transform(Xtr[sub])
        m = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-4, learning_rate_init=1e-3, batch_size=256, random_state=42)
        for ep in range(1, steps + 1):
            idx = _balanced_epoch(ytr, rng, task)
            m.partial_fit(Xt[idx], ytr[idx], classes=classes)
            log(ep, steps, m.predict_proba(Xs), m.predict_proba(Xv), loss=m.loss_)
    else:
        scaler = StandardScaler().fit(Xtr); Xt = scaler.transform(Xtr)
        m = LogisticRegression(C=1.0, class_weight='balanced', max_iter=steps)
        m.fit(Xt, ytr)
        log(steps, steps, m.predict_proba(scaler.transform(Xtr[sub])), m.predict_proba(scaler.transform(Xv_live)))

    S = dict(name=name, task=task, feats=list(feats), classes=classes, model=m, scaler=scaler, cap=int(cap),
             history=pd.DataFrame(hist), n_train=len(Xtr), n_val=len(Xva), seconds=time.time() - t0, full=D['full'])
    S['val_full'] = studio_score(task, yva, studio_predict(S, Xva), classes)
    return S


# --------------------------------------------------------------------------------------------------
# 5.8  Background Warm Up (heavy files load while the presenter is still on the first page)
# --------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def warm_up():
    import threading
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
        ctx = get_script_run_ctx()
    except Exception:
        add_script_run_ctx, ctx = None, None

    def job():
        time.sleep(1.5)                          # let the first page finish before loading in the background
        for f in (_signals, feature_stats, results_html, test_index, studio_pool):
            try:
                f()
            except Exception:
                pass
    t = threading.Thread(target=job, daemon=True)
    if add_script_run_ctx and ctx:
        add_script_run_ctx(t, ctx)
    t.start()
    return True


# ==================================================================================================
# 6  Pages
# ==================================================================================================
HOME_LIVE = """<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;font-family:__FONT__;background:transparent}
.wrap{background:#fff;border:1px solid #E1E7F0;border-radius:18px;padding:20px 24px 18px 24px;box-sizing:border-box}
.top{display:flex;justify-content:space-between;align-items:flex-start;gap:16px}
.h{font-family:__DISPLAY__;font-size:22px;font-weight:600;color:#13294B}
.sub{font-size:15px;color:#5E6A82;margin-top:4px}
.pill{font-family:__DISPLAY__;font-weight:700;font-size:18px;padding:9px 20px;border-radius:999px;white-space:nowrap}
.ok{background:#E3F4EF;color:#0B6B5D}
.live{display:inline-flex;align-items:center;gap:7px;font-size:13px;font-weight:700;color:#0E7C86;margin-left:10px;vertical-align:middle;padding:4px 11px;border-radius:999px;background:#E3F4F3}
.eq{display:inline-flex;align-items:flex-end;gap:2px;height:12px}.eq b{display:block;width:3px;background:#0E7C86;border-radius:1px;animation:eq .9s ease-in-out infinite}
.eq b:nth-child(2){animation-delay:.15s}.eq b:nth-child(3){animation-delay:.3s}.eq b:nth-child(4){animation-delay:.45s}
@keyframes eq{0%,100%{height:3px}50%{height:12px}}
.alarm{background:#E4572E;color:#fff;animation:p 1s infinite}
@keyframes p{50%{opacity:.72}}
canvas{width:100%;height:250px;display:block;margin-top:14px}
.legend{display:flex;flex-wrap:wrap;gap:26px;font-size:15px;color:#4F5B73;margin-top:8px}
.sw{display:inline-block;width:20px;height:4px;border-radius:2px;vertical-align:middle;margin-right:8px}
.row{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:16px}
.m{background:#F5F8FC;border-radius:14px;padding:12px 16px}
.mk{font-size:15px;color:#5E6A82;font-weight:600}
.mv{font-family:__DISPLAY__;font-size:28px;font-weight:700;color:#1F2A44;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
@media (max-width:700px){.row{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (prefers-reduced-motion:reduce){.alarm{animation:none}}
</style></head><body><div class="wrap">
<div class="top"><div><div class="h">__TITLE__ <span class="live"><span class="eq"><b></b><b></b><b></b><b></b></span>Streaming</span></div>
<div class="sub">__SUB__</div></div>
<div id="pill" class="pill ok">All Clear</div></div>
<canvas id="c"></canvas>
<div class="legend"><span><span class="sw" style="background:#3A86FF"></span>Acceleration (g)</span>
<span><span class="sw" style="background:#E4572E"></span>Fall Probability</span>
<span><span class="sw" style="height:0;border-top:4px dashed #13294B;border-radius:0;background:none"></span>Alarm Threshold</span></div>
<div class="row">
<div class="m"><div class="mk">Activity</div><div class="mv" id="act">Warming Up</div></div>
<div class="m"><div class="mk">Fall Probability</div><div class="mv" id="prob">0%</div></div>
<div class="m"><div class="mk">Body Tilt</div><div class="mv" id="tilt">0°</div></div>
<div class="m"><div class="mk">Stream Time</div><div class="mv" id="clock">0.0 s</div></div>
</div></div><script>
const D=__DATA__;
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
function size(){const r=cv.getBoundingClientRect(),d=window.devicePixelRatio||1;cv.width=r.width*d;cv.height=r.height*d;ctx.setTransform(d,0,0,d,0,0);}
size();window.addEventListener('resize',size);
const dur=D.svm.length/D.fs,loop=dur+3,span=10,ymax=Math.max(4,Math.ceil(Math.max(...D.svm)+0.5));
const t0=performance.now();
const pill=document.getElementById('pill'),eAct=document.getElementById('act'),eProb=document.getElementById('prob'),
      eTilt=document.getElementById('tilt'),eClock=document.getElementById('clock');
function lastWin(t){let i=-1;for(let j=0;j<D.tw.length;j++){if(D.tw[j]<=t)i=j;else break;}return i;}
function frame(now){
  let t=((now-t0)/1000)%loop; t=Math.min(t,dur);
  const W=cv.clientWidth,H=cv.clientHeight,pl=40,pr=46,pt=10,pb=28;
  ctx.clearRect(0,0,W,H);
  const x0=Math.max(0,t-span),X=s=>pl+(s-x0)/span*(W-pl-pr),Ya=v=>pt+(1-v/ymax)*(H-pt-pb),Yp=v=>pt+(1-v)*(H-pt-pb);
  ctx.font='13px sans-serif';ctx.lineWidth=1;
  for(let g=0;g<=ymax;g++){const y=Ya(g);ctx.strokeStyle='#EEF1F6';ctx.beginPath();ctx.moveTo(pl,y);ctx.lineTo(W-pr,y);ctx.stroke();
    ctx.fillStyle='#7D879B';ctx.textAlign='right';ctx.fillText(g+' g',pl-6,y+4);}
  ctx.textAlign='left';ctx.fillStyle='#E4572E';ctx.fillText('100%',W-pr+6,Yp(1)+4);ctx.fillText('0%',W-pr+6,Yp(0)+4);
  ctx.fillStyle='#7D879B';ctx.textAlign='center';
  for(let s=Math.ceil(x0/2)*2;s<=x0+span;s+=2){ctx.fillText(s+' s',X(s),H-8);}
  const i=lastWin(t);
  ctx.beginPath();let started=false,lastX=null;
  for(let j=0;j<=i;j++){if(D.tw[j]<x0)continue;const px=X(D.tw[j]),py=Yp(D.p[j]);
    if(!started){ctx.moveTo(px,Yp(0));ctx.lineTo(px,py);started=true;}else ctx.lineTo(px,py);lastX=px;}
  if(started){ctx.lineTo(lastX,Yp(0));ctx.closePath();ctx.fillStyle='rgba(228,87,46,0.16)';ctx.fill();
    ctx.beginPath();let s2=false;for(let j=0;j<=i;j++){if(D.tw[j]<x0)continue;const px=X(D.tw[j]),py=Yp(D.p[j]);if(!s2){ctx.moveTo(px,py);s2=true;}else ctx.lineTo(px,py);}
    ctx.strokeStyle='#E4572E';ctx.lineWidth=2;ctx.stroke();}
  ctx.setLineDash([6,5]);ctx.strokeStyle='#13294B';ctx.lineWidth=1.2;ctx.beginPath();ctx.moveTo(pl,Yp(D.thr));ctx.lineTo(W-pr,Yp(D.thr));ctx.stroke();ctx.setLineDash([]);
  const a=Math.floor(x0*D.fs),b=Math.min(Math.floor(t*D.fs),D.svm.length-1);
  ctx.beginPath();for(let k=a;k<=b;k++){const px=X(k/D.fs),py=Ya(Math.min(D.svm[k],ymax));if(k===a)ctx.moveTo(px,py);else ctx.lineTo(px,py);}
  ctx.strokeStyle='#3A86FF';ctx.lineWidth=2.2;ctx.stroke();
  const fired=D.alarms.filter(v=>v<=t);
  fired.forEach(v=>{if(v>=x0){ctx.setLineDash([3,4]);ctx.strokeStyle='#E4572E';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(X(v),pt);ctx.lineTo(X(v),H-pb);ctx.stroke();ctx.setLineDash([]);}});
  const ll=i>=0?D.ll[i]:0;
  if(fired.length){pill.className='pill alarm';pill.textContent=ll>=3?'Fall Detected, Call for Help':'Fall Detected';}
  else{pill.className='pill ok';pill.textContent='All Clear';}
  eAct.textContent=i>=0?D.act[i]:'Warming Up';eProb.textContent=(i>=0?Math.round(D.p[i]*100):0)+'%';
  eProb.style.color=(i>=0&&D.p[i]>=D.thr)?'#E4572E':'#1F2A44';
  eTilt.textContent=Math.round(D.tilt[Math.max(0,b)])+'°';eClock.textContent=t.toFixed(1)+' s';
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script></body></html>"""

HERO_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;background:transparent;font-family:__FONT__;overflow:hidden}
.hero{position:relative;overflow:hidden;border-radius:24px;padding:44px 52px 30px 52px;color:#fff;box-sizing:border-box;
 background:radial-gradient(900px 380px at 8% -20%,#1F6280 0%,rgba(31,98,128,0) 60%),radial-gradient(700px 300px at 100% 120%,#0F5560 0%,rgba(15,85,96,0) 60%),#13294B}
.glow{position:absolute;width:420px;height:420px;border-radius:50%;right:-120px;top:-160px;background:radial-gradient(circle,rgba(228,87,46,.28),rgba(228,87,46,0) 65%);animation:drift 9s ease-in-out infinite}
@keyframes drift{50%{transform:translate(-60px,40px)}}
h1{font-family:__DISPLAY__;font-size:62px;margin:0;font-weight:700;letter-spacing:-.5px;animation:in .8s cubic-bezier(.2,.7,.2,1) both}
.tag{font-size:22px;color:#D5E1F2;margin-top:10px;max-width:60ch;line-height:1.5;animation:in .8s .12s cubic-bezier(.2,.7,.2,1) both}
.by{margin-top:14px;font-size:16px;color:#A9BBD6;animation:in .8s .24s both}
.chip{position:absolute;right:52px;top:48px;font-size:14px;font-weight:700;color:#fff;background:rgba(228,87,46,.18);border:1px solid rgba(228,87,46,.6);
 padding:6px 14px 6px 28px;border-radius:999px}
.chip:before{content:'';position:absolute;left:11px;top:50%;width:9px;height:9px;margin-top:-4.5px;border-radius:50%;background:#E4572E;animation:p 1.2s infinite}
@keyframes p{50%{opacity:.3}}
@keyframes in{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:none}}
canvas{display:block;width:100%;height:110px;margin:22px 0 6px 0}
@media (max-width:700px){h1{font-size:44px}.tag{font-size:18px}.chip{display:none}.hero{padding:30px 26px}}
</style></head><body><div class="hero"><div class="glow"></div><div class="chip">Monitoring Live</div>
<h1>SafeStep</h1>
<div class="tag">A Waist Worn Phone that Detects Falls, Warns during the Fall and Alerts a Caregiver after a Long Lie on the Floor.</div>
<canvas id="c"></canvas>
<div class="by">__BY__</div></div><script>
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
function size(){const r=cv.getBoundingClientRect(),d=window.devicePixelRatio||1;cv.width=r.width*d;cv.height=r.height*d;ctx.setTransform(d,0,0,d,0,0);}
size();window.addEventListener('resize',size);
const FS=50,CYCLE=9.0;
function sig(t){const c=t%CYCLE;let v=1+0.22*Math.sin(2*Math.PI*1.8*t)+0.08*Math.sin(2*Math.PI*4.1*t+1.3);
 if(c>5.2&&c<5.6)v=1-1.7*(c-5.2)/0.4*0.45;
 if(c>=5.6&&c<5.75)v=0.25+(c-5.6)/0.15*3.4;
 if(c>=5.75&&c<6.0)v=3.65-(c-5.75)/0.25*2.4;
 if(c>=6.0)v=1+0.35*Math.exp(-(c-6.0)*3)*Math.sin(2*Math.PI*3*(c-6.0))+0.015*Math.sin(t*37);
 return v;}
function phase(t){const c=t%CYCLE;return c<5.2?'walk':c<5.6?'drop':c<6.4?'impact':'rest';}
const t0=performance.now();let ring=0,lastPhase='';
function draw(now){
 const t=(now-t0)/1000,W=cv.clientWidth,H=cv.clientHeight,span=8,nowX=W*0.78,pad=10;
 ctx.clearRect(0,0,W,H);
 const X=s=>nowX-(t-s)/span*nowX, Y=v=>H-pad-(Math.min(v,4)/4)*(H-2*pad);
 ctx.strokeStyle='rgba(255,255,255,.07)';ctx.lineWidth=1;
 for(let g=1;g<=3;g++){ctx.beginPath();ctx.moveTo(0,Y(g));ctx.lineTo(W,Y(g));ctx.stroke();}
 const grad=ctx.createLinearGradient(0,0,nowX,0);grad.addColorStop(0,'rgba(143,179,230,0)');grad.addColorStop(.25,'rgba(143,179,230,.75)');grad.addColorStop(1,'#B9D3F5');
 let prev=null;
 for(let s=t-span;s<=t;s+=1/FS){const v=sig(Math.max(s,0)),x=X(s),y=Y(v);
   if(prev){ctx.beginPath();ctx.moveTo(prev[0],prev[1]);ctx.lineTo(x,y);ctx.lineWidth=2.4;
     ctx.strokeStyle=(v>2||prev[2]>2)?'#F07A52':grad;ctx.stroke();}
   prev=[x,y,v];}
 ctx.setLineDash([2,6]);ctx.strokeStyle='rgba(255,255,255,.18)';ctx.beginPath();ctx.moveTo(nowX,4);ctx.lineTo(nowX,H-4);ctx.stroke();ctx.setLineDash([]);
 const ph=phase(t),v=sig(t),y=Y(v),alarm=(ph==='impact'||ph==='rest')&&(t%CYCLE)<8.6;
 if(ph==='impact'&&lastPhase!=='impact')ring=1;lastPhase=ph;
 if(ring>0){ctx.beginPath();ctx.arc(nowX,y,10+(1-ring)*34,0,2*Math.PI);ctx.strokeStyle='rgba(228,87,46,'+ring+')';ctx.lineWidth=2;ctx.stroke();ring-=0.018;}
 const col=alarm?'#E4572E':'#5EEAD4';
 const halo=ctx.createRadialGradient(nowX,y,0,nowX,y,18);halo.addColorStop(0,col);halo.addColorStop(1,'rgba(0,0,0,0)');
 ctx.fillStyle=halo;ctx.beginPath();ctx.arc(nowX,y,18,0,2*Math.PI);ctx.fill();
 ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(nowX,y,5.5,0,2*Math.PI);ctx.fill();
 ctx.fillStyle=col;ctx.beginPath();ctx.arc(nowX,y,3.5,0,2*Math.PI);ctx.fill();
 requestAnimationFrame(draw);}
requestAnimationFrame(draw);
</script></body></html>"""


def embed(html_text, height):
    html_text = html_text.replace('<style>', '<style>' + FONT_FACES.replace('{{', '{').replace('}}', '}'), 1)
    html_text = html_text.replace('__FONT__', FONT.replace("'", '"')).replace('__DISPLAY__', DISPLAY.replace("'", '"'))
    if hasattr(st, 'iframe'):
        st.iframe(html_text, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(html_text, height=height)


@st.cache_resource(show_spinner=False)
def home_stream_data(code='F01'):
    T = trials()
    pool = T[(T.split == 'test') & (T.code == code) & (T.age_group == 'young')]
    if pool.empty:
        pool = T[(T.split == 'test') & (T.type == 'Fall')]
    for tid in pool.index[:6]:
        res = analyze_trial(int(tid))
        if res['alarms']:
            break
    fs = 25; step = core.FS // fs
    return dict(fs=fs, svm=np.round(res['sig']['svm'][::step], 3).tolist(), tilt=np.round(res['posture'][::step], 1).tolist(),
                tw=np.round(res['t_end'], 2).tolist(), p=np.round(res['p_fall'], 3).tolist(),
                act=[act_name(a) for a in res['act']], ll=np.round(res['long_lie_s'], 1).tolist(),
                thr=float(res['thr']), alarms=[float(res['t_end'][a]) for a in res['alarms']])


def live_stream(code, title, sub, height=520):
    data = home_stream_data(code)
    embed(HOME_LIVE.replace('__DATA__', json.dumps(data)).replace('__TITLE__', title).replace('__SUB__', sub), height)


def headline_numbers():
    test = models()['cfg']['test']
    falls = int(round(test['falls'])); caught = int(round(test['sensitivity'] * falls))
    lo, hi = wilson(caught, falls)
    fa = int(round(test['false_alarms'])); hours = float(test['adl_hours'])
    return dict(test=test, falls=falls, caught=caught, lo=lo, hi=hi, fa=fa, hours=hours, ub=poisson_upper(fa, hours))


def page_home():
    embed(HERO_HTML.replace('__BY__', f'{AUTHOR}, {INSTITUTE}'), 450)
    gap(0.4)
    a, b = st.columns([2.3, 1], gap='large', vertical_alignment='center')
    with a:
        st.markdown('<div class="ss-intro"><div class="ss-intro-t">Meet SafeStep</div>'
                    '<div class="ss-intro-s">SafeStep Turns an Ordinary Smartphone into a Personal Fall Guardian. Clipped at the Waist, '
                    'It Reads Motion Every Half Second, Recognises Daily Activities, Spots a Fall as It Happens and Calls a '
                    'Caregiver when the Person cannot Get Up.</div></div>', unsafe_allow_html=True)
    with b:
        if st.button('Open the Dashboard', type='primary', width='stretch', icon=':material/dashboard:', key='home_cta_top'):
            st.switch_page(PAGES['dash'])

    section(1, 'How SafeStep Works')
    cfg = models()['cfg']
    steps = [('Phone Sensors', 'Accelerometer and Gyroscope at the Waist, in Any Orientation.'),
             ('Three Second Windows', 'A Fresh Window Every Half Second from the Live Stream.'),
             (f'{len(core.FEATURE_NAMES)} Features', 'Orientation Free Signals for Impact, Tilt, Rotation and Stillness.'),
             ('Two Models', f'{model_name(cfg["fall_model"])} for Falls and {cfg["activity_model"]} for Activities.'),
             ('Caregiver Alert', 'Alarm, Long Lie Check and a Call Request.')]
    arrow = ('<svg class="ss-step-arrow" viewBox="0 0 22 22"><circle cx="11" cy="11" r="11" fill="white" stroke="#E1E7F0"/>'
             f'<path d="M8 6 L14 11 L8 16" fill="none" stroke="{CORAL}" stroke-width="2.4" stroke-linecap="round"/></svg>')
    st.markdown('<div class="ss-flow">' + ''.join(
        f'<div class="ss-step" style="--i:{i}"><div class="ss-step-n">Step {i + 1}</div><div class="ss-step-t">{t}</div>'
        f'<div class="ss-step-s">{d}</div>{arrow if i < len(steps) - 1 else ""}</div>' for i, (t, d) in enumerate(steps)) + '</div>',
        unsafe_allow_html=True)

    section(2, 'Why It Matters')
    pillars = [('Falls are Silent', 'Many Older People cannot Reach a Phone after a Fall. SafeStep Calls for Them.'),
               ('Any Phone, Any Pocket', 'Orientation Free Features Keep Working However the Phone is Clipped.'),
               ('Explained Decisions', 'Every Alarm Comes with Plain Reasons a Caregiver can Trust.')]
    cols = st.columns(3, gap='large')
    for col, (t, d) in zip(cols, pillars):
        col.markdown(f'<div class="ss-card"><div class="ss-ct" style="font-family:{DISPLAY};font-size:1.3rem;font-weight:600;color:{NAVY}">{t}</div>'
                     f'<div class="ss-s" style="font-size:1.02rem;margin-top:8px">{d}</div></div>', unsafe_allow_html=True)
    gap(1.6)
    c1, _ = st.columns([1, 2.4])
    with c1:
        if st.button('Explore the Dashboard', type='primary', width='stretch', icon=':material/arrow_forward:', key='home_cta_bottom'):
            st.switch_page(PAGES['dash'])


def page_dashboard():
    header('Dashboard', 'Every Demonstration and Analysis in One Place.')
    h = headline_numbers()
    cards([('Falls Caught', pct(h['test']['sensitivity']), f"{h['caught']} of {h['falls']} Falls, 95% CI {pct(h['lo'])} to {pct(h['hi'])}", TEAL),
           ('False Alarms', f"{h['fa']}", f"In {h['hours']:.1f} h of Daily Activity, 95% Upper Bound {h['ub']:.2f} per Hour", NAVY),
           ('Reaction Time', f"{h['test']['median_delay']:.1f} s", 'Median Delay after Impact', CORAL),
           ('Features', f'{len(core.FEATURE_NAMES)}', 'Computed Every Half Second', PLUM)], anim=True)
    gap(1.0)
    live_stream('F06', 'Live Monitoring Preview', 'A Faint while Walking, Streamed Continuously through the Full SafeStep Pipeline', 500)
    section(1, 'Live Demonstrations')
    items = [(PAGES['live'], 'Live Phone Monitor', 'Wear a Phone and Watch SafeStep React over Wi-Fi.'),
             (PAGES['sim'], 'Fall Simulator', 'Sixteen Real Situations Played Back Live.'),
             (PAGES['challenge'], 'Beat the AI', 'Spot the Fall Faster and More Accurately than SafeStep.'),
             (PAGES['editor'], 'Fall Editor', 'Soften a Fall, Rotate the Phone or Add Sensor Noise.')]
    _link_cards(items)
    section(2, 'Inside SafeStep')
    items = [(PAGES['engine'], 'Feature Engine', f'All {len(core.FEATURE_NAMES)} Features Changing in Real Time.'),
             (PAGES['lab'], 'Model Lab', 'Train, Validate and Test Models Live.'),
             (PAGES['why'], 'Decision Explainer', 'Reasons behind Every Alarm.'),
             (PAGES['results'], 'Research Results', 'Benchmarks, Robustness and Explainability.')]
    _link_cards(items)


def _link_cards(items):
    cols = st.columns(len(items), gap='large')
    for i, (col, (pg, t, d)) in enumerate(zip(cols, items)):
        with col:
            st.markdown(f'<div class="ss-card ss-link-card ss-swing" style="--i:{i}"><div class="ss-ct">{t}</div>'
                        f'<div class="ss-s" style="margin-top:8px">{d}</div></div>', unsafe_allow_html=True)
            st.page_link(pg, label=f'Open {t}', icon=':material/arrow_forward:')


def page_live():
    header('Live Phone Monitor', 'Clip a Phone at the Waist, Then Walk, Sit, Lie Down or Fall on a Mattress and Watch SafeStep React.')
    ss = st.session_state
    ss.setdefault('running', False)
    ss.setdefault('live_src', 'Real Phone over Wi-Fi')
    src = st.segmented_control('Source', ['Real Phone over Wi-Fi', 'Demo Stream', 'Upload a Recording'],
                               label_visibility='collapsed', key='live_src') or 'Real Phone over Wi-Fi'
    gap(0.4)

    def start():
        ss.running = True; ss.engine = None; ss.pop('live_error', None)

    def stop():
        ss.running = False

    if src == 'Upload a Recording':
        ss.pop('live_error', None)
        c1, c2 = st.columns([2, 1], gap='large')
        files = c1.file_uploader('Phone CSV File, One File with Both Sensors or Two Separate Files', type=['csv', 'txt'],
                                 accept_multiple_files=True)
        demos = sorted((BUNDLE / 'demo_phone').glob('*.csv'))
        demo_label = {d.name: tc(d.stem.replace('demo_', '').replace('_', ' ')) for d in demos}
        demo = c2.selectbox('Or a Demo Phone File', ['None'] + list(demo_label),
                            format_func=lambda v: 'None' if v == 'None' else demo_label[v])
        calib = c2.toggle('Auto Calibration', value=True)
        d1, d2, _ = st.columns([1, 1, 1.4], gap='medium')
        d1.download_button('Download Sample Fall CSV', sample_csv('fall'), file_name='SafeStep_Sample_Fall.csv',
                           mime='text/csv', icon=':material/download:', width='stretch')
        d2.download_button('Download Sample Daily Life CSV', sample_csv('daily'), file_name='SafeStep_Sample_Daily_Life.csv',
                           mime='text/csv', icon=':material/download:', width='stretch')
        if not files and demo != 'None':
            files = [str(BUNDLE / 'demo_phone' / demo)]; name = demo_label[demo]
        else:
            name = ', '.join(f.name for f in files) if files else ''
        if not files:
            note('Upload a Phone Recording or Choose a Demo File to Begin. Accepted Formats: phyphox Exports, '
                 'Sensor Logger Exports or Any CSV with a Time Column and x, y, z Sensor Columns.')
            return
        try:
            acc, gyr, info, fs_in, gain = parse_phone(files, calibrate=calib); res = analyze(acc, gyr)
        except Exception as e:
            st.error(f'File Not Usable. {e}'); return
        cards([('Duration', f'{len(acc) / core.FS:.1f} s'), ('Phone Rate', f'{fs_in:.0f} Hz'),
               ('Calibration Gain', f'{gain:.3f}'),
               ('Fall Alarms', f"{len(res['alarms'])}", '', CORAL if res['alarms'] else TEAL)])
        gap(0.8)
        fancy_table(pd.DataFrame(info, columns=['Item', 'Value']), 'File Details', icon=':material/description:')
        gap(0.8)
        c1, c2 = st.columns([1, 4], gap='medium')
        speed = c1.select_slider('Speed', [1, 2, 4], value=2, format_func=lambda v: f'{v}x')
        go_ = c1.button('Play', type='primary', width='stretch', icon=':material/play_arrow:')
        t = c2.slider('Time (s)', float(core.WIN / core.FS), float(len(acc) / core.FS), float(len(acc) / core.FS), 0.5, key='up_time')
        ph = st.empty()
        if go_:
            play(res, name, ph, speed, key='up')
        else:
            with ph.container():
                dashboard(state_at(res, t, name), 'up_static')
        st.download_button('Download the Analysis', pd.DataFrame({'time_s': res['t_end'], 'fall_probability': res['p_fall'],
                                                                  'activity': res['act']}).to_csv(index=False).encode(),
                           file_name='safestep_analysis.csv', icon=':material/download:')
        return

    c1, c2, c3 = st.columns([2.2, 1, 1], gap='medium')
    tid = None
    if src == 'Real Phone over Wi-Fi':
        url = c1.text_input('Phone Address Shown by phyphox', value=ss.get('phone_url', '192.168.0.100:8080'))
        ss.phone_url = url
    else:
        T = trials(); pool = T[T.split == 'test']
        opts = pool.sort_values(['type', 'code']).index.tolist()
        tid = c1.selectbox('Recording to Stream', opts, format_func=lambda i: recording_label(T.loc[i]))
    minutes = c2.select_slider('Session Length', [1, 2, 3, 4, 5], value=2, format_func=lambda v: f'{v} min')
    c3.markdown('<div style="height:1.9rem"></div>', unsafe_allow_html=True)
    if ss.running:
        c3.button('Stop', on_click=stop, width='stretch', icon=':material/stop:', key='live_stop')
    else:
        c3.button('Start', on_click=start, type='primary', width='stretch', icon=':material/play_arrow:', key='live_start')
    if src == 'Real Phone over Wi-Fi':
        with st.expander('How to Connect the Phone', icon=':material/help:', expanded=False):
            st.markdown('1. Install the Free **phyphox** App on the Phone.\n'
                        '2. Download the SafeStep Experiment below and Open It in phyphox (Plus Button, Then *Add Experiment from File*).\n'
                        '3. Connect the Phone and the Laptop to the Same Wi-Fi or Hotspot, Then Enable **Remote Access** in the phyphox Menu.\n'
                        '4. Type the Address Shown by phyphox into the Box above, Press Start and Clip the Phone at the Waist.\n'
                        '5. Quick Check: Open the Same Address in the Laptop Browser. The phyphox Page must Load.')
            st.download_button('Download SafeStep Experiment', PHYPHOX_XML.encode('utf-8'), file_name='SafeStep.phyphox',
                               icon=':material/download:')
    if src != 'Real Phone over Wi-Fi':
        ss.pop('live_error', None)               # the phone guide belongs to the real phone source only
    elif ss.get('live_error'):
        phone_help(ss.live_error, ss.get('phone_url', ''))

    ph = st.empty(); snd = st.empty(); info = st.empty()
    if ss.running:
        try:
            source = Phyphox(ss.phone_url) if src == 'Real Phone over Wi-Fi' else Replay(tid)
            source.connect()
        except Exception:
            ss.running = False
            ss.live_error = 'unreachable'
            st.rerun()
        eng = Engine(calibrate=True); ss.engine = eng
        t_start = time.time(); k = 0; rang = 0; fails = 0; lost = False
        while ss.running and time.time() - t_start < minutes * 60:
            try:
                eng.add(*source.poll()); fails = 0
            except Exception:
                fails += 1
                if fails > 20:
                    lost = True; break
            S = eng.step()
            if S is not None and k % 2 == 0:
                with ph.container():
                    dashboard(S, f'lv{k}')
                info.caption(f'Received {eng.duration():.1f} s of Motion. Calibration Gain {eng.gain:.3f}. '
                             f'Processing Time {eng.ms:.0f} ms per Update.')
                if len(eng.alarms) > rang:
                    with snd.container():
                        play_siren()
                    rang = len(eng.alarms)
            if isinstance(source, Replay) and source.done():
                break
            k += 1; time.sleep(0.25)
        source.stop(); ss.running = False
        if lost:
            ss.live_error = 'lost'
        st.rerun()

    eng = ss.get('engine')
    if eng is not None and eng.hist:
        section(1, 'Session Report')
        acts = pd.Series([act_name(h[2]) for h in eng.hist])
        mins = (acts.value_counts() * 0.5 / 60).reset_index(); mins.columns = ['activity', 'minutes']
        cards([('Session Length', f'{eng.duration():.0f} s'),
               ('Fall Alarms', f'{len(eng.alarms)}', '', CORAL if eng.alarms else TEAL),
               ('Main Activity', acts.mode().iloc[0]), ('Calibration Gain', f'{eng.gain:.3f}')])
        gap(0.8)
        a, b = st.columns(2, gap='large')
        with a:
            chart(layout(px.pie(mins, values='minutes', names='activity', hole=0.6, color='activity',
                                color_discrete_map=ACT_COLORS), 380, 'Time per Activity', legend_pos='bottom'))
        with b:
            H = pd.DataFrame(eng.hist, columns=['t', 'p', 'activity'])
            fig = px.area(H, x='t', y='p', color_discrete_sequence=[CORAL])
            fig.update_yaxes(range=[0, 1.03], title='Fall Probability'); fig.update_xaxes(title='Time (s)')
            for t_a in eng.alarms:
                fig.add_vline(x=t_a, line=dict(color=CORAL, dash='dot', width=3))
            chart(layout(fig, 380, 'Fall Probability during the Session', legend=False))
        tbl = mins.assign(seconds=mins.minutes * 60).rename(columns={'activity': 'Activity', 'seconds': 'Time (s)'})
        tbl['Share'] = tbl['Time (s)'] / tbl['Time (s)'].sum()
        fancy_table(tbl[['Activity', 'Time (s)', 'Share']], 'Activity Breakdown', fmt={'Time (s)': '{:.1f}', 'Share': '{:.1%}'})
        rec = eng.recording()
        if rec is not None:
            gap(0.6)
            st.download_button('Download This Recording', rec.to_csv(index=False).encode(), file_name='safestep_session.csv',
                               type='primary', icon=':material/download:', help='Open It Later with Upload a Recording.')


def phone_help(kind, url):
    """Plain guidance for the presenter, never a technical error."""
    if kind == 'lost':
        title, sub = 'Phone Connection Paused', 'The Stream Stopped Arriving. Follow These Steps, Then Press Start Again.'
    else:
        title, sub = 'Let Us Connect Your Phone', f'SafeStep Waits for the Phone at <code>{url}</code>. Follow These Steps, Then Press Start.'
    st.markdown(f"""<div class="ss-help"><div class="ss-help-t">{title}</div><div class="ss-help-s">{sub}</div><ol>
      <li>Open <b>phyphox</b> on the Phone, Load the <b>SafeStep</b> Experiment and Press the Play Button.</li>
      <li>In the phyphox Menu, Choose <b>Allow Remote Access</b> and Copy the Exact Address It Shows, for Example <code>192.168.1.37:8080</code>.</li>
      <li>Keep the Phone and the Laptop on the Same Wi-Fi. Venue Wi-Fi Often Blocks Devices, so a <b>Phone Hotspot</b> Works Best.</li>
      <li>Keep the phyphox Screen Open and Unlocked during the Session.</li>
      <li>Quick Check: Open the Same Address in the Laptop Browser and Look for the phyphox Page.</li>
      <li>No Phone at Hand? Choose <b>Demo Stream</b> above for the Same Live Experience.</li></ol></div>""", unsafe_allow_html=True)
    gap(0.8)


@st.cache_resource(show_spinner=False)
def sample_csv(kind):
    """A phone style CSV made from an unseen test recording: rotated phone, about 100 Hz with timing jitter, m/s² and rad/s."""
    T = trials()
    codes = ['F05', 'F01', 'F02', 'F06'] if kind == 'fall' else ['D07', 'D01', 'D09', 'D15']
    pool = T[(T.split == 'test') & (T.code.isin(codes)) & (T.age_group == 'young')]
    order = {c: i for i, c in enumerate(codes)}
    pool = pool.assign(o=pool.code.map(order)).sort_values(['o', 'subject'])
    chosen = int(pool.index[0])
    for tid in pool.index[:12]:
        res = analyze_trial(int(tid))
        if (kind == 'fall') == bool(res['alarms']):
            chosen = int(tid); break
    x = trial_signal(chosen).astype(np.float64)
    rr = np.random.default_rng(7); R = rotation(float(rr.uniform(40, 140)), seed=11)
    t50 = np.arange(len(x)) / core.FS
    t = np.cumsum(rr.uniform(0.008, 0.012, int(t50[-1] / 0.01))); t = t[t < t50[-1]]
    acc = np.column_stack([np.interp(t, t50, (x[:, :3] @ R.T)[:, i]) for i in range(3)]) * G_CONST
    gyr = np.radians(np.column_stack([np.interp(t, t50, (x[:, 3:6] @ R.T)[:, i]) for i in range(3)]))
    df = pd.DataFrame({'time_s': np.round(t, 4), 'acc_x_ms2': acc[:, 0], 'acc_y_ms2': acc[:, 1], 'acc_z_ms2': acc[:, 2],
                       'gyro_x_rads': gyr[:, 0], 'gyro_y_rads': gyr[:, 1], 'gyro_z_rads': gyr[:, 2]}).round(5)
    return df.to_csv(index=False).encode()


SCENARIOS = [('Walking', 'D01'), ('Jogging', 'D03'), ('Stairs', 'D05'), ('Sit Down Fast', 'D10'),
             ('Lie Down on a Bed', 'D13'), ('Get into a Car', 'D17'), ('Stumble and Recover', 'D18'), ('Jump', 'D19'),
             ('Slip Forward', 'F01'), ('Slip Backward', 'F02'), ('Slip Sideways', 'F03'), ('Trip while Jogging', 'F05'),
             ('Faint while Walking', 'F06'), ('Fall while Getting Up', 'F08'), ('Fall while Sitting Down', 'F11'),
             ('Faint while Seated', 'F13')]


def page_simulator():
    header('Fall Simulator', 'Pick a Situation. A Real Recording of an Unseen Person Plays Live and SafeStep Decides.')
    T = trials(); ss = st.session_state
    c1, c2 = st.columns([1, 1], gap='large')
    age = c1.radio('Person', ['young', 'elderly'], horizontal=True, format_func=str.capitalize)
    speed = c2.select_slider('Speed', [0.5, 1, 2], value=1, format_func=lambda v: f'{v}x')

    section(1, 'Daily Life')
    for row in (SCENARIOS[0:4], SCENARIOS[4:8]):
        cols = st.columns(4, gap='medium')
        for col, (lab, code) in zip(cols, row):
            if col.button(lab, key=f'sc_{code}', width='stretch'):
                ss.sim = code; ss.sim_play = True
    section(2, 'Falls')
    for row in (SCENARIOS[8:12], SCENARIOS[12:16]):
        cols = st.columns(4, gap='medium')
        for col, (lab, code) in zip(cols, row):
            if col.button(lab, key=f'sc_{code}', width='stretch', type='primary'):
                ss.sim = code; ss.sim_play = True

    code = ss.get('sim')
    if code is None:
        note('Choose a Situation above to Start the Simulation.')
    else:
        pool = T[(T.code == code) & (T.split == 'test') & (T.age_group == age)]
        if pool.empty:
            pool = T[(T.code == code) & (T.split == 'test')]
        if pool.empty:
            pool = T[T.code == code]
        tid = int(pool.index[int(ss.get('sim_seed', 0)) % len(pool)])
        r = T.loc[tid]; res = analyze_trial(tid)
        title = f'{tc(r["name"])} (Person {r.subject})'
        section(3, title)
        b1, b2, _ = st.columns([1, 1, 4], gap='small')
        if b1.button('Play Again', width='stretch', icon=':material/replay:'):
            ss.sim_play = True
        if b2.button('Another Person', width='stretch', icon=':material/person:'):
            ss.sim_seed = ss.get('sim_seed', 0) + 1; ss.sim_play = True; st.rerun()
        ph = st.empty()
        if ss.get('sim_play'):
            ss.sim_play = False
            play(res, title, ph, speed, key=f'sim{tid}')
        else:
            with ph.container():
                dashboard(state_at(res, res['n'] / core.FS, title), 'sim_static')
        truth = r.type == 'Fall'; alarmed = bool(res['alarms']); ok = alarmed == truth
        gap(0.8)
        banner('Correct Decision' if ok else ('Missed Fall' if truth else 'False Alarm'), 'ok' if ok else 'warn',
               f'Ground Truth: {"Fall" if truth else "No Fall"}.  SafeStep: {"Alarm Raised" if alarmed else "Stayed Calm"}.')


def page_challenge():
    header('Beat the AI', 'Study the Motion Signal of an Unseen Person, Decide Fall or No Fall, Then Compare with SafeStep.')
    T = trials(); ss = st.session_state
    ss.setdefault('game', dict(round=0, human=0, ai=0, tid=None, revealed=False, guess=None, log=[]))
    gm = ss.game
    pool_fall = T[(T.split == 'test') & (T.type == 'Fall')].index.tolist()
    pool_hard = T[(T.split == 'test') & (T.code.isin(['D18', 'D19', 'D11', 'D13', 'D10', 'D12', 'D17']))].index.tolist()

    def new_round():
        rng = np.random.default_rng()
        gm['tid'] = int(rng.choice(pool_fall if rng.random() < 0.5 else pool_hard)); gm['revealed'] = False; gm['guess'] = None

    def guess(g):
        r = T.loc[gm['tid']]; res = analyze_trial(gm['tid'])
        truth = r.type == 'Fall'; ai = bool(res['alarms'])
        gm['round'] += 1; gm['human'] += int(g == truth); gm['ai'] += int(ai == truth)
        gm['revealed'] = True; gm['guess'] = g
        gm['log'].append({'Round': gm['round'], 'Activity': tc(r['name']), 'Ground Truth': 'Fall' if truth else 'No Fall',
                          'Your Answer': 'Fall' if g else 'No Fall', 'SafeStep': 'Fall' if ai else 'No Fall',
                          'You': 'Correct' if g == truth else 'Wrong', 'AI': 'Correct' if ai == truth else 'Wrong'})

    def reset():
        ss.game = dict(round=0, human=0, ai=0, tid=None, revealed=False, guess=None, log=[])

    cards([('Round', f"{gm['round']}"), ('Your Score', f"{gm['human']} / {gm['round']}", 'Correct Answers', SKY),
           ('SafeStep Score', f"{gm['ai']} / {gm['round']}", 'Correct Answers', CORAL)])
    gap(1.0)
    if gm['tid'] is None:
        st.button('Start the Game', type='primary', on_click=new_round, width='stretch', icon=':material/sports_esports:')
        return

    tid = gm['tid']; x = trial_signal(tid); t = np.arange(len(x)) / core.FS; r = T.loc[tid]
    fig = go.Figure()
    fig.add_scatter(x=t, y=np.linalg.norm(x[:, :3], axis=1), name='Acceleration (g)', line=dict(color=SKY, width=2))
    fig.add_scatter(x=t, y=np.linalg.norm(x[:, 3:6], axis=1) / 100, name='Rotation (100 °/s)', line=dict(color=PLUM, width=1.3))
    if gm['revealed']:
        res = analyze_trial(tid)
        for a in res['alarms']:
            fig.add_vline(x=res['t_end'][a], line=dict(color=CORAL, width=3, dash='dot'))
    fig.update_xaxes(title='Time (s)')
    chart(layout(fig, 420, f'Mystery Recording {gm["round"] + (0 if gm["revealed"] else 1)}'))

    if not gm['revealed']:
        c1, c2 = st.columns(2, gap='large')
        c1.button('Fall', type='primary', width='stretch', on_click=guess, args=(True,), icon=':material/personal_injury:')
        c2.button('No Fall', width='stretch', on_click=guess, args=(False,), icon=':material/directions_walk:')
    else:
        res = analyze_trial(tid); truth = r.type == 'Fall'; ai = bool(res['alarms'])
        c1, c2, c3 = st.columns(3, gap='large')
        with c1:
            card('Ground Truth', 'Fall' if truth else 'No Fall', tc(r['name']), CORAL if truth else TEAL)
        with c2:
            card('Your Answer', 'Correct' if gm['guess'] == truth else 'Wrong', 'Fall' if gm['guess'] else 'No Fall',
                 TEAL if gm['guess'] == truth else CORAL)
        with c3:
            card('SafeStep', 'Correct' if ai == truth else 'Wrong', f"Highest Fall Probability {res['p_fall'].max() * 100:.0f}%",
                 TEAL if ai == truth else CORAL)
        gap(0.8)
        c1, c2, c3 = st.columns(3, gap='medium')
        c1.button('Next Recording', type='primary', on_click=new_round, width='stretch', icon=':material/skip_next:')
        watch = c2.button('Watch What Happened', width='stretch', icon=':material/visibility:')
        c3.button('Reset Scores', on_click=reset, width='stretch', icon=':material/restart_alt:')
        if watch:
            play(res, tc(r['name']), st.empty(), 2, key=f'ch{tid}')
    if gm['log']:
        gap(1.0)
        good = lambda v: 'ss-good' if v == 'Correct' else 'ss-bad'
        fancy_table(pd.DataFrame(gm['log'])[::-1], f'Rounds Played ({len(gm["log"])})', expanded=True,
                    icon=':material/scoreboard:', tone={'You': good, 'AI': good})


def page_editor():
    header('Fall Editor', 'Take a Real Fall, Change It and Check whether SafeStep Still Catches It.')
    T = trials()
    pool = T[(T.split == 'test') & ((T.type == 'Fall') | T.code.isin(['D18', 'D19', 'D10', 'D13']))].sort_values(['type', 'code'])
    tid = st.selectbox('Recording', pool.index.tolist(), format_func=lambda i: recording_label(T.loc[i]))
    x = trial_signal(tid).astype(float); acc0, gyr0 = x[:, 0:3], x[:, 3:6]

    section(1, 'Change the Fall')
    c1, c2, c3 = st.columns(3, gap='large')
    strength = c1.slider('Impact Strength (%)', 20, 200, 100, 10)
    rot = c2.slider('Phone Worn Rotated by (°)', 0, 180, 0, 15)
    noise = c3.slider('Sensor Noise (g)', 0.0, 0.1, 0.0, 0.005)
    c4, c5, c6 = st.columns(3, gap='large')
    rate = c4.select_slider('Phone Sampling Rate (Hz)', [50, 25, 20, 10], value=50)
    gain = c5.slider('Sensor Calibration Error (Gain)', 0.8, 1.2, 1.0, 0.05)
    gyro_on = c6.toggle('Gyroscope Enabled', value=True)

    grav = core.lowpass(acc0, 0.3)
    acc = grav + (acc0 - grav) * strength / 100; gyr = gyr0 * (0.5 + 0.5 * strength / 100)
    if rot:
        acc, gyr = perturb(acc, gyr, 'rotation', rot, seed=3)
    if noise:
        acc, gyr = perturb(acc, gyr, 'noise', noise, seed=3)
    if rate < 50:
        acc, gyr = perturb(acc, gyr, 'rate', rate)
    if gain != 1.0:
        acc, gyr = perturb(acc, gyr, 'scale', gain)
    if not gyro_on:
        gyr = np.zeros_like(gyr)
    base, ed = analyze_trial(tid), analyze(acc, gyr)
    r = T.loc[tid]

    section(2, 'Decision')
    a, b = st.columns(2, gap='large')
    with a:
        card('Original Recording', 'Alarm' if base['alarms'] else 'No Alarm',
             f"Highest Fall Probability {base['p_fall'].max() * 100:.0f}%", CORAL if base['alarms'] else TEAL)
    with b:
        card('Edited Version', 'Alarm' if ed['alarms'] else 'No Alarm',
             f"Highest Fall Probability {ed['p_fall'].max() * 100:.0f}%", CORAL if ed['alarms'] else TEAL)
    gap(0.6)
    truth = r.type == 'Fall'; ok = bool(ed['alarms']) == truth
    verdict = 'Still Caught' if ok and truth else ('Still Calm' if ok else ('Missed' if truth else 'False Alarm'))
    banner(verdict, 'ok' if ok else 'warn', f'Ground Truth: {"Fall" if truth else "No Fall"}')

    section(3, 'Signal')
    t = np.arange(len(acc)) / core.FS
    fig = go.Figure()
    fig.add_scatter(x=t, y=np.linalg.norm(acc0, axis=1), name='Original', line=dict(color=GREY, width=2))
    fig.add_scatter(x=t, y=np.linalg.norm(acc, axis=1), name='Edited', line=dict(color=SKY, width=2))
    fig.update_xaxes(title='Time (s)'); fig.update_yaxes(title='Acceleration (g)')
    chart(layout(fig, 340, 'Acceleration Magnitude'))
    fig = go.Figure()
    fig.add_scatter(x=base['t_end'], y=base['p_fall'], name='Original', line=dict(color=GREY, width=3))
    fig.add_scatter(x=ed['t_end'], y=ed['p_fall'], name='Edited', line=dict(color=CORAL, width=3))
    fig.add_hline(y=base['thr'], line=dict(color=NAVY, dash='dash', width=1))
    fig.update_yaxes(range=[0, 1.03], title='Fall Probability'); fig.update_xaxes(title='Time (s)')
    chart(layout(fig, 320, 'Fall Probability'))

    section(4, 'Watch the Edited Version')
    if st.button('Play the Edited Recording', type='primary', icon=':material/play_arrow:'):
        play(ed, 'Edited Recording', st.empty(), 2, key=f'ed{tid}')


def page_engine():
    header('Feature Engine', f'{len(core.FEATURE_NAMES)} Features Computed Every Half Second from the Last 3 Seconds of Motion.')
    cat = catalog(); FN = core.FEATURE_NAMES
    mu, sd = feature_stats()
    tid = trial_picker('fe'); res = analyze_trial(tid)

    def tiles(z, vals):
        html = []
        for fam in FAMILY_NAMES:
            idx = np.where(cat.group.values == fam)[0]
            cells = []
            for j in idx:
                zz = float(z[j]); w = min(abs(zz) / 4, 1) * 100; col = CORAL if zz > 0 else SKY
                bg = (f'rgba(228,87,46,{min(abs(zz) / 5, 0.32):.2f})' if zz > 0
                      else f'rgba(58,134,255,{min(abs(zz) / 5, 0.32):.2f})')
                v = vals[j]; vs = f'{v:.3g}' if abs(v) < 1e4 else f'{v:.1e}'
                cells.append(f'<div class="ss-tile" style="background:{bg}" title="{feature_label(FN[j])}">'
                             f'<div class="ss-tn">{feature_words(FN[j])}</div><div class="ss-tid">{FN[j]}</div>'
                             f'<div class="ss-tv">{vs}</div>'
                             f'<div class="ss-bar"><div style="width:{w:.0f}%;height:5px;background:{col}"></div></div></div>')
            html.append(f'<div class="ss-fam">{FAMILY_NAMES[fam]} Features ({len(idx)})</div><div class="ss-grid">{"".join(cells)}</div>')
        return ''.join(html)

    def frame(w, holder, key):
        x = res['X'].iloc[w]; z = ((x - mu) / sd).clip(-6, 6).values
        with holder.container():
            c1, c2 = st.columns([1, 2.4], gap='large')
            with c1:
                chart(body(res['posture'][min(res['starts'][w] + core.WIN - 1, res['n'] - 1)],
                           'fall' if res['p_fall'][w] >= res['thr'] else 'safe', h=320), key=f'b_{key}')
                card('Window Ending', f"{res['t_end'][w]:.1f} s",
                     f"{act_name(res['act'][w])}, Fall Probability {res['p_fall'][w] * 100:.0f}%")
            with c2:
                chart(signal_fig(res, res['starts'][w] / core.FS, res['t_end'][w], h=270, title='The 3 Seconds being Described'),
                      key=f's_{key}')
                top = pd.Series(np.abs(z), index=FN).sort_values(ascending=False).head(3)
                cards([(feature_html(n), f'{x[n]:.3g}', f'{z[FN.index(n)]:+.1f} Standard Deviations from Typical')
                       for n in top.index])
            st.markdown(tiles(z, x.values), unsafe_allow_html=True)

    section(1, 'All Features, Live')
    c1, c2 = st.columns([1, 4], gap='medium')
    go_ = c1.button('Play', type='primary', width='stretch', icon=':material/play_arrow:')
    w0 = c2.slider('Window', 0, len(res['t_end']) - 1, int(np.argmax(res['p_fall'])), key='fe_w')
    holder = st.empty()
    if go_:
        for w in range(len(res['t_end'])):
            frame(w, holder, f'p{w}'); time.sleep(0.35)
    else:
        frame(w0, holder, 'static')

    section(2, 'Spin the Phone')
    st.caption('The Phone Turns All the Way Around while SafeStep Analyses the Same Moment Again. '
               'Phone Ready Features Stay Still, Device Axis Features Move.')
    if st.button('Spin', type='primary', icon=':material/360:'):
        base_x = res['X'].iloc[w0]; ph = st.empty(); dev = [f for f in FN if f not in core.INVARIANT_FEATURES]
        for deg in range(0, 361, 30):
            R = rotation(deg, seed=1); r2 = analyze(res['acc'] @ R.T, res['gyro'] @ R.T); x2 = r2['X'].iloc[w0]
            chg = (x2 - base_x).abs() / sd
            with ph.container():
                cards([('Phone Rotated', f'{deg}°'),
                       ('Phone Ready Features Changed', f'{chg[core.INVARIANT_FEATURES].mean():.3f}', 'Standard Deviations', TEAL),
                       ('Device Axis Features Changed', f'{chg[dev].mean():.3f}', 'Standard Deviations', CORAL),
                       ('Fall Probability', f"{r2['p_fall'][w0] * 100:.1f}%", f"Originally {res['p_fall'][w0] * 100:.1f}%")])
            time.sleep(0.3)


LAB_STEPS = ['Step 1  Data Split', 'Step 2  Live Training', 'Step 3  Validation', 'Step 4  Final Test']


def page_lab():
    from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score, average_precision_score
    from sklearn.model_selection import GroupKFold
    header('Model Lab', 'Split the People, Train Models Live, Tune Them on Validation People and Run the Final Test.')
    ss = st.session_state; ss.setdefault('lab', {})
    T = trials(); M = models(); cfg = M['cfg']
    step = st.segmented_control('Lab Step', LAB_STEPS, default=LAB_STEPS[0], key='lab_step', label_visibility='collapsed')
    step = step or LAB_STEPS[0]
    gap(0.6)

    if step == LAB_STEPS[0]:
        subj = T.groupby('subject').agg(split=('split', 'first'), age=('age_group', 'first'), recordings=('code', 'size')).reset_index()
        cols = st.columns(3, gap='large')
        for col, sp, nm in zip(cols, ['train', 'val', 'test'], ['Training', 'Validation', 'Final Test']):
            g = subj[subj.split == sp]; c = SPLIT_COLORS[sp]
            chips = ''.join(f'<span class="ss-chip" style="background:{c}1A;color:{c}">{s}</span>' for s in g.subject)
            with col:
                st.markdown(f'<div class="ss-card"><div class="ss-k">{nm}</div><div class="ss-v" style="color:{c}">{len(g)} People</div>'
                            f'<div class="ss-s">{(g.age == "young").sum()} Young, {(g.age == "elderly").sum()} Elderly, '
                            f'{g.recordings.sum()} Recordings</div><div style="margin-top:12px">{chips}</div></div>',
                            unsafe_allow_html=True)
        gap(1.0)
        sets = {s: set(subj[subj.split == s].subject) for s in ['train', 'val', 'test']}
        cards([('Shared by Training and Test', f"{len(sets['train'] & sets['test'])}", 'People, Expected Zero', TEAL),
               ('Shared by Training and Validation', f"{len(sets['train'] & sets['val'])}", 'People, Expected Zero', TEAL),
               ('Shared by Validation and Test', f"{len(sets['val'] & sets['test'])}", 'People, Expected Zero', TEAL)])
        gap(1.0)
        summary = T.groupby(['split', 'age_group', 'type']).size().unstack(fill_value=0).reset_index()
        summary['split'] = summary.split.map({'train': 'Training', 'val': 'Validation', 'test': 'Final Test'})
        summary['age_group'] = summary.age_group.str.capitalize()
        summary = summary.rename(columns={'split': 'Split', 'age_group': 'Age Group', 'ADL': 'Daily Activity Recordings',
                                          'Fall': 'Fall Recordings'})
        fancy_table(summary, 'Recordings per Split', icon=':material/group:')

    elif step == LAB_STEPS[1]:
        df, full = studio_pool()
        c1, c2, c3 = st.columns(3, gap='large')
        task = c1.selectbox('Task', list(STUDIO_TASKS), format_func=STUDIO_TASKS.get, key='lab_task')
        names = c2.multiselect('Models to Race', STUDIO_MODELS, default=['LightGBM', 'Random Forest', 'Neural Network (MLP)'],
                               key='lab_models')
        fsets = studio_feature_sets(); fs_name = c3.selectbox('Features', list(fsets), key='lab_fs')
        cap = st.select_slider('Training Windows', [5000, 10000, 20000, 40000, 80000], value=20000,
                               format_func=lambda v: f'{v:,}', key='lab_cap')
        st.caption(f'Training Pool: {"Every Training Window" if full else "20,000 Window Sample"} '
                   f'({len(df):,} Windows Available). The Live Curve Scores {LIVE_VAL_WINDOWS:,} Validation Windows; '
                   f'the Leaderboard Uses Every Validation Window.')
        metric = 'PR AUC' if task == 'fall' else 'Macro F1'
        if st.button('Start the Race', type='primary', width='stretch', icon=':material/flag:') and names:
            holder = st.empty(); status = st.empty(); curves = {n: ([], []) for n in names}

            def draw(key):
                fig = go.Figure()
                for n, (xx, yy) in curves.items():
                    fig.add_scatter(x=xx, y=yy, name=n, mode='lines+markers', line=dict(width=3))
                fig.update_xaxes(title='Training Progress (%)', range=[0, 102]); fig.update_yaxes(title=f'Validation {metric}')
                with holder.container():
                    chart(layout(fig, 460, 'Live Race on Validation People'), key=key)
            draw('race_empty')
            status.caption('Preparing the Training Data')
            studio_matrices(task, tuple(fsets[fs_name]), int(cap))
            for name in names:
                def on_step(step_i, total, H, name=name):
                    curves[name] = (list(H.step / total * 100), list(H.val))
                    draw(f'race_{name}_{step_i}')
                    status.caption(f'Training {name}: Step {step_i} of {total}')
                S = studio_train(name, task, fsets[fs_name], STUDIO_STEPS[name], cap, on_step)
                S['feature_set'] = fs_name; ss.lab[name] = S
            status.empty()
        lab = {k: v for k, v in ss.lab.items() if v['task'] == task}
        if lab:
            gap(0.6)
            LB = pd.DataFrame([(k, v['val_full'], v['history'].val.max(), v['seconds'], v['n_train'], v['feature_set'])
                               for k, v in lab.items()],
                              columns=['Model', f'Validation {metric}', f'Best Live {metric}', 'Training Time (s)',
                                       'Training Windows', 'Features']).sort_values(f'Validation {metric}', ascending=False)
            LB.insert(0, 'Rank', range(1, len(LB) + 1))
            fancy_table(LB, 'Leaderboard on Validation People', expanded=True, icon=':material/leaderboard:',
                        fmt={f'Validation {metric}': '{:.4f}', f'Best Live {metric}': '{:.4f}', 'Training Time (s)': '{:.1f}',
                             'Training Windows': '{:,}'})

    elif step == LAB_STEPS[2]:
        lab = ss.lab
        if not lab:
            note('Train at Least One Model in Step 2 to Unlock Validation.')
        else:
            name = st.selectbox('Model', list(lab), key='val_model'); S = lab[name]
            D = studio_matrices(S['task'], tuple(S['feats']), S['cap'])
            Xva, yva = D['Xva'], D['yva']
            P = studio_predict(S, Xva)
            if S['task'] == 'fall':
                thr = st.slider('Alarm Threshold', 0.01, 0.99, float(S.get('thr', 0.5)), 0.01, key=f'thr_{name}'); S['thr'] = thr
                pred = P[:, 1] >= thr
                tp, fn = int((pred & (yva == 1)).sum()), int((~pred & (yva == 1)).sum())
                fp, tn = int((pred & (yva == 0)).sum()), int((~pred & (yva == 0)).sum())
                gap(0.4)
                cards([('Falls Caught', f'{tp}', f'Out of {tp + fn} Fall Windows', TEAL),
                       ('Falls Missed', f'{fn}', 'Fall Windows below the Threshold', CORAL if fn else TEAL),
                       ('False Alarms', f'{fp}', f'Among {fp + tn:,} Normal Windows', CORAL if fp else TEAL),
                       ('Correctly Calm', f'{tn:,}', 'Normal Windows below the Threshold')])
            else:
                cl = S['classes']; pred = cl[P.argmax(1)]
                cards([('Validation Macro F1', f"{f1_score(yva, pred, average='macro'):.4f}", '', TEAL),
                       ('Balanced Accuracy', f'{balanced_accuracy_score(yva, pred):.4f}')])
                gap(0.6)
                names_ = [act_name(c) for c in cl]
                fig = px.imshow(confusion_matrix(yva, pred, labels=cl, normalize='true'), x=names_, y=names_,
                                text_auto='.2f', color_continuous_scale='Blues')
                chart(layout(fig, 580, 'Activity Confusion on Validation People', legend=False))
            section(1, 'Cross Validation by Person')
            k = st.select_slider('Folds', [3, 4, 5], value=5)
            if st.button('Run Cross Validation', type='primary', icon=':material/sync:'):
                import lightgbm as lgb
                df, full = studio_pool(); y = studio_labels(df, S['task'], full)
                ok = ((y != -1) if S['task'] == 'fall' else (y != 'ignore')) & (df.split.values == 'train'); idx = np.where(ok)[0]
                if len(idx) > 30000:
                    idx = np.sort(np.random.default_rng(0).choice(idx, 30000, replace=False))
                X, yy, grp = df[S['feats']].values[idx], y[idx], df.subject.values[idx]
                bar = st.progress(0.0); out = st.empty(); res = []
                for f, (a, b) in enumerate(GroupKFold(n_splits=k).split(X, yy, grp)):
                    cls = np.array(sorted(np.unique(yy)))
                    m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.08, num_leaves=31, class_weight='balanced',
                                           verbose=-1, random_state=0).fit(X[a], np.searchsorted(cls, yy[a]))
                    Pp = m.predict_proba(X[b])
                    sc = (average_precision_score(yy[b], Pp[:, 1]) if S['task'] == 'fall'
                          else f1_score(yy[b], cls[Pp.argmax(1)], average='macro'))
                    res.append((f'Fold {f + 1}', sc, len(set(grp[b])))); bar.progress((f + 1) / k, text=f'Fold {f + 1} of {k}')
                    R = pd.DataFrame(res, columns=['Fold', 'Score', 'Held Out People'])
                    with out.container():
                        chart(layout(px.bar(R, x='Fold', y='Score', text_auto='.4f', color_discrete_sequence=[NAVY],
                                            range_y=[0, 1.05]), 320, 'Score per Fold', legend=False), key=f'cv{f}')
                cards([('Mean Score', f'{R.Score.mean():.4f}', 'Across People Groups', TEAL),
                       ('Spread', f'{R.Score.std():.4f}', 'Smaller Means More Stable')])
                gap(0.6)
                fancy_table(R, 'Cross Validation Folds', expanded=True, fmt={'Score': '{:.4f}'})

    else:
        opts = ['Deployed SafeStep'] + [k for k, v in ss.lab.items() if v['task'] == 'fall']
        name = st.selectbox('Model for the Final Test', opts, key='test_model')
        if name == 'Deployed SafeStep':
            S = dict(name=cfg['fall_model'], task='fall', feats=cfg['features'], classes=np.array([0, 1]), model=M['fall'], scaler=None)
            thr, k = float(cfg['threshold']), int(cfg['consecutive_windows'])
        else:
            S = ss.lab[name]; thr, k = float(S.get('thr', 0.5)), 2
        st.caption(f'Alarm Threshold {thr:.2f} with {k} Consecutive Windows. Every Recording of the '
                   f'{T[T.split == "test"].subject.nunique()} Test People Streamed Like a Live Phone.')
        if st.button('Start the Final Test', type='primary', width='stretch', icon=':material/fact_check:'):
            TWs, groups = test_index()
            p = studio_predict(S, TWs[S['feats']].values)[:, 1]
            bar = st.progress(0.0); box = st.empty()
            caught = n_fall = fa = 0; hours = 0.0; per = []
            for j, (tid, idx, te, imp, typ, code, age, h) in enumerate(groups):
                al = [te[i] for i in alarm_indices(p[idx], thr, k, te)]
                if typ == 'Fall':
                    n_fall += 1; hit = any(imp <= a <= imp + core.WIN / core.FS + 0.5 for a in al); caught += hit
                    per.append((code, 'Fall', hit, 0))
                else:
                    fa += len(al); hours += h; per.append((code, 'Daily Activity', False, len(al)))
                if j % 40 == 0 or j == len(groups) - 1:
                    bar.progress((j + 1) / len(groups), text=f'{j + 1} of {len(groups)} Recordings Checked')
                    with box.container():
                        cards([('Falls Caught', f'{caught} / {n_fall}', pct(caught / max(n_fall, 1)), TEAL),
                               ('False Alarms', f'{fa}', f'In {hours:.1f} h of Daily Activity', CORAL if fa else TEAL),
                               ('False Alarms per Hour', f'{fa / max(hours, 1e-9):.2f}',
                                f'95% Upper Bound {poisson_upper(fa, hours):.2f}' if hours > 0 else '')])
            base = stream_eval((TWs['svm__max'].values >= 3.0).astype(float), 0.5, 1)
            gap(1.0)
            cmp_ = pd.DataFrame([(model_name(name), caught / max(n_fall, 1) * 100, fa / max(hours, 1e-9)),
                                 ('Classic 3 g Threshold', base['sensitivity'] * 100, base['fa_per_hour'])],
                                columns=['model', 'caught', 'fa'])
            a, b = st.columns(2, gap='large')
            with a:
                chart(layout(go.Figure(go.Bar(x=cmp_.model, y=cmp_.caught, marker_color=[TEAL, GREY],
                                              text=[f'{v:.1f}%' for v in cmp_.caught], textposition='outside')
                                       ).update_yaxes(range=[0, 115]), 360, 'Falls Caught (%)', legend=False))
            with b:
                chart(layout(go.Figure(go.Bar(x=cmp_.model, y=cmp_.fa, marker_color=[CORAL, GREY],
                                              text=[f'{v:.2f}' for v in cmp_.fa], textposition='outside')
                                       ).update_yaxes(range=[0, max(cmp_.fa.max() * 1.2, 1)]), 360, 'False Alarms per Hour', legend=False))
            P = pd.DataFrame(per, columns=['code', 'kind', 'hit', 'fa'])
            names_map = trials().drop_duplicates('code').set_index('code')['name']
            byc = P.groupby(['code', 'kind']).agg(Recordings=('hit', 'size'), Caught=('hit', 'sum'), FalseAlarms=('fa', 'sum')).reset_index()
            byc['Activity'] = byc.code.map(lambda c: f'{c}  {tc(names_map.get(c, c))}')
            byc['Result'] = np.where(byc.kind == 'Fall', byc.Caught.astype(int).astype(str) + ' of ' + byc.Recordings.astype(str) + ' Caught',
                                     byc.FalseAlarms.astype(int).astype(str) + ' False Alarm(s)')
            fancy_table(byc[['Activity', 'kind', 'Recordings', 'Result']].rename(columns={'kind': 'Type'}),
                        'Result per Activity Type', icon=':material/fact_check:',
                        tone={'Result': lambda v: 'ss-bad' if ('False' in v and not v.startswith('0')) or
                              ('Caught' in v and v.split(' of ')[0] != v.split(' of ')[1].split()[0]) else 'ss-good'})


def page_why():
    header('Decision Explainer', 'SafeStep Shows the Reasons behind Each Decision in Plain Words.')
    M = models()
    if 'explain' in M:
        st.caption(f'Explanations Use Exact TreeSHAP Values from a LightGBM Twin of the Deployed '
                   f'{model_name(M["cfg"]["fall_model"])}. Both Models Agree Closely on the Fall Score.')
    tid = trial_picker('why'); res = analyze_trial(tid)
    c1, c2 = st.columns([1, 4], gap='medium')
    go_ = c1.button('Play', type='primary', width='stretch', icon=':material/play_arrow:')
    w0 = c2.slider('Moment', 0, len(res['t_end']) - 1, int(np.argmax(res['p_fall'])), key='why_w')
    ph = st.empty()

    def frame(w, key):
        out = contributions(res['X'].iloc[[w]])
        with ph.container():
            p = res['p_fall'][w]; alarm = p >= res['thr']
            banner('Looks Like a Fall' if alarm else 'Looks Safe', 'alarm' if alarm else 'ok',
                   f'Fall Probability {p * 100:.0f}% at {res["t_end"][w]:.1f} s')
            gap(0.8)
            a, b = st.columns([1, 2], gap='large')
            with a:
                chart(body(res['posture'][min(res['starts'][w] + core.WIN - 1, res['n'] - 1)], 'fall' if alarm else 'safe', h=380),
                      key=f'wb{key}')
            with b:
                if out is not None:
                    s = out[0].iloc[0]; s = s.reindex(s.abs().sort_values(ascending=False).index)[:8][::-1]
                    fig = go.Figure(go.Bar(x=s.values, y=[feature_tick(f) for f in s.index], orientation='h',
                                           marker_color=[CORAL if v > 0 else TEAL for v in s.values]))
                    lim = max(8, float(np.abs(s.values).max()) * 1.1)
                    fig.update_xaxes(range=[-lim, lim], title='Contribution to the Fall Score')
                    fig.update_yaxes(automargin=True, tickfont=dict(size=13))
                    layout(fig, 470, 'Reasons behind This Decision', legend=False)
                    fig.add_annotation(xref='paper', yref='paper', x=0, y=1.0, yanchor='bottom', xanchor='left', showarrow=False,
                                       text='Toward Safe', font=dict(color=TEAL, size=14))
                    fig.add_annotation(xref='paper', yref='paper', x=1, y=1.0, yanchor='bottom', xanchor='right', showarrow=False,
                                       text='Toward Fall', font=dict(color=CORAL, size=14))
                    fig.update_layout(margin=dict(t=88))
                    chart(fig, key=f'wr{key}')
                else:
                    note('Explanation Model Not Included in This Bundle.')
            if out is not None:
                s_all = out[0].iloc[0]
                pos = s_all[s_all > 0].sort_values(ascending=False).head(4)
                neg = s_all[s_all < 0].sort_values().head(3)
                chips = lambda idx: ''.join(f'<span class="ss-feat"><b>{feature_words(f)}</b><div class="ss-fid">{f}</div></span>'
                                            for f in idx)
                st.markdown(f'<div class="ss-card"><div class="ss-k" style="color:{CORAL}">Toward a Fall</div>'
                            f'{chips(pos.index) or "Nothing Fall Like"}'
                            f'<div class="ss-k" style="color:{TEAL};margin-top:10px">Toward Safe</div>'
                            f'{chips(neg.index) or "Nothing Notable"}</div>', unsafe_allow_html=True)
                gap(0.6)
            chart(signal_fig(res, res['starts'][w] / core.FS, res['t_end'][w], h=280, title='The 3 Seconds behind This Decision'),
                  key=f'ws{key}')

    if go_:
        for w in range(len(res['t_end'])):
            frame(w, f'p{w}'); time.sleep(0.4)
    else:
        frame(w0, 's')
        out = contributions(res['X'].iloc[[w0]])
        if out is not None:
            s = out[0].iloc[0]; top = s.reindex(s.abs().sort_values(ascending=False).index)[:15]
            gap(0.6)
            fancy_table(pd.DataFrame({'Rank': range(1, len(top) + 1), 'Reason': [feature_html(f) for f in top.index],
                                      'Direction': ['Toward Fall' if v > 0 else 'Toward Safe' for v in top.values],
                                      'Contribution': top.values}),
                        'Top 15 Reasons at This Moment', fmt={'Contribution': '{:+.3f}'}, icon=':material/psychology:',
                        tone={'Direction': lambda v: 'ss-bad' if v == 'Toward Fall' else 'ss-good'}, html_cols=('Reason',))


# --------------------------------------------------------------------------------------------------
# 5.9  SafeStep Visual Library (hand built, animated SVG charts; every chart has its own entrance)
# --------------------------------------------------------------------------------------------------
import html as _h

VIZ_CSS = f"""
<style>
  .ss-viz {{ background: white; border: 1px solid #E1E7F0; border-radius: 18px; padding: 22px 24px 16px 24px; margin-bottom: 1.2rem; }}
  .ss-viz-t {{ font-family: {DISPLAY}; font-size: 1.32rem; font-weight: 600; color: {NAVY}; }}
  .ss-viz-s {{ font-size: 1.0rem; color: #5E6A82; margin: 4px 0 10px 0; line-height: 1.5; }}
  .ss-svg {{ width: 100%; height: auto; display: block; overflow: visible; }}
  .ss-svg text {{ font-family: {FONT}; fill: #4F5B73; }}
  .ss-svg .vt-b {{ fill: {INK}; font-weight: 700; }}
  .ss-svg .vt-s {{ fill: #8A94A6; }}
  .ss-svg .grid {{ stroke: #EEF1F6; stroke-width: 1; }}
  .ss-svg .axis {{ stroke: #CBD5E1; stroke-width: 1; }}
  .ss-leg {{ display: flex; flex-wrap: wrap; gap: 8px 26px; margin: 4px 0 6px 0; font-size: 0.98rem; color: #3B465E; }}
  .ss-leg i {{ display: inline-block; width: 22px; height: 4px; border-radius: 2px; vertical-align: middle; margin-right: 8px; }}
  .ss-leg i.dash {{ height: 0; border-top: 4px dashed; border-radius: 0; background: none !important; }}

  /* each chart owns a different entrance */
  @keyframes vz-spin   {{ from {{ opacity: 0; transform: rotate(-140deg) scale(0.7); }} }}
  @keyframes vz-rise   {{ from {{ transform: scaleY(0); }} }}
  @keyframes vz-draw   {{ from {{ stroke-dashoffset: 1; }} }}
  @keyframes vz-ping   {{ 0% {{ r: 5; opacity: 0.9; }} 100% {{ r: 22; opacity: 0; }} }}
  @keyframes vz-slide  {{ from {{ transform: translateX(var(--dx)); opacity: 0.2; }} }}
  @keyframes vz-stick  {{ from {{ transform: scaleX(0); }} }}
  @keyframes vz-pop    {{ 0% {{ transform: scale(0); }} 70% {{ transform: scale(1.35); }} 100% {{ transform: scale(1); }} }}
  @keyframes vz-chain  {{ from {{ transform: translateY(-26px); opacity: 0; }} }}
  @keyframes vz-fadeline {{ from {{ opacity: 0; }} }}
  @keyframes vz-grow   {{ from {{ transform: scaleX(0.02); }} }}
  @keyframes vz-radar  {{ from {{ transform: scale(0) rotate(-35deg); opacity: 0; }} }}
  @keyframes vz-blur   {{ from {{ filter: blur(7px); opacity: 0; }} }}
  @keyframes vz-run    {{ from {{ offset-distance: 0%; }} to {{ offset-distance: 100%; }} }}
  @keyframes vz-wave   {{ from {{ transform: scale(0.1); opacity: 0; }} }}
  @keyframes vz-blink  {{ 50% {{ opacity: 0.35; }} }}
  @keyframes vz-drop   {{ 0% {{ transform: translateY(-150px); opacity: 0; }} 60% {{ transform: translateY(6px); opacity: 1; }}
                         80% {{ transform: translateY(-3px); }} 100% {{ transform: none; }} }}
  @keyframes vz-gauge  {{ from {{ stroke-dashoffset: var(--len); }} }}
  @property --vznum {{ syntax: '<integer>'; initial-value: 0; inherits: false; }}
  @keyframes vz-count  {{ from {{ --vznum: 0; }} }}
  @keyframes vz-scan   {{ from {{ clip-path: inset(0 100% 0 0); }} to {{ clip-path: inset(0 0 0 0); }} }}
  @keyframes vz-spring {{ 0% {{ transform: scaleY(0); }} 65% {{ transform: scaleY(1.12); }} 85% {{ transform: scaleY(0.96); }} 100% {{ transform: scaleY(1); }} }}
  @keyframes vz-steps  {{ from {{ transform: scaleX(0); }} }}
  @keyframes vz-flip   {{ from {{ transform: perspective(700px) rotateY(85deg); opacity: 0; }} }}
  @keyframes vz-glow   {{ 50% {{ filter: drop-shadow(0 0 6px rgba(228,87,46,0.75)); }} }}

  .vz-spin   {{ transform-box: view-box; transform-origin: var(--ox) var(--oy); animation: vz-spin 1.1s cubic-bezier(.2,.8,.2,1) both; }}
  .vz-rise   {{ transform-box: fill-box; transform-origin: bottom; animation: vz-rise 0.7s ease-out both; animation-delay: var(--d); }}
  .vz-draw   {{ stroke-dasharray: 1; animation: vz-draw 1.8s ease-in-out both; animation-delay: var(--d); }}
  .vz-ping   {{ animation: vz-ping 1.6s ease-out infinite; }}
  .vz-slide  {{ animation: vz-slide 1.1s cubic-bezier(.3,.8,.3,1) both; animation-delay: var(--d); }}
  .vz-stick  {{ transform-box: fill-box; transform-origin: left; animation: vz-stick 0.7s ease-out both; animation-delay: var(--d); }}
  .vz-pop    {{ transform-box: fill-box; transform-origin: center; animation: vz-pop 0.5s ease-out both; animation-delay: var(--d); }}
  .vz-chain  {{ animation: vz-chain 0.45s cubic-bezier(.3,1.4,.5,1) both; animation-delay: var(--d); }}
  .vz-fadeline {{ animation: vz-fadeline 0.8s ease both; animation-delay: var(--d); }}
  .vz-grow   {{ transform-box: fill-box; transform-origin: left; animation: vz-grow 1.0s cubic-bezier(.2,.8,.2,1) both; animation-delay: var(--d); }}
  .vz-radar  {{ transform-box: view-box; transform-origin: var(--ox) var(--oy); animation: vz-radar 0.9s cubic-bezier(.2,.8,.2,1) both; animation-delay: var(--d); }}
  .vz-blur   {{ animation: vz-blur 1.0s ease-out both; animation-delay: var(--d); }}
  .vz-run    {{ animation: vz-run 4.5s linear infinite; animation-delay: var(--d); }}
  .vz-wave   {{ transform-box: fill-box; transform-origin: center; animation: vz-wave 0.5s ease-out both; animation-delay: var(--d); }}
  .vz-blink  {{ animation: vz-blink 1.8s ease-in-out infinite; }}
  .vz-drop   {{ animation: vz-drop 0.9s ease-out both; animation-delay: var(--d); }}
  .vz-gauge  {{ animation: vz-gauge 1.6s cubic-bezier(.2,.8,.2,1) both; animation-delay: var(--d); }}
  .vz-num    {{ --vznum: var(--to); counter-reset: vznum var(--vznum); animation: vz-count 1.6s cubic-bezier(.2,.8,.2,1) both;
                animation-delay: var(--d); font-variant-numeric: tabular-nums; }}
  .vz-num::before {{ content: counter(vznum); }}
  .vz-scan   {{ animation: vz-scan 1.4s ease-in-out both; }}
  .vz-spring {{ transform-box: fill-box; transform-origin: bottom; animation: vz-spring 1.0s ease-out both; animation-delay: var(--d); }}
  .vz-steps  {{ transform-box: fill-box; transform-origin: left; animation: vz-steps 1.0s steps(10, end) both; animation-delay: var(--d); }}
  .vz-flip   {{ animation: vz-flip 0.7s ease-out both; animation-delay: var(--d); }}
  .vz-glow   {{ animation: vz-glow 2.2s ease-in-out infinite; }}

  .ss-gauges {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; }}
  .ss-gauge {{ background: white; border: 1px solid #E1E7F0; border-radius: 18px; padding: 18px 18px 14px 18px; text-align: center; }}
  .ss-gauge-v {{ font-family: {DISPLAY}; font-size: 2.1rem; font-weight: 700; color: {INK}; margin-top: -54px; }}
  .ss-gauge-k {{ font-family: {DISPLAY}; font-size: 1.12rem; font-weight: 600; color: {NAVY}; margin-top: 8px; }}
  .ss-gauge-s {{ font-size: 0.9rem; color: #6B768C; margin-top: 4px; line-height: 1.45; }}
  .ss-mini {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px; }}
  .ss-mini-c {{ background: #FAFBFD; border: 1px solid #E5EAF2; border-radius: 14px; padding: 10px 12px; }}
  .ss-mini-t {{ font-weight: 600; color: {INK}; font-size: 0.95rem; }}
  .ss-mini-s {{ font-size: 0.82rem; color: #7D879B; }}
  @media (max-width: 900px) {{ .ss-gauges {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
  @media (prefers-reduced-motion: reduce) {{
    [class*="vz-"] {{ animation: none !important; }}
  }}
</style>
"""


def _e(s):
    return _h.escape(str(s))


def viz(title, sub, body_html, legend=None):
    leg = ''
    if legend:
        leg = '<div class="ss-leg">' + ''.join(
            f'<span><i class="{"dash" if dash else ""}" style="background:{c};border-color:{c}"></i>{_e(n)}</span>'
            for n, c, dash in [(l + (False,))[:3] for l in legend]) + '</div>'
    return f'<div class="ss-viz"><div class="ss-viz-t">{title}</div><div class="ss-viz-s">{sub}</div>{leg}{body_html}</div>'


def _svg(w, h, inner, label=''):
    return f'<svg class="ss-svg" viewBox="0 0 {w} {h}" role="img" aria-label="{_e(label)}">{inner}</svg>'


def _two_line(x, y, f, anchor='end', size=14):
    """Feature meaning in front, the technical name small underneath."""
    return (f'<text x="{x}" y="{y - 3}" text-anchor="{anchor}" font-size="{size}" class="vt-b">{_e(feature_words(f))}</text>'
            f'<text x="{x}" y="{y + size - 1}" text-anchor="{anchor}" font-size="{size - 4}" class="vt-s">{_e(f)}</text>')


# ---------------------------------------------------------------- 1 dataset composition (spin in)
def viz_dataset():
    T = trials()
    W, H, cx, cy = 640, 360, 190, 180
    outer = [('Daily Activities', int((T.type == 'ADL').sum()), NAVY), ('Falls', int((T.type == 'Fall').sum()), CORAL)]
    inner = [('Young People', int((T.age_group == 'young').sum()), TEAL), ('Elderly People', int((T.age_group == 'elderly').sum()), AMBER)]

    def ring(parts, r, w):
        C = 2 * np.pi * r; tot = sum(p[1] for p in parts); start = 0; out = ''
        for name, v, c in parts:
            L = C * v / tot
            out += (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{c}" stroke-width="{w}" '
                    f'stroke-dasharray="{L - 3:.1f} {C - L + 3:.1f}" stroke-dashoffset="{-start:.1f}" '
                    f'transform="rotate(-90 {cx} {cy})"><title>{_e(name)}: {v:,}</title></circle>')
            start += L
        return out
    g = (f'<g class="vz-spin" style="--ox:{cx}px;--oy:{cy}px">{ring(outer, 140, 34)}{ring(inner, 98, 26)}</g>'
         f'<text x="{cx}" y="{cy - 4}" text-anchor="middle" font-size="34" class="vt-b">{len(T):,}</text>'
         f'<text x="{cx}" y="{cy + 22}" text-anchor="middle" font-size="15">Recordings</text>')
    y = 70
    for name, v, c in outer + inner:
        share = v / len(T)
        g += (f'<rect x="380" y="{y - 14}" width="22" height="4" rx="2" fill="{c}" transform="translate(0 6)"/>'
              f'<text x="414" y="{y}" font-size="16" class="vt-b">{_e(name)}</text>'
              f'<text x="414" y="{y + 22}" font-size="14">{v:,} Recordings, {share:.0%}</text>')
        y += 64
    return viz('Who and What was Recorded', f'{T.subject.nunique()} Volunteers, {T.duration_s.sum() / 3600:.0f} Hours of Waist Motion. '
        'Outer Ring: Activity Type. Inner Ring: Age Group.', _svg(W, H, g, 'Dataset composition'))


# ---------------------------------------------------------------- 2 threshold problem (bars rise in a wave)
def viz_threshold():
    T = trials(); W, H, L, R, Tp, B = 640, 360, 56, 20, 20, 56
    bins = np.arange(0, 12.5, 0.5)
    pw = (W - L - R); ph = H - Tp - B
    series = [('ADL', NAVY, 'Daily Activities'), ('Fall', CORAL, 'Falls')]
    hist = {k: np.histogram(np.clip(T[T.type == k].peak_g.values, 0, 11.99), bins)[0] / max((T.type == k).sum(), 1) for k, _, _ in series}
    ymax = max(h.max() for h in hist.values()) * 1.1
    X = lambda v: L + v / 12 * pw; Y = lambda v: Tp + ph - v / ymax * ph
    g = ''.join(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                f'<text x="{L - 8}" y="{Y(v) + 5:.1f}" text-anchor="end" font-size="13">{v:.0%}</text>'
                for v in np.linspace(0, ymax / 1.1, 4))
    bw = pw / (len(bins) - 1)
    for k, c, nm in series:
        for i, v in enumerate(hist[k]):
            if v <= 0:
                continue
            g += (f'<rect class="vz-rise" style="--d:{i * 35}ms" x="{X(bins[i]) + 1:.1f}" y="{Y(v):.1f}" width="{bw - 2:.1f}" '
                  f'height="{Y(0) - Y(v):.1f}" fill="{c}" fill-opacity="0.62"><title>{nm}, {bins[i]:.1f} to {bins[i + 1]:.1f} g: {v:.1%}</title></rect>')
    g += ''.join(f'<text x="{X(v):.1f}" y="{H - B + 22}" text-anchor="middle" font-size="13">{v} g</text>' for v in range(0, 13, 2))
    g += f'<text x="{L + pw / 2}" y="{H - 10}" text-anchor="middle" font-size="14">Peak Acceleration of the Recording</text>'
    g += (f'<line x1="{X(3):.1f}" x2="{X(3):.1f}" y1="{Tp}" y2="{Y(0):.1f}" stroke="{INK}" stroke-width="2.5" stroke-dasharray="7 6"/>'
          f'<text x="{X(3) + 8:.1f}" y="{Tp + 16}" font-size="14" class="vt-b">Classic 3 g Threshold</text>')
    fall_low = (T[T.type == 'Fall'].peak_g < 3).mean(); adl_high = (T[T.type == 'ADL'].peak_g >= 3).mean()
    return viz('Why a Simple Threshold Falls Short', f'{fall_low:.0%} of Falls Stay below 3 g, while {adl_high:.0%} of Daily Activities Go above It.',
        _svg(W, H, g, 'Peak acceleration histogram'), [('Daily Activities', NAVY), ('Falls', CORAL), ('Classic 3 g Threshold', INK, True)])


# ---------------------------------------------------------------- 3 fall anatomy (lines draw, live impact ping)
@st.cache_resource(show_spinner=False)
def fall_anatomy_data():
    T = trials(); sig = _signals(); out = {}
    pre, post = 2 * core.FS, 3 * core.FS
    for d, g in T[T.type == 'Fall'].groupby('direction'):
        rows = []
        for _, r in g.iterrows():
            p = int(r.peak_idx)
            if p - pre < 0 or p + post > r.n_samples:
                continue
            x = sig[r.offset + p - pre: r.offset + p + post].astype(np.float32)
            rows.append(np.linalg.norm(x[:, :3], axis=1))
        if rows:
            out[d] = np.median(np.vstack(rows), axis=0)
    return out


def viz_anatomy():
    D = fall_anatomy_data(); W, H, L, R, Tp, B = 1000, 380, 60, 24, 24, 56
    cols = {'forward': SKY, 'backward': CORAL, 'lateral': TEAL, 'vertical': AMBER, 'assisted': PLUM}
    ymax = max(v.max() for v in D.values()) * 1.12
    n = len(next(iter(D.values()))); t = np.arange(n) / core.FS - 2
    X = lambda s: L + (s + 2) / 5 * (W - L - R); Y = lambda v: Tp + (1 - v / ymax) * (H - Tp - B)
    g = ''.join(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/><text x="{L - 8}" y="{Y(v) + 5:.1f}" '
                f'text-anchor="end" font-size="13">{v:.0f} g</text>' for v in range(0, int(ymax) + 1))
    g += ''.join(f'<text x="{X(s):.1f}" y="{H - B + 22}" text-anchor="middle" font-size="13">{f"{s:+d}" if s else "0"} s</text>' for s in range(-2, 4))
    g += f'<text x="{X(0.5):.1f}" y="{H - 8}" text-anchor="middle" font-size="14">Time Relative to Impact</text>'
    g += (f'<rect x="{X(-0.6):.1f}" y="{Tp}" width="{X(0) - X(-0.6):.1f}" height="{H - Tp - B}" fill="{AMBER}" fill-opacity="0.08"/>'
          f'<text x="{X(-0.3):.1f}" y="{Tp + 16}" text-anchor="middle" font-size="13" class="vt-b">Free Fall</text>'
          f'<rect x="{X(0.6):.1f}" y="{Tp}" width="{X(3) - X(0.6):.1f}" height="{H - Tp - B}" fill="{TEAL}" fill-opacity="0.05"/>'
          f'<text x="{X(1.8):.1f}" y="{Tp + 16}" text-anchor="middle" font-size="13" class="vt-b">Lying Still</text>')
    for i, (d, v) in enumerate(D.items()):
        pts = ' '.join(f'{X(a):.1f},{Y(b):.1f}' for a, b in zip(t[::2], v[::2]))
        g += (f'<polyline class="vz-draw" style="--d:{i * 250}ms" pathLength="1" points="{pts}" fill="none" stroke="{cols.get(d, GREY)}" '
              f'stroke-width="3" stroke-linejoin="round"><title>{_e(tc(d))} Falls</title></polyline>')
    peak = max(v.max() for v in D.values())
    g += (f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{Tp}" y2="{Y(0):.1f}" stroke="{INK}" stroke-width="1.5" stroke-dasharray="4 5"/>'
          f'<circle class="vz-ping" cx="{X(0):.1f}" cy="{Y(peak):.1f}" r="5" fill="none" stroke="{CORAL}" stroke-width="3"/>'
          f'<circle cx="{X(0):.1f}" cy="{Y(peak):.1f}" r="5" fill="{CORAL}"/>'
          f'<text x="{X(0) + 12:.1f}" y="{Y(peak) + 5:.1f}" font-size="14" class="vt-b">Impact</text>')
    return viz('Anatomy of a Fall', 'Median Acceleration of Every Recorded Fall, Aligned at the Moment of Impact. '
        'A Short Free Fall, a Sharp Impact, Then Stillness.', _svg(W, H, g, 'Fall anatomy'),
        [(tc(d) + ' Falls', cols.get(d, GREY)) for d in D])


# ---------------------------------------------------------------- 4 feature family ablation (dumbbells slide apart)
def viz_ablation():
    A = table('feature_ablation.csv'); A = A[A.kind == 'family']
    fam = A.pivot(index='name', columns='mode', values='activity_f1').sort_values('only')
    n_only = A[A['mode'] == 'only'].set_index('name').n
    W, L, R, rowh, Tp = 640, 190, 30, 50, 34; H = Tp + rowh * len(fam) + 50
    lo, hi = 0.4, 0.9
    X = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)
    g = ''.join(f'<line class="grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{Tp - 10}" y2="{H - 46}"/>'
                f'<text x="{X(v):.1f}" y="{H - 24}" text-anchor="middle" font-size="13">{v:.1f}</text>' for v in np.arange(0.4, 0.91, 0.1))
    g += f'<text x="{L + (W - L - R) / 2}" y="{H - 4}" text-anchor="middle" font-size="14">Activity Macro F1</text>'
    for i, (name, r) in enumerate(fam.iterrows()):
        y = Tp + i * rowh + rowh / 2; x1, x2 = X(r['only']), X(r['without'])
        g += (f'<text x="{L - 14}" y="{y - 2:.1f}" text-anchor="end" font-size="15" class="vt-b">{_e(FAMILY_NAMES.get(name, name))}</text>'
              f'<text x="{L - 14}" y="{y + 15:.1f}" text-anchor="end" font-size="12" class="vt-s">{int(n_only.get(name, 0))} Features</text>'
              f'<line class="vz-stick" style="--d:{i * 120 + 300}ms" x1="{x1:.1f}" x2="{x2:.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="#CBD5E1" stroke-width="6" stroke-linecap="round"/>'
              f'<circle cx="{x1:.1f}" cy="{y:.1f}" r="9" fill="{AMBER}"><title>Only This Family: {r["only"]:.3f}</title></circle>'
              f'<circle class="vz-slide" style="--d:{i * 120 + 300}ms;--dx:{x1 - x2:.1f}px" cx="{x2:.1f}" cy="{y:.1f}" r="9" fill="{NAVY}">'
              f'<title>Every Family except This One: {r["without"]:.3f}</title></circle>')
    return viz('No Single Feature Family is Enough', 'Activity Score Using One Family Alone versus Using Every Family except It. '
        'Families Work Best Together.', _svg(W, H, g, 'Family ablation'), [('This Family Alone', AMBER), ('All Others Together', NAVY)])


# ---------------------------------------------------------------- 5 top features by mutual information (lollipops)
def viz_top_mi():
    M = table('feature_mi_fall.csv'); M.columns = ['feature', 'mi']; M = M.head(10)
    W, L, R, rowh, Tp = 640, 300, 50, 46, 10; H = Tp + rowh * len(M) + 40
    X = lambda v: L + v / (M.mi.max() * 1.1) * (W - L - R)
    g = ''
    for i, r in enumerate(M.itertuples()):
        y = Tp + i * rowh + rowh / 2
        g += _two_line(L - 14, y, r.feature, 'end', 15)
        g += (f'<line class="vz-stick" style="--d:{i * 90}ms" x1="{L}" x2="{X(r.mi):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{TEAL}" stroke-width="4" stroke-linecap="round"/>'
              f'<circle class="vz-pop" style="--d:{i * 90 + 550}ms" cx="{X(r.mi):.1f}" cy="{y:.1f}" r="9" fill="{TEAL}"/>'
              f'<text x="{X(r.mi) + 16:.1f}" y="{y + 5:.1f}" font-size="13">{r.mi:.3f}</text>')
    g += f'<text x="{L + (W - L - R) / 2}" y="{H - 8}" text-anchor="middle" font-size="14">Mutual Information with Falls</text>'
    return viz('The Ten Most Telling Signals', 'How Much Each Feature Alone Reveals about a Fall. Peak Acceleration, Body Turning Speed and Drop Speed Lead.',
        _svg(W, H, g, 'Top features'))


# ---------------------------------------------------------------- 6 choosing the number of features (dots chain in)
def viz_kcurve():
    K = table('feature_k_curve.csv'); K = K[K.set.isin(['all', 'phone'])]
    ks = sorted(K.k.unique()); W, H, L, R, Tp, B = 640, 340, 56, 24, 30, 58
    lo, hi = K.activity_f1.min() - 0.02, K.activity_f1.max() + 0.02
    X = lambda k: L + ks.index(k) / (len(ks) - 1) * (W - L - R); Y = lambda v: Tp + (1 - (v - lo) / (hi - lo)) * (H - Tp - B)
    g = ''.join(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/><text x="{L - 8}" y="{Y(v) + 5:.1f}" '
                f'text-anchor="end" font-size="13">{v:.2f}</text>' for v in np.linspace(lo + 0.01, hi - 0.01, 4))
    g += ''.join(f'<text x="{X(k):.1f}" y="{H - B + 22}" text-anchor="middle" font-size="13">{k}</text>' for k in ks)
    g += f'<text x="{L + (W - L - R) / 2}" y="{H - 10}" text-anchor="middle" font-size="14">Number of Selected Features</text>'
    for j, (s, c) in enumerate([('all', NAVY), ('phone', CORAL)]):
        d = K[K.set == s].sort_values('k')
        pts = ' '.join(f'{X(k):.1f},{Y(v):.1f}' for k, v in zip(d.k, d.activity_f1))
        g += f'<polyline class="vz-fadeline" style="--d:{900 + j * 200}ms" points="{pts}" fill="none" stroke="{c}" stroke-width="3"/>'
        for i, (k, v) in enumerate(zip(d.k, d.activity_f1)):
            g += (f'<circle class="vz-chain" style="--d:{i * 110 + j * 60}ms" cx="{X(k):.1f}" cy="{Y(v):.1f}" r="6.5" fill="white" '
                  f'stroke="{c}" stroke-width="3"><title>{k} Features: {v:.3f}</title></circle>')
    sel = K[(K.set == 'phone') & (K.k == 120)]
    if len(sel):
        v = float(sel.activity_f1.iloc[0])
        g += (f'<circle cx="{X(120):.1f}" cy="{Y(v):.1f}" r="13" fill="none" stroke="{CORAL}" stroke-width="2" class="vz-glow"/>'
              f'<text x="{X(120):.1f}" y="{Y(v) + 34:.1f}" text-anchor="middle" font-size="13" class="vt-b">Chosen: 120</text>')
    return viz('How Many Features are Needed', 'Activity Score Climbs Quickly, Then Flattens. SafeStep Keeps 120 Phone Ready Features.',
        _svg(W, H, g, 'k curve'), [('Every Feature Type', NAVY), ('Phone Ready Features', CORAL)])


# ---------------------------------------------------------------- 7 model benchmark (bars grow, false alarm badges)
def viz_benchmark():
    A = table('all_models_arena.csv').copy(); base = table('baselines.csv')
    A['Model'] = A.model.map(model_name); A['fam'] = np.where(A.model.str.startswith('Deep'), 'Deep Learning', 'Classical')
    rows = A[['Model', 'fam', 'test_sens', 'test_fa_h']].rename(columns={'test_sens': 'sens', 'test_fa_h': 'fa'})
    if base is not None:
        B = base[base.split == 'test']
        rows = pd.concat([rows, pd.DataFrame({'Model': B.model.map(model_name), 'fam': 'Threshold Baseline',
                                              'sens': B.sensitivity.values, 'fa': B.fa_per_hour.values})])
    rows = rows.sort_values(['sens', 'fa'], ascending=[False, True]).reset_index(drop=True)
    deployed = model_name(models()['cfg']['fall_model'])
    cmap = {'Classical': NAVY, 'Deep Learning': PLUM, 'Threshold Baseline': GREY}
    W, L, R, rowh, Tp = 1000, 230, 170, 38, 30; H = Tp + rowh * len(rows) + 46
    lo = min(0.7, np.floor((rows.sens.min() - 0.05) * 10) / 10); X = lambda v: L + (v - lo) / (1 - lo) * (W - L - R)
    g = ''.join(f'<line class="grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{Tp - 8}" y2="{H - 40}"/>'
                f'<text x="{X(v):.1f}" y="{H - 20}" text-anchor="middle" font-size="13">{v:.0%}</text>' for v in np.arange(lo, 1.001, 0.1))
    g += f'<text x="{W - R + 86}" y="{Tp - 12}" text-anchor="middle" font-size="13" class="vt-b">False Alarms per Hour</text>'
    for i, r in rows.iterrows():
        y = Tp + i * rowh; me = r.Model == deployed; c = CORAL if me else cmap[r.fam]
        g += (f'<text x="{L - 12}" y="{y + rowh / 2 + 5:.1f}" text-anchor="end" font-size="15" class="{"vt-b" if me else ""}">'
              f'{_e(r.Model)}{" (Deployed)" if me else ""}</text>'
              f'<rect class="vz-grow" style="--d:{i * 70}ms" x="{L}" y="{y + 7}" width="{max(X(r.sens) - L, 2):.1f}" height="{rowh - 14}" rx="7" fill="{c}" '
              f'fill-opacity="{1 if me else 0.85}"><title>{_e(r.Model)}: {r.sens:.1%} of Falls Caught</title></rect>'
              + (f'<text x="{X(r.sens) - 10:.1f}" y="{y + rowh / 2 + 5:.1f}" text-anchor="end" font-size="13" style="fill:white;font-weight:700">{r.sens:.1%}</text>'
               if X(r.sens) - L > 80 else
               f'<text x="{X(r.sens) + 8:.1f}" y="{y + rowh / 2 + 5:.1f}" font-size="13" class="vt-b">{r.sens:.1%}</text>'))
        fa_col = TEAL if r.fa == 0 else (AMBER if r.fa < 2 else CORAL)
        g += (f'<rect x="{W - R + 36}" y="{y + 8}" width="100" height="{rowh - 16}" rx="11" fill="{fa_col}" fill-opacity="0.14"/>'
              f'<text x="{W - R + 86}" y="{y + rowh / 2 + 5:.1f}" text-anchor="middle" font-size="13.5" style="fill:{fa_col};font-weight:700">{r.fa:.2f}</text>')
    return viz('Every Model on Unseen People', 'Share of Falls Caught, with the False Alarm Rate beside Each Model. '
        'Learned Models Beat the Classic Threshold by a Wide Margin.', _svg(W, H, g, 'Benchmark'),
        [('SafeStep Deployed Model', CORAL), ('Classical Models', NAVY), ('Deep Learning', PLUM), ('Threshold Baseline', GREY)])


# ---------------------------------------------------------------- 8 model profile radar (polygons unfold)
def viz_radar():
    A = table('model_arena.csv').set_index('model')
    pick = [m for m in ['MLP', 'LightGBM', 'XGBoost', 'SVM-RBF'] if m in A.index]
    axes = [('Falls Caught', lambda r: (r.test_sens - 0.9) / 0.1), ('Few False Alarms', lambda r: 1 - min(r.test_fa_h, 1)),
            ('Activity F1', lambda r: (r.act_test_f1 - 0.75) / 0.15), ('Window PR AUC', lambda r: (r.win_prauc - 0.98) / 0.02),
            ('Speed', lambda r: 1 - min(np.log10(r.ms_per_1000_windows) / 3, 1))]
    W, H, cx, cy, Rr = 640, 420, 320, 215, 150
    cols = [CORAL, NAVY, TEAL, AMBER]
    ang = [-np.pi / 2 + 2 * np.pi * i / len(axes) for i in range(len(axes))]
    P = lambda a, v: (cx + Rr * v * np.cos(a), cy + Rr * v * np.sin(a))
    g = ''
    for lv in (0.25, 0.5, 0.75, 1.0):
        g += f'<polygon class="grid" fill="none" points="{" ".join(f"{P(a, lv)[0]:.1f},{P(a, lv)[1]:.1f}" for a in ang)}"/>'
    for a, (nm, _) in zip(ang, axes):
        x, y = P(a, 1.0); lx, ly = P(a, 1.12)
        anc = 'start' if np.cos(a) > 0.3 else 'end' if np.cos(a) < -0.3 else 'middle'
        dy = -8 if np.sin(a) < -0.9 else (18 if np.sin(a) > 0.3 else 5)
        g += (f'<line class="axis" x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}"/>'
              f'<text x="{lx:.1f}" y="{ly + dy:.1f}" text-anchor="{anc}" font-size="14" class="vt-b">{nm}</text>')
    for i, m in enumerate(pick):
        r = A.loc[m]; vals = [float(np.clip(f(r), 0.04, 1)) for _, f in axes]
        pts = ' '.join(f'{P(a, v)[0]:.1f},{P(a, v)[1]:.1f}' for a, v in zip(ang, vals))
        g += (f'<polygon class="vz-radar" style="--d:{i * 260}ms;--ox:{cx}px;--oy:{cy}px" points="{pts}" fill="{cols[i]}" '
              f'fill-opacity="0.12" stroke="{cols[i]}" stroke-width="3"><title>{_e(model_name(m))}</title></polygon>')
    return viz('Strength Profile of the Leading Models', 'Each Axis Scaled So the Outer Ring is Best. '
        'The Deployed MLP Balances Accuracy, Calm and Speed.', _svg(W, H, g, 'Radar'),
        [(model_name(m) + (' (Deployed)' if m == models()['cfg']['fall_model'] else ''), cols[i]) for i, m in enumerate(pick)])


# ---------------------------------------------------------------- 9 deep learning curves (blur in, live comets)
def viz_deep():
    hist = {n: table(f'deep_history_{k}.csv') for k, n in [('cnn', 'Deep CNN'), ('cnn_bilstm', 'Deep CNN BiLSTM'), ('hybrid', 'Deep Hybrid')]}
    hist = {k: v for k, v in hist.items() if v is not None}
    if not hist:
        return
    W, H, L, R, Tp, B = 640, 340, 56, 24, 24, 58
    emax = max(v.epoch.max() for v in hist.values()); lo = min(v.val_act_f1.min() for v in hist.values()) - 0.02; hi = 0.92
    X = lambda e: L + e / emax * (W - L - R); Y = lambda v: Tp + (1 - (v - lo) / (hi - lo)) * (H - Tp - B)
    g = ''.join(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/><text x="{L - 8}" y="{Y(v) + 5:.1f}" '
                f'text-anchor="end" font-size="13">{v:.1f}</text>' for v in np.arange(np.ceil(lo * 10) / 10, hi, 0.1))
    g += ''.join(f'<text x="{X(e):.1f}" y="{H - B + 22}" text-anchor="middle" font-size="13">{e}</text>'
                 for e in range(0, int(emax) + 1, max(int(emax) // 6, 1)))
    g += f'<text x="{L + (W - L - R) / 2}" y="{H - 10}" text-anchor="middle" font-size="14">Training Epoch</text>'
    cols = [SKY, PLUM, TEAL]
    for i, (n, d) in enumerate(hist.items()):
        path = 'M' + ' L'.join(f'{X(e):.1f},{Y(v):.1f}' for e, v in zip(d.epoch, d.val_act_f1))
        g += (f'<path class="vz-blur" style="--d:{i * 200}ms" d="{path}" fill="none" stroke="{cols[i]}" stroke-width="3"/>'
              f'<circle class="vz-run" style="offset-path:path(\'{path}\');--d:{i * 600}ms" r="6" fill="{cols[i]}" stroke="white" stroke-width="2"/>')
    return viz('Deep Networks Learning Live', 'Validation Activity Score per Epoch. The Moving Dots Retrace Each Network as It Trains.',
        _svg(W, H, g, 'Deep curves'), [(n, cols[i]) for i, n in enumerate(hist)])


# ---------------------------------------------------------------- 10 every test fall (waffle wave, missed cells blink)
def viz_waffle():
    R_ = table('final_stream_per_recording.csv'); F = R_[R_.type == 'Fall'].sort_values(['code', 'tid'])
    names = trials().drop_duplicates('code').set_index('code')['name']
    codes = sorted(F.code.unique()); ncol = int(F.groupby('code').size().max())
    L, cell, gapc, Tp = 330, 19, 3, 10; W = L + ncol * (cell + gapc) + 80; H = Tp + len(codes) * (cell + 6) + 10
    g = ''
    for i, c in enumerate(codes):
        y = Tp + i * (cell + 6); d = F[F.code == c]
        g += (f'<text x="{L - 14}" y="{y + 14}" text-anchor="end" font-size="14" class="vt-b">{_e(tc(names.get(c, c)))}</text>'
              f'<text x="{L + ncol * (cell + gapc) + 12}" y="{y + 14}" font-size="13">{int(d.detected.sum())} of {len(d)}</text>')
        for j, hit in enumerate(d.detected.values):
            g += (f'<rect class="vz-wave{"" if hit else " vz-blink"}" style="--d:{(i + j) * 22}ms" x="{L + j * (cell + gapc)}" y="{y}" '
                  f'width="{cell}" height="{cell}" rx="5" fill="{TEAL if hit else CORAL}"><title>{c}: {"Caught" if hit else "Missed"}</title></rect>')
    caught = int(F.detected.sum())
    return viz(f'All {len(F)} Test Falls, One Square Each', f'{caught} of {len(F)} Falls Caught on People SafeStep Never Saw '
        'during Training, across All Fifteen Fall Types.',
        _svg(W, H, g, 'Waffle'), [('Caught', TEAL), ('Missed', CORAL)])


# ---------------------------------------------------------------- 11 pre impact lead (dots drop in like falling bodies)
def viz_preimpact():
    P = table('preimpact_lead.csv'); ex = (jload('extra_results.json') or {}).get('preimpact', {})
    warned = P.dropna(subset=['lead_ms']); W, H, L, R, Tp, B = 1000, 300, 40, 30, 40, 56
    vmax = max(1500, warned.lead_ms.max())
    X = lambda v: L + v / vmax * (W - L - R)
    rng = np.random.default_rng(3)
    g = ''.join(f'<line class="grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{Tp}" y2="{H - B}"/>'
                f'<text x="{X(v):.1f}" y="{H - B + 22}" text-anchor="middle" font-size="13">{v} ms</text>' for v in range(0, int(vmax) + 1, 250))
    g += f'<text x="{L + (W - L - R) / 2}" y="{H - 10}" text-anchor="middle" font-size="14">Warning Time before Impact</text>'
    g += (f'<rect x="{X(0):.1f}" y="{Tp}" width="{X(150) - X(0):.1f}" height="{H - B - Tp}" fill="{AMBER}" fill-opacity="0.12"/>'
          f'<text x="{X(0):.1f}" y="{Tp - 10}" text-anchor="start" font-size="12" class="vt-b">Airbag Needs about 150 ms</text>')
    for i, (v, code) in enumerate(zip(warned.lead_ms.values, warned.code.values)):
        y = Tp + 12 + rng.uniform(0, H - B - Tp - 24)
        g += (f'<circle class="vz-drop" style="--d:{(i % 70) * 18}ms" cx="{X(v):.1f}" cy="{y:.1f}" r="5" fill="{PLUM}" fill-opacity="0.65">'
              f'<title>{code}: {v:.0f} ms</title></circle>')
    med = float(warned.lead_ms.median())
    g += (f'<line x1="{X(med):.1f}" x2="{X(med):.1f}" y1="{Tp}" y2="{H - B}" stroke="{CORAL}" stroke-width="3"/>'
          f'<text x="{X(med) + 8:.1f}" y="{Tp + 14}" font-size="14" class="vt-b">Median {med:.0f} ms</text>')
    return viz('Warning Arrives before the Body Lands', f'{ex.get("test_warned", len(warned) / len(P)):.0%} of Test Falls were Flagged during the Fall Itself. '
        'Each Dot is One Fall.', _svg(W, H, g, 'Pre impact'), [('One Warned Fall', PLUM), ('Median Lead Time', CORAL)])


# ---------------------------------------------------------------- 12 extended capabilities (gauges sweep, numbers count)
def viz_gauges():
    ex = jload('extra_results.json') or {}
    items = []
    if 'direction' in ex:
        items.append(('Fall Direction', ex['direction']['test_trial_acc'], 'Forward, Backward, Lateral, Vertical or Assisted', SKY))
    if 'context' in ex:
        items.append(('Fall Situation', ex['context']['test_trial_acc'], 'Walking, Fainting, Getting Up, Sitting Down or Seated', PLUM))
    if 'preimpact' in ex:
        items.append(('Pre Impact Warning', ex['preimpact']['test_warned'], f"Median {ex['preimpact']['test_median_lead_ms']:.0f} ms before Impact", AMBER))
    if 'long_lie' in ex:
        items.append(('Long Lie Confirmed', ex['long_lie']['confirmed_share'], f"Median {ex['long_lie']['median_confirm_s']:.1f} s after Impact", CORAL))
    r = 70; C = np.pi * r
    cells = ''
    for i, (k, v, s, c) in enumerate(items):
        L = C * v
        cells += (f'<div class="ss-gauge"><svg viewBox="0 0 180 104" style="width:100%;max-width:230px">'
                  f'<path d="M20,95 A{r},{r} 0 0 1 160,95" fill="none" stroke="#EEF1F6" stroke-width="16" stroke-linecap="round"/>'
                  f'<path class="vz-gauge" style="--len:{C:.1f};--d:{i * 180}ms" d="M20,95 A{r},{r} 0 0 1 160,95" fill="none" stroke="{c}" '
                  f'stroke-width="16" stroke-linecap="round" stroke-dasharray="{L:.1f} {C:.1f}" stroke-dashoffset="0"/></svg>'
                  f'<div class="ss-gauge-v"><span class="vz-num" style="--to:{int(round(v * 100))};--d:{i * 180}ms"></span>%</div>'
                  f'<div class="ss-gauge-k">{k}</div><div class="ss-gauge-s">{s}</div></div>')
    return f'<div class="ss-gauges">{cells}</div>'


# ---------------------------------------------------------------- 13 robustness (scan reveal)
def viz_robustness(kind):
    rb = table('robustness.csv'); d = rb[rb.perturbation == kind].copy(); d['Model'] = d.model.map(model_name)
    W, H, L, R, Tp, B = 1000, 380, 64, 30, 52, 60
    levels = sorted(d.level.unique(), reverse=(kind == 'rate'))
    X = lambda lv: L + 40 + (levels.index(lv) / max(len(levels) - 1, 1) if len(levels) > 1 else 0.5) * (W - L - R - 80)
    Y = lambda v: Tp + (1 - v) * (H - Tp - B)
    g = ''.join(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/><text x="{L - 8}" y="{Y(v) + 5:.1f}" '
                f'text-anchor="end" font-size="13">{v:.0%}</text>' for v in np.arange(0, 1.01, 0.25))
    unit = {'noise': ' g', 'rate': ' Hz', 'rotation': '°', 'scale': 'x', 'gyro_off': ''}[kind]
    g += ''.join(f'<text x="{X(lv):.1f}" y="{H - B + 22}" text-anchor="middle" font-size="13">'
                 f'{"Gyroscope Off" if kind == "gyro_off" else f"{lv:g}{unit}"}</text>' for lv in levels)
    g += f'<text x="{L + (W - L - R) / 2}" y="{H - 10}" text-anchor="middle" font-size="14">{PERTURB_TITLES[kind][1]}</text>'
    series = []
    cols = {('SafeStep (Phone Ready Features)', 'sensitivity'): CORAL, ('SafeStep (Phone Ready Features)', 'act_f1'): TEAL,
            ('Device Axis Features', 'sensitivity'): GREY, ('Device Axis Features', 'act_f1'): '#B4BCCB'}
    inner = ''
    for mdl, gm in d.groupby('Model'):
        gm = gm.set_index('level').loc[levels]
        for col, nm in [('sensitivity', 'Falls Caught'), ('act_f1', 'Activity F1')]:
            if gm[col].isna().all():
                continue
            c = cols.get((mdl, col), NAVY); series.append((f'{mdl}: {nm}', c))
            pts = ' '.join(f'{X(lv):.1f},{Y(v):.1f}' for lv, v in gm[col].items())
            inner += f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="3.5" stroke-linejoin="round"/>'
            for lv, v in gm[col].items():
                inner += (f'<circle cx="{X(lv):.1f}" cy="{Y(v):.1f}" r="7" fill="white" stroke="{c}" stroke-width="3">'
                          f'<title>{_e(mdl)}, {nm}: {v:.1%}</title></circle>')
            if col == 'sensitivity':
                for lv, fa in gm['fa_per_hour'].items():
                    inner += (f'<text x="{X(lv):.1f}" y="{Y(gm.loc[lv, col]) - 14:.1f}" text-anchor="middle" font-size="12" '
                              f'style="fill:{c};font-weight:700">{fa:.1f} FA/h</text>')
    g += f'<g class="vz-scan">{inner}</g>'
    return _svg(W, H, g, 'Robustness'), series


# ---------------------------------------------------------------- 14 young versus elderly (spring bars)
def viz_age():
    ca = table('cross_age.csv').copy()
    ca[['Metric', 'Age']] = ca.metric.str.split(' · ', expand=True); ca = ca[ca.Metric == 'falls caught']
    trains = [('young + elderly (SafeStep)', 'Young and Elderly (SafeStep)', NAVY), ('young only', 'Young Only', AMBER)]
    W, H, L, R, Tp, B = 640, 340, 56, 20, 30, 54
    lo = 0.6; Y = lambda v: Tp + (1 - (v - lo) / (1 - lo)) * (H - Tp - B)
    g = ''.join(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/><text x="{L - 8}" y="{Y(v) + 5:.1f}" '
                f'text-anchor="end" font-size="13">{v:.0%}</text>' for v in np.arange(0.6, 1.01, 0.1))
    groups = [('young', 'Young People'), ('elderly', 'Elderly People')]; gw = (W - L - R) / 2; bw = 90
    for gi, (age, lab) in enumerate(groups):
        cx0 = L + gi * gw + gw / 2
        g += f'<text x="{cx0:.1f}" y="{H - B + 26}" text-anchor="middle" font-size="15" class="vt-b">{lab}</text>'
        for ti, (key, nm, c) in enumerate(trains):
            v = float(ca[(ca.training == key) & (ca.Age == age)].value.iloc[0])
            x = cx0 + (ti - 1) * (bw + 10) + 5
            g += (f'<rect class="vz-spring" style="--d:{gi * 300 + ti * 150}ms" x="{x:.1f}" y="{Y(v):.1f}" width="{bw}" height="{Y(lo) - Y(v):.1f}" '
                  f'rx="8" fill="{c}"><title>{nm}, {lab}: {v:.1%}</title></rect>'
                  f'<text x="{x + bw / 2:.1f}" y="{Y(v) - 10:.1f}" text-anchor="middle" font-size="14" class="vt-b">{v:.1%}</text>')
    return viz('Elderly Daily Life Makes the Difference', 'Falls Caught by Age Group. Adding Elderly Daily Activity to Training Helps on Elderly Falls.',
        _svg(W, H, g, 'Age'), [(nm, c) for _, nm, c in trains])


# ---------------------------------------------------------------- 15 global feature importance (stepped bars)
def viz_shap():
    S = table('shap_global.csv').head(12)
    W, L, R, rowh, Tp = 1000, 380, 80, 44, 10; H = Tp + rowh * len(S) + 44
    X = lambda v: L + v / (S.mean_abs_shap.max() * 1.05) * (W - L - R)
    fam_col = {'fall_dynamics': CORAL, 'statistical': NAVY, 'spectral': PLUM, 'temporal': TEAL, 'cross_signal': AMBER, 'device_axis': GREY}
    g = ''
    for i, r in enumerate(S.itertuples()):
        y = Tp + i * rowh + rowh / 2; c = fam_col.get(r.group, NAVY)
        g += _two_line(L - 16, y, r.feature, 'end', 15)
        g += (f'<rect class="vz-steps" style="--d:{i * 80}ms" x="{L}" y="{y - 12:.1f}" width="{X(r.mean_abs_shap) - L:.1f}" height="24" rx="6" fill="{c}">'
              f'<title>{_e(feature_words(r.feature))}: {r.mean_abs_shap:.2f}</title></rect>'
              f'<text x="{X(r.mean_abs_shap) + 10:.1f}" y="{y + 5:.1f}" font-size="13">{r.mean_abs_shap:.2f}</text>')
    g += f'<text x="{L + (W - L - R) / 2}" y="{H - 8}" text-anchor="middle" font-size="14">Average Push on the Fall Decision (Mean Absolute SHAP)</text>'
    fams = [f for f in fam_col if f in set(S.group)]
    return viz('What SafeStep Looks at Most', 'The Twelve Features with the Strongest Influence on Fall Decisions, Coloured by Family.',
        _svg(W, H, g, 'SHAP'), [(FAMILY_NAMES[f], fam_col[f]) for f in fams])




PERTURB_TITLES = {'noise': ('Sensor Noise', 'Added Noise Level'), 'rate': ('Sampling Rate', 'Phone Sampling Rate'),
                  'gyro_off': ('Gyroscope Off', 'Gyroscope'), 'rotation': ('Phone Rotation', 'Phone Rotated by'),
                  'scale': ('Calibration Gain', 'Sensor Gain')}


@st.cache_resource(show_spinner='Drawing the Research Charts')
def results_html():
    """Every research chart is drawn once per app start, then served instantly."""
    out = {}
    for name in ['dataset', 'threshold', 'anatomy', 'top_mi', 'ablation', 'kcurve', 'benchmark', 'radar', 'deep', 'waffle',
                 'gauges', 'preimpact', 'age', 'shap']:
        try:
            out[name] = globals()[f'viz_{name}']() or ''
        except Exception:
            out[name] = ''
    return out


@st.cache_resource(show_spinner=False, max_entries=8)
def robustness_svg(kind):
    return viz_robustness(kind)


def show(html_text):
    if html_text:
        st.markdown(html_text, unsafe_allow_html=True)


def page_results():
    st.markdown(VIZ_CSS, unsafe_allow_html=True)
    header('Research Results', 'The Full Study behind SafeStep, from Raw Motion to a Trusted Alarm.')
    H = results_html()

    section(1, 'The Data')
    a, b = st.columns(2, gap='large')
    with a:
        show(H['dataset'])
    with b:
        show(H['threshold'])
    show(H['anatomy'])

    section(2, 'The Features')
    a, b = st.columns(2, gap='large')
    with a:
        show(H['top_mi'])
    with b:
        show(H['ablation'])
    show(H['kcurve'])

    section(3, 'The Models')
    show(H['benchmark'])
    a, b = st.columns(2, gap='large')
    with a:
        show(H['radar'])
    with b:
        show(H['deep'])
    show(H['waffle'])
    arena = table('all_models_arena.csv')
    if arena is not None:
        A = arena.copy(); A['Model'] = A.model.map(model_name)
        A['Family'] = np.where(A.model.str.startswith('Deep'), 'Deep Learning', 'Classical')
        bench = A[['Model', 'Family', 'test_sens', 'test_fa_h', 'test_delay_s', 'win_prauc', 'act_test_f1', 'ms_per_1000_windows']]
        bench.columns = ['Model', 'Family', 'Falls Caught', 'False Alarms per Hour', 'Median Delay (s)', 'Window PR AUC',
                        'Activity Macro F1', 'Inference (ms per 1000 Windows)']
        fancy_table(bench.sort_values('Falls Caught', ascending=False), 'Full Benchmark Table', icon=':material/table_chart:',
                    fmt={'Falls Caught': '{:.1%}', 'False Alarms per Hour': '{:.2f}', 'Median Delay (s)': '{:.2f}',
                         'Window PR AUC': '{:.4f}', 'Activity Macro F1': '{:.3f}', 'Inference (ms per 1000 Windows)': '{:.0f}'})

    section(4, 'Beyond Detection')
    show(H['gauges'])
    gap(1.2)
    show(H['preimpact'])

    section(5, 'Ready for the Real World')
    rb = table('robustness.csv')
    if rb is not None:
        kinds = [k for k in PERTURB_TITLES if k in set(rb.perturbation)]
        choice = st.segmented_control('Condition', [PERTURB_TITLES[k][0] for k in kinds], default=PERTURB_TITLES[kinds[0]][0],
                                      key='rb_kind', label_visibility='collapsed') or PERTURB_TITLES[kinds[0]][0]
        kind = next(k for k in kinds if PERTURB_TITLES[k][0] == choice)
        svg, series = robustness_svg(kind)
        show(viz(f'Stress Test: {PERTURB_TITLES[kind][0]}', 'Falls Caught and Activity Score as Conditions Get Harder. '
                 'Labels Show False Alarms per Hour (FA/h).', svg, series))
        d = rb[rb.perturbation == kind].copy(); d['Model'] = d.model.map(model_name)
        tbl = d[['Model', 'level', 'sensitivity', 'fa_per_hour', 'act_f1']].rename(
            columns={'level': PERTURB_TITLES[kind][1], 'sensitivity': 'Falls Caught', 'fa_per_hour': 'False Alarms per Hour',
                     'act_f1': 'Activity Macro F1'})
        fancy_table(tbl, f'{PERTURB_TITLES[kind][0]} Results Table', icon=':material/tune:',
                    fmt={'Falls Caught': '{:.1%}', 'False Alarms per Hour': '{:.2f}', 'Activity Macro F1': '{:.3f}',
                         PERTURB_TITLES[kind][1]: '{:g}'})
    gap(1.0)
    show(H['age'])

    section(6, 'Inside the Decision')
    show(H['shap'])


def page_setup_missing():
    header('Model Bundle Not Found', 'SafeStep Requires Its Model Bundle to Start.')
    z = locate_bundle_zip()
    note(f'<b>Expected Location:</b> {DEFAULT_BUNDLE}<br>'
         + (f'<b>Archive Found:</b> {z}<br>Run <b>setup.bat</b> Once to Extract It Automatically.'
            if z else f'Place <b>SafeStep_bundle.zip</b> or the <b>app_bundle</b> Folder inside {PROJECT_DIR}, '
                      f'Then Run <b>setup.bat</b>.'))


# ==================================================================================================
# 7  Navigation and Sidebar
# ==================================================================================================
style()
try:
    st.logo(str(ASSETS / 'logo.svg'), size='large', icon_image=str(ASSETS / 'mark.svg'))
except Exception:
    pass

if BUNDLE is None:
    page_setup_missing()
    st.stop()

load_core(BUNDLE)

PAGES = {
    'home': st.Page(page_home, title='Home', icon=':material/home:', url_path='home', default=True),
    'dash': st.Page(page_dashboard, title='Dashboard', icon=':material/dashboard:', url_path='dashboard'),
    'live': st.Page(page_live, title='Live Phone Monitor', icon=':material/sensors:', url_path='live-phone'),
    'sim': st.Page(page_simulator, title='Fall Simulator', icon=':material/directions_walk:', url_path='simulator'),
    'challenge': st.Page(page_challenge, title='Beat the AI', icon=':material/psychology:', url_path='challenge'),
    'editor': st.Page(page_editor, title='Fall Editor', icon=':material/tune:', url_path='editor'),
    'engine': st.Page(page_engine, title='Feature Engine', icon=':material/memory:', url_path='features'),
    'lab': st.Page(page_lab, title='Model Lab', icon=':material/science:', url_path='model-lab'),
    'why': st.Page(page_why, title='Decision Explainer', icon=':material/insights:', url_path='explainer'),
    'results': st.Page(page_results, title='Research Results', icon=':material/analytics:', url_path='results'),
}
nav = st.navigation({
    'Start': [PAGES['home'], PAGES['dash']],
    'Live Demonstrations': [PAGES['live'], PAGES['sim'], PAGES['challenge'], PAGES['editor']],
    'Inside SafeStep': [PAGES['engine'], PAGES['lab'], PAGES['why'], PAGES['results']],
}, position='sidebar', expanded=True)


SIDE_ICONS = {
    'models': ('#2ECC71', '<path d="M5 12l4 4 10-10"/>'),
    'rec': ('#F2A541', '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>'),
    'people': ('#60A5FA', '<circle cx="9" cy="8" r="4"/><path d="M2 21c0-3.9 3.1-7 7-7s7 3.1 7 7"/><path d="M16 4a4 4 0 0 1 0 8"/><path d="M19 14c1.9 1.2 3 3.4 3 7"/>'),
    'feat': ('#C084FC', '<path d="M12 2l10 5-10 5L2 7z"/><path d="M2 12l10 5 10-5"/><path d="M2 17l10 5 10-5"/>'),
    'fall': ('#FB7A55', '<path d="M12 2l8 4v6c0 5-3.5 8.5-8 10-4.5-1.5-8-5-8-10V6z"/><path d="M12 8v5"/><circle cx="12" cy="16.5" r="0.6"/>'),
    'act': ('#5EEAD4', '<path d="M2 12h4l3-8 4 16 3-8h6"/>'),
}


def side_item(key, text):
    c, path = SIDE_ICONS[key]
    return (f'<div class="ss-si"><span class="ss-si-i" style="background:{c}26;border-color:{c}66">'
            f'<svg viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">{path}</svg>'
            f'</span><span>{text}</span></div>')


with st.sidebar:
    try:
        _T = trials(); _cfg = models()['cfg']
        st.markdown(f"""
        <div class="ss-side">
          <div class="ss-side-h">System Status</div>
          <div class="ss-si" style="margin-top:10px"><span class="ss-si-i ss-si-live"><span class="ss-dot"></span></span><span>Models Ready</span></div>
          {side_item('rec', f"{len(_T):,} Recordings")}
          {side_item('people', f"{_T.subject.nunique()} People")}
          {side_item('feat', f"{len(core.FEATURE_NAMES)} Features per Window")}
          {side_item('fall', f"Fall Model: {model_name(_cfg['fall_model'])}")}
          {side_item('act', f"Activity Model: {_cfg['activity_model']}")}
        </div>
        <div class="ss-side-foot">{AUTHOR}<br>{INSTITUTE}</div>""", unsafe_allow_html=True)
    except Exception as e:                                       # keep navigation usable even if a file is damaged
        st.error(f'Model Bundle Not Loaded. Run setup.bat Again. Details: {e}')

try:
    warm_up()
except Exception:
    pass

nav.run()
