"""
How to download and load IBL (International Brain Laboratory) data
using the Open Neurophysiology Environment (ONE) API.

Installation (run once in terminal):
    pip install ONE-api

IBL public data server:
    base_url = 'https://openalyx.internationalbrainlab.org'
    password = 'international'
"""
import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import sklearn
import DateTime
import statsmodels.api as stat
import matplotlib.pyplot as plt

# =============================================================================
# 1. SETUP — Connect to IBL public data server
# =============================================================================
from one.api import ONE

# One-time setup (do this once, or whenever you need to reconfigure)
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)

# Connect — use password='international' for public data
one = ONE(password='international')

# =============================================================================
# 2. SEARCH — Find experiments / sessions of interest
# =============================================================================

# See what search terms are available
print(one.search_terms())

# Search for sessions by date range, required datasets, subject, lab, etc.
eids, info = one.search(
    date_range=['2020-08-01', '2020-08-31'],
    datasets='probes.description.json',  # sessions that have ephys probes
    details=True
)
print(f"Found {len(eids)} sessions:")
for eid in eids:
    print(f"  {eid}")

# Search by subject
eids = one.search(subject='KS023')

# Search by lab
eids = one.search(lab='steinmetzlab')

# Search by project
eids = one.search(project='brainwide')

# Search for sessions with a specific tag (data release)
eids = one.search(tag='2021_Q1_IBL_et_al_Behaviour')

# Search by multiple criteria — sessions with two probes in a project
eids = one.search(data=['probe00', 'probe01'], project='brainwide')

# =============================================================================
# 3. GET SESSION DETAILS
# =============================================================================
eid = eids[0]

# Get metadata about a session
session_details = one.eid2details(eid)
print(session_details.keys())  # subject, start_time, lab, task_protocol, etc.

# =============================================================================
# 4. LIST — See what datasets are available for a session
# =============================================================================

# List all datasets in the 'alf' collection (preprocessed data)
alf_datasets = one.list_datasets(eid, collection='alf')
print(alf_datasets)

# List datasets for a specific probe
probe_insertions = one.load_dataset(eid, 'probes.description')
probe_label = probe_insertions[0]['label']
probe_datasets = one.list_datasets(eid, collection=f'alf/{probe_label}*')
print(probe_datasets)

# =============================================================================
# 5. LOAD — Download & load specific datasets into memory
# =============================================================================

# --- Load a single dataset ---
probes = one.load_dataset(eid, 'probes.description')  # probe metadata

# --- Load an ALF object (all datasets sharing a name, e.g. all camera data) ---
cam = one.load_object(eid, 'leftCamera', collection='alf')
print(cam.keys())  # e.g. ['times', 'dlc', 'features']
print(cam['times'])       # camera frame timestamps
print(cam['dlc'].shape)   # DeepLabCut tracking points

# Load only specific attributes of an object
cam_times_dlc = one.load_object(
    eid, 'leftCamera',
    collection='alf',
    attribute=['times', 'dlc']
)

# --- Load spike-sorted data ---
spikes = one.load_object(eid, 'spikes', collection=f'alf/{probe_label}*')
# spikes['times'], spikes['clusters'], spikes['amps'], etc.

# --- Load behavioral / trials data ---
trials = one.load_object(eid, 'trials', collection='alf')
# trials['choice'], trials['contrastLeft'], trials['contrastRight'],
# trials['feedbackType'], trials['probabilityLeft'], trials['reactionTime'], etc.

# --- Load wheel data ---
wheel = one.load_object(eid, 'wheel', collection='alf')
# wheel['times'], wheel['position'], wheel['velocity'] (computed)

# --- Load clusters brain regions ---
clusters = one.load_object(eid, 'clusters', collection=f'alf/{probe_label}*')
# clusters['brainAcronyms'], clusters['channels'], etc.

# --- Load raw ephys data (specific chunks) ---
# one.load_dataset(eid, 'ephysData.raw.npy', ...)

# =============================================================================
# 6. DOWNLOAD ONLY (without loading into memory)
# =============================================================================

# Download datasets to local cache without loading — returns file paths
dataset_paths = one.load_datasets(
    eid,
    datasets=['trials.intervals.npy', 'trials.choice.npy', 'wheel.times.npy'],
    download_only=True
)
print(dataset_paths)

# =============================================================================
# 7. QUALITY CONTROL — Load only QC-passed data
# =============================================================================
dsets = one.list_datasets(eid, qc='WARNING', ignore_qc_not_set=True)
data, info = one.load_datasets(eid, dsets)

# =============================================================================
# 8. CACHE MANAGEMENT — Refresh the cache of all available data
# =============================================================================
one.load_cache()  # download fresh cache tables (refresh every 6 hours automatically)

# =============================================================================
# 9. WORKING OFFLINE — After downloading, you can work offline
# =============================================================================
# from one.api import One
# one_offline = One(cache_dir='/path/to/cache')
# one_offline.list_datasets(eid, collection='alf')

# =============================================================================
# 10. CONVENIENT BATCH LOADING MULTIPLE SESSIONS
# =============================================================================
import numpy as np

eids = one.search(date_range=['2020-08-01', '2020-08-31'],
                   datasets='probes.description.json')

all_trials = []
for eid in eids:
    try:
        trials = one.load_object(eid, 'trials', collection='alf')
        all_trials.append(trials)
    except Exception as e:
        print(f"Failed to load {eid}: {e}")

print(f"Loaded trials from {len(all_trials)} sessions")


# =============================================================================
# 11. FIND & SEPARATE VISION-RELATED VS SENSORY-RELATED NEURAL DATA
# =============================================================================
"""
IBL uses Neuropixels probes recording across the whole mouse brain.
We separate neurons (clusters) by their Allen Brain Atlas region:

VISION-RELATED regions (visual cortex + visual thalamus):
  - Visual cortical areas: VISp, VISl, VISal, VISam, VISpm, VISrl, VISli, VISpor, VISpl, VISa
  - Visual thalamus: LGd, LP, LD
  - Superior colliculus (visual layers): SCs, SCop, SCiw (superficial layers)

SOMATOSENSORY-RELATED regions:
  - Primary somatosensory: SSp-bfd, SSp-ll, SSp-m, SSp-n, SSp-tr, SSp-ul, SSp-un, SSp
  - Supplemental somatosensory: SSs
  - Somatosensory thalamus: VPM, VPL, PO, VPLpc, VPMpc
  - Whisker-related: any SSp-* barrel field subregions

We will:
  1) Search for all sessions with ephys probes
  2) For each session, load cluster brain region labels
  3) Classify each cluster as vision, somatosensory, or other
  4) Separate spike times & cluster IDs by modality
  5) Load trial-level visual stimulus info (contrastLeft/contrastRight)
"""

from collections import defaultdict
from pprint import pprint

# ---------------------------------------------------------------------------
# 11a. Define brain-region acronyms for each modality
# ---------------------------------------------------------------------------

VISION_ACRONYMS = {
    # Primary visual cortex & higher visual areas (Allen CCF)
    'VISp', 'VISl', 'VISal', 'VISam', 'VISpm', 'VISrl', 'VISli', 'VISpor',
    'VISpl', 'VISa', 'VIS',
    # Visual thalamus
    'LGd', 'LGv', 'LP', 'LD',
    # Superior colliculus (superficial / visual layers)
    'SCs', 'SCop', 'SCiw',
    # Pretectum / accessory optic
    'APN', 'NOT', 'OP',
}

SOMATOSENSORY_ACRONYMS = {
    # Primary somatosensory cortex (barrel, limb, mouth, nose, trunk, etc.)
    'SSp-bfd', 'SSp-ll', 'SSp-m', 'SSp-n', 'SSp-tr', 'SSp-ul', 'SSp-un',
    'SSp',
    # Supplemental somatosensory
    'SSs',
    # Somatosensory thalamus
    'VPM', 'VPL', 'PO', 'VPLpc', 'VPMpc',
    # Trigeminal / brainstem
    'SPVI', 'SPVC', 'SPVO', 'PSV', 'PRNc',
}

# ---------------------------------------------------------------------------
# 11b. Search for all ephys sessions (using a broad tag or project)
# ---------------------------------------------------------------------------
# Brain-wide map — the flagship IBL dataset with neuropixels across the brain
eids_ephys = one.search(
    project='brainwide',
    datasets='spikes.times.npy',
    query_type='remote'
)
print(f"\n{'='*70}")
print(f"Total ephys sessions found: {len(eids_ephys)}")
print(f"{'='*70}")

# ---------------------------------------------------------------------------
# 11c. For each session, classify clusters & separate spikes by modality
# ---------------------------------------------------------------------------
results = []

for eid in eids_ephys[:5]:  # ← limit to 5 for demo; remove [:5] for all
    print(f"\n--- Processing {eid} ---")

    # Get session metadata
    details = one.eid2details(eid)
    subject = details.get('subject', 'unknown')
    date = details.get('start_time', 'unknown')[:10]
    print(f"  Subject: {subject}, Date: {date}")

    # Load probe insertions to find probe labels
    try:
        probe_info = one.load_dataset(eid, 'probes.description')
    except Exception:
        print("  Skipping — no probe info")
        continue

    session_data = {
        'eid': eid,
        'subject': subject,
        'date': date,
        'probes': {}
    }

    for probe in probe_info:
        p_label = probe['label']
        print(f"  Probe: {p_label}")

        try:
            # Load cluster metadata (brain regions)
            clu = one.load_object(
                eid, 'clusters',
                collection=f'alf/{p_label}*',
                attribute=['acronyms', 'atlas_id']
            )
            # Load spike data
            spk = one.load_object(
                eid, 'spikes',
                collection=f'alf/{p_label}*',
                attribute=['times', 'clusters', 'amps']
            )
        except Exception as exc:
            print(f"    Skipping — {exc}")
            continue

        acronyms = clu['acronyms']        # array of strings, one per cluster
        spike_times = spk['times']        # spike times in seconds
        spike_clusters = spk['clusters']  # cluster ID for each spike

        n_clusters = len(acronyms)
        n_spikes = len(spike_times)

        # Classify each cluster
        cluster_modality = {}  # cluster_id -> 'vision' | 'somatosensory' | 'other'
        n_vision = 0
        n_somato = 0
        n_other = 0

        for clu_id in range(n_clusters):
            region = acronyms[clu_id]
            if region in VISION_ACRONYMS:
                cluster_modality[clu_id] = 'vision'
                n_vision += 1
            elif region in SOMATOSENSORY_ACRONYMS:
                cluster_modality[clu_id] = 'somatosensory'
                n_somato += 1
            else:
                cluster_modality[clu_id] = 'other'
                n_other += 1

        print(f"    Clusters — Vision: {n_vision}, "
              f"Somatosensory: {n_somato}, Other: {n_other}")

        # --- Separate spikes by modality ---
        # Build boolean mask: True for spikes that belong to vision clusters
        vision_mask = np.isin(spike_clusters,
                              [c for c, m in cluster_modality.items()
                               if m == 'vision'])

        somato_mask = np.isin(spike_clusters,
                              [c for c, m in cluster_modality.items()
                               if m == 'somatosensory'])

        # Vision-related spike data
        vision_spikes = {
            'times': spike_times[vision_mask],
            'clusters': spike_clusters[vision_mask],
        }
        if 'amps' in spk:
            vision_spikes['amps'] = spk['amps'][vision_mask]

        # Somatosensory-related spike data
        somato_spikes = {
            'times': spike_times[somato_mask],
            'clusters': spike_clusters[somato_mask],
        }
        if 'amps' in spk:
            somato_spikes['amps'] = spk['amps'][somato_mask]

        # --- Per-region breakdown ---
        region_counts = defaultdict(int)
        for clu_id in range(n_clusters):
            if cluster_modality[clu_id] != 'other':
                region_counts[acronyms[clu_id]] += 1

        print(f"    Vision regions: "
              f"{ {k:v for k,v in region_counts.items() if k in VISION_ACRONYMS} }")
        print(f"    Somato regions: "
              f"{ {k:v for k,v in region_counts.items() if k in SOMATOSENSORY_ACRONYMS} }")

        # Store
        session_data['probes'][p_label] = {
            'cluster_modality': cluster_modality,
            'acronyms': acronyms,
            'vision_spikes': vision_spikes,
            'somato_spikes': somato_spikes,
            'region_counts': dict(region_counts),
        }

    # -----------------------------------------------------------------------
    # 11d. Load trial-level visual-stimulus data for this session
    # -----------------------------------------------------------------------
    try:
        trials = one.load_object(
            eid, 'trials', collection='alf',
            attribute=['contrastLeft', 'contrastRight', 'choice',
                       'feedbackType', 'probabilityLeft', 'reactionTime',
                       'intervals']
        )
        session_data['trials'] = {
            'contrastLeft': trials['contrastLeft'],
            'contrastRight': trials['contrastRight'],
            'choice': trials['choice'],
            'feedbackType': trials['feedbackType'],
            'probabilityLeft': trials['probabilityLeft'],
            'reactionTime': trials['reactionTime'],
            'intervals': trials['intervals'],
            'n_trials': len(trials['choice']),
        }
        print(f"  Trials loaded: {session_data['trials']['n_trials']}")

        # Identify high-contrast vision trials vs low/no-contrast trials
        has_visual_stim = (
            (np.abs(trials['contrastLeft']) > 0) |
            (np.abs(trials['contrastRight']) > 0)
        )
        print(f"  Trials with visual stimulus: {has_visual_stim.sum()}")
        print(f"  Trials without visual stimulus: {(~has_visual_stim).sum()}")

        session_data['trials']['visual_stim_mask'] = has_visual_stim

    except Exception as exc:
        print(f"  No trial data — {exc}")

    # -----------------------------------------------------------------------
    # 11e. Load passive visual stimulus data if available
    # -----------------------------------------------------------------------
    try:
        # Passive replay — visual gratings replayed while mouse is passive
        passive_vis = one.load_object(
            eid, 'passiveVisual', collection='alf',
            attribute=['times', 'contrast']
        )
        session_data['passive_visual'] = {
            'times': passive_vis['times'],
            'contrast': passive_vis['contrast'],
        }
        print(f"  Passive visual data loaded: "
              f"{len(passive_vis['times'])} frames")
    except Exception:
        pass  # Not all sessions have passive replay

    # -----------------------------------------------------------------------
    # 11f. Load receptive-field mapping data if available
    # -----------------------------------------------------------------------
    try:
        rf = one.load_object(
            eid, 'passiveRFMapped', collection='alf',
            attribute=['times', 'contrast', 'posXY']
        )
        session_data['receptive_field_map'] = {
            'times': rf['times'],
            'contrast': rf['contrast'],
            'posXY': rf['posXY'],
        }
        print(f"  Receptive-field mapping data found "
              f"({len(rf['times'])} stimuli)")
    except Exception:
        pass

    results.append(session_data)

# =============================================================================
# 11g. SUMMARY — Aggregate vision vs somatosensory across all sessions
# =============================================================================
print(f"\n{'='*70}")
print("CROSS-SESSION SUMMARY")
print(f"{'='*70}")

total_vision_clusters = 0
total_somato_clusters = 0
total_vision_spikes = 0
total_somato_spikes = 0
all_vision_regions = defaultdict(int)
all_somato_regions = defaultdict(int)

for sess in results:
    for p_label, pdata in sess['probes'].items():
        n_vis = sum(1 for m in pdata['cluster_modality'].values()
                     if m == 'vision')
        n_som = sum(1 for m in pdata['cluster_modality'].values()
                      if m == 'somatosensory')
        total_vision_clusters += n_vis
        total_somato_clusters += n_som

        total_vision_spikes += len(pdata['vision_spikes']['times'])
        total_somato_spikes += len(pdata['somato_spikes']['times'])

        for region, count in pdata['region_counts'].items():
            if region in VISION_ACRONYMS:
                all_vision_regions[region] += count
            elif region in SOMATOSENSORY_ACRONYMS:
                all_somato_regions[region] += count

print(f"\nSessions processed: {len(results)}")
print(f"\nTotal vision clusters: {total_vision_clusters}")
print(f"  By region: {dict(sorted(all_vision_regions.items()))}")
print(f"\nTotal somatosensory clusters: {total_somato_clusters}")
print(f"  By region: {dict(sorted(all_somato_regions.items()))}")
print(f"\nTotal vision spikes: {total_vision_spikes:,}")
print(f"Total somatosensory spikes: {total_somato_spikes:,}")

# =============================================================================
# 11h. ACCESS EXAMPLE — Use the separated data for further analysis
# =============================================================================
if results:
    first = results[0]
    # e.g. get all vision spike times from the first probe of the first session
    for p_label, pdata in first['probes'].items():
        if len(pdata['vision_spikes']['times']) > 0:
            print(f"\nExample — {p_label} vision spikes shape: "
                  f"{pdata['vision_spikes']['times'].shape}")
        break

    # e.g. align vision spikes to trial visual-stimulus onsets
    if 'trials' in first:
        trial_starts = first['trials']['intervals'][:, 0]  # trial onset times
        # You can now epoch vision_spikes['times'] around trial_starts
        # for trials where contrastLeft or contrastRight > 0 (visual stimulus)
        print(f"Example — {len(trial_starts)} trial onsets available "
              f"for peri-stimulus time histogram (PSTH) analysis")


# =============================================================================
# 12. PRELIMINARY ANALYSIS & VISUALIZATION
# =============================================================================
"""
Visualizations for the vision- vs somatosensory-separated data:

  12a. Matplotlib setup
  12b. Basic spike statistics (firing rate, count per region)
  12c. Firing-rate distributions: vision vs somatosensory
  12d. Peri-stimulus time histograms (PSTH) aligned to stimulus onset
  12e. Spike raster plots (vision vs somato, stimulus-aligned)
  12f. Visual contrast tuning curves (vision neurons)
  12g. Cross-session region composition (stacked bar)
"""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'savefig.dpi': 150,
})

# Colors for the two modalities (consistent everywhere)
COLOR_VISION = '#1f77b4'   # blue
COLOR_SOMATO = '#d62728'   # red
COLOR_OTHER = '#7f7f7f'    # gray

# ---------------------------------------------------------------------------
# 12a. Helper: flatten all clusters/spikes across sessions by modality
# ---------------------------------------------------------------------------
def gather_spikes_by_modality(results):
    """Return dict with lists of per-cluster spike times for each modality."""
    vision = []   # list of arrays (one per cluster)
    somato = []
    for sess in results:
        for p_label, pdata in sess['probes'].items():
            acronyms = pdata['acronyms']
            modality = pdata['cluster_modality']
            # Build mapping cluster_id -> modality string
            for clu_id, mod in modality.items():
                if mod == 'other':
                    continue
                region = acronyms[clu_id]
                # Extract this cluster's spikes
                clu_spikes = pdata['vision_spikes'] if mod == 'vision' \
                    else pdata['somato_spikes']
                mask = clu_spikes['clusters'] == clu_id
                times = clu_spikes['times'][mask]
                if mod == 'vision':
                    vision.append({'region': region, 'times': times})
                else:
                    somato.append({'region': region, 'times': times})
    return vision, somato


vision_clusters, somato_clusters = gather_spikes_by_modality(results)

# ---------------------------------------------------------------------------
# 12b. Basic spike statistics
# ---------------------------------------------------------------------------
def firing_rate(times, t_start=None, t_end=None):
    """Firing rate (Hz) given spike times. Session window inferred if None."""
    if len(times) == 0:
        return 0.0
    t_start = times.min() if t_start is None else t_start
    t_end = times.max() if t_end is None else t_end
    duration = t_end - t_start
    if duration <= 0:
        return 0.0
    return len(times) / duration


vision_rates = [firing_rate(c['times']) for c in vision_clusters]
somato_rates = [firing_rate(c['times']) for c in somato_clusters]

print("\n" + "=" * 70)
print("FIRING-RATE SUMMARY")
print("=" * 70)
print(f"Vision clusters:      {len(vision_clusters):4d}  "
      f"median rate = {np.median(vision_rates):6.2f} Hz")
print(f"Somatosensory clusters: {len(somato_clusters):3d}  "
      f"median rate = {np.median(somato_rates):6.2f} Hz")

# --- Spike-count table per region ---
region_summary = defaultdict(lambda: {'vision': 0, 'somato': 0})
for c in vision_clusters:
    region_summary[c['region']]['vision'] += 1
for c in somato_clusters:
    region_summary[c['region']]['somato'] += 1

print("\nRegion | Vision clusters | Somato clusters")
print("-" * 45)
for region in sorted(region_summary):
    v = region_summary[region]['vision']
    s = region_summary[region]['somato']
    print(f"{region:>6} | {v:>16d} | {s:>15d}")

# ---------------------------------------------------------------------------
# 12c. Firing-rate distributions
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Histogram overlay
ax = axes[0]
bins = np.linspace(0, max(np.percentile(vision_rates + somato_rates, 95), 1), 40)
ax.hist(vision_rates, bins=bins, color=COLOR_VISION, alpha=0.6,
        label=f'Vision (n={len(vision_rates)})')
ax.hist(somato_rates, bins=bins, color=COLOR_SOMATO, alpha=0.6,
        label=f'Somatosensory (n={len(somato_rates)})')
ax.set_xlabel('Firing rate (Hz)')
ax.set_ylabel('Number of clusters')
ax.set_title('Firing-rate distribution')
ax.legend()

# Box plot
ax = axes[1]
box_data = [vision_rates, somato_rates]
bp = ax.boxplot(box_data, tick_labels=['Vision', 'Somatosensory'],
                patch_artist=True)
for patch, color in zip(bp['boxes'], [COLOR_VISION, COLOR_SOMATO]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_ylabel('Firing rate (Hz)')
ax.set_title('Firing-rate comparison')

# Mann-Whitney U test if both groups non-empty
if len(vision_rates) > 0 and len(somato_rates) > 0:
    u, p = scipy_stats.mannwhitneyu(vision_rates, somato_rates,
                                     alternative='two-sided')
    ax.set_xlabel(f'Mann-Whitney U p = {p:.3g}')

plt.tight_layout()
plt.savefig('fig12c_firing_rates.png', bbox_inches='tight')
plt.close(fig)
print("\nSaved: fig12c_firing_rates.png")

# ---------------------------------------------------------------------------
# 12d. PSTH aligned to visual-stimulus onset
# ---------------------------------------------------------------------------
def compute_psth(spike_times, event_times, t_pre=0.2, t_post=1.0, bin=0.05):
    """
    Return (bin_centers, psth_hz) for spikes aligned to event_times.
    """
    bins = np.arange(-t_pre, t_post + bin, bin)
    counts, _ = np.histogram(spike_times, bins=bins)  # placeholder
    # Vectorized: count spikes in each trial window
    n_events = len(event_times)
    counts_per_trial = []
    for t0 in event_times:
        rel = spike_times - t0
        in_win = rel[(rel >= -t_pre) & (rel < t_post)]
        c, _ = np.histogram(in_win, bins=bins)
        counts_per_trial.append(c)
    counts_per_trial = np.array(counts_per_trial)
    mean_counts = counts_per_trial.mean(axis=0)
    psth_hz = mean_counts / bin  # spikes/sec
    bin_centers = bins[:-1] + bin / 2
    return bin_centers, psth_hz, counts_per_trial


# Use the first session that has trial data
psth_fig, psth_ax = plt.subplots(figsize=(11, 5))
plotted = False

for sess in results:
    if 'trials' not in sess:
        continue
    trial_starts = sess['trials']['intervals'][:, 0]
    # Only use trials WITH visual stimulus
    if 'visual_stim_mask' in sess['trials']:
        stim_mask = sess['trials']['visual_stim_mask']
        vis_trial_starts = trial_starts[stim_mask]
    else:
        vis_trial_starts = trial_starts

    # Average vision PSTH and somato PSTH for this session
    all_vis_times = np.concatenate(
        [pdata['vision_spikes']['times']
         for pdata in sess['probes'].values()]
    )
    all_som_times = np.concatenate(
        [pdata['somato_spikes']['times']
         for pdata in sess['probes'].values()]
    )
    vis_center, vis_psth, _ = compute_psth(all_vis_times, vis_trial_starts)
    som_center, som_psth, _ = compute_psth(all_som_times, vis_trial_starts)

    psth_ax.plot(vis_center, vis_psth, color=COLOR_VISION, lw=2,
                 label='Vision (all clusters)')
    psth_ax.plot(som_center, som_psth, color=COLOR_SOMATO, lw=2,
                 label='Somatosensory (all clusters)')
    plotted = True
    break  # just first session for now

psth_ax.axvline(0, color='k', ls='--', lw=1, label='Stimulus onset')
psth_ax.set_xlabel('Time from stimulus onset (s)')
psth_ax.set_ylabel('Firing rate (Hz)')
psth_ax.set_title('PSTH aligned to visual-stimulus onset (first session)')
psth_ax.legend()
plt.tight_layout()
plt.savefig('fig12d_psth.png', bbox_inches='tight')
plt.close(psth_fig)
print("Saved: fig12d_psth.png")

# ---------------------------------------------------------------------------
# 12e. Raster plot — a few example neurons
# ---------------------------------------------------------------------------
def plot_raster(ax, spike_times_by_cluster, event_times, color, t_pre=0.2,
                t_post=1.0, max_clusters=20):
    """Plot raster of spikes aligned to event_times."""
    for i, clu_times in enumerate(spike_times_by_cluster[:max_clusters]):
        for t0 in event_times:
            rel = clu_times - t0
            in_win = rel[(rel >= -t_pre) & (rel < t_post)]
            if len(in_win) > 0:
                ax.plot(in_win, np.full_like(in_win, i), '|', color=color,
                        markersize=3, markeredgewidth=1)
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.set_xlim(-t_pre, t_post)
    ax.set_xlabel('Time from stimulus onset (s)')
    ax.set_ylabel('Cluster index')


if results:
    sess = results[0]
    if 'trials' in sess:
        trial_starts = sess['trials']['intervals'][:, 0]
        stim_mask = sess['trials'].get('visual_stim_mask',
                                       np.ones(len(trial_starts), dtype=bool))
        vis_starts = trial_starts[stim_mask]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                                 sharex=True, sharey=False)

        # Gather per-cluster times
        vis_times = []
        som_times = []
        for pdata in sess['probes'].values():
            for clu_id in pdata['cluster_modality']:
                mod = pdata['cluster_modality'][clu_id]
                if mod == 'vision':
                    m = pdata['vision_spikes']['clusters'] == clu_id
                    vis_times.append(pdata['vision_spikes']['times'][m])
                elif mod == 'somatosensory':
                    m = pdata['somato_spikes']['clusters'] == clu_id
                    som_times.append(pdata['somato_spikes']['times'][m])

        plot_raster(axes[0], vis_times, vis_starts, COLOR_VISION)
        axes[0].set_title(f'Vision clusters ({len(vis_times)})')
        plot_raster(axes[1], som_times, vis_starts, COLOR_SOMATO)
        axes[1].set_title(f'Somatosensory clusters ({len(som_times)})')

        plt.suptitle('Raster aligned to visual-stimulus onset (first session)')
        plt.tight_layout()
        plt.savefig('fig12e_raster.png', bbox_inches='tight')
        plt.close(fig)
        print("Saved: fig12e_raster.png")

# ---------------------------------------------------------------------------
# 12f. Visual contrast tuning (vision neurons)
# ---------------------------------------------------------------------------
if results:
    sess = results[0]
    if 'trials' in sess:
        trials = sess['trials']
        contrast = np.abs(trials['contrastLeft']) + \
                   np.abs(trials['contrastRight'])  # total contrast
        trial_starts = trials['intervals'][:, 0]

        # Bin trials by contrast level
        contrast_levels = np.unique(contrast[contrast > 0])
        if len(contrast_levels) >= 3:
            fig, ax = plt.subplots(figsize=(8, 5))
            all_vis_times = np.concatenate(
                [pdata['vision_spikes']['times']
                 for pdata in sess['probes'].values()]
            )
            means = []
            for c in contrast_levels:
                sel = contrast == c
                c_starts = trial_starts[sel]
                if len(c_starts) == 0:
                    continue
                _, psth, per_trial = compute_psth(all_vis_times, c_starts,
                                                   t_pre=0.0, t_post=0.5,
                                                   bin=0.05)
                means.append(psth.mean())

            ax.plot(contrast_levels[:len(means)], means, 'o-',
                    color=COLOR_VISION, lw=2, markersize=6)
            ax.set_xlabel('Stimulus contrast')
            ax.set_ylabel('Mean firing rate 0-500ms (Hz)')
            ax.set_title('Contrast-response function (vision neurons)')
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig('fig12f_contrast_tuning.png', bbox_inches='tight')
            plt.close(fig)
            print("Saved: fig12f_contrast_tuning.png")
        else:
            print("Skipped fig12f — not enough distinct contrast levels")

# ---------------------------------------------------------------------------
# 12g. Cross-session region composition (stacked bar)
# ---------------------------------------------------------------------------
if region_summary:
    fig, ax = plt.subplots(figsize=(10, 5))
    regions = sorted(region_summary)
    vision_counts = [region_summary[r]['vision'] for r in regions]
    somato_counts = [region_summary[r]['somato'] for r in regions]

    x = np.arange(len(regions))
    ax.bar(x, vision_counts, color=COLOR_VISION, label='Vision', alpha=0.8)
    ax.bar(x, somato_counts, bottom=vision_counts, color=COLOR_SOMATO,
           label='Somatosensory', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=45, ha='right')
    ax.set_ylabel('Number of clusters')
    ax.set_title('Cluster composition by brain region (all sessions)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('fig12g_region_composition.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig12g_region_composition.png")

print("\n" + "=" * 70)
print("ANALYSIS & VISUALIZATION COMPLETE")
print("Generated figures: fig12c_firing_rates.png, fig12d_psth.png, "
      "fig12e_raster.png, fig12f_contrast_tuning.png, "
      "fig12g_region_composition.png")


# =============================================================================
# 13. ML DECODING — Predict visual stimuli from neural activity
# =============================================================================
"""
Neural decoding: can we predict what the mouse saw from the neural data?

TASKS:
  Task A — Stimulus detection (binary):
      "Was a visual grating present on this trial?" → y ∈ {0, 1}

  Task B — Stimulus side (3-class):
      "Was the grating on the LEFT, RIGHT, or NEITHER side?" → y ∈ {-1, 0, +1}

  Task C — Contrast level (regression):
      Predict the actual contrast value from population activity → y ∈ ℝ

FEATURES:
  Spike-count vector per trial: count spikes in [50 ms, 550 ms] after
  stimulus onset for each neuron → X ∈ (n_trials, n_neurons)

MODELS COMPARED:
  - Logistic Regression (L2-regularized)
  - Linear SVM
  - Random Forest
  - Gradient Boosting (sklearn)
  - Multi-Layer Perceptron (2 hidden layers)

EVALUATION:
  - 5-fold Stratified Cross-Validation
  - Accuracy, F1, ROC-AUC (classification)
  - R², MAE (regression)
  - Permutation test (shuffle labels) → p-value against chance
  - Vision-only vs Somatosensory-only vs Combined feature sets
"""

# sklearn imports (install with: pip install scikit-learn)
from sklearn.model_selection import StratifiedKFold, cross_val_score, \
    cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, \
    RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, r2_score, mean_absolute_error,
                              classification_report, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ---------------------------------------------------------------------------
# 13a. Build trial-by-trial feature matrix from spike data
# ---------------------------------------------------------------------------
def build_feature_matrix(session_data, modality='vision',
                         t_latency=0.05, t_window=0.50):
    """
    Build (n_trials, n_neurons) spike-count matrix.

    Parameters
    ----------
    session_data : dict  — one element of `results`
    modality : str       — 'vision', 'somato', or 'combined'
    t_latency : float    — seconds after trial onset to start counting
    t_window : float     — seconds of spike-count window

    Returns
    -------
    X : np.ndarray (n_trials, n_neurons)
    neuron_regions : list[str] — brain region of each column in X
    """
    if 'trials' not in session_data:
        raise ValueError("Session has no trial data")

    trial_starts = session_data['trials']['intervals'][:, 0]
    n_trials = len(trial_starts)

    # Collect per-neuron spike times
    neuron_times = []   # list of (spike_times_array, region_name)
    for p_label, pdata in session_data['probes'].items():
        acronyms = pdata['acronyms']
        for clu_id, mod in pdata['cluster_modality'].items():
            if modality == 'vision' and mod != 'vision':
                continue
            if modality == 'somato' and mod != 'somatosensory':
                continue
            if mod == 'other':
                continue
            # Get this cluster's spike times
            src = pdata['vision_spikes'] if mod == 'vision' \
                else pdata['somato_spikes']
            mask = src['clusters'] == clu_id
            times = src['times'][mask]
            neuron_times.append((times, acronyms[clu_id]))

    if len(neuron_times) == 0:
        raise ValueError(f"No {modality} neurons found")

    n_neurons = len(neuron_times)
    X = np.zeros((n_trials, n_neurons))
    neuron_regions = []

    for j, (spk_times, region) in enumerate(neuron_times):
        neuron_regions.append(region)
        for i, t0 in enumerate(trial_starts):
            lo = t0 + t_latency
            hi = lo + t_window
            X[i, j] = np.sum((spk_times >= lo) & (spk_times < hi))

    return X, neuron_regions


# ---------------------------------------------------------------------------
# 13b. Define prediction targets
# ---------------------------------------------------------------------------
def build_targets(session_data):
    """
    Build label vectors from trial data.

    Returns dict with keys:
      - 'detection'  : binary (0 = no stimulus, 1 = stimulus present)
      - 'side'       : -1 left, 0 no stim, +1 right
      - 'side_binary': 0 left, 1 right (only stimulus trials)
      - 'contrast'   : float total contrast
    """
    trials = session_data['trials']
    cL = trials['contrastLeft']
    cR = trials['contrastRight']

    detection = ((np.abs(cL) > 0) | (np.abs(cR) > 0)).astype(int)

    side = np.zeros(len(cL), dtype=int)
    side[np.abs(cL) > 0] = -1
    side[np.abs(cR) > 0] = +1

    contrast = np.abs(cL) + np.abs(cR)

    # Binary side classification (left vs right, stim trials only)
    stim_idx = detection == 1
    side_binary_full = np.full(len(cL), -1, dtype=int)
    side_binary_full[side == 1] = 1  # only meaningful on stim trials

    return {
        'detection': detection,
        'side': side,
        'side_binary': side_binary_full,
        'contrast': contrast,
        'stim_idx': stim_idx,
    }


# ---------------------------------------------------------------------------
# 13c. Define models to compare
# ---------------------------------------------------------------------------
def get_classifiers():
    """Return dict of (name, Pipeline) for classification."""
    return {
        'LogisticReg (L2)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=5000, C=1.0,
                                        class_weight='balanced')),
        ]),
        'Linear SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='linear', C=1.0, probability=True,
                        class_weight='balanced')),
        ]),
        'Random Forest': Pipeline([
            ('clf', RandomForestClassifier(n_estimators=200, max_depth=10,
                                            class_weight='balanced',
                                            random_state=42)),
        ]),
        'Gradient Boosting': Pipeline([
            ('clf', GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                random_state=42)),
        ]),
        'MLP (2-layer)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(hidden_layer_sizes=(64, 32),
                                   max_iter=2000, early_stopping=True,
                                   random_state=42)),
        ]),
    }


def get_regressors():
    """Return dict of (name, Pipeline) for regression."""
    return {
        'Random Forest': Pipeline([
            ('reg', RandomForestRegressor(n_estimators=200, max_depth=10,
                                           random_state=42)),
        ]),
        'Gradient Boosting': Pipeline([
            ('reg', GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                random_state=42)),
        ]),
        'MLP (2-layer)': Pipeline([
            ('scaler', StandardScaler()),
            ('reg', MLPRegressor(hidden_layer_sizes=(64, 32),
                                  max_iter=2000, early_stopping=True,
                                  random_state=42)),
        ]),
    }


# ---------------------------------------------------------------------------
# 13d. Evaluation with cross-validation + permutation test
# ---------------------------------------------------------------------------
def evaluate_classifier(name, pipeline, X, y, cv=5, n_permutations=200):
    """
    Stratified K-fold CV for a classifier.
    Returns metrics dict + permutation p-value.
    """
    try:
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True,
                                       random_state=42)
        scoring = {'acc': 'accuracy', 'f1': 'f1_weighted',
                   'roc_auc': 'roc_auc_ovr_weighted'}
        scores = cross_validate(pipeline, X, y, cv=cv_splitter,
                                scoring=scoring, n_jobs=-1)
        result = {
            'accuracy': scores['test_acc'].mean(),
            'f1': scores['test_f1'].mean(),
            'roc_auc': scores['test_roc_auc'].mean(),
            'acc_std': scores['test_acc'].std(),
            'f1_std': scores['test_f1'].std(),
        }
    except Exception as exc:
        return {'accuracy': np.nan, 'f1': np.nan, 'roc_auc': np.nan,
                'error': str(exc)[:80]}

    # Permutation test — shuffle labels to estimate chance level
    try:
        perm_accs = []
        for _ in range(n_permutations):
            y_shuf = np.random.permutation(y)
            perm_acc = cross_val_score(pipeline, X, y_shuf,
                                       cv=StratifiedKFold(
                                           n_splits=min(cv, 3),
                                           shuffle=True,
                                           random_state=None),
                                       scoring='accuracy',
                                       n_jobs=-1)
            perm_accs.append(perm_acc.mean())
        perm_accs = np.array(perm_accs)
        result['chance_mean'] = perm_accs.mean()
        result['chance_std'] = perm_accs.std()
        result['perm_p_value'] = np.mean(perm_accs >= result.get('accuracy',
                                                                  0))
    except Exception:
        result['perm_p_value'] = np.nan

    return result


def evaluate_regressor(name, pipeline, X, y, cv=5):
    """K-fold CV for a regressor. Returns R² and MAE."""
    try:
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True,
                                       random_state=42)
        # Stratify by binned y for regression
        y_binned = pd.cut(y, bins=min(5, len(np.unique(y))),
                          labels=False).astype(int)
    except Exception:
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True,
                                       random_state=42)
        y_binned = np.digitize(y, np.percentile(y, [25, 50, 75]))

    try:
        scoring = {'r2': 'r2', 'mae': 'neg_mean_absolute_error'}
        scores = cross_validate(pipeline, X, y, cv=cv_splitter,
                                params={'reg__sample_weight': None},
                                scoring=scoring, n_jobs=-1,
                                error_score='raise')
        return {
            'r2': scores['test_r2'].mean(),
            'r2_std': scores['test_r2'].std(),
            'mae': -scores['test_mae'].mean(),
            'mae_std': scores['test_mae'].std(),
        }
    except Exception as exc:
        return {'r2': np.nan, 'mae': np.nan, 'error': str(exc)[:80]}


# ---------------------------------------------------------------------------
# 13e. Run full comparison: all models × all tasks × all feature sets
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ML DECODING — Predicting visual stimuli from neural activity")
print("=" * 70)

if not results or 'trials' not in results[0]:
    print("ERROR: No results with trial data available. "
          "Run sections 11 first.")
else:
    sess = results[0]
    targets = build_targets(sess)

    # Feature sets to compare
    feature_sets = {}
    for modality_name in ['vision', 'somato', 'combined']:
        try:
            X, regions = build_feature_matrix(sess, modality=modality_name)
            feature_sets[modality_name] = (X, regions)
            print(f"\nFeature set '{modality_name}': "
                  f"{X.shape[0]} trials × {X.shape[1]} neurons")
            if modality_name in ('vision', 'somato'):
                unique_regions = set(regions)
                print(f"  Regions: {sorted(unique_regions)}")
        except ValueError as e:
            print(f"  Skipping '{modality_name}': {e}")

    # =====================================================================
    # TASK A — Stimulus Detection (binary classification)
    # =====================================================================
    print("\n" + "-" * 50)
    print("TASK A: Stimulus Detection (stimulus present? yes/no)")
    print("-" * 50)

    y_detect = targets['detection']
    n_pos = y_detect.sum()
    n_neg = len(y_detect) - n_pos
    print(f"  Classes: {n_pos} positive, {n_neg} negative "
          f"(chance = {max(n_pos, n_neg) / len(y_detect):.1%})")

    detection_results = {}

    for fs_name, (X, _) in feature_sets.items():
        if X.shape[1] < 5:
            print(f"  Too few neurons in '{fs_name}' — skipping")
            continue
        print(f"\n  --- Feature set: {fs_name} ---")
        classifiers = get_classifiers()
        fs_results = {}
        for name, pipe in classifiers.items():
            res = evaluate_classifier(name, pipe, X, y_detect,
                                       cv=5, n_permutations=200)
            fs_results[name] = res
            stars = " ***" if res.get('perm_p_value', 1) < 0.001 else \
                    " **" if res.get('perm_p_value', 1) < 0.01 else \
                    " *" if res.get('perm_p_value', 1) < 0.05 else ""
            print(f"    {name:<22s} "
                  f"Acc={res.get('accuracy', 0):.3f}±{res.get('acc_std', 0):.3f}  "
                  f"F1={res.get('f1', 0):.3f}  "
                  f"ROC-AUC={res.get('roc_auc', 0):.3f}  "
                  f"p={res.get('perm_p_value', 1):.3f}{stars}")
        detection_results[fs_name] = fs_results

    # =====================================================================
    # TASK B — Stimulus Side (binary: left vs right)
    # =====================================================================
    print("\n" + "-" * 50)
    print("TASK B: Stimulus Side (left vs right, stim trials only)")
    print("-" * 50)

    stim_idx = targets['stim_idx']
    y_side = targets['side'][stim_idx]
    # Map -1→0 (left), +1→1 (right)
    y_side_bin = (y_side == 1).astype(int)
    n_left = (y_side_bin == 0).sum()
    n_right = (y_side_bin == 1).sum()
    print(f"  Classes: {n_left} left, {n_right} right "
          f"(chance = {max(n_left, n_right) / len(y_side_bin):.1%})")

    side_results = {}

    for fs_name, (X_all, _) in feature_sets.items():
        X = X_all[stim_idx]
        if X.shape[1] < 5 or X.shape[0] < 20:
            print(f"  Too few neurons/trials in '{fs_name}' — skipping")
            continue
        print(f"\n  --- Feature set: {fs_name} ---")
        classifiers = get_classifiers()
        fs_results = {}
        for name, pipe in classifiers.items():
            res = evaluate_classifier(name, pipe, X, y_side_bin,
                                       cv=min(5, X.shape[0] // 5),
                                       n_permutations=200)
            fs_results[name] = res
            stars = " ***" if res.get('perm_p_value', 1) < 0.001 else \
                    " **" if res.get('perm_p_value', 1) < 0.01 else \
                    " *" if res.get('perm_p_value', 1) < 0.05 else ""
            print(f"    {name:<22s} "
                  f"Acc={res.get('accuracy', 0):.3f}±{res.get('acc_std', 0):.3f}  "
                  f"F1={res.get('f1', 0):.3f}  "
                  f"ROC-AUC={res.get('roc_auc', 0):.3f}  "
                  f"p={res.get('perm_p_value', 1):.3f}{stars}")
        side_results[fs_name] = fs_results

    # =====================================================================
    # TASK C — Contrast Regression
    # =====================================================================
    print("\n" + "-" * 50)
    print("TASK C: Contrast Prediction (regression)")
    print("-" * 50)

    y_contrast = targets['contrast']
    print(f"  Contrast range: [{y_contrast.min():.3f}, "
          f"{y_contrast.max():.3f}], mean={y_contrast.mean():.3f}")

    contrast_results = {}

    for fs_name, (X, _) in feature_sets.items():
        if X.shape[1] < 5:
            print(f"  Too few neurons in '{fs_name}' — skipping")
            continue
        print(f"\n  --- Feature set: {fs_name} ---")
        regressors = get_regressors()
        fs_results = {}
        for name, pipe in regressors.items():
            res = evaluate_regressor(name, pipe, X, y_contrast, cv=5)
            fs_results[name] = res
            r2_str = f"R²={res.get('r2', 0):.3f}±{res.get('r2_std', 0):.3f}" \
                if not np.isnan(res.get('r2', np.nan)) else "R²=FAIL"
            print(f"    {name:<22s} {r2_str}  "
                  f"MAE={res.get('mae', 0):.4f}")
        contrast_results[fs_name] = fs_results

    # =====================================================================
    # 13f. Dummy (chance-level) baselines
    # =====================================================================
    print("\n" + "-" * 50)
    print("CHANCE BASELINES (Dummy classifiers)")
    print("-" * 50)

    for fs_name, (X, _) in feature_sets.items():
        if fs_name not in detection_results:
            continue
        dummy = DummyClassifier(strategy='stratified', random_state=42)
        dummy_acc = cross_val_score(dummy, X, y_detect, cv=5,
                                     scoring='accuracy').mean()
        print(f"  {fs_name:<10s} Dummy (stratified) accuracy: {dummy_acc:.3f}")

    # =====================================================================
    # 13g. Summary bar chart — best model per task per feature set
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Detection accuracy ---
    ax = axes[0]
    all_fs = []
    all_acc = []
    all_colors = []
    fs_color_map = {'vision': COLOR_VISION, 'somato': COLOR_SOMATO,
                    'combined': '#2ca02c'}
    for fs_name in ['vision', 'somato', 'combined']:
        if fs_name in detection_results:
            vals = [(r.get('accuracy', 0), r.get('acc_std', 0))
                    for r in detection_results[fs_name].values()
                    if not np.isnan(r.get('accuracy', np.nan))]
            if vals:
                best_acc, best_std = max(vals, key=lambda v: v[0])
                all_fs.append(fs_name.capitalize())
                all_acc.append(best_acc)
                all_colors.append(fs_color_map.get(fs_name, '#7f7f7f'))
                # Chance line from permutation test
                chance = next(iter(detection_results[fs_name].values())) \
                    .get('chance_mean', 0.5)
                ax.axhline(chance, color=fs_color_map.get(fs_name),
                           ls='--', lw=0.8, alpha=0.5)
    bars = ax.bar(all_fs, all_acc, color=all_colors, alpha=0.8, width=0.5)
    ax.axhline(max(n_pos, n_neg) / len(y_detect), color='gray',
               ls=':', lw=1, label='Majority baseline')
    ax.set_ylabel('Accuracy')
    ax.set_title('Task A: Stimulus Detection\n(best model per feature set)')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    # --- Side accuracy ---
    ax = axes[1]
    all_fs = []
    all_acc = []
    for fs_name in ['vision', 'somato', 'combined']:
        if fs_name in side_results:
            vals = [(r.get('accuracy', 0), r.get('acc_std', 0))
                    for r in side_results[fs_name].values()
                    if not np.isnan(r.get('accuracy', np.nan))]
            if vals:
                best_acc, best_std = max(vals, key=lambda v: v[0])
                all_fs.append(fs_name.capitalize())
                all_acc.append(best_acc)
    ax.bar(all_fs, all_acc,
           color=[fs_color_map.get(f.lower(), '#7f7f7f') for f in all_fs],
           alpha=0.8, width=0.5)
    ax.axhline(max(n_left, n_right) / len(y_side_bin), color='gray',
               ls=':', lw=1, label='Majority baseline')
    ax.set_ylabel('Accuracy')
    ax.set_title('Task B: Left vs Right Side\n(best model per feature set)')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    # --- Contrast R² ---
    ax = axes[2]
    all_fs = []
    all_r2 = []
    for fs_name in ['vision', 'somato', 'combined']:
        if fs_name in contrast_results:
            vals = [(r.get('r2', -999), r.get('r2_std', 0))
                    for r in contrast_results[fs_name].values()
                    if not np.isnan(r.get('r2', np.nan))]
            if vals:
                best_r2, best_std = max(vals, key=lambda v: v[0])
                all_fs.append(fs_name.capitalize())
                all_r2.append(best_r2)
    ax.bar(all_fs, all_r2,
           color=[fs_color_map.get(f.lower(), '#7f7f7f') for f in all_fs],
           alpha=0.8, width=0.5)
    ax.axhline(0, color='gray', ls='--', lw=1, label='Chance (R²=0)')
    ax.set_ylabel('R²')
    ax.set_title('Task C: Contrast Regression\n(best model per feature set)')
    ax.legend(fontsize=8)

    plt.suptitle('Neural Decoding Performance by Feature Set', fontsize=13,
                 y=1.02)
    plt.tight_layout()
    plt.savefig('fig13g_decoding_summary.png', bbox_inches='tight')
    plt.close(fig)
    print("\nSaved: fig13g_decoding_summary.png")

    # =====================================================================
    # 13h. Detailed model comparison heatmap (detection task)
    # =====================================================================
    if detection_results:
        model_names = list(next(iter(detection_results.values())).keys())
        fs_names = list(detection_results.keys())

        heatmap_data = np.zeros((len(model_names), len(fs_names))) * np.nan
        for i, model in enumerate(model_names):
            for j, fs in enumerate(fs_names):
                if fs in detection_results and model in detection_results[fs]:
                    heatmap_data[i, j] = \
                        detection_results[fs][model].get('accuracy', np.nan)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlBu_r',
                       vmin=0.4, vmax=1.0)

        ax.set_xticks(range(len(fs_names)))
        ax.set_xticklabels([f.capitalize() for f in fs_names])
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names)

        for i in range(len(model_names)):
            for j in range(len(fs_names)):
                val = heatmap_data[i, j]
                text = f'{val:.3f}' if not np.isnan(val) else '---'
                ax.text(j, i, text, ha='center', va='center',
                        fontsize=9, fontweight='bold')

        ax.set_title('Model × Feature Set: Detection Accuracy')
        plt.colorbar(im, ax=ax, label='Accuracy')
        plt.tight_layout()
        plt.savefig('fig13h_model_heatmap.png', bbox_inches='tight')
        plt.close(fig)
        print("Saved: fig13h_model_heatmap.png")

    # =====================================================================
    # 13i. Train final model & show confusion matrix (detection, vision)
    # =====================================================================
    if 'vision' in feature_sets:
        X_vis, _ = feature_sets['vision']
        X_train, X_test, y_train, y_test = train_test_split(
            X_vis, y_detect, test_size=0.2, stratify=y_detect,
            random_state=42
        )
        best_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=5000, C=1.0,
                                        class_weight='balanced')),
        ])
        best_pipe.fit(X_train, y_train)
        y_pred = best_pipe.predict(X_test)

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=['No Stim', 'Stim'],
            cmap='Blues', ax=ax
        )
        ax.set_title(f'Detection Confusion Matrix (Vision features)\n'
                     f'Accuracy = {accuracy_score(y_test, y_pred):.3f}')
        plt.tight_layout()
        plt.savefig('fig13i_confusion_matrix.png', bbox_inches='tight')
        plt.close(fig)
        print("Saved: fig13i_confusion_matrix.png")

    # =====================================================================
    # 13j. Vision vs Somatosensory: per-neuron decoding weights
    # =====================================================================
    if 'vision' in feature_sets and 'somato' in feature_sets:
        fig, ax = plt.subplots(figsize=(9, 5))
        X_v, reg_v = feature_sets['vision']
        X_s, reg_s = feature_sets['somato']

        all_X = np.hstack([X_v, X_s])
        scaler = StandardScaler().fit(all_X)
        X_all_scaled = scaler.transform(all_X)

        pipe = LogisticRegression(max_iter=5000, C=0.5,
                                   class_weight='balanced')
        pipe.fit(X_all_scaled, y_detect)
        weights = pipe.coef_[0]

        n_v = X_v.shape[1]
        colors = [COLOR_VISION] * n_v + [COLOR_SOMATO] * (len(weights) - n_v)
        ax.bar(range(len(weights)), np.abs(weights), color=colors, alpha=0.7)
        ax.axvline(n_v - 0.5, color='k', ls='--', lw=1)
        ax.text(n_v / 2, ax.get_ylim()[1] * 0.95, 'Vision', ha='center',
                fontweight='bold', color=COLOR_VISION)
        ax.text(n_v + (len(weights) - n_v) / 2, ax.get_ylim()[1] * 0.95,
                'Somato.', ha='center', fontweight='bold',
                color=COLOR_SOMATO)
        ax.set_xlabel('Neuron index')
        ax.set_ylabel('|Weight| (logistic regression)')
        ax.set_title('Feature importance: which neurons drive the decoder?')
        plt.tight_layout()
        plt.savefig('fig13j_neuron_weights.png', bbox_inches='tight')
        plt.close(fig)
        print("Saved: fig13j_neuron_weights.png")

    # =====================================================================
    # Final report
    # =====================================================================
    print("\n" + "=" * 70)
    print("ML DECODING COMPLETE")
    print("Generated figures: fig13g_decoding_summary.png, "
          "fig13h_model_heatmap.png, "
          "fig13i_confusion_matrix.png, "
          "fig13j_neuron_weights.png")
    print("=" * 70)
print("=" * 70)


