"""Plot direct decoding accuracy against adapter training parameters."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {'io':'I/O tuning','lora':'Projection LoRA', 'bc_lora':'B/C LoRA r=4',
          'bc_dt':'B/C LoRA + dt', 'full':'Gradual full FT',
          'sparse_sdt_m3':'Sparse state port','state_offset':'State-offset port',
          'memba_causal':'Causal membrane port'}


def read(directory, expected):
    verification = json.loads((directory/'verification.json').read_text())
    if verification['status'] != 'verified' or verification['fits'] != expected:
        raise ValueError(f'incomplete verified matrix: {directory}')
    with (directory/'groupedsummary.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    if any(int(row['n']) != (1 if row['method']=='none' else 3) for row in rows):
        raise ValueError('a method has an incomplete seed set')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main-dir',type=Path,required=True)
    parser.add_argument('--control-dir',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    main_rows = read(args.main_dir,50)
    control_rows = read(args.control_dir,6) if args.control_dir else []
    colors = dict(zip(LABELS,plt.get_cmap('tab10').colors))
    figure, axes = plt.subplots(1,2,figsize=(12,5.3),sharex=True)
    legend = {}
    for axis,task in zip(axes,('m1','m2')):
        for row in [r for r in main_rows if r['task']==task]:
            method = row['method']
            value = float(row['zero_allvalid_r2_mean'])
            if method=='none':
                handle = axis.axhline(value,linestyle='--',color='0.45',lw=1.2)
                legend['Source-only direct'] = handle
                continue
            handle = axis.errorbar(float(row['trainable_params_mean']),value,
                                   yerr=float(row['zero_allvalid_r2_std']),fmt='o',
                                   color=colors[method],capsize=3,markersize=6)
            legend[LABELS[method]] = handle
        for row in [r for r in control_rows if r['task']==task]:
            handle = axis.errorbar(float(row['trainable_params_mean']),float(row['zero_allvalid_r2_mean']),
                                   yerr=float(row['zero_allvalid_r2_std']),fmt='D',
                                   color=colors['bc_lora'],capsize=3,markersize=7,
                                   markerfacecolor='white')
            legend['B/C LoRA r=8 control'] = handle
        axis.set_xscale('log')
        axis.set_xlim(8_000,4_000_000)
        axis.set_title(task.upper())
        axis.set_xlabel('Trainable parameters (log scale)')
        axis.grid(alpha=.22)
    axes[0].set_ylabel('Direct variance-weighted R² on all valid query bins')
    figure.legend(list(legend.values()),list(legend),loc='lower center',ncol=4,fontsize=9)
    figure.suptitle('Mamba-3 target-prefix adaptation: mean ± SD over 3 adaptation seeds',fontsize=12)
    figure.tight_layout(rect=(0,.17,1,.95))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for extension in ('png','svg'):
        figure.savefig(args.output_dir/f'peft_parameter_tradeoff.{extension}',dpi=180,bbox_inches='tight')
    plt.close(figure)


if __name__=='__main__':
    main()
