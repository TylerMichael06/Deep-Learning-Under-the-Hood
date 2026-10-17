"""Build the final report's evidence tables and figures from saved experiment outputs.

No training or changes to the official detector/results. Requires the local audit
outputs and Kaggle data; the generated documentation assets are self-contained.
"""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from obdfault.config import CH, DT, KAGGLE_DATA
from obdfault.data import load_kaggle
from obdfault.scoring import WarmupRise
from obdfault.faults import delay_coolant

BLUE, ORANGE, GREEN = '#2563a6', '#d45a21', '#21836b'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit',type=Path,default=Path('results/thermostat_deep_2026-10-16/final_audit'))
    ap.add_argument('--output',type=Path,default=Path('docs/assets/thermostat_final'))
    ap.add_argument('--data',default=KAGGLE_DATA)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    evidence=a.output/'evidence';evidence.mkdir(exist_ok=True)
    files=['full_drive_summary.csv','full_drive_decisions.csv','seed_summary.csv','sensitivity_summary.csv',
           'descriptive_intervals.csv','paired_statistics.csv','team_log_coverage.csv','manifest.json']
    for name in files:shutil.copy2(a.audit/name,evidence/name)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','figure.facecolor':'white','savefig.facecolor':'white'})
    def save(fig,name):
        fig.savefig(a.output/name,dpi=180,bbox_inches='tight');plt.close(fig)
        print(name,flush=True)
    coverage=pd.read_csv(a.audit/'coverage.csv')
    candidates=coverage[coverage.baseline_evaluable & coverage.cold_candidate]
    counts=candidates.groupby(['car','split']).size().unstack(fill_value=0).reindex(
        sorted(coverage.loc[coverage.baseline_evaluable,'car'].unique()),fill_value=0)[['adaptation','calibration','test']]
    counts.to_csv(evidence/'candidate_counts.csv')
    totals=dict(segments=len(coverage),vehicles=int(coverage.car.nunique()),cold_candidates=int(coverage.cold_candidate.sum()),
                confirmed_candidates=int(coverage.confirmed_start_candidate.sum()),
                baseline_cold_candidates=len(candidates),test_sources=4,evaluated_vehicles=1)
    (evidence/'coverage_totals.json').write_text(json.dumps(totals,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(11,5.6));y=np.arange(len(counts));left=np.zeros(len(counts))
    for col,label,color in zip(counts.columns,['Training','Calibration','Test'],[BLUE,ORANGE,GREEN]):
        values=counts[col].to_numpy();ax.barh(y,values,left=left,label=label,color=color)
        for i,v in enumerate(values):
            if v:ax.text(left[i]+v/2,i,str(v),ha='center',va='center',color='white',weight='bold')
        left+=values
    ax.set_yticks(y,counts.index);ax.invert_yaxis();ax.set_xlim(0,23)
    ax.text(16.4,list(counts.index).index('car9'),'Only evaluable vehicle',va='center',fontsize=10)
    ax.text(13.4,list(counts.index).index('car8'),'Insufficient RPM/load\ncoverage for training',va='center',fontsize=9)
    ax.set_xlabel('Cold-state candidate recordings (before signal-coverage requirements)')
    ax.set_title('Usable data coverage limits the conclusion',loc='left',pad=30)
    ax.legend(ncol=3,loc='lower left',bbox_to_anchor=(0,1.01),frameon=False)
    fig.tight_layout();save(fig,'01_coverage.png')
    trips=load_kaggle(a.data)
    selected=coverage[(coverage.car=='car9')&coverage.cold_candidate]
    fit_ids=selected.loc[selected.split.eq('adaptation'),'trip_id'].tolist()
    model=WarmupRise().fit([trips[i] for i in fit_ids])
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for ax,i in zip(axes.flat,selected.loc[selected.split.eq('test'),'trip_id']):
        tr=trips[i];t=np.arange(len(tr['X']))*DT/60;e=tr['X'][:,CH.index('ECT')]
        A,_=model.inputs(tr)
        expected=e[np.flatnonzero(np.isfinite(e))[0]]+np.column_stack([np.ones(len(A)),(A-model.mean)/model.scale])@model.coef
        ax.plot(t,e,color=GREEN,label='Original recording',linewidth=2)
        ax.plot(t,delay_coolant(tr,.75)['X'][:,CH.index('ECT')],color=BLUE,label='Evolution at 75% rate')
        ax.plot(t,delay_coolant(tr,.5)['X'][:,CH.index('ECT')],color=ORANGE,label='Evolution at 50% rate')
        ax.plot(t,expected,color='#444444',linestyle='--',label='Expected temperature')
        ax.axvspan(2,5,color='#b7dbc9',alpha=.3)
        ax.set_title(f'Recording {i}'+(' — also alarms when clean' if i==394 else ''),fontsize=11)
        ax.set_xlim(0,10);ax.set_ylim(30,100);ax.grid(alpha=.15)
    for ax in axes[-1]:ax.set_xlabel('Minutes since recording start')
    for ax in axes[:,0]:ax.set_ylabel('Coolant temperature (°C)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,fontsize=10)
    fig.suptitle('Four source recordings; delayed copies are not real faults',weight='bold',y=.99)
    fig.tight_layout(rect=(0,.06,1,.95));save(fig,'02_curves.png')
    full=pd.read_csv(a.audit/'full_drive_summary.csv');full=full[full.seed.eq(1)].set_index('scenario')
    seed=pd.read_csv(a.audit/'seed_summary.csv');seed=seed[seed.seed.eq(1)]
    scenarios=['clean_cold','clean_warm','delay_0.75','delay_0.5']
    labels=['Clean cold\n(false alarms)','Clean warm\n(false alarms)','Mild delay\n(detections)','Strong delay\n(detections)']
    fig,axes=plt.subplots(1,2,figsize=(14,5.6),sharey=True)
    for ax,mode in zip(axes,['restricted','full']):
        for offset,method,col,color,lab in [(-.18,'baseline','baseline_alarms',BLUE,'Current model'),(.18,'baseline_plus_early_rise','combined_alarms',ORANGE,'With early check')]:
            if mode=='restricted':
                r=seed[seed.method.eq(method)].set_index('scenario').loc[scenarios];k=r.alarms.to_numpy();n=r.judged.to_numpy()
            else:k=full.loc[scenarios,col].to_numpy();n=full.loc[scenarios,'judged'].to_numpy()
            ax.bar(np.arange(4)+offset,k/n,width=.35,color=color,label=lab)
            for x,v,den in zip(np.arange(4)+offset,k,n):ax.text(x,v/den+.025,f'{int(v)}/{int(den)}',ha='center',fontsize=10)
        ax.set_xticks(np.arange(4),labels,fontsize=10);ax.set_ylim(0,1.17)
        ax.set_title('Earlier comparison: minutes 5–10' if mode=='restricted' else 'Comparison using the full-drive rule',fontsize=12)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel('Share of recordings with an alarm')
    axes[0].set_yticks([0,.25,.5,.75,1],['0%','25%','50%','75%','100%'])
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc='lower center',frameon=False)
    fig.suptitle('Detection improves, but the full-drive rule reveals an additional false alarm',weight='bold')
    fig.tight_layout(rect=(0,.06,1,.93));save(fig,'03_results.png')
    sens=pd.read_csv(a.audit/'sensitivity_summary.csv');sens=sens[sens.period.eq('early')].copy()
    sens['label']=[('Original' if v=='original' else ('Omit training ' if v=='omit_adaptation' else 'Omit calibration ')+str(int(i))) for v,i in zip(sens.variant,sens.omitted)]
    order=['Original']+sens.loc[sens.variant.ne('original'),'label'].drop_duplicates().tolist()
    table=sens.pivot(index='label',columns='scenario',values='combined_alarms').loc[order,['clean_cold','delay_0.75','delay_0.5']]
    fig,ax=plt.subplots(figsize=(10,7.5));im=ax.imshow(table.to_numpy(),vmin=0,vmax=4,cmap='Blues',aspect='auto')
    for i in range(len(table)):
        for j in range(3):ax.text(j,i,f'{int(table.iloc[i,j])}/4',ha='center',va='center',color='white' if table.iloc[i,j]>=3 else '#162b40')
    ax.set_yticks(range(len(table)),table.index);ax.set_xticks(range(3),['Clean cold\n(false alarms)','Mild delay\n(detections)','Strong delay\n(detections)'])
    ax.set_title('Sensitivity to removing one training or calibration recording\nOriginal restricted comparison; the same four test cases',fontsize=12,pad=18)
    ax.tick_params(length=0);fig.tight_layout();save(fig,'04_sensitivity.png')
    cis=pd.read_csv(a.audit/'descriptive_intervals.csv');cis=cis[cis.method.eq('baseline_plus_early_rise')].set_index('scenario').loc[['delay_0.5','delay_0.75','clean_cold']]
    fig,ax=plt.subplots(figsize=(10,4.3));y=np.arange(3)
    ax.errorbar(cis.rate,y,xerr=np.vstack([cis.rate-cis.low,cis.high-cis.rate]),fmt='o',color=BLUE,capsize=6,markersize=8,linewidth=2)
    ax.set_yticks(y,['Strong delay: 4/4','Mild delay: 2/4','Cold false alarms: 1/4']);ax.invert_yaxis()
    ax.set_xlim(-.02,1.03);ax.set_xticks(np.arange(0,1.01,.2),[f'{i}%' for i in range(0,101,20)])
    ax.set_xlabel('Observed proportion and exact 95% binomial interval')
    ax.set_title('Four recordings leave substantial uncertainty',loc='left',pad=15)
    ax.grid(axis='x',alpha=.2)
    fig.text(.5,.02,'Illustration assuming independence. Does not adjust for trip dependence or exploratory selection;\nnot an interval for performance across vehicles.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.12,1,1));save(fig,'05_uncertainty.png')


if __name__=='__main__':main()
