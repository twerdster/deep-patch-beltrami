#!/usr/bin/env python3
"""Review completed rollout training on CPU and write public-safe validation reports.

This does not freeze checkpoints or read BSD68. It refuses a nonterminal queue.
All output names are exclusive; raw checkpoints and private paths are not copied.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from beltrami_rollout.train import source_pins

SCHEMA = 'beltrami-long-rollout-validation-report-v1'
PARENT_SHA = '9a085c0745136eaee86bfaf0b2e1eb31926eb3bb0526169134fa3a89cf2cfed5'
RUNS = ('rollout-5-001', 'rollout-32-001', 'rollout-64-001')
MODEL_FIELDS = ('width','stages','metric','zero_dc','step_scale','reaction',
                'metric_precision','checkpoint_blocks','refinement_scale')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(x) for x in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(x) for x in value)
    return not isinstance(value, float) or math.isfinite(value)


def utc_elapsed(utc, start):
    seconds = (datetime.fromisoformat(utc) - datetime.fromisoformat(start)).total_seconds()
    require(math.isfinite(seconds) and seconds >= 0, 'Invalid UTC chronology')
    return seconds


def phase_counts(step, transitions, target):
    require(isinstance(step, int) and step >= 0 and transitions, 'Invalid phase history')
    require(transitions[0]['step'] == 0, 'Missing initial curriculum phase')
    result = {}
    previous = -1
    for index, event in enumerate(transitions):
        begin = event['step']
        require(begin > previous and begin >= 0 and event['active_stages'] <= target,
                'Curriculum transitions are unordered or invalid')
        end = transitions[index + 1]['step'] if index + 1 < len(transitions) else step
        amount = max(0, min(step, end) - begin)
        if amount:
            key = str(event['active_stages'])
            result[key] = result.get(key, 0) + amount
        previous = begin
    require(sum(result.values()) == step, 'Incomplete phase chronology')
    return result


def select_validation(records, minimum=20):
    require(minimum >= 20, 'Full-depth eligibility threshold is below the protocol minimum')
    eligible = []
    seen = set()
    for row in records:
        step, full = row['step'], row['full_depth_updates']
        require(isinstance(step, int) and step not in seen and 0 <= full <= step,
                'Invalid or repeated validation step')
        require(math.isfinite(row['mean_psnr']), 'Nonfinite validation mean')
        require(row['selection_eligible'] == (full >= minimum), 'Incorrect selection eligibility')
        seen.add(step)
        if row['selection_eligible']:
            eligible.append(row)
    require(eligible, 'No eligible full-depth validation')
    return min(eligible, key=lambda row: (-row['mean_psnr'], row['step']))


def paired(candidate, reference):
    c, r = np.asarray(candidate, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    require(c.shape == r.shape and c.ndim == 1 and len(c) == 32 and np.isfinite(c).all()
            and np.isfinite(r).all(), 'Expected paired finite validation32 scores')
    delta = c-r
    total = float(delta.sum())
    largest = sorted(delta.tolist(), reverse=True)[:2]
    return {'mean_db': float(delta.mean()), 'median_db': float(np.median(delta)),
            'wins': int((delta > 0).sum()), 'losses': int((delta < 0).sum()),
            'ties': int((delta == 0).sum()), 'minimum_db':float(delta.min()),
            'maximum_db':float(delta.max()), 'largest_two_signed_changes_db': largest,
            'largest_two_share_of_positive_net_percent': float(sum(largest)/total*100) if total > 0 else None}


def comparisons(arms):
    by_name = {arm['name']:arm for arm in arms}
    result = []
    for candidate, reference in [('rollout-32','rollout-5'), ('rollout-64','rollout-5'),
                                 ('rollout-64','rollout-32')]:
        c, r = by_name[candidate], by_name[reference]
        cc, rr = c['selected'], r['selected']
        common_seconds = min(c['charged_seconds'],r['charged_seconds'])
        cb = select_validation([x for x in c['history'] if x['elapsed_lease_wall_seconds'] <= common_seconds])
        rb = select_validation([x for x in r['history'] if x['elapsed_lease_wall_seconds'] <= common_seconds])
        crows = {x['step']:x for x in c['history'] if x['selection_eligible']}
        rrows = {x['step']:x for x in r['history'] if x['selection_eligible']}
        common = []
        for step in sorted(crows.keys() & rrows.keys()):
            cv, rv = crows[step], rrows[step]
            common.append({'step':step, 'candidate_full_depth_updates':cv['full_depth_updates'],
                'reference_full_depth_updates':rv['full_depth_updates'],
                'candidate_elapsed_seconds':cv['elapsed_lease_wall_seconds'],
                'reference_elapsed_seconds':rv['elapsed_lease_wall_seconds'],
                **paired(cv['per_image_psnr'],rv['per_image_psnr'])})
        result.append({'candidate':candidate, 'reference':reference,
            'selected_candidate_step':cc['step'], 'selected_reference_step':rr['step'],
            'selected_checkpoint_comparison':paired(cc['per_image_psnr'],rr['per_image_psnr']),
            'actual_charge_candidate_seconds':c['charged_seconds'],
            'actual_charge_reference_seconds':r['charged_seconds'],
            'common_completed_wall_seconds':common_seconds,
            'descriptive_common_wall_maxima':{'candidate_step':cb['step'],'reference_step':rb['step'],
                **paired(cb['per_image_psnr'],rb['per_image_psnr'])},
            'common_update_comparisons':common,
            'scope':'conditional validation only; unequal phases/updates/selection opportunities; not a causal depth estimate'})
    return result


def verified_checkpoint(path, cfg, cfg_sha, pins, train_sha, val_sha, transitions):
    before = sha(path)
    value = torch.load(path,map_location='cpu',weights_only=False)
    require(sha(path)==before,'Completed checkpoint changed during review')
    require(finite(value['model']) and finite(value['optimizer']), 'Nonfinite saved model/optimizer')
    require(value['config']==cfg and value['config_sha256']==cfg_sha and value['source_pins']==pins,
            'Checkpoint source/config mismatch')
    require(value['manifest_sha256']==train_sha and value['validation_manifest_sha256']==val_sha,
            'Checkpoint data-manifest mismatch')
    require(value['initialization']['source_sha256']==PARENT_SHA,'Checkpoint parent mismatch')
    counts = phase_counts(value['step'],transitions,cfg['model']['stages'])
    require(counts==value['phase_updates'] and counts.get(str(cfg['model']['stages']),0)==value['full_depth_updates'],
            'Checkpoint phase counts disagree with events')
    require(value['full_depth_updates']>=cfg['minimum_full_depth_updates'],'Checkpoint ineligible')
    fields = ('step','best_step','best_psnr','phase_updates','full_depth_updates','gpu_seconds','last_evaluation')
    result = {key:value[key] for key in fields}
    result.update({'sha256':before, 'finite_model':True,'finite_optimizer':True,
                   'parameters':sum(x.numel() for x in value['model'].values())})
    del value
    return result


def inspect_validation(path, cfg, manifest, transitions, events, lease):
    value=read(path)
    step=value['step']
    require(path.name==f'validation-{step:07d}.json','Validation filename step mismatch')
    require(value['stages']==cfg['model']['stages'] and value['crop']==0 and not value['smoke_one_image']
            and value['clipping']=='output-only float[0,255]','Invalid validation protocol')
    expected=[(r['path'],r['pixels_sha256']) for r in manifest['images'] if r['split']=='validation']
    require(len(expected)==32 and len(set(expected))==32,'Expected validation32')
    require([(r['image'],r['pixels_sha256']) for r in value['images']]==expected,'Validation identities differ')
    scores=[]
    for row in value['images']:
        require(math.isfinite(row['mse']) and row['mse']>0 and math.isfinite(row['psnr']) and
                abs(row['psnr']-10*math.log10(255**2/row['mse']))<1e-10,'Invalid validation metric')
        scores.append(row['psnr'])
    require(value['mean_psnr']==float(np.mean(scores)),'Incorrect validation mean')
    counts=phase_counts(step,transitions,cfg['model']['stages'])
    full=counts.get(str(cfg['model']['stages']),0)
    require(value['full_depth_updates']==full,'Validation full-depth count differs from curriculum')
    require(value['selection_eligible']==(full>=cfg['minimum_full_depth_updates']),'Incorrect eligibility')
    event,=[e for e in events if e['event']=='validation' and e['step']==step]
    require(event['mean_psnr']==value['mean_psnr'] and event['full_depth_updates']==full,
            'Validation event differs from saved validation')
    return {'step':step,'mean_psnr':value['mean_psnr'],'full_depth_updates':full,
            'phase_updates':counts,'selection_eligible':value['selection_eligible'],
            'utc':value['utc'],'elapsed_lease_wall_seconds':utc_elapsed(value['utc'],lease['started_utc']),
            'training_timer_seconds':event['gpu_seconds'],'validation_sha256':sha(path),
            'per_image_psnr':scores}


def inspect_arm(root, run_id, queue_receipt, ledger, pins, manifest_shas, manifest):
    run=root/'runs'/run_id
    cfg_path=root/'configs'/f'{run_id}.json'
    cfg,status,metadata=read(cfg_path),read(run/'status.json'),read(run/'metadata.json')
    cfg_sha=sha(cfg_path)
    require(cfg['source_pins']==pins and cfg['warmstart_sha256']==PARENT_SHA,'Configuration source/parent mismatch')
    require(metadata['config']==cfg and metadata['config_sha256']==cfg_sha,'Metadata config mismatch')
    require(status['state']=='complete' and status['reason']!='signal' and status['finite_model']
            and status['finite_optimizer'],'Arm not cleanly completed')
    leases=[e for e in ledger['entries'] if e['run_id']==run_id]
    require(len(leases)==1,'Expected exactly one arm lease')
    lease=leases[0]
    require(lease['state']=='complete' and lease['returncode']==0 and lease.get('termination_reason') is None,
            'Arm lease failed or was terminated')
    require(lease['allowance_seconds']==4500 and 0<lease['charged_seconds']<=4500,'Arm allowance/exposure differs')
    events=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
    require(events and events[-1]['event']=='complete' and events[-1]['reason']=='budget',
            'Missing clean budget-stop event')
    require(all(a['utc']<=b['utc'] and a['step']<=b['step'] for a,b in zip(events,events[1:])),
            'Event chronology is not monotonic')
    progress=[e for e in events if e['event']=='progress']
    require(progress and all(finite(e) and e['images_seen']==2*e['step'] for e in progress),
            'Nonfinite or inconsistent training progress')
    require(all(a['step']<b['step'] and a['gpu_seconds']<b['gpu_seconds'] for a,b in zip(progress,progress[1:])),
            'Training progress did not advance')
    transitions=[e for e in events if e['event']=='curriculum']
    require([e['active_stages'] for e in transitions]==[x['stages'] for x in cfg['curriculum']],
            'Observed curriculum phases differ from configuration')
    last=verified_checkpoint(run/'last.pt',cfg,cfg_sha,pins,*manifest_shas,transitions)
    best=verified_checkpoint(run/'best.pt',cfg,cfg_sha,pins,*manifest_shas,transitions)
    require(last['step']==status['step']==queue_receipt['terminal_step'] and
            last['last_evaluation']['step']==last['step'],'Terminal validation/checkpoint mismatch')
    require(last['sha256']==queue_receipt['last_sha256'] and best['sha256']==queue_receipt['best_sha256']
            and lease['charged_seconds']==queue_receipt['charged_seconds'],'Queue receipt mismatch')
    history=[inspect_validation(p,cfg,manifest,transitions,events,lease) for p in sorted(run.glob('validation-*.json'))]
    selected=select_validation(history,cfg['minimum_full_depth_updates'])
    require(history[-1]['step']==last['step'] and all(v['step']<=last['step'] for v in history),'Missing terminal validation')
    require(best['step']==status['best_step']==last['best_step']==selected['step'] and
            best['best_psnr']==status['best_psnr']==selected['mean_psnr'] and
            best['last_evaluation']['step']==selected['step'] and
            best['last_evaluation']['mean_psnr']==selected['mean_psnr'],'BEST is not maximum eligible/earliest tie')
    eligible=[v for v in history if v['selection_eligible']]
    quarter=max(1,len(eligible)//4)
    phases=[{'step':e['step'],'active_stages':e['active_stages'],'utc':e['utc'],
             'training_timer_seconds':e['gpu_seconds'],
             'elapsed_lease_wall_seconds':utc_elapsed(e['utc'],lease['started_utc'])} for e in transitions]
    selected={**selected,'checkpoint_sha256':best['sha256'],'phase_updates':best['phase_updates'],
              'training_timer_seconds':best['gpu_seconds']}
    return {'name':f"rollout-{cfg['model']['stages']}",'run_id':run_id,'state':'complete',
        'stages':cfg['model']['stages'],'parameters':best['parameters'],
        'model_config':{k:cfg['model'][k] for k in MODEL_FIELDS if k in cfg['model']},
        'training_seed':cfg['seed'],'validation_noise_seed':cfg['validation_seed'],
        'configuration_sha256':cfg_sha,'started_utc':lease['started_utc'],'ended_utc':lease['ended_utc'],
        'allowance_seconds':lease['allowance_seconds'],'charged_seconds':lease['charged_seconds'],
        'terminal_step':last['step'],'terminal_full_depth_updates':last['full_depth_updates'],
        'terminal_phase_updates':last['phase_updates'],'terminal_checkpoint_sha256':last['sha256'],
        'terminal_training_timer_seconds':last['gpu_seconds'],
        'images_seen':last['step']*2,'training_pixels_seen':last['step']*2*180*180,
        'selected':selected,'history':history,'phase_transitions':phases,
        'validation_records':len(history),'eligible_validation_records':len(eligible),
        'ineligible_steps':[v['step'] for v in history if not v['selection_eligible']],
        'initialization_psnr':history[0]['mean_psnr'],'all_saved_model_optimizer_finite':True,
        'logged_progress_records':len(progress),'all_logged_losses_gradients_finite':True,
        'logged_gradient_norm_range':[min(e['gradient_norm'] for e in progress),max(e['gradient_norm'] for e in progress)],
        'quarter_trend':{'eligible_observations':len(eligible),'quarter_observations':quarter,
           'rounding':'q=max(1,n//4); initialization/partial-curriculum validations excluded',
           'last_minus_first_quarter_mean_db':float(np.mean([x['mean_psnr'] for x in eligible[-quarter:]])-
                                                    np.mean([x['mean_psnr'] for x in eligible[:quarter]])),
           'scope':'descriptive only; fixed stopping forbids further training in this round'}}


def image_audit(root, manifest, directory):
    for row in manifest['images']:
        path=root/'data'/directory/row['path']
        require(sha(path)==row['file_sha256'],'Dataset file hash differs')
        with Image.open(path) as image:
            array=np.asarray(image,dtype=np.uint8)
            require(image.mode=='L' and list(array.shape)==row['shape_hw'],'Dataset mode/shape differs')
            require(hashlib.sha256(array.tobytes()).hexdigest()==row['pixels_sha256'],'Dataset pixel hash differs')
    return len(manifest['images'])


def load_study(root):
    require(not torch.cuda.is_initialized(),'Run on CPU before any CUDA initialization')
    queue=read(root/'rollout-status.json')
    require(queue['state']=='complete' and queue.get('current_run') is None and queue.get('lease_pid') is None,
            'PROVISIONAL: training queue is not terminal; no final report produced')
    require([r['run_id'] for r in queue['results']]==list(RUNS),'Expected all three completed arm receipts')
    proc=Path('/proc')/str(queue['pid'])/'cmdline'
    require(not(proc.exists() and b'run_rollout_plan.py' in proc.read_bytes()),'Queue driver is still live')
    ledger=read(root/'BUDGET.json')
    require(ledger['authorized_gpu_seconds']==43200 and ledger['status']=='idle'
            and not any(e['state']=='running' for e in ledger['entries']),'LIVE ledger is not idle/unchanged')
    require(math.isclose(ledger['consumed_gpu_seconds'],sum(e['charged_seconds'] for e in ledger['entries']),abs_tol=1e-6),
            'Ledger sum differs')
    pins=source_pins()
    plan=read(root/'rollout-plan.json')
    require(pins==plan['source_pins'] and sha(root/'rollout-plan.json')==queue['plan_sha256'],'Source/plan mismatch')
    require(all(sha(p)==s for p,s in queue['guards'].items()),'A frozen queue guard changed')
    manifests=[root/'data/train400.json',root/'data/validation32.json']
    train,val=map(read,manifests)
    counts={'train':image_audit(root,train,'original400'),'validation':image_audit(root,val,'validation32')}
    require(counts=={'train':400,'validation':32},'Unexpected training/validation counts')
    require(not({r['pixels_sha256'] for r in train['images']}&{r['pixels_sha256'] for r in val['images']}),'Dataset overlap')
    arms=[inspect_arm(root,r,receipt,ledger,pins,[sha(p) for p in manifests],val) for r,receipt in zip(RUNS,queue['results'])]
    report={'schema':SCHEMA,'created_utc':datetime.now(timezone.utc).isoformat(),'status':'complete-validation-review',
        'scope':'validation32 only; no BSD68 access; no final selection freeze',
        'source_pins':pins,'plan_sha256':sha(root/'rollout-plan.json'),
        'queue_terminal_receipt_sha256':sha(root/'rollout-status.json'),
        'generator_sha256':sha(__file__),'training_manifest_sha256':sha(manifests[0]),
        'validation_manifest_sha256':sha(manifests[1]),'actual_file_pixel_shape_checks':counts,
        'parent_safe_weights_sha256':PARENT_SHA,
        'budget':{'authorized_seconds':43200,'consumed_seconds_at_review':ledger['consumed_gpu_seconds'],
            'ledger_sha256':sha(root/'BUDGET.json'),'lease_count_at_review':len(ledger['entries']),
            'failed_leases_at_review':sum(e['state']=='failed' for e in ledger['entries']),
            'new_training_charge_seconds':sum(a['charged_seconds'] for a in arms)},
        'elapsed_axis':'UTC timestamp difference from each supervised lease start; total charge is authoritative monotonic supervisor wall time, not GPU busy time',
        'arms':arms,'comparisons':comparisons(arms),
        'limitations':['one training seed and fixed validation-noise realization; model selection uncertainty excluded',
          'depth, curriculum, refinements and auxiliary endpoints form a combined recipe; no isolated causal depth effect',
          'equal maximum wall allowances yield unequal updates, phase exposure and selection opportunities',
          '180px training crops do not provide full nominal spatial context at 32 or 64 stages',
          'positive metrics and finite training do not prove global stability or convergence',
          'BSD68 is already exposed; subsequent scores are a reused benchmark, not untouched held-out evidence']}
    require(not torch.cuda.is_initialized(),'Unexpected CUDA initialization')
    return report


def markdown(report):
    lines=['# Long-rollout validation review','',report['scope']+'.',
        'Selection review: maximum eligible validation32 mean, earliest exact tie; at least20 full-depth updates.',
        'This report does not freeze or change the final evaluation cohort.','',
        '| Arm | Terminal updates | Full-depth updates | Selected update | Validation dB | Eligible observations | Charged seconds |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for a in report['arms']:
        lines.append(f"| {a['name']} | {a['terminal_step']} | {a['terminal_full_depth_updates']} | {a['selected']['step']} | {a['selected']['mean_psnr']:.12f} | {a['eligible_validation_records']} | {a['charged_seconds']:.6f} |")
    lines+=['','Validation comparisons below condition on separately selected checkpoints. They are not test results.','',
        '| Candidate − reference | Mean dB | Median dB | Image wins/losses | Two largest changes / positive net |',
        '| --- | ---: | ---: | ---: | ---: |']
    for c in report['comparisons']:
        p=c['selected_checkpoint_comparison']; share=p['largest_two_share_of_positive_net_percent']
        share_label=f'{share:.2f}%' if share is not None else 'net nonpositive'
        lines.append(f"| {c['candidate']} − {c['reference']} | {p['mean_db']:+.12f} | {p['median_db']:+.12f} | {p['wins']}/{p['losses']} | {share_label} |")
    lines+=['', 'The JSON includes every validation point, eligibility, curriculum phases, selected/terminal tensor pins, actual charges, descriptive common-wall comparisons and common-update paired comparisons.', '',
        'Plot time uses recorded UTC differences from each lease start. The ledger reports total conservative supervisor wall time; neither quantity is GPU busy time. All saved BEST/LAST model and optimizer states and all logged losses/gradients were finite. All 400 training and 32 validation files, pixels and shapes were checked.', '', 'Limitations:', '']
    lines += ['- '+item+'.' for item in report['limitations']]
    return '\n'.join(lines)+'\n'


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    colors={'rollout-5':'#26547c','rollout-32':'#b14d00','rollout-64':'#633d91'}
    for arm in report['arms']:
        color=colors[arm['name']]
        for ax,key in zip(axes,('step','elapsed_lease_wall_seconds')):
            h=arm['history']
            ax.plot([r[key] for r in h],[r['mean_psnr'] for r in h],color=color,lw=1.5,label=arm['name'])
            ineligible=[r for r in h if not r['selection_eligible']]
            ax.scatter([r[key] for r in ineligible],[r['mean_psnr'] for r in ineligible],color=color,marker='x',s=22)
            ax.scatter(arm['selected'][key],arm['selected']['mean_psnr'],color=color,marker='*',s=110,zorder=4)
            for event in arm['phase_transitions'][1:]:
                ax.axvline(event[key],color=color,ls=':',alpha=.45)
            ax.set_ylabel('Validation32 mean PSNR (dB)')
            ax.grid(alpha=.2)
    axes[0].set_xlabel('Training updates')
    axes[1].set_xlabel('Elapsed leased wall time (s; recorded UTC)')
    axes[0].legend(frameon=False)
    fig.suptitle('Deep Beltrami long rollouts: validation only\nStars: selected eligible maxima; crosses: ineligible; dotted: depth transitions',fontsize=11)
    fig.savefig(output,dpi=180,metadata={'Software':'Matplotlib'})
    plt.close(fig)


def write_report(report, output, make_plot=True):
    # Deliberately forbid appending to an existing report directory.
    output.mkdir(parents=True,exist_ok=False)
    (output/'validation-rollouts.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'validation-rollouts.md').write_text(markdown(report))
    if make_plot:
        plot(report,output/'validation-rollouts.png')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--project-root',type=Path)
    inputs.add_argument('--report-json',type=Path,help='Regenerate docs/plot from an already reviewed public-safe JSON; no training assets needed')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--no-plot',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(1)
    if args.project_root is not None:
        report=load_study(args.project_root.resolve())
    else:
        report=read(args.report_json)
        require(report['schema']==SCHEMA and report['status']=='complete-validation-review',
                'Expected a completed public-safe validation report')
    write_report(report,args.output,not args.no_plot)
    print(json.dumps({'output':str(args.output),'status':report['status'],'arms':len(report['arms'])}))


if __name__=='__main__':
    main()
