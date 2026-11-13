"""Synthetic CPU tests for validation-only rollout review and sanitized reports."""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

MODULE=Path(__file__).resolve().parents[1]/'tools/report_rollout_validation.py'
spec=importlib.util.spec_from_file_location('validation_report',MODULE)
report=importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False))


def iso(seconds):
    return (datetime(2026,11,13,tzinfo=timezone.utc)+timedelta(seconds=seconds)).isoformat()


def validation_row(step,full,score=28.,eligible=None):
    return {'step':step,'full_depth_updates':full,'mean_psnr':score,
            'selection_eligible':full>=20 if eligible is None else eligible,
            'elapsed_lease_wall_seconds':float(step),'per_image_psnr':[score]*32}


@pytest.mark.parametrize('records',[
    [validation_row(0,0,29.),validation_row(30,30,28.4),validation_row(50,50,28.4)],
    [validation_row(50,50,28.4),validation_row(30,30,28.4),validation_row(0,0,29.)]])
def test_maximum_eligible_earliest_tie(records):
    assert report.select_validation(records)['step']==30


@pytest.mark.parametrize('records',[
    [validation_row(0,0,29.,True)], [validation_row(30,19,28.,True)],
    [validation_row(30,30,float('nan'))],
    [validation_row(30,30),validation_row(30,30)]])
def test_invalid_or_nonfinite_selection_rejected(records):
    with pytest.raises(ValueError): report.select_validation(records)


def test_no_full_depth_observations_rejected():
    with pytest.raises(ValueError,match='No eligible'):
        report.select_validation([validation_row(0,0),validation_row(50,0)])


def test_phase_counts_include_exact_boundary():
    phases=[{'step':0,'active_stages':16},{'step':750,'active_stages':32}]
    assert report.phase_counts(750,phases,32)=={'16':750}
    assert report.phase_counts(1750,phases,32)=={'16':750,'32':1000}
    phases.append({'step':1100,'active_stages':64})
    assert report.phase_counts(1200,phases,64)=={'16':750,'32':350,'64':100}


def test_repeated_or_missing_phase_rejected():
    with pytest.raises(ValueError): report.phase_counts(50,[{'step':1,'active_stages':5}],5)
    with pytest.raises(ValueError):
        report.phase_counts(50,[{'step':0,'active_stages':16},{'step':0,'active_stages':32}],32)


def test_outlier_sensitive_pairwise_summary():
    c=np.zeros(32); c[:2]=.1; c[2:]=-.003
    p=report.paired(c,np.zeros(32))
    assert p['wins']==2 and p['losses']==30 and p['median_db']<0 and p['mean_db']>0
    assert p['largest_two_share_of_positive_net_percent']>100


def test_pairwise_invalid_or_nonpositive_net():
    with pytest.raises(ValueError): report.paired([1]*31,[1]*31)
    with pytest.raises(ValueError): report.paired([float('inf')]*32,[1]*32)
    assert report.paired([0]*32,[1]*32)['largest_two_share_of_positive_net_percent'] is None


@pytest.fixture
def study(tmp_path,monkeypatch):
    pins={'beltrami_operator':'o','beltrami_precision':'p','beltrami_rollout':'r'}
    monkeypatch.setattr(report,'source_pins',lambda:pins)
    monkeypatch.setattr(report,'image_audit',lambda root,manifest,directory:400 if directory=='original400' else 32)
    train={'images':[{'path':'train.png','pixels_sha256':'trainhash'}]}
    val={'images':[{'path':f'{i}.png','pixels_sha256':f'hash{i}','split':'validation'} for i in range(32)]}
    dump(tmp_path/'data/train400.json',train); dump(tmp_path/'data/validation32.json',val)
    manifest_shas=[report.sha(tmp_path/'data'/name) for name in ('train400.json','validation32.json')]
    dump(tmp_path/'rollout-plan.json',{'source_pins':pins})
    receipts=[]; entries=[]
    for stages in (5,32,64):
        name=f'rollout-{stages}-001'; run=tmp_path/'runs'/name
        model={'stages':stages,'width':80}
        if stages==5:
            curricula=[{'until_fraction':1.,'stages':5}]
            phase_events=[{'event':'curriculum','step':0,'active_stages':5,'gpu_seconds':3.,'utc':iso(4)}]
        elif stages==32:
            curricula=[{'until_fraction':.15,'stages':16},{'until_fraction':1.,'stages':32}]
            phase_events=[{'event':'curriculum','step':0,'active_stages':16,'gpu_seconds':3.,'utc':iso(4)},
                          {'event':'curriculum','step':10,'active_stages':32,'gpu_seconds':12.,'utc':iso(13)}]
        else:
            curricula=[{'until_fraction':.1,'stages':16},{'until_fraction':.25,'stages':32},
                       {'until_fraction':1.,'stages':64}]
            phase_events=[{'event':'curriculum','step':0,'active_stages':16,'gpu_seconds':3.,'utc':iso(4)},
                          {'event':'curriculum','step':5,'active_stages':32,'gpu_seconds':7.,'utc':iso(8)},
                          {'event':'curriculum','step':10,'active_stages':64,'gpu_seconds':12.,'utc':iso(13)}]
        cfg={'model':model,'source_pins':pins,'warmstart_sha256':report.PARENT_SHA,'minimum_full_depth_updates':20,
             'seed':20261113,'validation_seed':742913,'curriculum':curricula}
        cfg_path=tmp_path/'configs'/f'{name}.json'; dump(cfg_path,cfg); cfg_sha=report.sha(cfg_path)
        dump(run/'metadata.json',{'config':cfg,'config_sha256':cfg_sha})
        status={'state':'complete','reason':'budget_or_steps','finite_model':True,'finite_optimizer':True,
                'step':50,'best_step':30,'best_psnr':28.5}
        dump(run/'status.json',status)
        events=list(phase_events)
        for step,timer in ((10,11.),(30,60.),(50,98.)):
            events.append({'event':'progress','step':step,'gpu_seconds':timer,'utc':iso(timer+1),
                           'images_seen':2*step,'loss':.01,'gradient_norm':.02})
        for step,score,timer in ((0,28.2,2.),(30,28.5,61.),(50,28.4,99.)):
            counts=report.phase_counts(step,phase_events,stages)
            full=counts.get(str(stages),0)
            values={'step':step,'mean_psnr':score,'stages':stages,'full_depth_updates':full,
                    'selection_eligible':full>=20,'smoke_one_image':False,'crop':0,
                    'clipping':'output-only float[0,255]','utc':iso(timer+1),
                    'images':[{'image':x['path'],'pixels_sha256':x['pixels_sha256'],
                               'psnr':score,'mse':255**2/10**(score/10)} for x in val['images']]}
            # Use NumPy's reduction exactly as in the frozen trainer.
            values['mean_psnr']=float(np.mean([x['psnr'] for x in values['images']]))
            dump(run/f'validation-{step:07d}.json',values)
            events.append({'event':'validation','step':step,'mean_psnr':values['mean_psnr'],
                           'full_depth_updates':full,'gpu_seconds':timer,'utc':iso(timer+1)})
        # Validation mean 28.5 is exactly representable, matching status/BEST.
        for key,step,score,timer in [('best',30,28.5,61.),('last',50,28.4,99.)]:
            counts=report.phase_counts(step,phase_events,stages)
            value={'step':step,'config':cfg,'config_sha256':cfg_sha,'source_pins':pins,
                   'manifest_sha256':manifest_shas[0],'validation_manifest_sha256':manifest_shas[1],
                   'initialization':{'source_sha256':report.PARENT_SHA},
                   'model':{'weight':torch.tensor([1.])},'optimizer':{'state':{'exp_avg':torch.tensor([.1])}},
                   'phase_updates':counts,'full_depth_updates':counts.get(str(stages),0),
                   'best_step':30,'best_psnr':28.5,'gpu_seconds':timer,
                   'last_evaluation':{'step':step,'mean_psnr':score}}
            torch.save(value,run/f'{key}.pt')
        events.append({'event':'complete','step':50,'gpu_seconds':100.,'utc':iso(101),'reason':'budget'})
        events.sort(key=lambda e:e['utc'])
        (run/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
        receipt={'run_id':name,'terminal_step':50,'last_sha256':report.sha(run/'last.pt'),
                 'best_sha256':report.sha(run/'best.pt'),'charged_seconds':110.}
        receipts.append(receipt)
        entries.append({'run_id':name,'state':'complete','returncode':0,'allowance_seconds':4500.,
                        'charged_seconds':110.,'started_utc':iso(0),'ended_utc':iso(110)})
    queue={'state':'complete','current_run':None,'lease_pid':None,'pid':999999999,
           'results':receipts,'plan_sha256':report.sha(tmp_path/'rollout-plan.json'),'guards':{}}
    dump(tmp_path/'rollout-status.json',queue)
    dump(tmp_path/'BUDGET.json',{'authorized_gpu_seconds':43200,'status':'idle','consumed_gpu_seconds':330.,'entries':entries})
    return tmp_path


def test_completed_cohort_report_no_private_paths(study):
    value=report.load_study(study)
    assert len(value['arms'])==3 and len(value['comparisons'])==3
    assert value['actual_file_pixel_shape_checks']=={'train':400,'validation':32}
    assert [a['selected']['step'] for a in value['arms']]==[30]*3
    assert all(a['eligible_validation_records']==2 for a in value['arms'])
    assert str(study) not in json.dumps(value)
    assert 'BSD68' in report.markdown(value) and 'not test results' in report.markdown(value)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('mutation,match',[
    ('live','PROVISIONAL'),('bad-charge','Ledger sum'),('nonfinite','Nonfinite saved'),
    ('parent','parent mismatch'),('phase','curriculum'),('missing-validation','Missing terminal')])
def test_failure_evidence_rejected(study,mutation,match):
    if mutation=='live':
        p=study/'rollout-status.json'; v=report.read(p); v['state']='running'; dump(p,v)
    elif mutation=='bad-charge':
        p=study/'BUDGET.json'; v=report.read(p); v['consumed_gpu_seconds']=329.; dump(p,v)
    elif mutation in ('nonfinite','parent'):
        p=study/'runs/rollout-5-001/best.pt'; v=torch.load(p,weights_only=False)
        if mutation=='nonfinite':v['optimizer']['state']['exp_avg'][0]=float('nan')
        else:v['initialization']['source_sha256']='wrong'
        torch.save(v,p)
    elif mutation=='phase':
        p=study/'runs/rollout-64-001/events.jsonl'; values=[json.loads(x) for x in p.read_text().splitlines()]
        values=[x for x in values if not(x['event']=='curriculum' and x['active_stages']==32)]
        p.write_text(''.join(json.dumps(x)+'\n' for x in values))
    else:
        (study/'runs/rollout-5-001/validation-0000050.json').unlink()
    with pytest.raises(ValueError,match=match):report.load_study(study)


def test_write_exclusive_output(study):
    value=report.load_study(study); out=study/'public-output'
    report.write_report(value,out,make_plot=False)
    assert report.read(out/'validation-rollouts.json')['schema']==report.SCHEMA
    with pytest.raises(FileExistsError):report.write_report(value,out,make_plot=False)


def test_reproducible_png_from_sanitized_json(study):
    pytest.importorskip('matplotlib')
    value=report.load_study(study)
    p,q=study/'plot1.png',study/'plot2.png'
    report.plot(value,p)
    report.plot(json.loads(json.dumps(value)),q)
    assert p.read_bytes().startswith(b'\x89PNG\r\n\x1a\n') and p.stat().st_size>10000
    assert report.sha(p)==report.sha(q)


def test_common_updates_require_both_eligible(study):
    value=report.load_study(study)
    for pair in value['comparisons']:
        assert [x['step'] for x in pair['common_update_comparisons']]==[30,50]
        assert pair['common_completed_wall_seconds']==110.


def test_utc_elapsed_and_negative_rejected():
    assert report.utc_elapsed(iso(10),iso(0))==10
    with pytest.raises(ValueError):report.utc_elapsed(iso(0),iso(10))
