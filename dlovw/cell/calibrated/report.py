"""Generate quantitative Markdown only from saved structured run evidence."""
import json
from pathlib import Path
from .config import Config


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def render(config=Config(out='outputs/cell_calibrated_20261113/v1')):
    root = config.path
    state = read(root/'pipeline_state.json')
    accepted = read(root/'acceptance.json')
    text = ['# Calibrated cell training and deployment acceptance', '',
            f'Protocol: `{config.provenance()["protocol"]}`.', '',
            f'Pipeline state: `{state["status"] if state else "not_started"}`; '
            f'phase: `{state.get("phase", "unknown") if state else "unknown"}`.', '',
            'Hardware actions executed: **no**. Acceptance applies to the frozen simulation task distribution, '
            'real-input adapter fault replay and CPU latency audit.', '',
            '## Data and screening', '',
            '| Split | Proposed | Accepted | Source |', '|---|---:|---:|---|']
    for split in ('train', 'validation', 'test'):
        path = root/'tasks'/split/'scenes.json'
        record = read(path)
        if record:
            summary = record['summary']
            text.append(f'| {split} | {summary["proposed"]} | {summary["accepted"]} | [manifest]({path.relative_to(root)}) |')
        else:
            text.append(f'| {split} | pending | pending | — |')
    text += ['', 'Accepted means the screened oracle found a valid plan. Oracle search failure is not a proof '
             'of physical impossibility. Policy success is conditional on this declared screening, and all rejection '
             'counts remain in the manifests.', '', '## Training', '',
             '| Seed | Completed rounds | Stage | Evidence |', '|---|---:|---|---|']
    for seed in (0, 1):
        folder = root/'policy'/f'seed_{seed}'
        complete = read(folder/'completed.json')
        if complete:
            text.append(f'| {seed} | {complete["rounds"]} | completed | [record]({(folder/"completed.json").relative_to(root)}) |')
        else:
            log = folder/'log.jsonl'
            lines = log.read_text().splitlines() if log.exists() else []
            last = json.loads(lines[-1]) if lines else {}
            text.append(f'| {seed} | {last.get("round", 0)} | {last.get("phase", "pending")} | '
                        f'{last.get("step", "—")} updates in current stage |')
    for split in ('validation', 'test'):
        label = 'Development validation' if split == 'validation' else 'Frozen final test'
        text += ['', f'## {label}', '',
                 '| Seed | Condition | Arm | N | Success % | Wilson 95% | Collisions | Timeouts | Errors | Decision P95 ms |',
                 '|---|---|---|---:|---:|---|---:|---:|---:|---:|']
        sources = []
        for seed in (0, 1):
            path = root/'policy'/f'seed_{seed}'/f'eval_{split}'/'summary.json'
            record = read(path)
            if not record:
                continue
            for condition, arms in record['groups'].items():
                for arm, summary in arms.items():
                    lo, hi = summary['success_wilson95']
                    outcome = summary['outcomes']
                    latency = summary['decision_p95_ms']
                    ms = f'{latency:.2f}' if latency is not None else '—'
                    text.append(f'| {seed} | {condition} | {arm} | {summary["n"]} | '
                        f'{summary["success_rate"]*100:.2f} | [{lo*100:.2f}, {hi*100:.2f}] | '
                        f'{outcome.get("clearance_violation", 0)} | {outcome.get("timeout", 0)} | '
                        f'{outcome.get("error", 0)} | {ms} |')
            sources.append(f'Seed {seed} source: [{path.name}]({path.relative_to(root)}). '
                           f'Checkpoint: `{record["checkpoint"]}`; evaluator code `{record["config"]["code_commit"]}`.')
        text += ['', *sources, '']
    text += ['', '## Deployment input and CPU timing', '',
             '| Seed | All faults passed | Decision samples | Full decision P95 ms | Accepted |',
             '|---|---|---:|---:|---|']
    for seed in (0, 1):
        path = root/'policy'/f'seed_{seed}'/'deployment_check.json'
        record = read(path)
        if record:
            text.append(f'| {seed} | {all(record["faults"].values())} | {record["decisions"]} | '
                        f'{record["decision_p95_ms"]:.2f} | {record["accepted"]} |')
    text += ['', 'Latency replay uses static rendered RGB with fresh timestamps; it tests the real-input controller '
             'without supplying a simulation centreline. Full dynamic task timing is reported separately above. '
             'This does not certify real camera timing, robot driver behavior, all self-collisions, or physical safety.', '',
             '## Acceptance', '', f'Overall: **{accepted["accepted"] if accepted else "not completed"}**.', '',
             'Targets: both seeds complete 20 rounds; each frozen-test deploy success ≥95%; zero sphere-rope '
             'clearance violations and workspace violations; speed ≤0.05 m/s and acceleration ≤0.15 m/s²; '
             '1000 full decision timing samples per seed with P95≤80 ms; input-fault replay passes.', '',
             '## Reproduction and raw evidence', '',
             '`MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/dlo-uv-cache '
             'uv run --no-project --python ../ImaginationDLO/.venv/bin/python python -u '
             f'scripts/cell/run_calibrated_pipeline.py --out {config.out} --workers 10`', '',
             'This report is generated by `scripts/cell/report_calibrated.py`; no metrics are manually inserted. '
             'Seeds: scene generation 20261113/14/15, policy 0/1. '
             'Inputs, checksums, code version and effective configuration: [config.json](config.json). '
             'Commands, timestamps, return codes and logs: [pipeline_state.json](pipeline_state.json), `logs/`. '
             'Raw evaluation episodes are in each `eval_*/rows.jsonl`; failed collector jobs remain in the demo index and logs.', '']
    target = root/'report.md'
    target.write_text('\n'.join(text))
    return target
