#!/usr/bin/env python3
"""Read-only temperature/feature diagnostic for the supplied Laplace trainer.

Place in the repository root beside train_laplace.py and laplace_loss.py.
Uses one GPU, no optimizer updates, bounded feature-extraction batches, and CPU
feature storage. Reuses the loss's affinity function; computes its shared branch
scale over ALL positions before sampling positions for the reported statistics.
Run --self-test on the login node first. Runtime outputs go to --workdir.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import torch
from laplace_loss import _laplace_unit_field, laplace_unit_field_loss


@torch.no_grad()
def feature_chunks(apply, params, x, kwargs, batch_size, device):
    """Process independent examples in small batches; assemble detached CPU output."""
    result = {}
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        feats = apply(params, x[start:stop].to(device), **kwargs)
        if start and set(feats) != set(result):
            raise RuntimeError('Feature keys changed between extraction batches')
        for key in feats:
            part = feats[key].detach().cpu()
            if start == 0:
                result[key] = torch.empty((len(x), *part.shape[1:]), dtype=part.dtype)
            result[key][start:stop].copy_(part)
            del part
        del feats
    return result


def spatial_slice(x, indices, device):
    # x: [label groups, candidates, positions, dimensions], stored on CPU.
    f = x.shape[2]
    return x[indices // f, :, indices % f, :].to(device=device, dtype=torch.float32)


def weight_slice(cfg, indices, positions, ng, np_, nu, device):
    c = cfg[indices // positions].to(device=device, dtype=torch.float32)
    return (torch.ones((len(indices), ng), device=device),
            c[:, None].expand(-1, np_), (c - 1)[:, None].expand(-1, nu))


@torch.no_grad()
def branch_scale(g, p, u, cfg, epsilon, chunk, device):
    # Exactly the scalar estimator in laplace_unit_field_loss. Streaming sums
    # have minor floating-point reduction differences; no per-position rescaling.
    count = g.shape[0] * g.shape[2]
    dist_sum = weight_sum = 0.0
    dist_count = weight_count = 0
    for start in range(0, count, chunk):
        idx = torch.arange(start, min(start + chunk, count))
        gg, pp, uu = (spatial_slice(x, idx, device) for x in (g, p, u))
        wg, wp, wu = weight_slice(cfg, idx, g.shape[2], g.shape[1], p.shape[1], u.shape[1], device)
        targets = torch.cat((gg, uu, pp), dim=1)
        weights = torch.cat((wg, wu, wp), dim=1)
        distances = torch.cdist(gg, targets)
        dist_sum += (distances * weights[:, None, :]).sum(dtype=torch.float64).item()
        weight_sum += weights.sum(dtype=torch.float64).item()
        dist_count += distances.numel()
        weight_count += weights.numel()
    value = (dist_sum / dist_count) / (weight_sum / weight_count + epsilon)
    return torch.tensor(max(value, epsilon), dtype=torch.float32, device=device)


@torch.no_grad()
def branch_statistics(g, p, u, cfg, taus, epsilon, chunk, max_positions, seed, device):
    scale = branch_scale(g, p, u, cfg, epsilon, chunk, device)
    count = g.shape[0] * g.shape[2]
    if max_positions > 0 and count > max_positions:
        idx_all = torch.randperm(count, generator=torch.Generator().manual_seed(seed))[:max_positions]
    else:
        idx_all = torch.arange(count)
    records = {(tau, name): {'ess': [], 'peak': [], 'active': []}
               for tau in taus for name in ('pos', 'gen', 'uncond')}
    for start in range(0, len(idx_all), chunk):
        idx = idx_all[start:start + chunk]
        gg, pp, uu = (spatial_slice(x, idx, device) for x in (g, p, u))
        wg, wp, wu = weight_slice(cfg, idx, g.shape[2], g.shape[1], p.shape[1], u.shape[1], device)
        for tau in taus:
            for name, target, weights, diagonal in (
                ('pos', pp, wp, False), ('gen', gg, wg, True), ('uncond', uu, wu, False)
            ):
                _, aff = _laplace_unit_field(gg, target, weights, scale, tau, epsilon, diagonal)
                prob = aff / aff.sum(-1, keepdim=True).clamp_min(1e-30)
                r = records[tau, name]
                r['ess'].append((1 / prob.square().sum(-1).clamp_min(1e-30)).flatten().cpu())
                r['peak'].append(prob.max(-1).values.flatten().cpu())
                active = (weights.mean(-1) > epsilon)[:, None].expand(-1, gg.shape[1])
                r['active'].append(active.flatten().cpu())
                del aff, prob
    return float(scale), {key: {k: torch.cat(v) for k, v in values.items()} for key, values in records.items()}


def self_test():
    # CPU only: compare streaming scale/ESS with the actual loss diagnostics.
    torch.manual_seed(42)
    g, p, u = torch.randn(2, 5, 3, 4), torch.randn(2, 7, 3, 4), torch.randn(2, 3, 3, 4)
    cfg = torch.tensor([1.0, 1.8])
    flatten = lambda x: x.permute(0, 2, 1, 3).reshape(-1, x.shape[1], x.shape[-1])
    gg, pp, uu = map(flatten, (g, p, u))
    c = cfg.repeat_interleave(3)
    scale, stats = branch_statistics(g, p, u, cfg, [0.05, 0.2], 1e-8, 2, 0, 42, 'cpu')
    for tau in (0.05, 0.2):
        _, info = laplace_unit_field_loss(gg, pp, uu,
            weight_gen=torch.ones(6, 5), weight_pos=c[:, None].expand(-1, 7),
            weight_neg=(c - 1)[:, None].expand(-1, 3),
            tau_init=tau, tau_final=tau, current_step=0)
        torch.testing.assert_close(torch.tensor(scale), info['scale'], rtol=2e-5, atol=2e-6)
        for label, prefix, max_support in [('pos', 'pos', 7), ('gen', 'gen', 4), ('uncond', 'neg', 3)]:
            ess = stats[tau, label]['ess']
            torch.testing.assert_close(ess.mean(), info[f'laplace_eff_{prefix}_mean'], rtol=3e-4, atol=3e-5)
            assert ess.min() >= 1 - 1e-5 and ess.max() <= max_support + 1e-4
    def toy_apply(params, x, **kwargs):
        return {'raw': x, 'mean': x.mean(1, keepdim=True)}
    data = torch.randn(7, 3, 5)
    out = feature_chunks(toy_apply, None, data, {}, 2, 'cpu')
    for key, value in toy_apply(None, data).items():
        torch.testing.assert_close(out[key], value)
    print('PASS: scale and effective-neighbour statistics match the actual loss; chunk assembly passed.', flush=True)


def stage_group(branch):
    for prefix in ('conv1', 'layer1', 'layer2', 'layer3', 'layer4'):
        if branch == prefix or branch.startswith(prefix + '_'):
            return prefix
    return branch


@torch.no_grad()
def run_diagnostic(args, trainer, kw):
    from utils.dist_util import process_count, set_local_device
    from dataset.dataset import infinite_sampler
    from memory_bank import ArrayMemoryBank
    if process_count() != 1:
        raise ValueError('Run with python on ONE allocated GPU, not four-process torchrun.')
    device = set_local_device(0)
    if device.type == 'cpu':
        raise RuntimeError('The model diagnostic needs a GPU allocation; --self-test is CPU-only.')
    model = kw['model'].cpu()
    checkpoint_step = 0
    if args.checkpoint_workdir:
        # Reuse the trainer's checkpoint interface rather than guessing its format.
        root = Path(args.checkpoint_workdir).resolve(strict=True)
        if root == Path(args.workdir).resolve():
            raise ValueError('Diagnostic workdir must differ from checkpoint workdir.')
        ema = __import__('copy').deepcopy(model)
        opt = kw['optimizer'](model.parameters())
        state = trainer.TrainState(0, model, opt, ema, float(kw.get('ema_decay', 0.999)))
        state = trainer.restore_checkpoint(state=state, workdir=str(root))
        checkpoint_step = int(state.step)
        if checkpoint_step <= 0:
            raise RuntimeError('No positive-step checkpoint restored; refusing silent random initialization.')
        model = state.ema_model if args.weights == 'ema' else state.model
        del state, opt, ema
    else:
        print('INITIALIZATION DIAGNOSTIC: no trained checkpoint selected. Repeat after training before choosing final temperatures.', flush=True)
    model = model.to(device)
    model.train()  # Match training-style generation, but all operations are under no_grad.
    loader = kw['train_loader']
    stream = infinite_sampler(loader, 0)
    pos_bank = ArrayMemoryBank(num_classes=1000, max_size=kw.get('positive_bank_size', 64))
    neg_bank = ArrayMemoryBank(num_classes=1, max_size=kw.get('negative_bank_size', 512))
    def refill():
        batch = kw['preprocess_fn'](next(stream))
        images, labels = batch['images'], batch['labels']
        pos_bank.add(images, labels)
        neg_bank.add(images, labels * 0)
        return labels
    for _ in range(args.bank_batches):
        labels = refill()
    forward = kw.get('forward_dict', {})
    n_gen = int(forward.get('gen_per_label', 8))
    n_pos, n_neg = int(kw.get('pos_per_sample', 32)), int(kw.get('neg_per_sample', 16))
    if min(n_pos, n_neg) < 1 or n_gen < 2:
        raise ValueError('Diagnostic expects positives and unconditional candidates and at least two generated samples.')
    eps = float((kw.get('laplace_kwargs') or {}).get('epsilon', 1e-8))
    act = kw.get('activation_kwargs', {})
    all_stats, scales, shapes, trial_meta = {}, {}, {}, []
    for trial in range(args.batches):
        if trial:
            labels = refill()
        if len(labels) < args.groups:
            raise ValueError('Not enough labels in loader batch for --groups.')
        rng_cpu = torch.Generator().manual_seed(args.seed + trial)
        selected = labels[torch.randperm(len(labels), generator=rng_cpu)[:args.groups]]
        pos = torch.as_tensor(pos_bank.sample(selected, n_samples=n_pos)).cpu()
        neg = torch.as_tensor(neg_bank.sample(selected * 0, n_samples=n_neg)).cpu()
        rng = torch.Generator(device=device).manual_seed(args.seed + trial)
        low, high = float(forward.get('cfg_min', 1)), float(forward.get('cfg_max', 4))
        power = 1 - float(forward.get('neg_cfg_pw', 1))
        frac = torch.rand(args.groups, generator=rng, device=device)
        cfg = torch.exp(math.log(low) + frac * math.log(high / low)) if abs(power) < 1e-6 else (low**power + frac * (high**power - low**power))**(1/power)
        frac2 = torch.rand(args.groups, generator=rng, device=device)
        cfg = torch.where(frac2 < float(forward.get('no_cfg_frac', 0)), torch.ones_like(cfg), cfg)
        input_labels = torch.as_tensor(selected, device=device, dtype=torch.long).repeat_interleave(n_gen)
        generated = model(c=input_labels, cfg_scale=cfg.repeat_interleave(n_gen), deterministic=False, train=True, rng=rng)['samples'].cpu()
        real = torch.cat((pos, neg), dim=1)
        real_flat = real.reshape(-1, *real.shape[2:])
        params, apply = kw['feature_params'], kw['activation_fn']
        rf = feature_chunks(apply, params, real_flat, act, args.feature_batch, device)
        gf = feature_chunks(apply, params, generated, act, args.feature_batch, device)
        if set(rf) != set(gf):
            raise RuntimeError('Generated and real feature keys differ.')
        print(f'Batch {trial+1}/{args.batches}: {len(gf)} actual feature branches.', flush=True)
        if shapes and set(shapes) != set(gf):
            raise RuntimeError('Feature branches changed between batches.')
        trial_meta.append({'labels': torch.as_tensor(selected).tolist(), 'cfg': cfg.cpu().tolist(),
            'unique_positive_candidates': [int(torch.unique(x.reshape(n_pos, -1), dim=0).shape[0]) for x in pos],
            'unique_unconditional_candidates': [int(torch.unique(x.reshape(n_neg, -1), dim=0).shape[0]) for x in neg]})
        del generated, real, real_flat, pos, neg
        for index, name in enumerate(list(gf)):
            g0, r0 = gf.pop(name), rf.pop(name)
            g = g0.reshape(args.groups, n_gen, *g0.shape[1:])
            r = r0.reshape(args.groups, n_pos+n_neg, *r0.shape[1:])
            p, u = r[:, :n_pos], r[:, n_pos:]
            if g.shape[2:] != p.shape[2:]:
                raise RuntimeError(f'Real/generated feature shapes differ for {name}')
            shapes[name] = {'positions': g.shape[2], 'dimensions': g.shape[3], 'group': stage_group(name)}
            scale, stats = branch_statistics(g, p, u, cfg.cpu(), args.taus, eps, args.position_chunk, args.max_positions,
                                             args.seed + trial*10000 + index, device)
            scales.setdefault(name, []).append(scale)
            for (tau, interaction), values in stats.items():
                dest = all_stats.setdefault((name, tau, interaction), {k: [] for k in values})
                for k, v in values.items():
                    dest[k].append(v)
            del g0, r0, g, r, p, u
        del gf, rf
    outdir = Path(args.workdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    counts = {'pos': n_pos, 'gen': n_gen-1, 'uncond': n_neg}
    for (name, tau, interaction), pieces in all_stats.items():
        values = {k: torch.cat(v).float() for k, v in pieces.items()}
        eff = values['ess']
        quantiles = torch.quantile(eff, torch.tensor([0.1, 0.5, 0.9]))
        rows.append({'branch': name, 'group': shapes[name]['group'], 'tau': tau, 'interaction': interaction,
            'candidate_count': counts[interaction], 'query_count': eff.numel(),
            'ess_mean': eff.mean().item(), 'ess_p10': quantiles[0].item(), 'ess_p50': quantiles[1].item(), 'ess_p90': quantiles[2].item(),
            'fraction_ess_le_4': (eff <= 4).float().mean().item(), 'max_weight_mean': values['peak'].mean().item(),
            'active_weight_fraction': values['active'].mean().item(), 'distance_scale_mean': sum(scales[name])/len(scales[name])})
    csv_path = outdir / 'tau_diagnostic.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    metadata = {'checkpoint_step': checkpoint_step, 'checkpoint_workdir': args.checkpoint_workdir, 'weights': args.weights,
        'feature_branch_count': len(shapes), 'branches': shapes, 'feature_config': {'mae_path': args.mae_path_record},
        'groups_per_scale_estimate': args.groups, 'feature_extraction_batch': args.feature_batch,
        'bank_warmup_batches': args.bank_batches, 'seed': args.seed, 'taus': args.taus, 'batches': trial_meta,
        'notes': ['Distances/weights/diagonal masking use the actual loss implementation.',
                  'Shared branch distance scale uses all positions; statistics may sample positions.',
                  'Quantiles pool sampled query rows across batches; p50 uses interpolated torch.quantile.',
                  'Banks are reconstructed, not restored from training; duplicate candidates affect ESS.',
                  'Zero-strength unconditional rows remain in ESS statistics; see active_weight_fraction.',
                  'No optimizer updates; CPU feature storage and small no-grad MAE batches bound GPU usage.']}
    (outdir / 'tau_diagnostic_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'Saved {csv_path}; actual branch count = {len(shapes)}', flush=True)
    if hasattr(kw.get('logger'), 'finish'):
        kw['logger'].finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--config')
    parser.add_argument('--workdir')
    parser.add_argument('--checkpoint-workdir')
    parser.add_argument('--weights', choices=['model', 'ema'], default='model')
    parser.add_argument('--taus', type=float, nargs='+', default=[0.02, 0.05, 0.1, 0.2])
    parser.add_argument('--groups', type=int, default=2)
    parser.add_argument('--batches', type=int, default=4)
    parser.add_argument('--feature-batch', type=int, default=8)
    parser.add_argument('--position-chunk', type=int, default=16)
    parser.add_argument('--max-positions', type=int, default=64)
    parser.add_argument('--bank-batches', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    if not args.config or not args.workdir:
        parser.error('--config and a separate --workdir are required')
    if any(t <= 0 or not math.isfinite(t) for t in args.taus):
        parser.error('Temperatures must be finite and positive')
    if min(args.groups, args.batches, args.feature_batch, args.position_chunk, args.bank_batches) < 1:
        parser.error('Batch/group/chunk arguments must be positive')
    if args.checkpoint_workdir and Path(args.workdir).resolve() == Path(args.checkpoint_workdir).resolve():
        parser.error('--workdir must differ from --checkpoint-workdir')
    import train_laplace as trainer
    config = trainer.load_config(args.config)
    assert config.train.get('use_laplace_unit', False), 'Laplace must be enabled inside train'
    for key in ('grad_accum_steps', 'use_laplace_unit', 'laplace_kwargs', 'ema_decay', 'push_per_step'):
        if key in config:
            raise ValueError(f'{key} is incorrectly at the YAML root instead of inside train')
    import inspect
    inspect.signature(trainer.train_gen).bind_partial(**dict(config.train))
    inspect.signature(trainer.train_step).bind_partial(**dict(config.train.get('forward_dict', {})))
    inspect.signature(laplace_unit_field_loss).bind_partial(**dict(config.train.get('laplace_kwargs', {})))
    args.mae_path_record = str(config.feature.get('mae_path', ''))
    trainer.train_gen = lambda **kw: run_diagnostic(args, trainer, kw)
    trainer.main_gen(config, output_dir=args.workdir)


if __name__ == '__main__':
    main()
