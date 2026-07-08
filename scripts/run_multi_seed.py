from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run CC-SGCL for multiple seeds')
    parser.add_argument('--config', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--data', type=int, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--gpu', default='')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script = Path(__file__).resolve().parent / 'run_cc_sgcl.py'
    for seed in args.seeds:
        cmd = [
            sys.executable,
            str(script),
            '--config', args.config,
            '--data_root', args.data_root,
            '--data', str(args.data),
            '--seed', str(seed),
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--output_dir', args.output_dir,
        ]
        env = None
        if args.gpu != '':
            env = dict(**__import__('os').environ)
            env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print('Running:', ' '.join(cmd))
        subprocess.run(cmd, check=True, env=env)


if __name__ == '__main__':
    main()
