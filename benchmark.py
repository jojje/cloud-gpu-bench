import argparse
import json
import sys
import torch
import torchvision.models
from pytorch_benchmark import benchmark

if not torch.cuda.is_available():
    print("FATAL: CUDA is not available, can't benchmark GPU")
    sys.exit(1)


def main():
    opts = parse_args()
    print('[*] initializing benchmark ...')

    model_type = getattr(torchvision.models, opts.model)

    model = model_type().to('cuda')
    sample = torch.randn(opts.batch_size, 3, opts.size, opts.size).to('cuda')
    result = benchmark(
        model,
        sample,
        num_runs=opts.steps,
        batch_size=opts.batch_size,
        sample_with_batch_size1=False)
    
    speed = result['timing'][f'batch_size_{opts.batch_size}']['total']['human_readable']['batches_per_second']
    flops = result['flops']
    params = result['params']

    info = result['machine_info']
    kernel = info['system']['release']
    
    cpu = info['cpu']
    cpu_model = cpu['model']
    cpu_cores = '{} physical ({} virtual)'.format(cpu['cores']['physical'], cpu['cores']['total'])
    cpu_freq = cpu['frequency']

    ram = info['memory']['total']
    ram_free = info['memory']['available']

    gpu = ', '.join(f'{g["name"]} (VRAM: {g["memory"]})' for g in info['gpus'])
    ngpu = len(info['gpus'])
    
    report = report_json if opts.json else report_text
    report(speed, flops, opts.model, opts.size, opts.batch_size, opts.steps, params, gpu, ngpu, cpu_model, cpu_cores, ram, ram_free, kernel)

def report_text(speed, flops, model, size, batch_size, steps, params, gpu, ngpu, cpu_model, cpu_cores, ram, ram_free, kernel):
    lines = ['',
        '==[ Benchmark result ]=====',
        f'Batches/second  : {speed}',
        f'MFLOPS          : {flops / (1<<20):.0f}',
        '',
        '==[ Test configuration ]==',
        f'Model           : {model}',
        f'Image dimension : ({size}, {size})',
        f'Batch size      : {batch_size}',
        f'Training steps  : {steps}',
        f'Model params    : {params}',
        '',
        '==[ Machine spec ]========',
        f'GPU             : {gpu}',
        f'GPU #           : {ngpu}',
        f'CPU             : {cpu_model}, Cores: {cpu_cores}',
        f'RAM             : {ram} total, {ram_free} free',
        f'Kernel          : {kernel}',
    ]
    print('\n'.join(lines))


def report_json(speed, flops, model, size, batch_size, steps, params, gpu, ngpu, cpu_model, cpu_cores, ram, ram_free, kernel):
    speed_mean = float(speed.split(" ")[0])
    speed_high = float(speed.split("[")[-1].split(",")[0])
    speed_low = float(speed.split(",")[-1].split("]")[0])

    print("\n" + json.dumps({
        'result': {
            'batches_per_second': speed_mean,
            'batches_per_second_min': speed_low,
            'batches_per_second_max': speed_high,
            'flops': flops,
        },
        'config': {
            'model': model,
            'image_dimension': (size, size),
            'batch_size': batch_size,
            'training_steps': steps,
            'model_params': params,
        },
        'matchine_specs': {
            'gpu': gpu,
            'gpu_count': ngpu,
            'cpu': cpu_model,
            'ram_gib': float(ram.split(" ")[0]),
            'ram_free_gib': float(ram_free.split(" ")[0]),
            'kernel': kernel,
        },
    }))

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Simple pytorch GPU benchmark'
    )

    model_names = [m for m in dir(torchvision.models) 
                   if getattr(torchvision.models, m).__class__.__name__ == 'function']

    parser.add_argument('--steps', default=1000, type=int, metavar='n',
                        help='Number of training steps (batches) to run')
    parser.add_argument('--size', default=224, type=int, metavar='n',
                        help='Image size (produces a square dimension image) to train')
    parser.add_argument('--batch-size', default=8, type=int, metavar='n',
                        help='Number of training steps (batches) to run')
    parser.add_argument('--model', default='resnet50', choices=model_names, metavar='model',
                        help=f'Model to use for the benchmark. Available models: {", ".join(model_names)}')
    parser.add_argument('--json', action='store_true', help='Output report in json instead of text form')

    return parser.parse_args()


if __name__ == '__main__':
    main()

