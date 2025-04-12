#######################################################################
# This code was written by:
# Mojtaba AlShams
#
# For questions and comments, Mojtaba can be reached on:
# University email: mojtaba.alshams@kaust.edu.sa
# Personal email: m.f.shams@hotmail.com
#######################################################################

import torch
import time
import numpy as np
from tqdm import tqdm

def profile_model(args, model, inputs):
    if args.device in ['cuda','jetson']:
        device = torch.device('cuda')
    elif args.device in ['intl_cpu','cpu']:
        device = torch.device('cpu')
    if args.device == 'mps':
        device = torch.device('mps')
        time_list, energy_list = profile_model_mps(args, model, inputs, device)
    elif args.device == 'cuda':
        time_list, energy_list = profile_model_nvidia(args, model, inputs, device)
    elif args.device == 'jetson':
        time_list, energy_list = profile_model_jetson(args, model, inputs, device)
    elif args.device == 'intl_cpu':
        time_list, energy_list = profile_model_intlcpu(args, model, inputs, device)
    elif args.device == 'cpu':
        time_list, energy_list = profile_model_cpu(args, model, inputs, device)
    else:
        raise ValueError("Unknown device!")
    return time_list, energy_list, (np.average(time_list), np.std(time_list)), (np.average(energy_list), np.std(energy_list))

def profile_model_mps(args, model, inputs, device):
    # assumes a M2 MAC Metal Performance Shaders (MPS)
    model = model.to(device=device)
    inputs = inputs.to(device=device)
    time_list, energy_list = [], []
    if args.profile_type in ['time','all']:
        for _ in range(args.warmup_iters):
            outputs = model(inputs)
            start_time = time.time()
        torch.mps.synchronize()  # Ensure all pending tasks are complete before starting
        for _ in tqdm(range(args.iterations), desc ='model latency profiling...'):
        # for _ in range(args.iterations):
            start_time = time.time()
            outputs = model(inputs)
            torch.mps.synchronize()  # Ensure all tasks are complete before ending
            end_time = time.time()
            time_list.append(end_time-start_time)
    if args.profile_type in ['energy','all']:
        print('WARNING: the modle was not profiled for energy.\nmps is not yet supported for that!')
    time_list = np.array(time_list)
    return time_list, energy_list

def profile_model_nvidia(args, model, inputs, device):
    # Assumes a nvidia GPU mounted using a PCIe card
    # It also assumes cuda-0 is being used
    model = model.to(device=device)
    inputs = inputs.to(device=device)
    time_list, energy_list = [], []
    if args.profile_type in ['time','all']:
        time_list = []
        for _ in range(args.warmup_iters):
            outputs = model(inputs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()  # Ensure all pending tasks are complete before starting
        for _ in tqdm(range(args.iterations), desc ='model latency profiling...'):
        # for _ in range(args.iterations):
            start.record()
            outputs = model(inputs)
            torch.cuda.synchronize()  # Ensure all tasks are complete before ending
            end.record()
            elapsed_time = start.elapsed_time(end)
            time_list.append(elapsed_time)
    time_list = np.array(time_list)
    if args.profile_type in ['energy','all']:
        from pynvml.smi import nvidia_smi
        nvsmi = nvidia_smi.getInstance()
        power_list = []
        for _ in range(args.warmup_iters):
            outputs = model(inputs)
        torch.cuda.synchronize()  # Ensure all pending tasks are complete before starting
        for _ in tqdm(range(args.iterations), desc ='model power profiling...'):
        # for _ in range(args.iterations):
            outputs = model(inputs)
            power_list.append(nvsmi.DeviceQuery('power.draw')['gpu'][0]['power_readings']['power_draw'])
        energy_list = np.array(power_list)*time_list
    return time_list, energy_list

def profile_model_jetson(args, model, inputs, device):
    # Assumes a nvidia Jetson device/kit
    model = model.to(device=device)
    inputs = inputs.to(device=device)
    time_list, energy_list = [], []
    if args.profile_type in ['time','all']:
        time_list = []
        for _ in range(args.warmup_iters):
            outputs = model(inputs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()  # Ensure all pending tasks are complete before starting
        for _ in tqdm(range(args.iterations), desc ='model latency profiling...'):
        # for _ in range(args.iterations):
            start.record()
            outputs = model(inputs)
            torch.cuda.synchronize()  # Ensure all tasks are complete before ending
            end.record()
            elapsed_time = start.elapsed_time(end)
            time_list.append(elapsed_time)
    time_list = np.array(time_list)
    if args.profile_type in ['energy','all']:
        from jtop import jtop
        with jtop() as jetson:
            power_list = []
            for _ in range(args.warmup_iters):
                outputs = model(inputs)
                if jetson.ok():
                    inst_power = jetson.power['tot']['power']
            torch.cuda.synchronize()  # Ensure all pending tasks are complete before starting
            count = 0
            for _ in tqdm(range(args.iterations), desc ='model power profiling...'):
            # for _ in range(args.iterations):
                outputs = model(inputs)
                if jetson.ok():
                    power_list.append(jetson.power['tot']['power'])
                    count += 1
            if count<args.iterations:
                print(f'WARNING: the model was profiled for {count} instead of {args.iterations} times!')
        energy_list = np.array(power_list)*time_list
    return time_list, energy_list

def profile_model_intlcpu(args, model, inputs, device):
    # Assumes an intel cpu that runs on a linux OS
    model = model.to(device=device)
    inputs = inputs.to(device=device)
    time_list, energy_list = [], []
    if args.profile_type in ['time','all']:
        for _ in range(args.warmup_iters):
            outputs = model(inputs)
            start_time = time.time()
        for _ in tqdm(range(args.iterations), desc ='model latency profiling...'):
        # for _ in range(args.iterations):
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            time_list.append(end_time-start_time)
    if args.profile_type in ['energy','all']:
        from pyJoules.energy_meter import measure_energy
        from pyJoules.device.rapl_device import RaplPackageDomain
        from pyJoules.device.rapl_device import RaplDramDomain
        from pyJoules.handler.csv_handler import CSVHandler
        print('WARNING: the modle was not profiled for energy.\nthis feature is yet completed!')
    time_list = np.array(time_list)
    return time_list, energy_list
def profile_model_cpu(args, model, inputs, device):
    model = model.to(device=device)
    inputs = inputs.to(device=device)
    time_list, energy_list = [], []
    if args.profile_type in ['time','all']:
        for _ in range(args.warmup_iters):
            outputs = model(inputs)
            start_time = time.time()
        for _ in tqdm(range(args.iterations), desc ='model latency profiling...'):
        # for _ in range(args.iterations):
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            time_list.append(end_time-start_time)
    if args.profile_type in ['energy','all']:
        print('WARNING: the modle was not profiled for energy.\nonly intel cpu running on linux is supported!')
    time_list = np.array(time_list)
    return time_list, energy_list