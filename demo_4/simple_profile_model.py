#######################################################################
# This code was written by:
# Mojtaba AlShams
#
# LAST code modification on 15Jan2025
#
# This code generate the model computation latency and energy (power converted to eng.) profiles
# with different batch-sizes when running DNN models.
# Current considered devices are MAC GPU (MPS), Nvidia GPUs, Jetson devices, CPUs
#
# For questions and comments, Mojtaba can be reached on:
# University email: mojtaba.alshams@kaust.edu.sa
# Personal email: m.f.shams@hotmail.com
#######################################################################
import torch
import time
import argparse
import os
import json

import models as m
import profilers as p

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="DNN models profiler supported by heterogenous accelerators/devices",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-dn", "--device-name", type=str,
                        help="Device name; e.g.: 'M2GPU' or 'A100'")
    parser.add_argument("-d", "--device", type=str,
                        help="Device type; e.g.: 'cuda' or 'mps'")
    parser.add_argument("-m", "--model-name", type=str,
                        choices=m.get_models_names(),
                        help="the neural network model for loading")
    parser.add_argument("-pt", "--profile-type", type=str,
                        choices=['time','energy','all'],
                        help="the model profiling type")
    parser.add_argument("-b", "--batch-size", default=1, type=int, help="batch size")
    parser.add_argument("-i", "--iterations", default=100, type=int,
                        help="iterations to average profiling for")
    parser.add_argument("-w", "--warmup", action="store_true", default=True,
                        help="perform a warmup iteration (strongly recommended)")
    parser.add_argument("--no-warmup", action="store_false", dest="warmup",
                        help="don't perform a warmup iteration")

    args = parser.parse_args()
    if args.warmup:
        args.warmup_iters = 20
    else:
        args.warmup_iters = 0
    if args.model_name=='ResNet152':
        model = m.resnet152()
        inputs = torch.randn(args.batch_size, 3, 224, 224)
    model.eval()
    tok = time.time()
    time_list, energy_list, time_avg_std, energy_avg_std = p.profile_model(args, model, inputs)
    print('Profiling is DONE!')
    tik = time.time()
    print(f'it took {(tik-tok)/60:0.00f} minutes to finish')
    write_profile_json(args, time_list, energy_list, time_avg_std, energy_avg_std)

def write_profile_json(args, time_list, energy_list, time_avg_std, energy_avg_std):
    fpath = find_fpath(args)
    profile = {}
    profile['config'] = m.get_cnfig(args.model_name)
    profile['time avg'] = time_avg_std[0]
    profile['time std'] = time_avg_std[1]
    profile['energy avg'] = energy_avg_std[0]
    profile['energy std'] = energy_avg_std[1]
    profile['time_list'] = time_list
    profile['energy_list'] = energy_list
    json_object = json.dumps(profile, indent=4)
    with open(fpath, "w") as outfile:
        outfile.write(json_object)

def find_fpath(args):
    folder_path = 'profiles'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    folder_path = os.path.join(folder_path, args.device_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    folder_path = os.path.join(folder_path, args.model_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    m_config = m.get_cnfig(args.model_name)
    f_name = folder_path+'/devName_'+args.device_name+'_modelName_'+args.model_name+'_numLayers_'+m_config['layers']+'_inputSize_'+m_config['inputSize']+'_dtypeSize_'+m_config['dtypeSize']+'_batchSize'+str(args.batch_size)+ '_iterations' +str(args.iterations)
    return f_name

if __name__=="__main__":
    main()