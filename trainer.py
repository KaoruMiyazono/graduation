import collections
import json
import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from datasets import get_dataset
from methods import WiSR
from methods_me import Baseline,Baseline_with_arc,Baseline_two_attn,Baseline_with_arc_concat
from evaluator import Evaluator
from lib import misc
from lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from torch.utils.data import DataLoader, ConcatDataset


def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch
def json_handler(v):
    if isinstance(v, (Path, range)):
        return str(v)
    raise TypeError(f"`{type(v)}` is not JSON Serializable")


def train(target_env, args, hparams, n_steps, checkpoint_freq, logger):
    if torch.cuda.is_available():
        # device = "cuda"
        device="cuda:0"
    else:
        device = "cpu"
    
    logger.info("")

    #######################################################
    # 防止warning
    ######################################################
    # import torch.backends.cudnn as cudnn
    # cudnn.benchmark = False
    # cudnn.deterministic = True
    # torch.backends.cudnn.allow_tf32 = False 
    # torch.backends.cuda.matmul.allow_tf32 = False


    #######################################################
    # setup dataset & loader
    #######################################################
    test_envs=[target_env]
    # print("我到了")
    dataset, train_dataset, test_dataset = get_dataset(args, hparams)



    train_envs = args.source_domains
    batch_size = args.batch_size
    
    n_traindata_env=[]    
    n_testdata_env=[]
    for env in train_dataset:
        n_traindata_env.append(len(env[0])) #算每个domain的样本个数
    for env in test_dataset:
        n_testdata_env.append(len(env[0]))
    len_train_data = sum(n_traindata_env)
    len_test_data = sum(n_testdata_env)
    if min(n_traindata_env)<batch_size:
        batch_size=min(n_traindata_env)
    hparams['batch_size']=batch_size

    logger.info(f"Batch size: {batch_size} ")

    combined_dataset = ConcatDataset([env for env, _ in train_dataset])
    env_weights=None
    train_loaders = DataLoader(
        dataset=combined_dataset,
        batch_size=batch_size,
        num_workers=dataset.N_WORKERS,
        shuffle=True,  # 如果需要随机采样
        drop_last=True
    )
    train_minibatches_iterator = infinite_dataloader(train_loaders)

    combined_dataset_test = ConcatDataset([env for env, _ in test_dataset])
    test_loaders = DataLoader(
        dataset=combined_dataset_test,
        batch_size=batch_size,
        num_workers=dataset.N_WORKERS,
        shuffle=True,  # 如果需要随机采样
        drop_last=True
    )
        
    test_minibatches_iterator = infinite_dataloader(test_loaders)
    
    # train_minibatches_iterator = zip(*train_loaders)
    
    # dataloaders 锁死
    ################这里修改了##############
    # train_loaders = [InfiniteDataLoader(
    #         dataset=env,
    #         weights=env_weights,
    #         batch_size=batch_size,
    #         num_workers=dataset.N_WORKERS)
    #         for env, env_weights in train_dataset]
    # train_minibatches_iterator = zip(*train_loaders)
    ################这里修改了##############
    
    eval_loaders = [FastDataLoader(
            dataset=env,
            batch_size=min(min(n_testdata_env),64),
            num_workers=dataset.N_WORKERS)
            for env, _ in test_dataset]


    
    # print(train_loaders)
    
    eval_loader_names = ["target_env{}".format(i) for i in range(len(test_dataset))]
    eval_meta = list(zip(eval_loader_names, eval_loaders, [None]*len(test_dataset)))
    # print(eval_loaders)
    # print(eval_meta)
    evaluator = Evaluator(
        test_envs,
        eval_meta,
        train_envs,
        logger,
        evalmode=args.evalmode,
        debug=args.debug,
        target_env=target_env,
    )

    #######################################################
    # setup algorithm (model)
    #######################################################
    # algorithm = WiSR(
    #     dataset.input_shape,
    #     dataset.num_classes,
    #     len(train_dataset),
    #     hparams,
    # )

    # algorithm = Baseline(
    #     dataset.input_shape,
    #     dataset.num_classes,
    #     len(train_dataset),
    #     hparams,
    # )

    # algorithm = Baseline_two_attn(
    #     dataset.input_shape,
    #     dataset.num_classes,
    #     len(train_dataset),
    #     hparams,
    # )

    # algorithm = Baseline_with_arc(
    #     dataset.input_shape,
    #     dataset.num_classes,
    #     len(train_dataset),
    #     hparams,
    # )

    # algorithm = Baseline_with_arc_concat(
    #     dataset.input_shape,
    #     dataset.num_classes,
    #     len(train_dataset),
    #     hparams,
    # )
    
    
    

    algorithm = Baseline_with_arc(
        dataset.input_shape,
        dataset.num_classes,
        len(train_dataset),
        hparams,
    )

    algorithm.to(device)

    n_params = sum([p.numel() for p in algorithm.parameters()])
    logger.info("# of params = %d" % n_params)

    checkpoint_vals = collections.defaultdict(lambda: [])

    # calculate steps per epoch
    steps_per_epoch = int(len_train_data/args.batch_size)
    if steps_per_epoch==0:
        steps_per_epoch=1
    logger.info(f"Number of steps per epoch: {steps_per_epoch}")
    epochs = int(n_steps / steps_per_epoch)

    last_results_keys = None
    records = []
    epochs_path = args.out_dir / "results.jsonl"
    best_test_acc=0

    for step in range(n_steps):
        step_start_time = time.time()
        epoch = step / steps_per_epoch

        # minibatches_device = [
        #         (x.to(device), y.to(device)) for ((x, y, d),idx) in next(train_minibatches_iterator)
        #     ]

        sample = next(train_minibatches_iterator)
        (x,y,d),idx=sample
        minibatches_device = [x.to(device), y.to(device)]

        sample_test=next(test_minibatches_iterator)
        (x_test,_,_),idx=sample_test
        # print("到了")
        # exit(0)
        step_vals = algorithm.update(minibatches_device,x_test.to(device),args)
        

        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if step % checkpoint_freq == 0 or (step == n_steps - 1):
            results = {
                "step": step,
                #"epoch": step / steps_per_epoch,
            }


            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            accuracies, summaries,losses = evaluator.evaluate(algorithm)
            # print(accuracies["target_env0"],losses["target_env0"])
            if accuracies["target_env0"]>best_test_acc:
                best_test_acc = accuracies["target_env0"]
            
            results_keys = list(summaries.keys()) + sorted(accuracies.keys()) + list(results.keys())
            # merge results
            results.update(summaries)
            results.update(accuracies)

            # print
            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                logger.info(misc.to_row(results_keys))
                last_results_keys = results_keys
            logger.info(misc.to_row([results[key] for key in results_keys]))
            records.append(copy.deepcopy(results))

            # update results to record
            results.update({
                "hparams": dict(hparams), 
                "args": vars(args)
                })


            checkpoint_vals = collections.defaultdict(lambda: [])


            if args.model_save and step >= args.model_save:
                ckpt_dir = args.out_dir / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)

                test_env_str = ",".join(map(str, test_envs))
                filename = "TE{}_{}.pth".format(test_env_str, step)
                if len(test_envs) > 1 and target_env is not None:
                    train_env_str = ",".join(map(str, train_envs))
                    filename = f"TE{target_env}_TR{train_env_str}_{step}.pth"
                path = ckpt_dir / filename

                save_dict = {
                    "args": vars(args),
                    "model_hparams": dict(hparams),
                    "test_envs": test_envs,
                    "model_dict": algorithm.cpu().state_dict(),
                }
                algorithm.to(device)
                if not args.debug:
                    torch.save(save_dict, path)
                else:
                    logger.debug("DEBUG Mode -> no save (org path: %s)" % path)


    return [len_train_data,len_test_data,best_test_acc]

    
