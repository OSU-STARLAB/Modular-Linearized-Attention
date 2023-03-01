# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import soundfile as sf
import sys
import torch
import torchaudio

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.logging import progress_bar
from fairseq.tasks.text_to_speech import plot_tts_output
from fairseq.data.audio.text_to_speech_dataset import TextToSpeechDataset

import cProfile

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_parser():
    parser = options.get_speech_generation_parser()
    parser.add_argument("--dump-features", action="store_true")
    parser.add_argument("--dump-waveforms", action="store_true")
    parser.add_argument("--dump-attentions", action="store_true")
    parser.add_argument("--dump-eos-probs", action="store_true")
    parser.add_argument("--dump-plots", action="store_true")
    parser.add_argument("--dump-target", action="store_true")
    parser.add_argument("--output-sample-rate", default=22050, type=int)
    parser.add_argument("--teacher-forcing", action="store_true")
    parser.add_argument(
        "--audio-format", type=str, default="wav", choices=["wav", "flac"]
    )
    return parser


def postprocess_results(
        dataset: TextToSpeechDataset, sample, hypos, resample_fn, dump_target
):
    def to_np(x):
        return None if x is None else x.detach().cpu().numpy()

    sample_ids = [dataset.ids[i] for i in sample["id"].tolist()]
    texts = sample["src_texts"] if "src_texts" in sample else [""] * len(hypos)
    attns = [to_np(hypo["attn"]) for hypo in hypos]
    eos_probs = [to_np(hypo.get("eos_prob", None)) for hypo in hypos]
    feat_preds = [to_np(hypo["feature"]) for hypo in hypos]
    wave_preds = [to_np(resample_fn(h["waveform"])) for h in hypos]
    if dump_target:
        feat_targs = [to_np(hypo["targ_feature"]) for hypo in hypos]
        wave_targs = [to_np(resample_fn(h["targ_waveform"])) for h in hypos]
    else:
        feat_targs = [None for _ in hypos]
        wave_targs = [None for _ in hypos]

    return zip(sample_ids, texts, attns, eos_probs, feat_preds, wave_preds,
               feat_targs, wave_targs)


def dump_result(
        is_na_model,
        args,
        vocoder,
        sample_id,
        text,
        attn,
        eos_prob,
        feat_pred,
        wave_pred,
        feat_targ,
        wave_targ,
):
    sample_rate = args.output_sample_rate
    out_root = Path(args.results_path)
    if args.dump_features:
        feat_dir = out_root / "feat"
        feat_dir.mkdir(exist_ok=True, parents=True)
        np.save(feat_dir / f"{sample_id}.npy", feat_pred)
        if args.dump_target:
            feat_tgt_dir = out_root / "feat_tgt"
            feat_tgt_dir.mkdir(exist_ok=True, parents=True)
            np.save(feat_tgt_dir / f"{sample_id}.npy", feat_targ)
    if args.dump_attentions:
        attn_dir = out_root / "attn"
        attn_dir.mkdir(exist_ok=True, parents=True)
        #np.save(attn_dir / f"{sample_id}.npy", attn.numpy())
        np.save(attn_dir / f"{sample_id}.npy", attn)
    if args.dump_eos_probs and not is_na_model:
        eos_dir = out_root / "eos"
        eos_dir.mkdir(exist_ok=True, parents=True)
        np.save(eos_dir / f"{sample_id}.npy", eos_prob)

    if args.dump_plots:
        images = [feat_pred.T] if is_na_model else [feat_pred.T, attn]
        names = ["output"] if is_na_model else ["output", "alignment"]
        if feat_targ is not None:
            images = [feat_targ.T] + images
            names = [f"target (idx={sample_id})"] + names
        if is_na_model:
            plot_tts_output(images, names, attn, "alignment", suptitle=text)
        else:
            plot_tts_output(images, names, eos_prob, "eos prob", suptitle=text)
        plot_dir = out_root / "plot"
        plot_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(plot_dir / f"{sample_id}.png")
        plt.close()

    if args.dump_waveforms:
        ext = args.audio_format
        if wave_pred is not None:
            wav_dir = out_root / f"{ext}_{sample_rate}hz_{vocoder}"
            wav_dir.mkdir(exist_ok=True, parents=True)
            sf.write(wav_dir / f"{sample_id}.{ext}", wave_pred, sample_rate)
        if args.dump_target and wave_targ is not None:
            wav_tgt_dir = out_root / f"{ext}_{sample_rate}hz_{vocoder}_tgt"
            wav_tgt_dir.mkdir(exist_ok=True, parents=True)
            sf.write(wav_tgt_dir / f"{sample_id}.{ext}", wave_targ, sample_rate)


def main(args):
    assert(args.dump_features or args.dump_waveforms or args.dump_attentions
           or args.dump_eos_probs or args.dump_plots)
    if args.max_tokens is None and args.batch_size is None:
        args.max_tokens = 8000
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu
    task = tasks.setup_task(args)
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.path],
        task=task,
        arg_overrides=ast.literal_eval(args.model_overrides),
    )
    model = models[0].cuda() if use_cuda else models[0]
    # use the original n_frames_per_step
    task.args.n_frames_per_step = saved_cfg.task.n_frames_per_step
    task.load_dataset(args.gen_subset, task_cfg=saved_cfg.task)

    data_cfg = task.data_cfg
    sample_rate = data_cfg.config.get("features", {}).get("sample_rate", 22050)
    resample_fn = {
        False: lambda x: x,
        True: lambda x: torchaudio.sox_effects.apply_effects_tensor(
            x.detach().cpu().unsqueeze(0), sample_rate,
            [['rate', str(args.output_sample_rate)]]
        )[0].squeeze(0)
    }.get(args.output_sample_rate != sample_rate)
    if args.output_sample_rate != sample_rate:
        logger.info(f"resampling to {args.output_sample_rate}Hz")

    generator = task.build_generator([model], args)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=(sys.maxsize, sys.maxsize),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
        data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)

    Path(args.results_path).mkdir(exist_ok=True, parents=True)
    is_na_model = getattr(model, "NON_AUTOREGRESSIVE", False)
    dataset = task.dataset(args.gen_subset)
    #print(dataset, flush=True)
    vocoder = task.args.vocoder
    logger.info(f"Inference on test set is beginning.")
    with progress_bar.build_progress_bar(args, itr) as t:
        for sample in t:
            # manual information for processed tokens, a better solution should be used later
            lut_pho = {
                "W" : 6.3,
                "AY1" : 13.52,
                "L" : 7.97,
                "spn" : 47.84,
                "F" : 9.7,
                "R" : 6.61,
                "AH0" : 5.24,
                "Z" : 9.74,
                "D" : 6.0,
                "IH1" : 6.44,
                "JH" : 8.99,
                "N" : 5.94,
                "T" : 7.6,
                "IY0" : 9.09,
                "EH0" : 6.79,
                "G" : 6.92,
                "EY1" : 12.47,
                "P" : 7.94,
                "AA1" : 10.07,
                "HH" : 7.16,
                "ER0" : 9.18,
                "S" : 10.66,
                "EH1" : 7.88,
                "IH0" : 5.31,
                "M" : 6.9,
                "OW1" : 11.61,
                "AE1" : 9.76,
                "K" : 8.37,
                "UW1" : 9.71,
                "EY0" : 11.87,
                "AH1" : 6.04,
                "DH" : 3.99,
                "ER1" : 11.13,
                "IY1" : 10.76,
                "sp" : 15.01,
                "AW1" : 14.96,
                "B" : 5.53,
                "AE0" : 6.73,
                "V" : 6.26,
                "AH2" : 6.73,
                "UH1" : 5.9,
                "AO1" : 10.8,
                "Y" : 7.08,
                "UW0" : 6.86,
                "NG" : 8.51,
                "AW2" : 11.33,
                "IH2" : 5.53,
                "SH" : 10.39,
                "TH" : 9.72,
                "CH" : 10.08,
                "OW0" : 9.06,
                "ZH" : 8.32,
                "EH2" : 6.11,
                "EY2" : 9.83,
                "AY0" : 9.3,
                "AY2" : 13.84,
                "OW2" : 8.84,
                "AO2" : 7.15,
                "IY2" : 7.83,
                "OY1" : 14.51,
                "AE2" : 10.21,
                "AO0" : 6.27,
                "UH2" : 5.95,
                "AA2" : 8.56,
                "AA0" : 7.07,
                "UW2" : 7.15,
                "UH0" : 3.06,
                "ER2" : 7.18,
                "AW0" : 15.29,
                "OY2" : 15.93,
                "OY0" : 19.25,
            }

            lut = {}
            for key in lut_pho:
                lut[task.src_dict.index(key)] = lut_pho[key]

            #print(lut)

            # hard code lut for now, add a switch later
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            #cProfile.runctx('generator.generate(model, sample, has_targ=args.dump_target, lut=lut)', globals(), locals(), 'restats')
            hypos = generator.generate(model, sample, has_targ=args.dump_target, lut=lut)
            for result in postprocess_results(
                    dataset, sample, hypos, resample_fn, args.dump_target
            ):
                dump_result(is_na_model, args, vocoder, *result)
    
    logger.info(f"Inference is finished.")

def cli_main():
    parser = make_parser()
    args = options.parse_args_and_arch(parser)
    main(args)

if __name__ == "__main__":
    cli_main()
