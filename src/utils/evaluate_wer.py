"""
date: 12/04/2025 
Author: Lorenzo Concina

This script compute only WER. Used to evaluate MLC SLM decodings
For ASR: WER
"""
import os
import re
import argparse
from load_params import load_params
from evaluate import load
from whisper_normalizer.basic import BasicTextNormalizer
# for english use:
# from whisper_normalizer.english import EnglishTextNormalizer


def parse_file(params, output_dir):

    ckpt_path = params.decode.ckpt_path
    
    ground_truth_file = os.path.join(output_dir, ckpt_path, params.evaluate.ground_truth_file)
    prediction_file = os.path.join(output_dir, ckpt_path, params.evaluate.prediction_file)

    with open(ground_truth_file, 'r') as gt, open(prediction_file, 'r') as pred:
        gt_lines = gt.readlines()
        pred_lines = pred.readlines()
    
    gt_keys, gt_transcripts = [], []
    pred_keys, pred_transcripts = [], []

    for gt_line, pred_line in zip(gt_lines, pred_lines):
        print(gt_line)
        gt_parts = gt_line.strip().split("\t", 1)
        print(gt_parts)
        pred_parts = pred_line.strip().split("\t", 1)
        print(pred_parts)
        print(len(gt_parts))
        if len(gt_parts) == 2 and len(pred_parts) == 2:
            print("inside")
            try:
                #extract key
                gt_keys.append(gt_parts[0])
                pred_keys.append(pred_parts[0])
              

                gt_transcripts.append( gt_parts[1])
                pred_transcripts.append(pred_parts[1])
               
            except:
                print("Invalid lines.")
    #check lists lenghts
    assert len(gt_keys) == len(pred_keys), "Error: The keys lists have different lengths!"
    assert len(gt_transcripts) == len(pred_transcripts), "Error: The keys lists have different lengths!"
    
    return gt_keys, gt_transcripts, pred_keys, pred_transcripts


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args_parser.add_argument('--output_dir', required=True)
    args = args_parser.parse_args()

    #load parameters from params.yaml file
    params = load_params(params_path=args.config)

    # load normalizer
    normalizer = BasicTextNormalizer()

    # parse files
    _, ground_truth_texts, _, prediction_texts = parse_file(params, args.output_dir)

    """
    WER for ASR computations
    """
    # normalize
    ground_truth_texts = [normalizer(text) for text in ground_truth_texts]
    prediction_texts = [normalizer(text) for text in prediction_texts]
    # load and compute word error rate
    wer_metric = load("wer")
    wer = wer_metric.compute(references=ground_truth_texts, predictions=prediction_texts)
    wer = 100*wer
    # get WER percentage
    print("Computed WER for this prediction experiment:", wer)

   
    """
    Open and write output log file
    """
    output_file = params.evaluate.evaluate_log
    output_file_path = os.path.join(args.output_dir, params.decode.ckpt_path, output_file)
    with open(output_file_path, "w") as file:
        wer_line = "CER: "+ str(wer) + "\n"
        file.write(wer_line)
        
        