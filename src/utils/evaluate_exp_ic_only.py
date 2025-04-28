"""
date: 18/3/2025 
Author: Lorenzo Concina

This script compute several metrics for evaluating the decode stage output. 
For ASR: WER
For Intent Classification: Accuracy
For Slot Filling: TODO

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
    
    gt_keys, gt_intents = [], []
    pred_keys, pred_intents = [], []

    for gt_line, pred_line in zip(gt_lines, pred_lines):
        gt_parts = gt_line.strip().split("\t", 1)
        pred_parts = pred_line.strip().split("\t", 1)
        if len(gt_parts) == 2 and len(pred_parts) == 2:
            try:
                #extract key
                gt_keys.append(gt_parts[0])
                pred_keys.append(pred_parts[0])
                #debug prints
                """
                print("Ground truth key: ", gt_parts[0])
                print("Ground truth asr+ic:", gt_parts[1])
                print("Pred truth key: ", pred_parts[0])
                print("Pred truth asr+ic:", pred_parts[1])
                """

                #extract transcription, intent and others...
                gt_match = re.match(r'.*Intent class: (.*)', gt_parts[1].strip())
                pred_match = re.match(r'.*Intent class: (.*)', pred_parts[1].strip())
               
                gt_intent_class = gt_match.groups()
                pred_intent_class = pred_match.groups()
                gt_intents.append(gt_intent_class[0])
                pred_intents.append(pred_intent_class[0])
                #debug prints
                """
                print("gt match: ", gt_match)
                print("pred match: ", pred_match)
                print("gt intent: ", gt_intent_class)
                print("pred intent: ", pred_intent_class)
                """
            except:
                print("Invalid lines.")
    #check lists lenghts
    assert len(gt_keys) == len(pred_keys), "Error: The keys lists have different lengths!"
    assert len(pred_intents) == len(pred_intents), "Error: The keys lists have different lengths!"
   
    return gt_keys, gt_intents, pred_keys, pred_intents


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
    _, ground_truth_intents, _, prediction_intents= parse_file(params, args.output_dir)

    
    """
    Accuracy for Intent Classification computation
    """
    # normalize intents
    ground_truth_intents = [normalizer(text) for text in ground_truth_intents]
    prediction_intents = [normalizer(text) for text in prediction_intents]

    #correct_intents = sum(1 for gt, pred in zip(ground_truth_intents, prediction_intents) if gt == pred)
    correct_intents = 0

    for gt, pred in zip(ground_truth_intents, prediction_intents):
        if gt == pred:
            correct_intents += 1
            #print(f"Correct prediction: ground truth = {gt}, prediction = {pred}")
        #else:
        #    print(f"Mismatch: ground truth = {gt}, prediction = {pred}")

    intent_accuracy = 100 * (correct_intents / len(ground_truth_intents))
    print("Computed Intent Classification Accuracy:", intent_accuracy)


    """
    Open and write output log file
    """
    output_file = params.evaluate.evaluate_log
    output_file_path = os.path.join(args.output_dir, params.decode.ckpt_path, output_file)
    with open(output_file_path, "w") as file:
        ic_line = "Intent Classification Accuracy: " + str(intent_accuracy) + "\n"
        file.write(ic_line)
        