"""
date: 18/3/2025 
Author: Lorenzo Concina

This script compute several metrics for evaluating the decode stage output. 
For ASR: WER
For Intent Classification: Accuracy
For Slot Filling: TODO

"""
import re
import argparse
from load_params import load_params
from evaluate import load
from whisper_normalizer.basic import BasicTextNormalizer
# for english use:
# from whisper_normalizer.english import EnglishTextNormalizer


def parse_file(params):

    ckpt_path = params.decode.ckpt_path
    
    ground_truth_file = params.evaluate.ground_truth_file
    prediction_file = params.evaluate.prediction_file

    with open(ground_truth_file, 'r') as gt, open(prediction_file, 'r') as pred:
        gt_lines = gt.readlines()
        pred_lines = pred.readlines()
    
    gt_keys, gt_transcripts, gt_intents = [], [], []
    pred_keys, pred_transcripts, pred_intents = [], [], []

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
                gt_match = re.match(r'Transcript: (.*?)\.?\s+Intent class: (.*)', gt_parts[1].strip())
                pred_match = re.match(r'Transcript: (.*?)\.?\s+Intent class: (.*)', pred_parts[1].strip())
               
                gt_transcript, gt_intent_class = gt_match.groups()
                pred_transcript, pred_intent_class = pred_match.groups()
                gt_transcripts.append(gt_transcript)
                gt_intents.append(gt_intent_class)
                pred_transcripts.append(pred_transcript)
                pred_intents.append(pred_intent_class)
                #debug prints
                """
                print("gt match: ", gt_match)
                print("pred match: ", pred_match)
                print("gt transcript: ", gt_transcript)
                print("pred transcript: ", pred_transcript)
                print("gt intent: ", gt_intent_class)
                print("pred intent: ", pred_intent_class)
                """
            except:
                print("Invalid lines.")
    #check lists lenghts
    assert len(gt_keys) == len(pred_keys), "Error: The keys lists have different lengths!"
    assert len(gt_transcripts) == len(pred_transcripts), "Error: The keys lists have different lengths!"
    assert len(pred_intents) == len(pred_intents), "Error: The keys lists have different lengths!"
        
    return gt_keys, gt_transcripts, gt_intents, pred_keys, pred_transcripts, pred_intents


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    #load parameters from params.yaml file
    params = load_params(params_path=args.config)

    # load normalizer
    normalizer = BasicTextNormalizer()

    # parse files
    _, ground_truth_texts, ground_truth_intents, _, prediction_texts, prediction_intents= parse_file(params)

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
    Accuracy for Intent Classification computation
    """
    #TODO implement 


    """
    Open and write output log file
    """
    output_file = params.evaluate.evaluate_log
    with open(output_file, "w") as file:
        wer_line = "WER: "+ str(wer)
        file.write(wer_line)
        