"""
Compute the evaluation results for POPE-n900
"""

######################################################################################
# Standard library imports
import json

# Local application imports
## Logger
from logger import logger
######################################################################################

def safe_division(numerator, denominator):
    """Safely perform division, avoiding division by zero"""
    
    return numerator / denominator if denominator != 0 else 0.0

def calculate_metrics(TP, FP, TN, FN):
    """Calculate precision, recall, F1, and accuracy"""

    # For positive (yes) class
    precision_p = safe_division(TP, TP + FP)
    recall_p = safe_division(TP, TP + FN)
    f1_p = safe_division(2 * precision_p * recall_p, precision_p + recall_p)

    # For negative (no) class
    precision_n = safe_division(TN, TN + FN)
    recall_n = safe_division(TN, TN + FP)
    f1_n = safe_division(2 * precision_n * recall_n, precision_n + recall_n)

    acc = safe_division(TP + TN, TP + TN + FP + FN)

    return precision_p, recall_p, f1_p, precision_n, recall_n, f1_n, acc

def compute_result(args, argw, output_folder, ans_file, label_file, tortureList):
    """Evaluate model's answers against correct labels"""

    logger.info("Start evaluating")

    answers = [json.loads(q) for q in open(ans_file, "r")]
    answers = answers[0:args.sample] # depend on user-specified values, "--sample"
    label_list = [json.loads(q)["label"] for q in open(label_file, "r")]

    for answer in answers:
        text = answer["answer"]

        if text.find(".") != -1:
            text = text.split(".")[0]

        text = text.replace(",", "")
        words = text.split(" ")
        if "No" in words or "not" in words or "no" in words or "don\'t" in words or "cannot" in words:
            answer["answer"] = "no"
        else:
            answer["answer"] = "yes"

    for i in range(len(label_list)):
        if label_list[i] == "no":
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer["answer"] == "no":
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)
    yes_ratio *= 100

    # NOTE: track samples that are not predicted correctly (FP and FN)
    TP, TN, FP, FN = 0, 0, 0, 0
    i = 1 # for tracking samples (qid)
    FP_list = []
    FN_list = []
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
            FP_list.append(i)
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1
            FN_list.append(i)
        i += 1

    precision_p, recall_p, f1_p, precision_n, recall_n, f1_n, acc = calculate_metrics(TP, FP, TN, FN)
    precision_p *= 100
    recall_p *= 100
    f1_p *= 100
    precision_n *= 100
    recall_n *= 100
    f1_n *= 100
    acc *= 100

    # Print results
    logger.info("FP samples list (qid): %s", FP_list)
    logger.info("FN samples list (qid): %s", FN_list)
    logger.info("Toruture samples list (qid): %s", tortureList)
    logger.info("-"*20 + args.subset + "-"*20)
    logger.info("TP\tFP\tTN\tFN\t")
    logger.info("{}\t{}\t{}\t{}".format(TP, FP, TN, FN))
    logger.info("Accuracy:\t\t%.2f", acc)
    logger.info("Precision-P:\t\t%.2f", precision_p)
    logger.info("Recall-P:\t\t%.2f", recall_p)
    logger.info("F1 score-P:\t\t%.2f", f1_p)
    logger.info("Precision-N:\t\t%.2f", precision_n)
    logger.info("Recall-N:\t\t%.2f", recall_n)
    logger.info("F1 score-N:\t\t%.2f", f1_n)
    logger.info("F1 Macro:\t\t%.2f", (f1_p+f1_n)/2)
    logger.info("Yes ratio:\t\t%.2f", yes_ratio)
    logger.info("Runtime:\t\t%.2f", argw.total_runtime)
    logger.info("-"*20 + args.subset + "-"*20)

    logger.info("Finish evaluating")

    # Save results to txt file
    eval_result_file = output_folder + "eval_result.txt"
    with open(eval_result_file, "w") as f:
        f.write("FP samples list (qid): " + str(FP_list) + "\n")
        f.write("FN samples list (qid): " + str(FN_list) + "\n")
        f.write("Torture samples list (qid): " + str(tortureList) + "\n")
        f.write("-"*20 + args.subset + "-"*20 + "\n")
        f.write("TP\tFP\tTN\tFN\t")
        f.write("\n{}\t{}\t{}\t{}\n".format(TP, FP, TN, FN))
        f.write("Accuracy:\t\t" + str(acc) + "\n")
        f.write("Precision-P:\t" + str(precision_p) + "\n")
        f.write("Recall-P:\t\t" + str(recall_p) + "\n")
        f.write("F1 score-P:\t\t" + str(f1_p) + "\n")
        f.write("Precision-N:\t" + str(precision_n) + "\n")
        f.write("Recall-N:\t\t" + str(recall_n) + "\n")
        f.write("F1 score-N:\t\t" + str(f1_n) + "\n")
        f.write("F1 Macro:\t\t " + str((f1_p+f1_n)/2) + "\n")
        f.write("Yes ratio:\t\t" + str(yes_ratio) + "\n")
        f.write("Runtime (min):\t" + str(argw.total_runtime) + "\n")

    # Pass results to argw
    argw.tp = TP
    argw.fp = FP
    argw.tn = TN
    argw.fn = FN
    argw.acc = acc
    argw.precision_p = precision_p
    argw.recall_p = recall_p
    argw.f1_p = f1_p
    argw.precision_n = precision_n
    argw.recall_n = recall_n
    argw.f1_n = f1_n
    argw.f1_macro = (f1_p+f1_n)/2
    argw.yes_ratio = yes_ratio
