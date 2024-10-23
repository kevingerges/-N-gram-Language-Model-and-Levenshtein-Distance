import argparse
import os
import pandas as pd
import numpy as np
import Levenshtein
import tqdm
import json


def main(args):
    
    print("Reading input files...")
    with open(args.gold_fn, 'r') as f:
        gold_df = pd.read_csv(f)
    with open(args.pred_fn, 'r') as f:
        pred_df = pd.read_csv(f)
    assert len(gold_df) == len(pred_df)
    if args.ostrich_summary_perf_fn is None:
        print('... NOTE: assuming calculating Ostrich metrics')
        ostrich_perf = None
    else:
        with open(args.ostrich_summary_perf_fn, 'r') as f:
            ostrich_perf = json.load(f)
    print("... done")


    print("Evaluating predictions from '%s' against gold messages in '%s'..." % (args.pred_fn, args.gold_fn))
    d_eval = []
    for idx in tqdm.tqdm(pred_df.index):
        gold = gold_df['gold_msg'][idx]
        pred = pred_df['pred_msg'][idx]
        entry = {}

        if gold == pred:
            entry['exact_match'] = 1
        else:
            entry['exact_match'] = 0

        # Full string Levenshtein similarity ratio (1 perfect, 0 none)
        ls = Levenshtein.ratio(gold, pred)
        entry['character_levenshtein_similarity'] = ls
        
        # Full string Levenshtein distance (unbounded)
        ld = Levenshtein.distance(gold, pred)
        entry['character_levenshtein_distance'] = ld

        # Word error rate calculation requires dynamic sequence alignment, for which we'll use Lev sim per token
        gtkns = gold.split()
        ptkns = pred.split()
        uww = Levenshtein.distance(gtkns, ptkns)
        entry['word_levenshtein_distance'] = uww
        entry['word_error_rate'] = uww / len(gtkns)

        d_eval.append(entry)
    print("... done")

    # Write performances to files
    instance_perf_fn = os.path.join(args.out_dir, '%s_inst.csv' % args.out_pref)
    summary_perf_fn = os.path.join(args.out_dir, '%s_summary.json' % args.out_pref)
    print("Writing CSV of per-instance outputs to '%s'..." % instance_perf_fn)
    out_df = pd.DataFrame(d_eval)
    with open(instance_perf_fn, 'w') as f:
        out_df.to_csv(f)
    print("... done")
    print("Writing summary stats to '%s'..." % summary_perf_fn)
    with open(summary_perf_fn, 'w') as f:
        per_exact_match = 100. * sum(out_df['exact_match']) / len(out_df)
        avg_character_levenshtein_similarity = sum(out_df['character_levenshtein_similarity']) / len(out_df)
        avg_word_error_rate = sum(out_df['word_error_rate']) / len(out_df)
        json.dump({
                'per_exact_match': per_exact_match,
                'avg_character_levenshtein_similarity': avg_character_levenshtein_similarity,
                'avg_word_error_rate': avg_word_error_rate,
                'per_exact_match_improvement': 
                    (per_exact_match - ostrich_perf['per_exact_match']) / (100 - ostrich_perf['per_exact_match'])
                    if ostrich_perf is not None else None,
                'avg_character_levenshtein_similarity_improvement': 
                    (avg_character_levenshtein_similarity - ostrich_perf['avg_character_levenshtein_similarity']) / 
                        (1 - ostrich_perf['avg_character_levenshtein_similarity'])
                    if ostrich_perf is not None else None,
                'avg_word_error_rate_reduction': 
                    (ostrich_perf['avg_word_error_rate'] - avg_word_error_rate) / 
                        ostrich_perf['avg_word_error_rate']
                    if ostrich_perf is not None else None
            }, f, indent=4)
    print("... done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_fn", type=str, help="Disc location of CSV with gold messages", required=True)
    parser.add_argument("--pred_fn", type=str, help="Disc location of CSV with predicted messages", required=True)
    parser.add_argument("--out_dir", type=str, help="Directory to write performance files", required=True)
    parser.add_argument("--out_pref", type=str, help="File prefix for output files", required=True)
    parser.add_argument("--ostrich_summary_perf_fn", type=str, help="Location of ostrich alg summary perf", required=False)

    args = parser.parse_args()
    main(args)