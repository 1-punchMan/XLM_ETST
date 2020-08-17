import rouge
import numpy as np
import matplotlib.pyplot as plt
import re
import json
import pickle
import matplotlib
matplotlib.use('AGG')


evaluator = rouge.Rouge(metrics=['rouge-n'],
                        max_n=1,
                        limit_length=True,
                        length_limit=100,
                        length_limit_type='words',
                        apply_avg=True,
                        apply_best=False,
                        alpha=0.5,  # Default F1_score
                        weight_factor=1.2,
                        stemming=True)


def limit_float_decimal(f):

    return float("{:.2f}".format(f))


def plot_learning_curve(learning_logs, metric, labels, save):

    plt.xlabel("Step (K)")
    plt.ylabel("Valid ROUGE-1")

    # plt.title()

    epoch = 180
    colors = ['-k', '-r', '-b', '-g', 'tab:orange']
    for i, learning_log in enumerate(learning_logs):
        x = []
        y = []

        for j, log in enumerate(learning_log[: epoch]):
            if j % 5 == 0:
                x.append(log['step'] / 1000)
                y.append(log[metric])

        plt.plot(x, y, colors[i], label=labels[i])

    plt.legend(loc='lower right')
    plt.savefig(save)
    plt.show()


def parse_log(log_file):

    logs = open(log_file, 'r', encoding='utf-8').readlines()
    # new_logs = open("dumped/clts-elmo-zhen/sbcfjivn4w/train.log", 'w', encoding='utf-8')

    # {
    #   'epoch':,
    #   'loss':,
    #   'valid_zh-en_mt_ppl': ,
    #   'valid_zh-en_mt_acc': ,
    #   'valid_zh-en_mt_bleu': ,
    #   'valid_zh-en_mt_rouge1': ,
    #   'test_zh-en_mt_ppl': ,
    #   'test_zh-en_mt_acc': ,
    #   'test_zh-en_mt_bleu': ,
    #   'test_zh-en_mt_rouge1': ,
    # }

    # valid_all_references = open(f'dumped/clts-elmo-zhen/sbcfjivn4w/hypotheses/ref.zh-en.valid.txt', 'r',
    #                         encoding='utf-8').read().strip().split('\n')
    # test_all_references = open(f'dumped/clts-elmo-zhen/sbcfjivn4w/hypotheses/ref.zh-en.test.txt', 'r',
    #                         encoding='utf-8').read().strip().split('\n')
    learning_log = []
    for i, log in enumerate(logs):
        # for i in range(209):
        if 'End of epoch' in log:
            step = float(logs[i - 1].split(' - ')[3].strip())
            loss = sum([float(logs[i - n].split(' - ')[6].strip().split('  ')[-1])
                        for n in range(1, 11)]) / 10.
        if not '__log__' in log:
            # new_logs.write(log)
            continue

        # epoch = log_json['epoch']

        # log_json = {
        #     'epoch': i,
        #     'steps': (i + 1) * 6250
        # }

        # all_hypothesis = open(f'dumped/clts-elmo-zhen/sbcfjivn4w/hypotheses/hyp{i}.zh-en.valid.txt', 'r', encoding='utf-8').read().strip().split('\n')
        # scores = evaluator.get_scores(all_hypothesis, valid_all_references)
        # log_json['valid_zh-en_mt_rouge1'] = scores['rouge-1']['f'] * 100
        # print(f'dumped/clts-elmo-zhen/sbcfjivn4w/hypotheses/hyp{i}.zh-en.valid.txt', scores['rouge-1']['f'] * 100)

        # all_hypothesis = open(f'dumped/clts-elmo-zhen/sbcfjivn4w/hypotheses/hyp{i}.zh-en.test.txt', 'r', encoding='utf-8').read().strip().split('\n')
        # scores = evaluator.get_scores(all_hypothesis, test_all_references)
        # log_json['test_zh-en_mt_rouge1'] = scores['rouge-1']['f'] * 100
        # print(f'dumped/clts-elmo-zhen/sbcfjivn4w/hypotheses/hyp{i}.zh-en.test.txt', scores['rouge-1']['f'] * 100)
        # print()
        # new_logs.write('__log__:' + json.dumps(log_json) + '\n')

        log_json = json.loads(re.search("{.+}", log.strip()).group())
        if 'step' not in log_json:
            log_json['step'] = step
        # log_json['loss'] = loss
        log_json = {k: limit_float_decimal(v) for k, v in log_json.items()}

        learning_log.append(log_json)

    # new_logs.close()
    return learning_log


def load_learning_log(log_file):
    learning_log = None
    with open(log_file, 'rb') as inFile:
        learning_log = pickle.load(inFile)

    return learning_log


if __name__ == '__main__':

    # labels = ['clts-xenc-3M', 'clts-xenc-100w',
    #           'clts-xenc-50w', 'clts-xenc-30w']
    # exps = ['clts-xencoder-zhen', 'clts-xencoder-zhen',
    #         'clts-xencoder-zhen', 'clts-xencoder-zhen']
    # expnames = ['lyz8t3jr20', 'zv8ijj9kwo', 'a02dt74jqc', 'aby93sga8t']

    labels = ['clts-xenc-3000k', 'clts-xenc-1000k',
              'clts-xenc-500k', 'clts-xenc-300k', 'clts-xenc-scratch']
    exps = ['clts-xencoder-zhen', 'clts-xencoder-zhen',
            'clts-xencoder-zhen', 'clts-xencoder-zhen', 'clts-xencoder-zhen']
    expnames = ['btobrthg29', 'zv8ijj9kwo',
                'a02dt74jqc', 'aby93sga8t', 'thxeib07j0']

    log_files = [f"dumped/{exp}/{expname}/train.log" for exp,
                 expname in zip(exps, expnames)]
    for log_file in log_files:
        learning_log = parse_log(log_file)

        with open(log_file.replace('train.log', 'learning_log.pickle'), 'wb') as outFile:
            pickle.dump(learning_log, outFile)

    learning_log_files = [
        f"dumped/{exp}/{expname}/learning_log.pickle" for exp, expname in zip(exps, expnames)]
    learning_logs = [load_learning_log(f) for f in learning_log_files]

    # metric = 'test_zh-en_mt_rouge1'
    metric = 'valid_zh-en_mt_rouge1'
    save = f"valid_learning_curve.png"
    plot_learning_curve(learning_logs, metric, labels, save)
