import rouge


def prepare_results(p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


def get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram = text_length - n
    for i in range(max_index_ngram + 1):
        ngram_set.add(tuple(text[i:i+n]))
    return ngram_set


def rouge_n(evaluated_sentences, reference_sentences, n=2):  # 默认rouge_2
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        return 0

    evaluated_ngrams = get_ngrams(n, evaluated_sentences)
    reference_ngrams = get_ngrams(n, reference_sentences)
    reference_ngrams_count = len(reference_ngrams)
    if reference_ngrams_count == 0:
        return 0

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_ngrams_count = len(overlapping_ngrams)
    return overlapping_ngrams_count / reference_ngrams_count


def rouge_1(evaluated_sentences, reference_sentences):
    evaluated_sentences = evaluated_sentences.split()
    reference_sentences = reference_sentences.split()
    return rouge_n(evaluated_sentences, reference_sentences, n=1)

def rouge_2(evaluated_sentences, reference_sentences):
    evaluated_sentences = evaluated_sentences.split()
    reference_sentences = reference_sentences.split()
    return rouge_n(evaluated_sentences, reference_sentences, n=2)

def rouge_3(evaluated_sentences, reference_sentences):
    evaluated_sentences = evaluated_sentences.split()
    reference_sentences = reference_sentences.split()
    return rouge_n(evaluated_sentences, reference_sentences, n=3)

def rouge_4(evaluated_sentences, reference_sentences):
    evaluated_sentences = evaluated_sentences.split()
    reference_sentences = reference_sentences.split()
    return rouge_n(evaluated_sentences, reference_sentences, n=4)


def F_1(evaluated_sentences, reference_sentences, beta=1):
    evaluated_sentences = evaluated_sentences.split()
    reference_sentences = reference_sentences.split()
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        return 0

    evaluated_ngrams = get_ngrams(
        beta, evaluated_sentences)  # equal to retrieved set
    reference_ngrams = get_ngrams(
        beta, reference_sentences)  # equal to relevant set
    evaluated_ngrams_num = len(evaluated_ngrams)
    reference_ngrams_num = len(reference_ngrams)

    if reference_ngrams_num == 0 or evaluated_ngrams_num == 0:
        return 0

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_ngrams_num = len(overlapping_ngrams)
    if overlapping_ngrams_num == 0:
        return 0
    return 2*overlapping_ngrams_num / (reference_ngrams_num + evaluated_ngrams_num)


def ncls_rouge(hypotheses, references):
    rg1list = []
    rg2list = []
    rg3list = []
    rg4list = []
    for hypo, ref in zip(hypotheses, references):
        rouge1 = F_1(hypo, ref, beta=1)
        rouge2 = F_1(hypo, ref, beta=2)
        rouge3 = F_1(hypo, ref, beta=3)
        rouge4 = F_1(hypo, ref, beta=4)
        rg1list.append(rouge1)
        rg2list.append(rouge2)
        rg3list.append(rouge3)
        rg4list.append(rouge4)
    rg1 = sum(rg1list) / len(rg1list)
    rg2 = sum(rg2list) / len(rg2list)
    rg3 = sum(rg3list) / len(rg3list)
    rg4 = sum(rg4list) / len(rg4list)
    return rg1 * 100, rg2 * 100, rg3 * 100, rg4 * 100


evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],  # , 'rouge-w'],
                        max_n=4,
                        limit_length=True,
                        length_limit=100,
                        length_limit_type='words',
                        apply_avg=True,
                        apply_best=False,
                        alpha=0.5,  # Default F1_score
                        weight_factor=1.2,
                        stemming=True)


# exp = 'clts_zhen-mlm_enzh-scratch'
# exp = 'clts-ft-zhen'
lang1 = 'en'
lang2 = 'zh'
exp = f'clts-ft-{lang1}{lang2}'
exp_name = 'owbcufzet3'
epoch = '159'
datatype = 'test'  # 'valid', 'test'
print(f'exp_name: {exp_name} | epoch: {epoch}')
all_hypothesis = open(f'/home/chiamin/python-projects/XLM-master/dumped/{exp}/{exp_name}/hypotheses/hyp{epoch}.{lang1}-{lang2}.{datatype}.txt',
                      'r', encoding='utf-8').read().strip().split('\n')
all_references = open(f'/home/chiamin/python-projects/XLM-master/dumped/{exp}/{exp_name}/hypotheses/ref.{lang1}-{lang2}.{datatype}.txt', 'r',
                      encoding='utf-8').read().strip().split('\n')

print(all_hypothesis[0])
print(all_references[0])
########## py-rouge ############
scores = evaluator.get_scores(all_hypothesis, all_references)

for metric, results in sorted(scores.items(), key=lambda x: x[0]):
    print(prepare_results(results['p'], results['r'], results['f']))
print()
################################

########## NCLS Rouge ##########
# rouge1, rouge2, rouge3, rouge4 = ncls_rouge(all_hypothesis, all_references)
# print(rouge1, rouge2, rouge3, rouge4)
################################