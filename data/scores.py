from rouge_score import rouge_scorer
from bert_score import score

def compute_rouge_l(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores["rougeL"]

def compute_bert_score(reference, hypothesis):
    P, R, F1 = score([hypothesis], [reference], lang="en")
    return P,R,F1

# Example usage
reference_text = "Patient has been extubated . Lungs are clear . Normal cardiomediastinal and hilar silhouettes and pleural surfaces ."
hypothesis_text = ["Patient's lungs show multiple opacities. Cardiomediastinal and hilar structures are not visible. There are infiltrates noted in the mediastinum. Left and right lymph nodes are visible."]

for s in hypothesis_text:
    rouge_l_score = compute_rouge_l(reference_text, s)
    print("ROUGE-L Precision: {:.2f}".format(rouge_l_score.precision))
    print("ROUGE-L Recall: {:.2f}".format(rouge_l_score.recall))
    print("ROUGE-L F1 Score: {:.2f}".format(rouge_l_score.fmeasure))


for s in hypothesis_text:
    P, R, F1 = compute_bert_score(reference_text, s)
    print("BERTScore Precision:", P.mean().item())
    print("BERTScore Recall:", R.mean().item())
    print("BERTScore F1:", F1.mean().item())
