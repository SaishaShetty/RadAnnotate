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
hypothesis_text = ["Patient has a history of smoking. Significant calcification observed in the apical segment of the left lower lobe.",
"Patient presents with persistent chest pain. No signs of abnormalities in the heart or major vessels. Pleural effusion is noted on the left side.",
"Patient's CT scan shows a possible nodule in the right upper lobe. Further evaluation recommended.", 
"Patient has a normal EKG. No signs of ischemia or infarction.",
"Patient presents with shortness of breath. There is a mass in the right lung apex."]

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
