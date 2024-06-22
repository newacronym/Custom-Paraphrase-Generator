# Custom-Paraphrase-Generator

### Run: 
python install -r requirements.txt
python main.py


### LLM Based Paraphraser
``` https://huggingface.co/tuner007/pegasus_paraphrase ```


### Metrics and Evaluation

ON CPU:
```
LLM Method Results:
Time: 157.96116971969604 seconds
BLEU Score: 0.11611027055084448
ROUGE Scores: {'rouge1': Score(precision=0.9485294117647058, recall=0.7747747747747747, fmeasure=0.8528925619834711), 'rouge2': Score(precision=0.7970479704797048, recall=0.6506024096385542, fmeasure=0.7164179104477612), 'rougeL': Score(precision=0.8529411764705882, recall=0.6966966966966966, fmeasure=0.7669421487603305)}

Custom Method Results:
Time: 1.4861249923706055 seconds
BLEU Score: 0.09365693806955869
ROUGE Scores: {'rouge1': Score(precision=0.8272727272727273, recall=0.8198198198198198, fmeasure=0.8235294117647058), 'rouge2': Score(precision=0.6413373860182371, recall=0.6355421686746988, fmeasure=0.6384266263237519), 'rougeL': Score(precision=0.796969696969697, recall=0.7897897897897898, fmeasure=0.7933634992458521)}
```

ON GPU: T4
```
LLM Method Results:
Time: 143.391583442688 seconds
BLEU Score: 0.11611027055084448
ROUGE Scores: {'rouge1': Score(precision=0.9485294117647058, recall=0.7747747747747747, fmeasure=0.8528925619834711), 'rouge2': Score(precision=0.7970479704797048, recall=0.6506024096385542, fmeasure=0.7164179104477612), 'rougeL': Score(precision=0.8529411764705882, recall=0.6966966966966966, fmeasure=0.7669421487603305)}

Custom Method Results:
Time: 0.16471219062805176 seconds
BLEU Score: 0.09365693806955869
ROUGE Scores: {'rouge1': Score(precision=0.8272727272727273, recall=0.8198198198198198, fmeasure=0.8235294117647058), 'rouge2': Score(precision=0.6413373860182371, recall=0.6355421686746988, fmeasure=0.6384266263237519), 'rougeL': Score(precision=0.796969696969697, recall=0.7897897897897898, fmeasure=0.7933634992458521)}
```

