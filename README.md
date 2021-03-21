<div align="center">
 <img src="images/logo.png" width="250"> 
</div>

Research code for evaluating category knowledge acquired by word embedding models.

## Metrics

- Classification Performance

A measure how well pairs of learned representations can be correctly judged to belong to the same category

- Divergence-from-Prototype

A measure of how abstract learned representations are.

-  Category-Spread 

A measure of spread between learned representations that belong to the same category

-  Silhouette Score

A measure of how well the similarity structure of learned representations captures the gold category structure 


## Usage

```python
from categoryeval.ba import BAScorer

probe2cat = {'door': 'FURNITURE', 'cat': 'ANIMAL'}
scorer = BAScorer(probe2cat)

gold_sims = scorer.gold_sims
balanced_accuracy = scorer.calc_score(pred_sims, gold_sims)  # predicted, and gold similarity matrices for probe words
```
## Compatibility

Developed on Ubuntu 18.04 and Python 3.7