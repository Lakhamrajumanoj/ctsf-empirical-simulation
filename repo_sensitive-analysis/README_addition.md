## Sensitivity and Ablation Analysis


```
                  configuration  precision  recall    f1   fpr
       Original weights (paper)      0.971    1.00 0.985 0.008
      Equal weights (0.20 each)      1.000    1.00 1.000 0.000
           Drop temporal signal      0.952    1.00 0.976 0.012
        Drop behavioural signal      0.943    1.00 0.971 0.015
         Drop volumetric signal      0.980    0.99 0.985 0.005
         Drop geo_device signal      0.943    1.00 0.971 0.015
          Drop role_norm signal      0.962    1.00 0.980 0.010
   temporal +50% (renormalized)      0.980    1.00 0.990 0.005
   temporal -50% (renormalized)      0.962    1.00 0.980 0.010
behavioural +50% (renormalized)      0.971    1.00 0.985 0.008
behavioural -50% (renormalized)      0.971    1.00 0.985 0.008
 volumetric +50% (renormalized)      0.952    1.00 0.976 0.012
 volumetric -50% (renormalized)      0.980    1.00 0.990 0.005
 geo_device +50% (renormalized)      0.943    1.00 0.971 0.015
 geo_device -50% (renormalized)      0.971    1.00 0.985 0.008
  role_norm +50% (renormalized)      0.980    1.00 0.990 0.005
  role_norm -50% (renormalized)      0.962    1.00 0.980 0.010
```

### Interpretation

F1-score ranges from 0.971 to 1.000 across every configuration tested.
No single signal dominates model performance — removing any individual
signal degrades F1 by at most 0.014, and +/-50% perturbation of any single
weight changes F1 by at most 0.009. This indicates the paper's weight
configuration, while informed by the relative diagnostic value of each
signal, is not narrowly over-fit, and the model is robust to reasonable
variation in weight assignment.
