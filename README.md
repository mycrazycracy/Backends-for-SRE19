# # Backends for NIST SRE 2019

This script shows different backends to generate scores for a speaker verification systems. It assumes that you have extracted embeddings for utterances. The embedding could be i-vector or x-vector, or whatever vector representations. If you have any problem extracting the embedding, please refer to Kaldi (egs/sre16/v2) or my [tf-kaldi-speaker](https://github.com/mycrazycracy/tf-kaldi-speaker) project for more details. The experiment results here is evaluated using the baseline ETDNN network trained by  Kaldi.

*Note that these results are obtained after the evaluation, and are different from the THUEE official submission. This is our official results: [system description](https://arxiv.org/abs/1912.11585)*

----

## Details

I implemented 10 backends. 

The datasets used include:

* SRE04-10 training set: SRE_04_10. Out-of-domain (ooD) data. This set is augmented in the experiments.
* SRE18 evaluation set: SRE18-eval. In-domain (inD) data. This set is augmented in one backend.
* SRE18 unlabel data: SRE18-unlabel. In-domain data without labels (used for mean estimation)
* SRE18 dev set: SRE18-dev-enroll and SRE18-dev-test. This is the development set.
* SRE19 eval set: SRE19-eval-enroll and SRE19-eval-test. This is the evaluation set.

All these backends subtract the x-vectors with the corresponding mean vectors first. The mean for the inD data is estimated using the SRE18-unlabel. 

1. LDA + PLDA. Both the LDA and PLDA models are trained using the ooD data.
2. LDA + PLDA. Both the LDA and PLDA models are trained using the inD data.
3. LDA + PLDA. The models are trained using the mixed version of ooD and inD data.
4. LDA + PLDA (adapted). The models are first trained using the mixed ooD and inD data. Then the PLDA is adapted using the mixed data. The adaptation adjusts the covariances, making them more suitable for the inD data.
5. LDA + PLDA (adapted) + AS-Norm. Same as S4. Add adaptive score normalization.
6. LDA + whitening + PLDA (interpolated). The PLDA is trained on ooD and inD data independently. Apply the supervised PLDA adaptation. 
7. LDA + whitening + PLDA (interpolated) + AS-Norm.
8. CORAL + LDA + whitening + PLDA (interpolated). CORAL is a feature-level adapation technique. The ooD data is first transformed using CORAL, then the backend is applied.
9. CORAL + LDA + whitening + PLDA (interpolated) + AS-Norm. 
10. CORAL + LDA + whitening + PLDA (interpolated) + AS-Norm. The inD data is further augmented before training.

The references are showed in the script. See the script for more details.

## Results

|  ID  | SRE18 dev EER (%) | minDCF | SRE19 eval EER (%) | minDCF |
| :--: | :---------------: | :----: | :----------------: | :----: |
|  1   |       8.79        | 0.560  |        8.98        | 0.556  |
|  2   |       6.90        | 0.413  |        6.01        | 0.467  |
|  3   |       7.04        | 0.529  |        6.94        | 0.482  |
|  4   |       6.27        | 0.500  |        6.34        | 0.451  |
|  5   |       4.86        | 0.359  |        5.12        | 0.397  |
|  6   |       5.13        | 0.319  |        4.34        | 0.354  |
|  7   |       5.31        | 0.370  |        4.29        | 0.359  |
|  8   |       5.12        | 0.286  |        4.35        | 0.359  |
|  9   |       5.14        | 0.312  |        4.16        | 0.340  |
|  10  |       4.90        | 0.242  |        4.09        | 0.324  |



## Requirements

This is an extension of Kaldi. Install Kaldi first. 

```
mkdir -p kaldi/egs/sre19
cd kaldi/egs/sre19
```

Git clone this repository in the sre19 directory. 
To score the systems, you need to download the Key files provided by NIST. I only include the scoring tools provided during NIST SRE19. The tools are developed by NIST.

