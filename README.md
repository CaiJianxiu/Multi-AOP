# Multi-AOP: A Lightweight Multi-View Deep Learning Framework for Antioxidant Peptide Discovery

## Description
We presented Multi-AOP, a novel lightweight multi-view deep learning framework that enhances AOP discovery through integrated sequence and graph learning. Specifically, we employed Extended Long Short-Term Memory (xLSTM), a parameter-efficient network to generate sequence embeddings of peptides. Concurrently, we transformed peptide sequences into SMILES representations and extracted molecular graph features using a Message Passing Neural Network (MPNN), thereby capturing the intrinsic physicochemical properties of the peptides. By leveraging both sequence patterns and structural information, Multi-AOP demonstrated improved predictive performance. Comprehensive evaluations across three benchmark datasets revealed that our method consistently outperforms conventional machine learning algorithms and state-of-the-art deep learning approaches. Furthermore, we constructed a unified AOP dataset by integrating these benchmark datasets,  facilitating future development of generalizable AOP models.

### Dataset
We collected three distinct existing datasets as the benchmark datasets: the AnOxPePred dataset from Olson et al., the AnOxPP dataset from Qin et al., and the AOPP dataset from Li et al. The AnOxPePred dataset, sourced from the BIOPEP-UWM database in 2020, comprises 676 free radical scavenger (FRS) AOP as positive samples and 728 non-AOPs as negative samples. The negative set includes 218 experimentally validated non-AOP supplemented with 500 randomly generated sequences. The AnOxPP dataset, compiled in 2023 from the DFBP and BIOPEP-UWM databases, contains 1060 peptides with documented radical scavenging activities as positive samples, balanced with 1060 randomly generated sequences serving as negative samples. The most recent AOPP dataset represents the most comprehensive collection, integrating data from multiple repositories including DFBP, BIOPEP-UWM, Antimicrobial Peptide Database, PlantPepDB, and FermFooDb. This dataset encompasses 1511 validated AOPs and an equivalent number of randomly generated sequences as negative controls. 

To enhance the practical utility of our proposed model, we developed the final prediction model trained on a comprehensive dataset combining all three benchmark data. After removing duplicated sequences, our consolidated dataset comprised 5,235 peptides, including 2597 AOPs as the positive set and 2638 random sequences as the negative set. Furthermore, we evaluated its performance on an external validation set of 54 newly published AOPs collected from diverse sources including DEBP, BIOPEP-UWM database, Antimicrobial Peptide database, Plant-PepDB, and FermFooDB. 
* Dataset from AnOxPePred is in /data/AnOxPePred/ (5-CV split)
* Dataset from AnOxPP is in /data/AnOxPP (5-CV split)
* Dataset from AOPP is in /data/AOPP (5-CV split)
* To reproduce the comparison machine learning (ML) model is in /reproduce/ml_base/aop_ml_train_test.py
* To reproduce the final model is in /final_model/multi_aop_train.py

## Getting Started

### Python packages

* pytorch=2.2.1=py3.8_0
* biopython==1.81
* xlstm==2.0.2

### Executing program
* run /reproduce/multi-aop/hierarchical_train.py to reproduce the experimental result
* run /reproduce/ml_base/aop_ml_train_test.py to reproduce the experimental result
* run reproduce/mulit-aop/predict.py to predict peptide antioxidant acitivity
* run /final_model/multi_aop_train.py to obtrain a final model and validate by the external dataset


## Acknowledgments

Inspiration, code snippets, etc.
*Beck, Maximilian, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter, and Sepp Hochreiter. "xlstm: Extended long short-term memory." Advances in Neural Information Processing Systems 37 (2024): 107547-107603.
