# Topic modelling 

Used for finding topic (group of words) from a collection of documents that represents the information in the collection or in other words assigning multiple tags to a text. (Unsupervised Document Classification)
Methods Used : 
1) NMF - Non-negative Matrix Factorization 
2) LSA - Latent Semantic Analysis
3) LDA - Latent Dirichlet Allocation

Steps: 
Create the term-document matrix. 
Perform topic modelling by giving number of topics.
Get the topics (groups of words).
Assign labels to each topic.
To retrieve similar documents: Perform classification to retrieve the top n similar documents using unsupervised nearest neighbours.

Conclusions:
1) For classification tasks, Topic modelling can be used as features.
2) LSA is much faster to train than LDA.
3) But LSA has lower accuracy.
4) LDA performs better comparatively.

