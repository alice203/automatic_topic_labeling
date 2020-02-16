# master thesis

This repository contains the code for my final research project to complete my Master’s degree. 

The research question of my project asks ’Can topic labels for TED talk transcripts be generated in a data-driven way using Latent Dirichlet Allocation and exogenous reference texts?'

Abstract

Automatic topic labeling of TED talk transcripts: A data-driven approach

Alicia Horsch

The amount of available audio data has been rising exponentially in the last decade, and with it, the need to process and manage it in an automatic way. Especially businesses see themselves confronted with a large amount of speech data that holds valuable information about their customers and products. One powerful tool that structures audio data in a meaningful way and has explicitly proven successful on audio transcripts is the Latent Dirichlet Allocation. It is used to decompose a collection of documents into latent topics that are syntactically similar. However, a common drawback of the Latent Dirichlet Allocation is that the latent topics are described as a distribution over words, instead of a simple descriptive name. This often requires human reviewing to characterize each created topic, which is subjective and labor-intensive. Prior research developed several approaches to overcome this interpretation issue, but until today, no established and automatic process generates a single label for a topic. In this thesis, a data-driven approach is proposed to automatically annotate topics and eventually transcripts with single labels using Latent Dirichlet Allocation and an exogenous reference corpus, on the example of TED talk transcripts. The results show that roughly 30% of the transcripts are labeled with a topic that matches one of the manual annotations of a transcript. The results from a semantic similarity analysis between predicted and actual labels further suggest that a small part of the discrepancy between data-driven and manual approach can be explained by the ambiguity in language. Eventually, several suggestions for further improvement of the approach but also the evaluation of the approach are presented and discussed.

Keywords: Latent Dirichlet Allocation, audio transcripts, text mining, natural language processing, machine learning

This research analysis is based on two text corpora. The main corpus, a collection of TED talk transcripts is available on Kaggle. The reference corpus, a collection of Wikipedia articles, was manually scraped from Wikipedia based on the topics available on the TED talk website. 

The corresponding research paper is available under:
