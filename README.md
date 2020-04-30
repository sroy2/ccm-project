# nyu-ds1016
Group Project for NYU-DSGA 1016, Computational Cognitive Modeling (2020 Spring Semester)

## Group Members
* Wangrui (Wendy) Hou  |  wh916
* Gabriella (Gaby) Hurtado  |  gh1408
* Stephen Roy  |  sr5388

## Topic: Neural Networks - Language
_Selected from [CCM Project Site](https://brendenlake.github.io/CCM-site/final_project_ideas.html)_  
>Exploring lexical and grammatical structure in BERT or GPT2. What do powerful pre-trained models learn about lexical and grammatical structure? Explore the learned representations of a state-of-the-art language model (BERT, GPT2, etc.) in systematic ways, and discuss the learned representation in relation to how children may acquire this structure through learning.

## Approved Proposal
>Exploring connections between BERT and child language acquisition. Using Kumon center data, we plan to examine in context masked word prediction using the hugging-face pre-trained BERT model. While we do not know what specific mistakes were made by Kumon students, we know which exercises they performed well and poorly at. We plan to divide these exercises into 3 groups: 1) students do perfectly, 2) half of the students do poorly, 3) most students do poorly. We want to know if BERT performs perfectly on all exercises or, if it makes mistakes, what kinds of mistakes they are. 
> 
>Additionally, we would love to explore the performances of each BERT layer when performing these masked-word predictions. We are currently looking at how to analyze BERTs attention at specific hidden-layers and if there are other metrics of performance we should consider without having to retrain BERT

## Timeline
_Adpated from [CCM Site](https://brendenlake.github.io/CCM-site/#final-project)_
1. (26-Mar) Initial Meeting
2. (02-Apr) Proposal Review
3. (06-Apr) Proposal Submission [complete](#approved-proposal)
  * The final project proposal is due Monday, April 6 (one half page written). Please submit via email to instructors-ccm-spring2020@nyuccl.org with the file name lastname1-lastname2-lastname3-ccm-proposal.pdf.
  * https://piazza.com/class/k5cskqm4l1d4ei?cid=87
4. (01-May) Friday 1PM EST [Zoom meeting](https://nyu.zoom.us/j/5079167320) with [Prof Cournane](https://wp.nyu.edu/cournane/)
5. (13-May) Final Project Due

## Dataset sources: 
* Kumon

## Working Documents
* [Google Drive](https://drive.google.com/drive/folders/16DHSToewAcIkIytzBF9Lzkr-OV284a1c)

## Road Map
- [x] set up git
- [x] update proposal
- [x] lit review draft
- [ ] clean dataset
- [x] bert masked token model [demo](./src/demo.ipynb)
- [ ] select easy/medium/hard tasks
- [ ] model easy/medium/hard tasks
- [ ] compare bert conclusions to childhood development stages
- [ ] analyze bert layer attention


## References:
* [Transformers Quickstart](https://huggingface.co/transformers/quickstart.html)
* [Ganesh Jawahar - intpret_bert upstream repo](https://ganeshjawahar.github.io/)
* [Ganesh speaking about interpret_bert](https://vimeo.com/384961703)
* [Children First Language Acquisition At Age 1-3 Years Old In Balata](http://www.iosrjournals.org/iosr-jhss/papers/Vol20-issue8/Version-5/F020855157.pdf)
* [Caregivers' Role in Child's Language Acquisition](https://dspace.univ-adrar.dz/jspui/handle/123456789/2476)
* [The Acquisition of Syntax](https://linguistics.ucla.edu/people/hyams/28%20Hyams-Orfitelli.final.pdf)
* [Studies in Child Language: An Anthropological View: A First Language: The Early Stages](https://www.researchgate.net/publication/249422499_Studies_in_Child_Language_An_Anthropological_View_A_First_Language_The_Early_Stages_Roger_Brown_Language_Acquisition_and_Communicative_Choice_Susan_Ervin-Tripp_Studies_of_Child_Language_Development_Ch)
* [What do you Learn from Context? Probing for Sentence Structure in Contextualized Word Representations](https://openreview.net/pdf?id=SJzSgnRcKX)
* [Stanford WordBank Dataset](http://wordbank.stanford.edu/analyses)
* [A Structural Probe for Finding Syntax in Word Representations](https://nlp.stanford.edu/pubs/hewitt2019structural.pdf)
