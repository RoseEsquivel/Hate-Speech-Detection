# Hate-Speech-Detection
 Natural language processing project focused on bigotry based harassment in text. Ideal implementation would be to detect hate speech and harassment on social media platforms.

I'm starting with sexism since that seems to be the hardest to quantify. Racism, homophobia, and many other common forms of bigotry rely heavily (though not exclusively) on slurs which any simple bot could detect and existing name entity recognition technology can address. In my research, I want to take bigotry detection beyond key words and phrases and ultimately train models to use extended interaction context (for example, on a social media platform with full access to messaging text data) to detect sutbler not no less significant forms of harassment and hate speech. If this can be successfully achieved through for misogyny, then the technological advancement could most likely be applied to other forms of hate speech as well.

Fundamentally, there are 3 different tasks to be completed:
- Task A: Determine if a text sample is sexist or not. 
- Task B: Classifty sexist text samples as threats, degrogation, animosity, or prejudiced discussion. (This will inform the the content moderation team of the social media platform of the nature of the hate speech. In the case of threats, there may be programmatic freezing or muting of the malicious user's account. Accuracy here is important.)
- Task C: Further granulized classification meant to aide in the monitoring and response to sexist behavior on a social media platform. Retain the taxonomical hierarchy of classication such that if granular classification is not possible, a general classificaion is assigned instead. The more granular and explainable an individual instance of flagging content as sexist is, the more interpretable and trustworthy the flag appears to the content moderation team and the social media platform users. 

  
![image](https://github.com/RoseEsquivel/Hate-Speech-Detection/assets/100469804/fdb5123d-e037-4e3e-9f2c-7e59409035c3)


## Goals

- [x] Create 3 "baseline" models on top of 4 different different feature sets using training datasets from [Samory et al. (2020)](https://arxiv.org/abs/2004.12764).  The goal here is to determine how informative the 4 feature sets are for sexism detection based on their impact on classifier performance. We're using the VADER sentiment analysis technique designed for social media text from [Hutto et al. (2014)](http://eegilbert.org/papers/icwsm14.vader.hutto.pdf), the Stanford Parser type dependency relationships based on [de Marneffe et al. (2006)](https://nlp.stanford.edu/~wcmac/papers/td-lrec06.pdf), term frequency weights of word n-grams from scikit-learn, and document embeddings from the BERT language representation model as described by [Devlin et al. (2018)](https://arxiv.org/abs/1810.04805).

Example: Trains logistic regression on sentiment and uni-gram features. To replicate the results, run the scripts in the 'experiments/scripts.txt'
```
export PARAMS_FILE='experiments/params.json'
export HYPERPARAMS_FILE='experiments/hyperparams.json'

python run.py \
  --params_file=$PARAMS_FILE \
  --hyperparams_file=$HYPERPARAMS_FILE \
```


Example: Obtains pre-trained BERT language representation model embeddings.
```
export DATA_FILE=/path/data.csv

python run_bert_feature_extraction.py \
	--data_file=$DATA_FILE \
```


- [x] Build a custom, annotated dataset with data collected via Reddit API. (Ask Maddie, Vanessa, and Elissa to annotate data.)
- [x] Solicit input from a statistially significant number of Gen Z women on "how misogynistic" each individual text sample is to attribute a number value and assign a primary and granular classification label to each sample. (The goal here is see how well out baseline model performs on this new test dataset.)
- [ ] Implement Bayesian deep Convolutional Neural Network to better quantify uncertainity in sentiment analysis, named entity recognition, and language modeling tasks per [Xiao et al. (2018)](https://arxiv.org/abs/1811.07253) while being self-supervising (since this is ultimately a classification task) following the guidance of [Shen et al. (2021)](https://arxiv.org/abs/2102.08946).
- [ ] Figure out the relative natural of sexism compared to other forms of hate speech, such as racism, homophobia, xenophobia to see how much of the research so far will work as a launching pad for expanding into detecting other forms of hate speech. (I work would love to find some throughlines across all forms of hate speech that could be capitalized on for their detection.)

