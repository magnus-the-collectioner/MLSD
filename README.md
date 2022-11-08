# MLSD

**Introduce yourself**:
"From the engineering standpoint I have been working mostly with machine learning, doing stuff with GANs and RL, some amount of NLP. From the product standpoint, I have a lot of exposure to biotechnology and quantum computing, primarily in Drug Discovery. Right at the moment I am a Principal Machine Learning Architect in a company of around a thousand employees working with various pharma companies."

**Ask clarifying questions on the problem and rephrase the problem** (On top of the board - key concepts):
Try to make assumptions instead of direct questions, ask if the assumptions are off.
The purpose of our business is to maximize profits - how do we achieve that? 

**Formulate a plan for the presentation** (5 equal vertical segments on the board):
+ Business goals and metrics
+ Data collection and labelling
+ Feature selection and engineering
+ Modelling baseline and solution
+ Deployment and monitoring

## 1) Business

**If there are several types of users, separate their requirements.** (e.g., driver and passenger)

**Functional requirements (whole service)**:
Possible actions in the system.

**Non-functional requirements (whole service)**:
What does the user care about?

**Decouple goals of the ML part** (e.g. for NewsFeed):
> • Filter out spam
> • Filter out NSFW content
> • Filter out misinformation
> • Rank posts by quality
> • Rank posts by engagement

**Business metrics we want to maximize for these goals**:
Take-Rate(netflix), Clicks, Conversion, Delay, Coverage, MAU (WAU, DAU). How are we constrained on some of this? These metrics might be used for validation and updating the model.

Business metrics to ML metrics for these goals:
What is X and Y? RMSE/MAPE/MSE/MAE, AUC, DCG, etc. Prioritize type-1 and type-2 errors. Think about metric to loss function. Think about model calibration.
It's possible to adapt loss function to a problem (somehow change the shape) - if there is time for it.

## 2) Data

Methods of collecting\labelling data, risks and constraints. Impact on the end user is crucial.

User input data? 
Lots of preprocessing and could be sensitive - think about privacy - what is absolutelly necessary and what is not.

System-generated data?
Could be used raw most of the time, might be user-generated and still require some thinking on limitations.

How to collect?
Prepare the service for data mining - plan in advance the architecture and data formats.
Perform mining on other platforms - could be a sanity-check or a cold-start.
Any ready-to-use datasets for a better cold-start?

How to label? (from fast to slow iterations, easy to hard)
Natural - best scenario if available right away, feedback loop might be too long
User - should be simplified, requires double-check, but no privacy concerns.
Team - imposes a risk of privacy leak, expensive and long, but the most controllable.
Algo - not precise, but cheap and fast, could be good for a cold-start.
If algo: Weak-supervision, semi-supervised (close distance), transfer and active learning

What about quality-control of labelling?
Consistency - changing source might cause incompatible distributions.
Redundancy -  minimize ambiguity, e.g., use multiple labelers per sample.
Coverage - explain corner-cases to the labelling team or encode in the algo.

How to sample? 
Non-probability - by availability, by experts, by quotas per category, by known legitimate profiles\connections.
Probability - simple random, stratified per category, weight by experts.
Reservoir - if online and does not fit into memory - K positions, change with K/N probability.
Importance - if you already have some distribution and want to shift it.

Class imbalance?
Data-level - under-sampling (Tomek links - merge close points), over-sampling (SMOTE - mix close points), train on the over-sampled and fine-tune on the original.
Metric-level - design metrics for what you care about or look at confusion matrix.
Algo-level - loss-matrix (one confusion is better than another), balance per class, focal loss (focus on low-performing samples)

Data augmentation?
Label-preserving transformation (flip, synonyms), perturbation (noise or close points), generative (permutations or blends of points)

How to split?
Prevent data leakage by splitting by time (if non-stationary process) and (or) by location.
Don't perform preprocessing before splitting - might induce biases with imputing or scaling.
Think about duplicates or close-related data points not to be split between train/test.

## 3) Features

Very well rounded ML features. Pros and cons of encoding choices. Implications for the final solution.

What are the features?
-User features (location, history, interest, etc)
-Object features (location, popularity, etc)
-User-object features (distance to object, frequency of user interactions, etc)
-Context features (time of a day, day in a year, other object features, etc.)

How to preprocess features?
-Hashing features to some space - if the number of features might be growing through time.
-Combine features - more meaningful\interpretable, might overcomplicate the solution - design by hand.
-Imputation vs deletion - missing values because of the feature, because of another feature(s), missing completely at random
-Preparing for modelling - scaling, discretization, standartization, tokenization, cropping, etc.

How to encode features?
If a custom training\tuning is proposed - mentioned briefly, get back to it in the end. 
Images - VGG-based encoder, maybe small EfficientNet, could be a SimCLR (contrastive learning) or even a custom training/tuning for a closer problem. 
Text - Transformer-based encoder, maybe fine-tune a TinyBERT, could be an adaption of MLM\NSP for a custom fine-tune. 
Tabular\time series - for cascade of columns it could be AVG\MEAN\GROUP(ANY), for circular features - sin-cos encoding, standartization to a range. 

Important features?
Make some hypotheses on the product level. Refer to validation metrics that might be used to evaluate if features are improving the score.
Sometimes feature importance scores, most likely attention maps or gradient flow. Think about SHAP or LIME for interpretability.

## 4) Models

Modelling - choice of 2-3 models and figure out trade-offs between them. What are the risks, how to mitigate - pros and cons. Make a clear decision on a trade off. Limitations, bottlenecks of the final model. 

Do we have a strict latency requirement? Data and computations sufficient? Enough time to perform training? Interpretability?

Baseline (cold start algorithmic solution or even random as theoretical minimum)?
Meaningful heuristics\distances that might help with data collection\labelling. Could be followed up with a simple ML model soon after.

Is it possible to improve on top of the baseline? Should we switch the model or even train silently online?

Ensembling such as boosting, stacking or bagging? Decomposing the problem into subproblems?
Ensembles are harder to deploy and mantain, but worth it in money-flowing services such as click-through rate.
Bagging - improves on data imbalance. Boosting - natural focus on errors. Stacking - system decomposition.

Assumptions made?
E.g. IID, Smooth (close points = close results), normal distribution?


## 5) Deployment

Model debugging - A/B testing!, online and offline evaluation. Pro-actively consider solutions! Justify metric choices and how to compute them.
Tests are used not only for data drift, but also for changing business priorities or hypotheses. 

Reliability: How do we handle if the system fails somewhere? e.g., wrong translation or wrong ban.
Scalability: How to scale up and down the infrastructure and how to handle artifacts?
Maintainability: How to make documentation and versioning as accessible as possible?
Adaptability: How to make an easy shift if there is a change in data distribution or even business direction?

Experiment tracking and versioning?
MLFlow, DVC, W&B for managing and controlling artifacts. Not only loss, samples and metrics, but also CPU/GPU, speed, even weights sometimes.

Distributed training and inference? (Data vs model parallelism)
Data - model is copied, data is splitted. Async gradients converge well due to sparse updates in large models. Batch is capped due to computational profits.
Model - vise-versa. Pipeline parallelism (like hardware instruction pipelining) - partial computations are sent forward to minimize stalling.

Soft AutoML (hyperparameter) vs Hard (architecture)

Costs of deployment

Offline Evaluation (Tests/Measurements)
Perturbation - adding noise or clipping samples in tst does not change predictions.
Invariance - changing sensitive features (age, race, etc) does not change predictions.
Directional Expectation - decreasing area does not increase price (illogical behavior).
Model calibration - prediction threshhold vs real frequency.
Confidence Measurement - sample-level threshold for confidence.
Slice-based - prioritize cancer diagnosis or fairly handle blacks\whites. (Simpson Paradox)

Slicing user categories by?
Heuristics - Desktop\Browser\Location. 
Error analysis - manually find pattens.
Automatic - search\clustering of groups.

load balancing, caching, databases,

Are there any limitations to testing?
E.g., can't show different prices for the same item - change prices for similar(!) items or slice by regions and approximate.

What to do with feedback loop?
E.g., recommending the prices based on previous recommendations and their influence on buying behavior - propose the price based on elasticity, or predicted demand, or simply by a margin on top of the manufacturing cost. Add multiarmed bandits to adapt on various types of products. It could be epsilon-greedy. 

How to compare if it's good or bad? 
Business metrics.
ML metrics.
Infrastructure metrics - how often to redeploy, how fast is the response.

How to run statistical (e.g., A/B) tests?
Control\test groups. How many points\day. Metrics. Problems: Multiple comparison. Data bugs (A/A test). Cannibalization of similar products? Overstocking behavior in time?

Offline validation.
Most likely historical data in time.

Online validation.
Real-time data flow.

Monitoring.

What to do if we can't run tests all the time?
We can try to parametrize some prior around an approximate optimal point and run tests around it.

## 0) Start iterating on details if there is time. Ask the interviewer if he wants to focus on something in particular or should you proceed on your notes.
