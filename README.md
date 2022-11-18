# MLSD

**Introduce yourself**:
From the engineering standpoint I have been working mostly with machine learning, doing stuff with GANs and RL, some amount of NLP. From the product standpoint, I have a lot of exposure to biotechnology and quantum computing, primarily in Drug Discovery. Right at the moment I am a Principal Machine Learning Architect in a company of around a thousand employees working with various pharma companies. 

**Ask clarifying questions on the problem and rephrase the problem** (On top of the board - key concepts):
Try to make assumptions instead of direct questions, ask if the assumptions are off.
The purpose of our business is to maximize profits - how do we achieve that? 

Formulate a plan for the presentation (5 equal vertical segments on the board):
Business goals and metrics, data collection and labelling, feature selection and engineering, modelling and deployment, monitoring and validation

## 1) Business goals and metrics

If there are several types of users, separate their requirements. (e.g., driver and passenger)

Functional requirements (whole service):
Possible actions in the system.

Non-functional requirements (whole service):
What does the user care about?

Given requirements, decouple goals of the ML system:
E.g. for NewsFeed:
• Filter out spam
• Filter out NSFW content
• Filter out misinformation
• Rank posts by quality
• Rank posts by engagement

Business metrics we want to maximize with these goals:
Take-Rate(netflix), Clicks, Conversion, Delay, Coverage, MAU (WAU, DAU). How are we constrained on some of this? These metrics might be used for validation and updating the model.

Business metrics to ML metrics for these goals:
What is X and Y? RMSE/MAPE/MSE/MAE, AUC, DCG, etc. Prioritize type-1 and type-2 errors. Think about metric to loss function. Think about model calibration.
It's possible to adapt loss function to a problem (somehow change the shape) - if there is time for it.

## 2) Data collection and labelling

Methods of collecting\labelling data, risks and constraints. Impact on the end user is crucial.

User input data? 
Lots of preprocessing and could be sensitive - think about privacy.

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

## 3) Feature selection and engineering

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

## 4) Modelling and Deployment

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

Distributed training and inference? (Data vs model parallelism)
Data - model is copied, data is splitted. Async gradients converge well due to sparse updates in large models. Batch is capped due to computational profits.
Model - vise-versa. Pipeline parallelism (like hardware instruction pipelining) - partial computations are sent forward to minimize stalling.

Online (generated on the fly) vs Offline (by schedule/event) predictions? Feature engineering equal between online/offline?

Deployment by layers? (FAANG has their own stuff most of the time)
Storage and compute - Characterized by how much memory (TB/GB) and how fast it runs (FLOPS\cores). Nearly impossible 100% utilization.
Resource management - Schedulers (when to run and what - DAGs, quotas, queues) and Orchestrators (where to get - instances, clusters)
ML Platform - model stores saving artifacts of models (MLFlow) and feature stores saving artifacts of data pipelines (DataHub).
Dev Environment - Versioning of data and code from Git to DVC to W&B. CI/CD like Git Actions or CircleCI. Standardize tools!

Model compression? 
Low-Rank Factorization (Convs), Knowledge Distillation, Architecture Pruning, Quantization.

Cloud vs Edge? 
Costs, latency, privacy.

Hardware optimization? 
Vectorization, Parallelization, Loop Tiling (Sequential Access), Operator Fusion (Redundancy), Vertical/Horizontal Graph Optimization

Offline Evaluation (Tests/Measurements)
Perturbation - adding noise or clipping samples in tst does not change predictions.
Invariance - changing sensitive features (age, race, etc) does not change predictions.
Directional Expectation - decreasing area does not increase price (illogical behavior).
Model calibration - prediction threshhold vs real frequency.
Confidence Measurement - sample-level threshold for confidence.
Slice-based - prioritize cancer diagnosis or fairly handle blacks\whites. (Simpson Paradox)


## 5) Monitoring and validation

Model debugging - A/B testing!, online and offline evaluation. Pro-actively consider solutions! Justify metric choices and how to compute them.

Reliability: How do we handle if the system fails somewhere? e.g., wrong translation or wrong ban.
Scalability: How to scale up and down the infrastructure and how to handle artifacts?
Maintainability: How to make documentation and versioning as accessible as possible?
Adaptability: How to make an easy shift if there is a change in data distribution or even business direction?

How the model might fail in production?
Edge cases - safety-critical predictions on extreme data samples.
Feedback loop - predictions influence the feedback and, thus, labeling. 
Data shift - Specific events, seasonal changes, growing trends. 

Dealing with edge cases? 
Devise a problem-specific heuristic, validation schema or even human-in-the-loop.

Dealing with degenerate feedback loops (homohenization of user behavior)?
Detect by different accuracy of the model across popularity bins. (by frequency of interactions)
Fix by randomization (bandits, epsilon-greedy) or causal inference. Include positional features (e.g., showed 1st, 2nd, ...). Separate prediction models.
E.g. propose the price based on elasticity, or predicted demand, or simply by a margin on top of the manufacturing cost. 

Types of data shifts P(X, Y) = P(Y|X)*P(X) = P(X|Y)*P(Y)?
Covariate shift P(X) (features)
Label shift P(Y) (labels)
Сoncept drift P(Y|X) (process) - seasonal\cyclic\event
Features - adding\removing features, changing their range or values.
Label Schemas - extending labels with new classes and values.

Dealing with data drift (source to target distribution difference)? 
Usually looking at P(X) distribution as P(Y) is delayed by feedback. 
Compare distribution statistics between distributions (variance, mean, skew, ...)
Statistical two-sample significance tests (e.g., Least Squares Density Difference, Chi-Squared).
Spatial (group of users, visible changes in clusters) vs Temporal shifts (interpret as time-series, choose windows wisely).

How to perform monitoring (Logs, Dashboards, Alerts)? 
Service Level Objectives (SLO), e.g. AWS EC2 uptime guarantee. It's all about metrics:
Software - latency; throughput; requests per minute, hour, day; the percentage of requests that return with a 2xx code; CPU/GPU utilization; memory utilization;
Accuracy-related metrics - click-rate, upvotes, shares, time-spent, completion rate.
Predictions - low-dimensional so easy to run statistical tests or spot drastic changes.
Features - validation by predefined schema for ranges and predefined sets of values.
Raw inputs - hard to unify, usually relies on the data engineering team.

Slicing user categories by?
Heuristics - Desktop\Browser\Location. 
Error analysis - manually find patterns.
Automatic - search\clustering of groups.

How frequently to update the model? 
Faster feedback loops or changes in behavior - faster updates.
Fine-tuning most of the time, occasional training from scratch for sanity-check. 
Rare special events (black friday) require updates in minutes, unlike other days.
Continuous cold start - not only new users, but also locations/devices/month.

Model iteration vs data iteration? More of a heuristic when to update the model or data.
Neural networks are better suited for online learning on new batches unlike matrixes\trees.

Deployment types?
Shadow - deploy in parallel, only log predictions and analyse offline. Expensive, heavy and slow, but reliant. 
Canary - deploy in parallel, gradually raise routing to the new model. Could be done in parallel with A/B tests.

Online Evaluation?
Test split on benchmark-dataset, backtest on frequently updated data points. Three types of production tests: 
A\B Testing (stateless) - route (randomly!) traffic between models, in some cases - alternate from day to day (e.g., pricing). Statistical significance with two-sample tests.
Interleaving Experiments - divide not users, but the predictions for the same user. Shows only preferences and is not directly connected to business metrics.
Bandits (stateful) - exploration\exploitation tradeoff while routing between models, i.e. finding best models while maximizing the accuracy. Outperforms A/B when feedback is fast.


## 0) Start iterating on details if there is time. Ask the interviewer if he wants to focus on something in particular or should you proceed on your notes.
