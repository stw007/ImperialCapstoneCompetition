**Initial Approach**

My aim was to produce a [Jupyter](https://jupyter.org/) Notebook to
analyse the inputs and outputs of an unknown function and recommend the
best candidate input to try next, in order to maximise it. I wanted to
create a general solution which could be used for arbitrary functions.

Rather than copying an existing implementation, I was keen to use the
exercise to understand more about optimisation in practice. I came
across the paper "[*Vanilla Bayesian Optimization Performs Great in High
Dimensions*](https://arxiv.org/pdf/2402.02229v3)" (Hvarfner et al) and
noted its conclusion that "*Our modification... reveals that standard
Bayesian optimization works drastically better than previously thought
in high dimensions, clearly outperforming existing state-of-the-art
algorithms on multiple commonly considered real-world high-dimensional
tasks.*" I decided to use their approach as my starting point, along
with their customisation of the "[BoTorch](https://botorch.org/)"
library to implement tuned Bayesian Optimisation (module 12) in PyTorch.

I was also interested to see how well various AI models could understand
technical content and implement code, so I tested them on the paper.
ChatGPT struggled to understand it, and its attempts at implementation
were either too simplistic or plagued with versioning problems, possibly
because BoTorch had evolved substantially over time. Gemini projected a
sophisticated understanding, but after producing ever more complex code
it became apparent that it was hallucinating and actively ignoring the
documentation. Claude proved the most reliable: it demonstrated a
reasonably analytical understanding of the paper and produced reliable
code, which became more sophisticated as I refined my prompts.

My initial codebase implemented a BayesianOptimiser class which took the
training data and used BoTorch to fit a model and suggest candidate
inputs to try next. BoTorch has various models, acquisition functions
and methods to help with this, and to begin with I simply used Expected
Improvement (EI), Upper Confidence Bound (UCB) and Potential Improvement
(PI) (module 12). My code found the top few candidates recommended by
each, and plotted the results in terms of projected mean, uncertainty
and acquisition level, using Principal Component Analysis (PCA - module
16) for the higher dimension functions. For the first submissions, I
trusted EI to pick the best candidate, balancing exploitation and
exploration.

This was my picture of the functions after a few submissions:

F1: I knew the functional form from the kick-off discussion.

F2: This seemed to have a narrow ridge with several peaks, and I was
finding new maxima.

F3: This seemed to have one high-value region, and I was finding new
maxima.

F4: I misunderstood the kick-off discussion so had flipped the sign,
i.e. was minimising...

F5: This had high values along the upper bounds of input space, where I
was finding new maxima.

F6: I was finding high values but struggling to find a new maximum.

F7: I was struggling to find high values at all.

F8: I was finding consistently high values and often a new maximum.

**Refinements**

I became concerned that lots of candidates were tightly clustered,
rather than exploring input space. Also, the algorithms sometimes seemed
to get "stuck" at the edges of that space. Therefore, my first
refinement was to boost the diversity of candidates. I explored a few
different methodologies, and found the most successful was to generate a
large number of random candidates; select those with the highest
acquisition values; apply k-means clustering (module 15); and then
select the best point from each cluster. For the next couple of
submissions, I used judgment to decide which of these to try, adding
more visualisations to help with the assessment, including dimensional
analysis and methods like "Parallel Coordinates" and "[t-distributed
Stochastic Neighbour
Embedding](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)"
(t-SNE) to find patterns in higher dimensions and see whether good
candidates were clustered or far apart. However, the overall approach
was less successful than I had hoped, as was switching to PI to see if a
higher projected mean would be more successful than a higher EI.

My next refinement was to implement bootstrapping (module 4). It had not
occurred to me that I could use this for candidate selection, but I
realised that I could analyse which candidates were suggested most often
across the bootstrap samples. The idea was that these might be more
robust suggestions, so I used bootstrapping entirely for one submission.
It was only moderately successful but importantly it found a new maximum
for F7 for the first time.

Researching alternative algorithms to help with Bayesian Optimisation, I
came across [HyperOpt](http://hyperopt.github.io/hyperopt/) and its
"[Tree-structured Parzen
Estimator](https://proceedings.neurips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf)"
(TPE), which models the probability distributions of inputs which
produce good vs bad outputs. I refactored my BayesianOptimizer class to
handle more than one model and enabled a number of different acquisition
functions and methods to suggest candidates, including TPE - which found
a new maximum for the previously tricky F7. I added
[scikit-learn](https://scikit-learn.org/)'s [Gaussian Process
Regressor](https://scikit-learn.org/stable/modules/gaussian_process.html)
(GPR) model so that I could try some different kernels (module 14),
starting with Matern and Radial Basis Function (RBF) and using k-fold
cross-validation (module 7) to select the one with the best fit (R^2^
score). My optimiser produced details of the fits and metrics (such as
projected mean and EI) for all the candidates, with a ranking and suite
of visualisations to help assess them. All of this increased my success
rate again, with new maxima being found for several functions each week.

I next added the ability to simulate a few steps forward before
suggesting a candidate, and tried that for some of the submissions, but
ultimately decided it was too error-prone because the simulation had to
rely on modelled outputs. I also realised it would have been useful to
keep track of which method I had been using each week, to compare their
track records, so I refactored the code to add analysis of the
optimisation approach itself, along with a method to try to reverse
engineer which approach I had used for each candidate submitted so far,
but this was error-prone and the analysis was time-consuming and
inconclusive.

I implemented some alternative regressors based on K-Nearest Neighbour
(module 8), Random Forest (modules 9-10), Gradient Boosting and Support
Vector Regression (module 14), but none performed as well as my existing
approaches. However, I

realised that I could be more ambitious with my choice of kernel, so I
added a dictionary of kernels with "Automatic Relevance Determination"
versions of RBF and Matern plus others with noise, periodicity, etc. and
various combinations. I also extended my code to handle logs of output
values, to help with some of the spikier functions. The resulting fits
for some functions were dramatically better, and it was interesting to
see how different kernels performed. Finally, I added some more advanced
BoTorch acquisition techniques, including "[Knowledge
Gradient](https://botorch.org/docs/tutorials/one_shot_kg/)", "[Max-value
Entropy Search](https://botorch.org/docs/tutorials/max_value_entropy/)"
and "[Thompson
Sampling](https://botorch.org/docs/tutorials/thompson_sampling/)" as
well as some Monte-Carlo methods, to work alongside my other candidate
selection techniques. All of this boosted my success for several
submissions.

After a while, though, I found that many of the candidates now had lower
projected values than my current best. Sometimes, this was because I had
found a reliable maximum, e.g. F5 seemed to be a multiplicative
function, maximised when all its inputs were; but in a couple of cases
the algorithm was just struggling. I decided to investigate feature
engineering (module 16), and added an optional preprocessing step to add
bespoke features to each function, as this seemed a difficult process to
generalise. I used Claude to suggest features, keeping the numbers
fairly small and in proportion to the amount of data. The implementation
was more complicated than I had hoped: with scikit-learn, it was
straightforward to operate in the real function dimensions for most of
my code, including candidate selection, but to preprocess to include
features when interacting with the model; BoTorch, however, has a
complicated tensor set-up designed to handle batch operations and the
interaction with input transformation was difficult to master. It was
illuminating to see that Claude and other AI models also struggled with
BoTorch's tensor handling. I did explore using the higher-level
"[Ax](https://ax.dev/)" framework to avoid the problem, and built a
working version, but Ax did not have quite enough flexibility. I could
now run my code with and without features, and they succeeded in
identifying new candidates with higher projected means. I was now seeing
much more progress with F6 and F7, and although I was struggling to beat
the current maximum with F4 (having finally realised that I should not
be flipping the sign...), the features were helping me to get close.

As I was now only finding new maxima for a couple of functions each
week, I did some more research to see if there was a way of using Deep
Learning and Neural Networks (modules 17-18), and found that I could do
so through "[Deep
Kernels](https://proceedings.mlr.press/v51/wilson16.html)", which build
a deep learning architecture into a kernel. However, the fits were not
as good as other kernels I was using, presumably because the datasets
were relatively small. While investigating them, I also came across
"[warped
kernels](https://botorch.org/docs/tutorials/bo_with_warped_gp/)", which
transform the input space to make optimisation easier, so I added those
alongside the other kernels and found that they did sometimes give the
best fits. I also added "[SHAP dependence
plots](https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/scatter.html)"
to further analyse the effect of inputs, and restructured the code to
save all the visualisations to file with an index of thumbnails, as with
plots of various combinations of input dimensions I was now generating a
large number of images.

At this stage, my code was stable and I simply updated the feature
engineering each week, to keep it current, as otherwise it could start
to be counterproductive. As the competition neared completion and most
of the EI measures had dropped close to zero, I focussed on the
projected mean of the various candidates, i.e. exploitation. While most
of the functions seemed to be at or around a maximum, there were one or
two -- notably F6 and F7 -- where I was still finding new maxima,
presumably because it took longer to identify good regions in the early
stages. I would have stuck with the same approach if the competition had
continued, particularly for those functions.

F1: From the functional form, I knew I had found the maximum.

F2: This was a complicated function of one variable with the other
almost irrelevant, so I was struggling to find the global maximum on a
high ridge, given the noise. I tried analysing it as a 1D function
instead.

F3: This seemed to have multiple high areas but I had found a maximum
very close to zero.

F4: The function seemed fairly well behaved but with one significant
interaction (SHAP curves splitting into two), and I had found a
reasonable maximum.

F5: This seemed to be a straightforward multiplicative function and I
had found the maximum.

F6: The function seemed fairly well behaved but with one significant
interaction (SHAP curves splitting into two), and I had found a
reasonable maximum close to zero.

F7: I was finally finding new maxima every week, but the function was
complex.

F8: The function seemed fairly well behaved, and I had found a good
maximum.

**Lessons Learned**

Some of the techniques I added were dead ends, at least for this
competition where the dataset was relatively small. For instance, some
of the more compute-intensive BoTorch acquisition functions based on
Monte-Carlo did not contribute much, and neither did the deep kernels;
the diverse candidate selection was also superseded when I added a wider
variety of candidate identification techniques. However, most of the
enhancements were valuable, particularly the kernel selection;
bootstrapping; feature engineering; and the visualisations, which were
very helpful in getting to grips with the functions. I also found it
helpful to have different models to compare: BoTorch with its own kernel
and in-built hyperparameter optimisation (module 19) vs scikit-learn
with my own kernel optimisation. I would use all these techniques from
the beginning, were I to repeat the exercise, and certainly for future
competitions.

**Jupyter Notebook**

See XXX and comments therein.

**Visualisations**

See the example at Visualisations/index.html XXX
