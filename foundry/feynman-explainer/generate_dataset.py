"""
Generate a Feynman-style explanation dataset using Google Gemini.
Fully synthetic — each thread gets a pre-assigned specific concept,
guaranteeing diversity with no duplicate topics.

Uses ThreadPoolExecutor for parallel API calls (I/O-bound, safe).

Run from repo root:
    export GOOGLE_API_KEY=...
    python foundry/feynman-explainer/generate_dataset.py
    python foundry/feynman-explainer/generate_dataset.py --workers 10
"""

import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL = "gemini-2.5-flash-lite"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_FILE = os.path.join(DATA_DIR, "raw_feynman.jsonl")

# Each entry is (concept, category) — pre-assigned for zero duplication
CONCEPTS = [
    # Machine Learning & AI (120)
    ("How does gradient descent find the minimum of a loss function?", "ML & AI"),
    ("What is backpropagation and how does it compute gradients?", "ML & AI"),
    ("Why does overfitting happen and how do we detect it?", "ML & AI"),
    ("What is a learning rate and why does it matter so much?", "ML & AI"),
    ("How does a convolutional neural network detect edges in images?", "ML & AI"),
    ("What is the vanishing gradient problem?", "ML & AI"),
    ("Why does batch normalization help training?", "ML & AI"),
    ("How does dropout prevent overfitting?", "ML & AI"),
    ("What is an embedding and why do we need it for text?", "ML & AI"),
    ("How does a transformer encoder represent a sentence?", "ML & AI"),
    ("What is attention in neural networks?", "ML & AI"),
    ("Why do we need multi-head attention instead of just one?", "ML & AI"),
    ("How does positional encoding work in transformers?", "ML & AI"),
    ("What is a residual connection and why does it help deep networks?", "ML & AI"),
    ("How does a language model predict the next word?", "ML & AI"),
    ("What is temperature in text generation and how does it affect output?", "ML & AI"),
    ("What is beam search and why is it better than greedy decoding?", "ML & AI"),
    ("How does RLHF make language models more helpful?", "ML & AI"),
    ("What is a reward model in reinforcement learning from human feedback?", "ML & AI"),
    ("Why does fine-tuning a pretrained model work so well?", "ML & AI"),
    ("What is LoRA and how does it reduce the parameters we need to train?", "ML & AI"),
    ("How does a random forest make a decision?", "ML & AI"),
    ("Why is a random forest more accurate than a single decision tree?", "ML & AI"),
    ("What is boosting and how does XGBoost use it?", "ML & AI"),
    ("How does K-means clustering find groups in data?", "ML & AI"),
    ("What is a support vector machine and what is the margin?", "ML & AI"),
    ("How does a recommendation system decide what to show you?", "ML & AI"),
    ("What is collaborative filtering?", "ML & AI"),
    ("What is transfer learning?", "ML & AI"),
    ("What is the difference between generative and discriminative models?", "ML & AI"),
    ("How does a GAN work — the generator vs the discriminator?", "ML & AI"),
    ("What is a variational autoencoder?", "ML & AI"),
    ("How does CLIP connect images and text?", "ML & AI"),
    ("What is a knowledge graph?", "ML & AI"),
    ("What is reinforcement learning?", "ML & AI"),
    ("What is the exploration vs exploitation tradeoff?", "ML & AI"),
    ("How does Q-learning work?", "ML & AI"),
    ("What is the difference between model-based and model-free RL?", "ML & AI"),
    ("What is a confusion matrix and what does each cell mean?", "ML & AI"),
    ("What is precision and recall and why do they trade off?", "ML & AI"),
    ("What is the F1 score?", "ML & AI"),
    ("What is ROC-AUC and what does it actually measure?", "ML & AI"),
    ("Why do we normalize features before training?", "ML & AI"),
    ("What is the curse of dimensionality?", "ML & AI"),
    ("How does PCA reduce dimensions?", "ML & AI"),
    ("What is t-SNE and why does it cluster differently than PCA?", "ML & AI"),
    ("What is regularization and why does it prevent overfitting?", "ML & AI"),
    ("What is the difference between L1 and L2 regularization?", "ML & AI"),
    ("Why do we use cross-entropy loss for classification?", "ML & AI"),
    ("What is the difference between sigmoid and softmax?", "ML & AI"),
    ("What is a hyperparameter and why can't we just train it?", "ML & AI"),
    ("How does cross-validation work?", "ML & AI"),
    ("What is data augmentation and why does it help?", "ML & AI"),
    ("What is tokenization in NLP?", "ML & AI"),
    ("How does word2vec learn word embeddings?", "ML & AI"),
    ("What is the difference between BERT and GPT architectures?", "ML & AI"),
    ("What is a context window in a language model?", "ML & AI"),
    ("What is hallucination in LLMs and why does it happen?", "ML & AI"),
    ("What is prompt engineering?", "ML & AI"),
    ("How does RAG (retrieval-augmented generation) work?", "ML & AI"),
    ("What is a vector database?", "ML & AI"),
    ("What is semantic search?", "ML & AI"),
    ("How does cosine similarity measure the relationship between embeddings?", "ML & AI"),
    ("What is the bias-variance tradeoff?", "ML & AI"),
    ("How does a neural network initialize its weights and why does it matter?", "ML & AI"),
    ("What is an activation function and why do we need non-linearity?", "ML & AI"),
    ("What is the difference between ReLU and sigmoid activations?", "ML & AI"),
    ("What is a recurrent neural network?", "ML & AI"),
    ("Why do RNNs struggle with long sequences?", "ML & AI"),
    ("What is an LSTM and how does it solve the vanishing gradient problem?", "ML & AI"),
    ("What is model distillation?", "ML & AI"),
    ("What is quantization in the context of ML models?", "ML & AI"),
    ("What is federated learning?", "ML & AI"),
    ("What is the difference between online and batch learning?", "ML & AI"),
    ("What is a Naive Bayes classifier?", "ML & AI"),
    ("What is logistic regression despite its name?", "ML & AI"),
    ("What is the difference between supervised, unsupervised, and self-supervised learning?", "ML & AI"),
    ("How does label smoothing help training?", "ML & AI"),
    ("What is a mixture of experts model?", "ML & AI"),
    ("How does flash attention make transformers faster?", "ML & AI"),
    ("What is sparse attention?", "ML & AI"),
    ("What is curriculum learning?", "ML & AI"),
    ("How does an image diffusion model generate images?", "ML & AI"),
    ("What is a latent space?", "ML & AI"),
    ("What is the encoder-decoder architecture?", "ML & AI"),
    ("How does machine translation work?", "ML & AI"),
    ("What is named entity recognition?", "ML & AI"),
    ("What is zero-shot learning?", "ML & AI"),
    ("What is few-shot prompting?", "ML & AI"),
    ("How does chain-of-thought prompting improve reasoning?", "ML & AI"),
    ("What is a decision boundary?", "ML & AI"),
    ("What is semi-supervised learning?", "ML & AI"),
    ("How does active learning reduce labeling cost?", "ML & AI"),
    ("What is catastrophic forgetting in neural networks?", "ML & AI"),
    ("What is a Siamese network?", "ML & AI"),
    ("How does object detection differ from image classification?", "ML & AI"),
    ("What is semantic segmentation?", "ML & AI"),
    ("Why is data quality more important than model complexity?", "ML & AI"),
    ("What is feature importance and how do we measure it?", "ML & AI"),
    ("What is SHAP and how does it explain model predictions?", "ML & AI"),
    ("What is concept drift?", "ML & AI"),
    ("What is MLOps?", "ML & AI"),
    ("How does A/B testing work for ML models?", "ML & AI"),
    ("What is shadow deployment for ML?", "ML & AI"),
    ("How do we monitor a machine learning model in production?", "ML & AI"),
    ("What is the difference between a model and a pipeline?", "ML & AI"),
    ("Why does more data often beat a better algorithm?", "ML & AI"),
    ("What is the no free lunch theorem?", "ML & AI"),
    ("What is Occam's razor and how does it apply to model selection?", "ML & AI"),
    ("What is the difference between a parameter and a statistic?", "ML & AI"),
    ("How does k-nearest neighbors make predictions?", "ML & AI"),
    ("What is anomaly detection?", "ML & AI"),
    ("What is time series forecasting?", "ML & AI"),
    ("How does a transformer handle variable-length inputs?", "ML & AI"),
    ("What is the softmax function doing geometrically?", "ML & AI"),
    ("What is layer normalization vs batch normalization?", "ML & AI"),
    ("How does weight decay work?", "ML & AI"),
    ("What is momentum in gradient descent?", "ML & AI"),
    ("How does Adam optimizer work?", "ML & AI"),
    # Statistics & Probability (100)
    ("What is a p-value and why do people misuse it?", "Statistics"),
    ("What is the central limit theorem?", "Statistics"),
    ("What is the law of large numbers?", "Statistics"),
    ("What is Bayes theorem and when should you use it?", "Statistics"),
    ("What is conditional probability?", "Statistics"),
    ("What is the difference between correlation and causation?", "Statistics"),
    ("What is statistical significance?", "Statistics"),
    ("What is a confidence interval?", "Statistics"),
    ("What is the difference between frequentist and Bayesian statistics?", "Statistics"),
    ("What is variance and why do we care about it?", "Statistics"),
    ("What is standard deviation vs standard error?", "Statistics"),
    ("What is a normal distribution and why is it everywhere?", "Statistics"),
    ("What is the Poisson distribution?", "Statistics"),
    ("What is a binomial distribution?", "Statistics"),
    ("What is sampling bias?", "Statistics"),
    ("What is Simpson's paradox?", "Statistics"),
    ("What is multiple hypothesis testing and why is it a problem?", "Statistics"),
    ("What is effect size?", "Statistics"),
    ("What is statistical power?", "Statistics"),
    ("What is a Type I vs Type II error?", "Statistics"),
    ("What is a null hypothesis?", "Statistics"),
    ("What is chi-squared test?", "Statistics"),
    ("What is an ANOVA test?", "Statistics"),
    ("What is linear regression?", "Statistics"),
    ("What is the coefficient of determination R-squared?", "Statistics"),
    ("What is heteroscedasticity?", "Statistics"),
    ("What is autocorrelation?", "Statistics"),
    ("What is multicollinearity?", "Statistics"),
    ("What is a random variable?", "Statistics"),
    ("What is expected value?", "Statistics"),
    ("What is entropy in information theory?", "Statistics"),
    ("What is mutual information?", "Statistics"),
    ("What is a Markov chain?", "Statistics"),
    ("What is Monte Carlo simulation?", "Statistics"),
    ("What is bootstrapping in statistics?", "Statistics"),
    ("What is a t-test?", "Statistics"),
    ("What is the difference between population and sample?", "Statistics"),
    ("What is regression to the mean?", "Statistics"),
    ("What is selection bias?", "Statistics"),
    ("What is survivorship bias?", "Statistics"),
    ("What is the gambler's fallacy?", "Statistics"),
    ("What is the birthday paradox?", "Statistics"),
    ("What is a Bayesian prior?", "Statistics"),
    ("What is a likelihood function?", "Statistics"),
    ("What is maximum likelihood estimation?", "Statistics"),
    ("What is the law of total probability?", "Statistics"),
    ("What is a joint probability distribution?", "Statistics"),
    ("What is covariance vs correlation?", "Statistics"),
    ("What is an outlier and how should we handle it?", "Statistics"),
    ("What is the difference between parametric and non-parametric tests?", "Statistics"),
    ("What is Kullback-Leibler divergence?", "Statistics"),
    ("What is Jensen's inequality?", "Statistics"),
    ("What is the median vs mean and when does each matter?", "Statistics"),
    ("What is a box plot and what does each part show?", "Statistics"),
    ("What is a z-score?", "Statistics"),
    ("What is a percentile?", "Statistics"),
    ("What is the geometric mean?", "Statistics"),
    ("What is a power law distribution?", "Statistics"),
    ("What is a long tail distribution?", "Statistics"),
    ("What is Zipf's law?", "Statistics"),
    ("What is the Monty Hall problem?", "Statistics"),
    ("What is the inspection paradox?", "Statistics"),
    ("What is Benford's law?", "Statistics"),
    ("What is the base rate fallacy?", "Statistics"),
    ("What is a lurking variable?", "Statistics"),
    ("What is the ecological fallacy?", "Statistics"),
    ("What is an instrumental variable?", "Statistics"),
    ("What is difference-in-differences?", "Statistics"),
    ("What is a randomized controlled trial?", "Statistics"),
    ("What is propensity score matching?", "Statistics"),
    ("What is causal inference?", "Statistics"),
    ("What is counterfactual reasoning?", "Statistics"),
    ("What is the stable unit treatment value assumption (SUTVA)?", "Statistics"),
    ("What is hazard rate in survival analysis?", "Statistics"),
    ("What is a Kaplan-Meier curve?", "Statistics"),
    ("What is time-to-event data?", "Statistics"),
    ("What is Bayesian updating?", "Statistics"),
    ("What is a conjugate prior?", "Statistics"),
    ("What is the Dirichlet distribution?", "Statistics"),
    ("What is hierarchical modeling?", "Statistics"),
    ("What is mixed effects modeling?", "Statistics"),
    ("What is empirical Bayes?", "Statistics"),
    ("What is the James-Stein estimator?", "Statistics"),
    ("What is regularization from a Bayesian perspective?", "Statistics"),
    ("What is an unbiased estimator?", "Statistics"),
    ("What is minimum variance unbiased estimation?", "Statistics"),
    ("What is the Cramér-Rao lower bound?", "Statistics"),
    ("What is a sufficient statistic?", "Statistics"),
    ("What is the likelihood ratio test?", "Statistics"),
    ("What is the Wald test?", "Statistics"),
    ("What is a permutation test?", "Statistics"),
    ("What is cross-validation in statistics?", "Statistics"),
    ("What is the AIC model selection criterion?", "Statistics"),
    ("What is the BIC model selection criterion?", "Statistics"),
    ("What is the false discovery rate?", "Statistics"),
    ("What is the Bonferroni correction?", "Statistics"),
    ("What is meta-analysis?", "Statistics"),
    ("What is publication bias?", "Statistics"),
    # Physics (80)
    ("Why does entropy always increase?", "Physics"),
    ("Why does ice float on water?", "Physics"),
    ("Why is the sky blue?", "Physics"),
    ("What is quantum entanglement?", "Physics"),
    ("What is the wave-particle duality of light?", "Physics"),
    ("How does a laser work?", "Physics"),
    ("What is Heisenberg's uncertainty principle?", "Physics"),
    ("What is a black hole?", "Physics"),
    ("How does gravity bend light?", "Physics"),
    ("What is time dilation?", "Physics"),
    ("What is the photoelectric effect?", "Physics"),
    ("How does a transistor work?", "Physics"),
    ("What is electrical resistance?", "Physics"),
    ("How does electricity flow through a wire?", "Physics"),
    ("What is a magnetic field?", "Physics"),
    ("How does a motor turn electricity into motion?", "Physics"),
    ("What is thermodynamics?", "Physics"),
    ("What is heat vs temperature?", "Physics"),
    ("What is a phase transition?", "Physics"),
    ("Why does water boil at 100C?", "Physics"),
    ("What is the Doppler effect?", "Physics"),
    ("What is resonance?", "Physics"),
    ("Why does a spinning top not fall over?", "Physics"),
    ("What is centripetal force?", "Physics"),
    ("What is conservation of momentum?", "Physics"),
    ("What is the double slit experiment?", "Physics"),
    ("What is a photon?", "Physics"),
    ("What is nuclear fission?", "Physics"),
    ("What is radioactive decay?", "Physics"),
    ("What is the strong nuclear force?", "Physics"),
    ("What is plasma?", "Physics"),
    ("How does a superconductor work?", "Physics"),
    ("What is a semiconductor?", "Physics"),
    ("What is the Pauli exclusion principle?", "Physics"),
    ("What is spin in quantum mechanics?", "Physics"),
    ("What is a wavefunction?", "Physics"),
    ("What is superposition in quantum mechanics?", "Physics"),
    ("How does an MRI machine work?", "Physics"),
    ("What is dark matter?", "Physics"),
    ("What is dark energy?", "Physics"),
    ("What is the Big Bang?", "Physics"),
    ("What is cosmic inflation?", "Physics"),
    ("Why do things fall at the same rate regardless of mass?", "Physics"),
    ("What is special relativity?", "Physics"),
    ("What is the speed of light limit?", "Physics"),
    ("What is energy and why is it conserved?", "Physics"),
    ("What is potential energy vs kinetic energy?", "Physics"),
    ("How does a nuclear reactor work?", "Physics"),
    ("What is sound?", "Physics"),
    ("How does a rainbow form?", "Physics"),
    ("What is total internal reflection?", "Physics"),
    ("Why is the ocean blue?", "Physics"),
    ("How does a microwave oven heat food?", "Physics"),
    ("What is a standing wave?", "Physics"),
    ("What is Bernoulli's principle?", "Physics"),
    ("How does an airplane wing generate lift?", "Physics"),
    ("What is surface tension?", "Physics"),
    ("How does capillary action work?", "Physics"),
    ("What is osmotic pressure?", "Physics"),
    ("What is Brownian motion?", "Physics"),
    ("What is diffusion?", "Physics"),
    ("What is a heat pump?", "Physics"),
    ("How does a refrigerator work?", "Physics"),
    ("What is an electromagnetic wave?", "Physics"),
    ("What is the electromagnetic spectrum?", "Physics"),
    ("How does WiFi transmit data wirelessly?", "Physics"),
    ("What is interference in waves?", "Physics"),
    ("What is polarization of light?", "Physics"),
    ("Why is glass transparent?", "Physics"),
    ("What is refractive index?", "Physics"),
    ("How does GPS know where you are?", "Physics"),
    ("What is the equivalence principle in general relativity?", "Physics"),
    ("What is Hawking radiation?", "Physics"),
    ("What is the Casimir effect?", "Physics"),
    ("What is the Coriolis effect?", "Physics"),
    ("Why does the moon always show the same face to Earth?", "Physics"),
    ("What is tidal locking?", "Physics"),
    ("What is the anthropic principle?", "Physics"),
    # Mathematics & Calculus (80)
    ("What is a derivative and what does it actually mean?", "Math"),
    ("What is an integral and what is it computing?", "Math"),
    ("What is the fundamental theorem of calculus?", "Math"),
    ("What is a limit in calculus?", "Math"),
    ("What is a Taylor series?", "Math"),
    ("What is a Fourier transform?", "Math"),
    ("What is an eigenvector?", "Math"),
    ("What is a matrix multiplication actually doing?", "Math"),
    ("What is a determinant of a matrix?", "Math"),
    ("What is the rank of a matrix?", "Math"),
    ("What is the null space of a matrix?", "Math"),
    ("What is a dot product?", "Math"),
    ("What is a cross product?", "Math"),
    ("What is a gradient in vector calculus?", "Math"),
    ("What is the chain rule?", "Math"),
    ("What is a partial derivative?", "Math"),
    ("What is a convex function?", "Math"),
    ("What is the Lagrangian and Lagrange multipliers?", "Math"),
    ("What is an optimization problem?", "Math"),
    ("What is the difference between local and global minima?", "Math"),
    ("What is a saddle point?", "Math"),
    ("What is compounding interest and why does it grow so fast?", "Math"),
    ("What is a logarithm?", "Math"),
    ("What is e (Euler's number)?", "Math"),
    ("What is the natural logarithm?", "Math"),
    ("What is a complex number?", "Math"),
    ("What is Euler's formula e^(i*pi) + 1 = 0?", "Math"),
    ("What is a prime number and why are they important?", "Math"),
    ("What is modular arithmetic?", "Math"),
    ("What is a proof by contradiction?", "Math"),
    ("What is a recursive function?", "Math"),
    ("What is infinity in mathematics?", "Math"),
    ("What is the halting problem?", "Math"),
    ("What is Gödel's incompleteness theorem?", "Math"),
    ("What is graph theory?", "Math"),
    ("What is a spanning tree?", "Math"),
    ("What is a Hamiltonian path?", "Math"),
    ("What is the four color theorem?", "Math"),
    ("What is a fractal?", "Math"),
    ("What is the Mandelbrot set?", "Math"),
    ("What is a topological space?", "Math"),
    ("What is a manifold?", "Math"),
    ("What is the Pythagorean theorem and why is it true?", "Math"),
    ("What is a vector space?", "Math"),
    ("What is linear independence?", "Math"),
    ("What is a basis of a vector space?", "Math"),
    ("What is orthogonality?", "Math"),
    ("What is a projection in linear algebra?", "Math"),
    ("What is singular value decomposition?", "Math"),
    ("What is a convolution?", "Math"),
    ("What is the Laplace transform?", "Math"),
    ("What is a differential equation?", "Math"),
    ("What is a system of equations?", "Math"),
    ("What is the Gaussian elimination?", "Math"),
    ("What is the simplex method?", "Math"),
    ("What is dynamic programming?", "Math"),
    ("What is the traveling salesman problem?", "Math"),
    ("What is P vs NP?", "Math"),
    ("What is a Monte Carlo method?", "Math"),
    ("What is number theory?", "Math"),
    ("What is the Riemann hypothesis?", "Math"),
    ("What is a topology?", "Math"),
    ("What is measure theory?", "Math"),
    ("What is a Hilbert space?", "Math"),
    ("What is an inner product?", "Math"),
    ("What is functional analysis?", "Math"),
    ("What is the spectral theorem?", "Math"),
    ("What is a random walk?", "Math"),
    ("What is a martingale?", "Math"),
    ("What is a Gaussian process?", "Math"),
    ("What is a Fourier series?", "Math"),
    ("What is the convolution theorem?", "Math"),
    ("What is a Dirac delta function?", "Math"),
    ("What is Parseval's theorem?", "Math"),
    ("What is the Nyquist sampling theorem?", "Math"),
    ("What is numerical stability?", "Math"),
    ("What is floating point arithmetic?", "Math"),
    ("What is gradient of a scalar field?", "Math"),
    ("What is the Jacobian matrix?", "Math"),
    ("What is the Hessian matrix?", "Math"),
    # Computer Science (80)
    ("What is a hash function?", "CS"),
    ("What is Big O notation?", "CS"),
    ("What is recursion?", "CS"),
    ("What is a stack vs a heap in memory?", "CS"),
    ("What is garbage collection?", "CS"),
    ("What is a pointer in C?", "CS"),
    ("What is a binary search tree?", "CS"),
    ("What is a heap data structure?", "CS"),
    ("What is a graph and how is it represented?", "CS"),
    ("What is breadth-first vs depth-first search?", "CS"),
    ("What is dynamic programming?", "CS"),
    ("What is memoization?", "CS"),
    ("What is a greedy algorithm?", "CS"),
    ("What is divide and conquer?", "CS"),
    ("What is a sorting algorithm and why does quicksort work?", "CS"),
    ("What is a hash table collision?", "CS"),
    ("What is a linked list?", "CS"),
    ("What is a queue vs a stack?", "CS"),
    ("What is a binary heap?", "CS"),
    ("What is a trie?", "CS"),
    ("What is cache memory and why is it fast?", "CS"),
    ("What is CPU cache hit vs cache miss?", "CS"),
    ("What is virtual memory?", "CS"),
    ("What is a process vs a thread?", "CS"),
    ("What is a deadlock?", "CS"),
    ("What is a race condition?", "CS"),
    ("What is a mutex?", "CS"),
    ("What is a semaphore?", "CS"),
    ("What is an operating system kernel?", "CS"),
    ("How does a computer boot?", "CS"),
    ("What is a compiled vs interpreted language?", "CS"),
    ("What is a type system?", "CS"),
    ("What is functional programming?", "CS"),
    ("What is immutability?", "CS"),
    ("What is a closure in programming?", "CS"),
    ("What is a monad?", "CS"),
    ("What is object-oriented programming?", "CS"),
    ("What is polymorphism?", "CS"),
    ("What is inheritance vs composition?", "CS"),
    ("What is a design pattern?", "CS"),
    ("What is a REST API?", "CS"),
    ("What is HTTP?", "CS"),
    ("What is TCP vs UDP?", "CS"),
    ("What is DNS?", "CS"),
    ("How does SSL/TLS encryption work?", "CS"),
    ("What is a public key and private key?", "CS"),
    ("What is a blockchain?", "CS"),
    ("What is a smart contract?", "CS"),
    ("What is a database index?", "CS"),
    ("What is SQL ACID?", "CS"),
    ("What is eventual consistency?", "CS"),
    ("What is a distributed system?", "CS"),
    ("What is the CAP theorem?", "CS"),
    ("What is sharding in databases?", "CS"),
    ("What is a message queue?", "CS"),
    ("What is a microservice?", "CS"),
    ("What is containerization and Docker?", "CS"),
    ("What is Kubernetes?", "CS"),
    ("What is a load balancer?", "CS"),
    ("What is a CDN?", "CS"),
    ("What is a compiler?", "CS"),
    ("What is an abstract syntax tree?", "CS"),
    ("What is garbage collection?", "CS"),
    ("What is tail call optimization?", "CS"),
    ("What is a regular expression?", "CS"),
    ("What is a finite state machine?", "CS"),
    ("What is a Turing machine?", "CS"),
    ("What is the lambda calculus?", "CS"),
    ("What is a bit and a byte?", "CS"),
    ("What is two's complement for negative numbers?", "CS"),
    ("What is a floating point number?", "CS"),
    ("What is a race condition in distributed systems?", "CS"),
    ("What is consensus in distributed systems?", "CS"),
    ("What is the Paxos algorithm?", "CS"),
    ("What is the Raft consensus algorithm?", "CS"),
    ("What is a Bloom filter?", "CS"),
    ("What is consistent hashing?", "CS"),
    ("What is a vector clock?", "CS"),
    ("What is backpressure in streaming systems?", "CS"),
    ("What is a binary number system?", "CS"),
    # Biology & Chemistry (60)
    ("How does DNA store information?", "Biology"),
    ("What is protein folding?", "Biology"),
    ("How does natural selection work?", "Biology"),
    ("What is the cell membrane and why is it selective?", "Biology"),
    ("How does ATP store energy?", "Biology"),
    ("What is photosynthesis at a molecular level?", "Biology"),
    ("How does the immune system recognize invaders?", "Biology"),
    ("What is CRISPR?", "Biology"),
    ("What is epigenetics?", "Biology"),
    ("How does a vaccine train the immune system?", "Biology"),
    ("What is a neuron and how does it fire?", "Biology"),
    ("How do synapses work?", "Biology"),
    ("What is the blood-brain barrier?", "Biology"),
    ("How does a hormone signal work?", "Biology"),
    ("What is enzyme catalysis?", "Biology"),
    ("What is a chemical bond?", "Biology"),
    ("What is electronegativity?", "Biology"),
    ("What is pH?", "Biology"),
    ("What is osmosis?", "Biology"),
    ("What is diffusion vs active transport?", "Biology"),
    ("How does oxygen bind to hemoglobin?", "Biology"),
    ("What is evolution at the gene level?", "Biology"),
    ("What is genetic drift?", "Biology"),
    ("What is horizontal gene transfer?", "Biology"),
    ("What is the endosymbiotic theory?", "Biology"),
    ("How do antibiotics work?", "Biology"),
    ("What is antibiotic resistance?", "Biology"),
    ("How does a virus replicate?", "Biology"),
    ("What is mRNA and how does it differ from DNA?", "Biology"),
    ("What is the central dogma of molecular biology?", "Biology"),
    ("What is a mutation?", "Biology"),
    ("What is cancer at the cellular level?", "Biology"),
    ("How does the heart pump blood?", "Biology"),
    ("What is cholesterol?", "Biology"),
    ("What is a chemical equilibrium?", "Biology"),
    ("What is activation energy?", "Biology"),
    ("What is a catalyst?", "Biology"),
    ("What is oxidation and reduction?", "Biology"),
    ("What is hydrogen bonding?", "Biology"),
    ("What makes water such a special molecule?", "Biology"),
    ("What is chirality in chemistry?", "Biology"),
    ("What is a polymer?", "Biology"),
    ("What is the periodic table organizing?", "Biology"),
    ("What is valence electron?", "Biology"),
    ("What is ionic vs covalent bonding?", "Biology"),
    ("What is quantum tunneling in chemistry?", "Biology"),
    ("What is photocatalysis?", "Biology"),
    ("What is the nitrogen cycle?", "Biology"),
    ("What is fermentation?", "Biology"),
    ("How do solar panels convert light to electricity?", "Biology"),
    ("What is entropy in chemistry?", "Biology"),
    ("What is enthalpy?", "Biology"),
    ("What is Gibbs free energy?", "Biology"),
    ("What is Le Chatelier's principle?", "Biology"),
    ("What is a buffer solution?", "Biology"),
    ("What is titration?", "Biology"),
    ("How does chromatography separate compounds?", "Biology"),
    ("What is mass spectrometry?", "Biology"),
    ("What is NMR spectroscopy?", "Biology"),
    ("What is a stereoisomer?", "Biology"),
    # Economics & Finance (60)
    ("What is compounding interest and why does it matter?", "Economics"),
    ("What is inflation?", "Economics"),
    ("What is the time value of money?", "Economics"),
    ("What is opportunity cost?", "Economics"),
    ("What is supply and demand?", "Economics"),
    ("What is price elasticity?", "Economics"),
    ("What is a Nash equilibrium?", "Economics"),
    ("What is a Prisoner's dilemma?", "Economics"),
    ("What is moral hazard?", "Economics"),
    ("What is adverse selection?", "Economics"),
    ("What is an externality?", "Economics"),
    ("What is the tragedy of the commons?", "Economics"),
    ("What is a public good?", "Economics"),
    ("What is GDP?", "Economics"),
    ("What is monetary policy?", "Economics"),
    ("What is quantitative easing?", "Economics"),
    ("What is fractional reserve banking?", "Economics"),
    ("How does compound interest create inequality?", "Economics"),
    ("What is a bond and how does its price relate to interest rates?", "Economics"),
    ("What is a stock?", "Economics"),
    ("What is the efficient market hypothesis?", "Economics"),
    ("What is arbitrage?", "Economics"),
    ("What is leverage in finance?", "Economics"),
    ("What is a derivative in finance?", "Economics"),
    ("What is an option in finance?", "Economics"),
    ("What is beta in finance?", "Economics"),
    ("What is diversification?", "Economics"),
    ("What is the capital asset pricing model (CAPM)?", "Economics"),
    ("What is the risk-free rate?", "Economics"),
    ("What is a yield curve?", "Economics"),
    ("What is a recession?", "Economics"),
    ("What is deflation and why is it bad?", "Economics"),
    ("What is a trade deficit?", "Economics"),
    ("What is comparative advantage?", "Economics"),
    ("What is a monopoly?", "Economics"),
    ("What is network effects?", "Economics"),
    ("What is diminishing returns?", "Economics"),
    ("What is the law of one price?", "Economics"),
    ("What is purchasing power parity?", "Economics"),
    ("What is a Ponzi scheme?", "Economics"),
    ("What is game theory?", "Economics"),
    ("What is a dominant strategy?", "Economics"),
    ("What is mechanism design?", "Economics"),
    ("What is behavioral economics?", "Economics"),
    ("What is loss aversion?", "Economics"),
    ("What is anchoring bias?", "Economics"),
    ("What is the sunk cost fallacy?", "Economics"),
    ("What is hyperbolic discounting?", "Economics"),
    ("What is the winner's curse?", "Economics"),
    ("What is Goodhart's law?", "Economics"),
    ("What is Gresham's law?", "Economics"),
    ("What is the Laffer curve?", "Economics"),
    ("What is a natural monopoly?", "Economics"),
    ("What is price discrimination?", "Economics"),
    ("What is the multiplier effect?", "Economics"),
    ("What is creative destruction?", "Economics"),
    ("What is a bubble in financial markets?", "Economics"),
    ("What is mean reversion?", "Economics"),
    ("What is liquidity?", "Economics"),
    ("What is systemic risk?", "Economics"),
]

PROMPT_TEMPLATE = """\
Explain the following concept in the style of Richard Feynman:

Concept: {concept}

Rules:
- Open with a concrete everyday analogy or scenario BEFORE any abstraction
- Build from the ground up — never assume jargon is known
- Short, punchy sentences. One idea per sentence.
- Use phrases like: "Here's the thing...", "Imagine you're...", "The key insight is...",
  "Most people get confused because...", "Now here's where it gets interesting..."
- Unpack every technical term immediately in plain English
- End with the core insight in 1-2 crystallizing sentences
- NO bullet points. NO headers. NO markdown. Pure flowing prose.
- Length: 180–320 words."""

_print_lock = threading.Lock()
_counter = {"done": 0, "total": 0}


def log(msg: str):
    with _print_lock:
        print(msg, flush=True)


def generate_one(client: genai.Client, concept: str, category: str, idx: int,
                 retries: int = 4) -> dict | None:
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=PROMPT_TEMPLATE.format(concept=concept),
                config=types.GenerateContentConfig(
                    max_output_tokens=600,
                    temperature=0.80,
                ),
            )
            text = response.text.strip()
            if len(text) > 100:
                with _print_lock:
                    _counter["done"] += 1
                    done = _counter["done"]
                    total = _counter["total"]
                    print(f"  ✓ [{done:4d}/{total}] [{category}] {concept[:55]}", flush=True)
                return {"instruction": concept, "response": text, "category": category}
        except Exception as e:
            err = str(e)
            if "503" in err or "quota" in err.lower() or "429" in err or "rate" in err.lower():
                wait = 2 ** attempt * 4
                log(f"  ⏳ [{idx}] Rate limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                log(f"  ✗ [{idx}] Error: {err[:80]}")
                time.sleep(1)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel threads (default 8, lower if rate-limited)")
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY environment variable.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    os.makedirs(DATA_DIR, exist_ok=True)

    _counter["total"] = len(CONCEPTS)

    print(f"\n{'='*60}")
    print(f"Feynman Dataset Generator")
    print(f"Model   : {MODEL}")
    print(f"Concepts: {len(CONCEPTS)}")
    print(f"Workers : {args.workers} parallel threads")
    print(f"Output  : {OUT_FILE}")
    print(f"{'='*60}\n")

    records = []
    failed = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(generate_one, client, concept, cat, i): (concept, cat)
            for i, (concept, cat) in enumerate(CONCEPTS, 1)
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                records.append(result)
            else:
                failed += 1

    elapsed = time.time() - start

    with open(OUT_FILE, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\n{'='*60}")
    print(f"Completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Generated : {len(records)} / {len(CONCEPTS)}")
    print(f"Failed    : {failed}")
    print()
    cats = {}
    for r in records:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {n:4d}  {cat}")
    print(f"\nOutput: {OUT_FILE}")
    print(f"Next  : python foundry/feynman-explainer/prepare_data.py")


if __name__ == "__main__":
    main()
