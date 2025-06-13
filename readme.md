# Explainable Vision Reinforcement Learning with PPO and Vision Transformers

This repository contains the implementation and results of a thesis project focused on enhancing the explainability of reinforcement learning (RL) agents using vision-based observations. 
The project investigates the use of Proximal Policy Optimization (PPO) combined with Vision Transformers (ViTs) and proposes novel methods for improving model interpretability.

 Our analysis reveals a significant improvement, with a 41% lower in mean squared error (MSE) loss between segmentation and embeddings correlation.  
 Furthermore, agent behavior interpretability is analyzed using tools such as decision trees. 
 Experimental results demonstrate that the proposed methods significantly enhance both the explainability of the models and the stability of the training process.
 
![segmentation gen](explain/segmentations_second-2.gif

![rewards](plots/reward-comp.png)

![explainer](plots/comp-explainer-loss.png)


# Materials and Methods
$$
L_t^{\text{CLIP}+\text{VF}+\text{ENT}+\text{SEG}}(\theta) = \mathbb{E}_t[L_t^\text{CLIP}(\theta)+c_1L_t^{\text{VF}}(\theta)+c_2L^\text{ENT}[\pi_\theta](s_t)]+c_3L_t^\text{SEG}(\theta)
$$

## Segmentation extraction
In this work, we propose two methods for extracting these features:
$$
\begin{enumerate}
    \item Create the extractor neural network $f_\theta(\text{embed}) \rightarrow \mathbf{v}(r, g, b), \quad \text{where } \mathbf{v} \in \mathbb{R}^3$, which maps the embedding to an RGB image representing segments.
    \item Create the linear projection map $g_\theta(\text{embed}) \rightarrow \mathbb{R}^{10}$ that maps the embedding to the logarithmic probabilities of belonging to one of the segment classes.
\end{enumerate}
$$

