---
title: Lecture Note - Overview of Modern Approach of Causal Inference
date: 2024-07-27 22:43:52
tags: knowledge
---

This is one of my notes of the online Causal Inference Course in Columbia University, taught by Michael E. Sobel who is a professor in the Department of Statistics. It would be good to have this overview of causal inference regarding the framework in mind before getting into statistical and theoretical details, especially for beginners. A clear approach framework is very important that it’s not only because rigorous experiment and analysis methods could be developed with this well-defined framework, but also because it can guide you to deal with challenging situations in a correct way, while being clear about the limitations and assumptions at the same time. This is why it became a Science.

Modern Approach for Causal Inference
--------------------------------------------------------------------------------------------------------------------

The modern dominant approach for causal inference, significantly influenced by Donald Rubin’s contributions, primarily revolves around the following key ideas:

1.  Potential Outcomes Framework:
    
    *   **Potential Outcomes Notation**: Introduced by Neyman and further developed by Rubin, this framework involves conceptualizing the outcomes that would occur both with and without the treatment for each unit. Each unit has a potential outcome under treatment and a potential outcome under control, but only one of these outcomes is observed for each unit.
    *   **Average Treatment Effects**: The focus is on estimating the average causal effect of a treatment across a population. This involves comparing the average outcomes of treated and untreated groups, taking into account the potential outcomes framework.
2.  Randomization and Its Analogs:
    
    *   **Role of Randomization**: In experimental studies, random assignment of treatments is crucial for ensuring that the treatment groups are comparable, allowing for unbiased estimation of causal effects.
    *   **Randomization-like Conditions in Observational Studies**: Rubin extended the framework to observational studies by arguing that causal inferences can be made if these studies fulfill conditions similar to randomization. This involves controlling for confounding variables that influence both the treatment and the outcome, often through methods like matching, regression adjustment, or instrumental variables.
3.  Counterfactual Reasoning:
    
    *   **Counterfactual Conditionals**: Causal relationships must satisfy counterfactual conditions. This means that for a cause to be deemed responsible for an effect, it should be demonstrable that if the cause had not occurred, the effect would not have occurred. This is formalized through the potential outcomes framework.

### Key Features of Rubin’s Approach:

*   **Application to Both Experimental and Observational Studies**: Rubin’s framework is versatile and can be applied to both types of studies, providing a unified approach to causal inference.
*   **Focus on Estimating Causal Effects**: The primary goal is to estimate the causal effect of treatments or interventions, rather than simply identifying associations.
*   **Use of Statistical Methods**: The approach leverages statistical methods to control for confounding variables and to estimate causal effects, emphasizing the importance of rigorous statistical analysis.

### Two Key Criteria of Modern Causal Inference

The modern dominant approach to causal inference primarily builds on two key criteria:

1.  Causation at the Singular Level:
    
    *   This criterion allows for the possibility that causation can be specific to individual subjects or units, acknowledging effect heterogeneity. It means that a cause may produce an effect in one individual but not necessarily in another, depending on various conditions.
2.  Satisfaction of Counterfactual Conditionals:
    
    *   A causal relationship must sustain a counterfactual conditional. This means that for a cause to be deemed responsible for an effect, it should be demonstrable that if the cause had not occurred, the effect would not have occurred. This criterion is essential for defining and reasoning about causal relationships in both experimental and observational studies.

### Impact on Empirical Research:

Rubin’s contributions have led to more careful and precise inferences about causal effects in various disciplines, particularly in the social sciences. Researchers now more rigorously design studies and analyze data to ensure that their conclusions about causality are well-founded within this robust statistical framework.

### Challenges in Randomized Studies:

1.  Non-compliance with Treatment Assignments:
    
    *   **Example**: In a study by the University of Michigan in the 1990s, unemployed persons were assigned to receive or not receive assistance in job searching. A significant percentage of those assigned to the treatment group did not actually take the treatment, complicating the comparison between groups and potentially overestimating the treatment’s effectiveness.
2.  Intermediate Variables:
    
    *   **Example**: An educational researcher wants to know the effect of encouragement to study on test scores. While the researcher can estimate the effect of encouragement, estimating the direct effect of study time is more complicated because it involves intermediate variables (encouragement affecting study time, which in turn affects test scores).
3.  Breakdown of Random Assignment:
    
    *   **Example**: If a subject’s treatment adherence is influenced by their perception of treatment benefits, comparing only those who comply can lead to biased estimates.

### Challenges in Observational Studies:

1.  Identifying and Measuring All Covariates:
    
    *   **Example**: When studying the effect of education on earnings, researchers must account for various covariates that affect both education levels and earnings. Failure to identify or measure all relevant covariates can lead to biased estimates.
2.  Estimating Average Treatment Effects:
    
    *   **Example**: In observational studies, various methods like matching, weighting, and regression are used to estimate treatment effects. Each method has its own set of practical issues and assumptions that need to be carefully managed.
3.  Longitudinal Observational Studies:
    
    *   **Example**: When treatments administered in different periods depend on previous treatments and outcomes, analysis becomes more complicated.
4.  Interference:
    
    *   **Example**: In a housing experiment conducted by the U.S. government, participants assigned to move from housing projects to suburbs knew each other. If the treatment assignment of one participant influenced the decision or outcome of another, traditional analysis methods might not be adequate.

These examples illustrate that both randomized and observational studies require careful consideration of various factors to ensure accurate and reliable causal inferences.

Potential Outcomes, Unit, and Average Effect
------------------------------------------------------------------------------------------------------------------------------------------

### Potential Outcomes Framework

1.  Potential Outcomes:
    *   Each unit (e.g., individual) has two potential outcomes: one if treated and one if not treated. However, only one outcome can be observed for each unit.
    *   This leads to the “fundamental problem of causal inference,” where we cannot observe both potential outcomes for a single unit.
2.  Notation and Unit Effects:
    *   For a unit $i$, denote the outcome as $Y\_i (1)$ if treated and $Y\_i (0)$ if not treated.
    *   The unit effect is defined as the difference $Y\_i (1)−Y\_i (0)$.
    *   The observed outcome $Y_i$ is determined by the treatment assignment $Z\_i$, where $Z\_i=1$ if the unit is treated and $Z\_i=0$ if not.
3.  Randomized vs. Observational Studies:
    *   In randomized experiments, treatment assignment $Z\_i$ is random.
    *   In observational studies, subjects choose their treatment, introducing potential biases.

### Average Treatment Effects

1.  Sample Average Treatment Effect (SATE):
    *   The average of the unit effects for the sample.
2.  Finite Population Average Treatment Effect (FATE):
    *   The average treatment effect for a finite population from which the sample is drawn.
3.  Average Treatment Effect (ATE):
    *   The average treatment effect in an infinite or large population. This is treated as an expectation of the potential outcomes.
4.  Estimands of Interest:
    *   Various estimands depend on the marginal distributions of potential outcomes, such as ATE and Average Treatment Effect on the Treated (ATT).
5.  Challenges and Assumptions:
    *   Estimating these effects requires assumptions like the Stable Unit Treatment Value Assumption (SUTVA), which ensures that the potential outcomes are well-defined and not affected by other units’ treatments.

### Practical Implications

1.  Decision-Making:
    *   Knowledge of average treatment effects aids decision-making in contexts like medical treatments and policy implementations.
2.  Ignorability Conditions:
    *   Under certain conditions, known as ignorability or unconfoundedness, it is possible to use observed data to estimate causal effects reliably.
3.  Extensions and Assumptions:
    *   The framework extends to multiple treatments and continuous treatments, though additional assumptions may be required.
    *   SUTVA assumes no alternative representations of treatment and no interference between units, which may need adjustments in certain studies.

Conditions Allow Average Effects be Unbiasedly/ Consistently Estimated
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Key Concepts

1.  Average Treatment Effect (ATE) Estimation:
    *   **Random Sampling**: Drawing random samples of treated ($y\_1$) and untreated ($y\_0$) units to estimate their respective means.
    *   **Sample Means**: The means of treated ($\\bar{y}\_1$) and untreated ($\bar{y}\_0$) samples serve as unbiased and consistent estimators of the population means.
2.  Unconfoundedness:
    *   **Definition**: Treatment assignment $z$ is independent of potential outcomes ($y\_0$ and $y\_1$).
    *   **Intuition**: In randomized experiments, treatment assignment is blind to potential outcomes, ensuring unconfoundedness. In observational studies, treatment assignment might depend on factors related to potential outcomes, potentially confounding the estimates.

### Examples

1.  Randomized Experiment vs. Observational Study:
    *   **Randomized Experiment**: Treatment assignment is random (e.g., coin flip), ensuring $z$ is independent of $y\_0$ and $y\_1$.
    *   **Observational Study**: Treatment assignment may depend on patient characteristics, potentially leading to biased estimates.
2.  Age and Treatment Example:
    *   **Scenario**: Older patients might forego treatment believing it’s less beneficial, while younger patients might opt for treatment believing it’s more beneficial.
    *   **Consequence**: Naive comparison between treated and untreated groups might overestimate the treatment effect due to confounding by age.

### Ignorability Condition

1.  Condition: $y\_0$ and $y\_1$ are independent of $z$ given covariates $x$ (e.g., age).
    *   **Stratified Analysis**: In both randomized experiments and observational studies, stratifying on covariates like age can help achieve conditional unconfoundedness.
2.  Adjusting for Covariates:
    *   **Randomized Experiment**: Can stratify on covariates either before or after the experiment.
    *   **Observational Study**: Treat it as a stratified randomized experiment by conditioning on covariates related to treatment status and potential outcomes.

### Practical Implications

1.  Comparison of Groups:
    *   In a stratified randomized experiment, compare treated and control groups within each stratum (e.g., age group) to estimate ATE.
    *   In observational studies, stratify on covariates to reduce bias and estimate ATE as if it were a stratified randomized experiment.
2.  Challenges in Observational Studies:
    *   **Unknown Assignment Mechanism**: Unlike randomized experiments, the assignment mechanism in observational studies is not controlled, making it harder to ensure unconfoundedness.
    *   **Measurement of Confounders**: It’s crucial to measure and account for all relevant confounders, though it may not always be possible.
