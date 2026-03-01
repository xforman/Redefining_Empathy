# JaEm: Evaluating Computational Empathy in Large Language Models

This repository implements **JaEm**, a framework for evaluating **Computational Empathy** in Large Language Models (LLMs). The project introduces a theory-grounded approach to empathy drawn from psychology and philosophy, and operationalizes it for systematic, bias-aware LLM evaluation.

---

## Overview

Empathy in NLP has often been reduced to simplified emotional labels or surface-level sentiment. Such approaches overlook two critical aspects:

* The **complexity** of emotional and situational understanding
* The **subjective and context-dependent** nature of empathy

This repository addresses these limitations by defining and measuring **Computational Empathy**, a structured formulation that separates empathy as a construct from the metrics used to measure it.

JaEm provides tools to evaluate how LLMs interpret people’s situations, especially under changing **social contexts**, revealing patterns of bias, assumptions, and context sensitivity.

---

## Computational Empathy

The framework models empathy along **two primary dimensions**:

### 1. Cognitive Empathy

The ability to understand another person’s situation, perspective, and inferred mental states.

### 2. Affective Empathy

Sensitivity to and alignment with another person’s emotional experience.

To avoid conflating empathy with its measurement, the framework defines **distinct metric classes** for:

* Cognitive empathy
* Affective empathy
* Empathic responding

This distinction ensures that theoretical constructs remain separate from operational evaluation signals.

---

## JaEm Evaluation Framework

JaEm evaluates how LLMs reason about **how and why** a person is in a given situation.

A core evaluation pattern involves prompts such as:

> *“How did the person get into that situation?”*

The **social context** of the scenario (e.g., background, identity cues, circumstances) is systematically varied. The framework then analyzes:

* Changes in causal explanations
* Sensitivity to contextual cues
* Bias in reasoning across social contexts

This enables assessment of:

* Context-aware interpretation
* Cognitive and affective empathy
* Bias in model reasoning

---

## Repository Structure

```
/JaEmS/
/framework_data/
/Llama_33_70B_analysis.ipynb
/deepseek_analysis.ipynb
/example_model_eval.ipynb
/JaEmS/framework/
/JaEmS/llm_eval_utils/
/JaEmS/load_utils/
```

### Core Package

**`/JaEmS/`** — Main implementation of the JaEm framework.

* **`framework/`**
  Core components for scenario construction, empathy dimensions, and metric computation.

  * **`llm_eval_utils/model_eval.py`**
    Contains loader functions for the LLMs from Hugging Face, evaluated locally, possibly using vLLM for efficiency. However, as of 5/2025 there were dependency issues between numpy and vLLM, caution is advised.  

* **`llm_eval_utils/`**
  Utilities for running evaluations with LLMs, including prompt handling and response processing.

* **`load_utils/`**
  Data loading and preprocessing tools for evaluation scenarios and context variations.

---

### Data

**`/framework_data/`**
Contains datasets and structured inputs used for evaluation scenarios and context manipulation.

---

### Notebooks

* **`example_model_eval.ipynb`**
  Demonstrates the end-to-end JaEm evaluation pipeline, including prompt generation, model interaction, and metric computation.

* **Model analysis notebooks**

  * `Llama_33_70B_analysis.ipynb`
  * `deepseek_analysis.ipynb`

  These notebooks analyze evaluation results for:

  * `meta-llama/Llama-3.3-70B-Instruct`
  * `meta-llama/Llama-3.1-8B-Instruct`
  * `neuralmagic/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8`

They explore how model responses vary under different social contexts and what this reveals about empathy-related reasoning and bias.

---

## Key Findings

Applying JaEm to multiple LLMs shows that model explanations of how a person reached a situation are **sensitive to changes in social context**. These shifts highlight:

* Potential **bias in causal reasoning**
* **Ethical risks** related to stereotype-driven interpretation
* **Security concerns** tied to contextual manipulation

At the same time, the results demonstrate the value of **Computational Empathy** and JaEm for deeper, theory-informed evaluation of LLMs.

---

## Summary

JaEm provides a **theory-driven, context-aware evaluation framework** for analyzing how LLMs understand human situations. It moves beyond emotion labeling toward assessing:

* Subjective interpretation
* Context sensitivity
* Bias in empathic reasoning
