# Design Document: Bayesian Active Inquiry for Behavior Reproduction

## 1. Problem Statement
Current personality assessment methods relies on rigid, hour-long surveys that reduce complex human behaviors to low-resolution numerical stats (e.g., "Big 5 Tratis: Openness = 5"). These metrics are insufficient for training computational models (LLMs) to accurately reproduce specific human behaviors.

## 2. Objective
Develop a system that "optimizes the process of collecting rich information about individuals."
Instead of static surveys, we use **Bayesian Optimization** to drive an **active inquiry process**. An LLM-based agent iteratively generates and selects the "next best question" to maximize information gain about the user, building a rich, nuance textual profile.

## 3. Methodology & Architecture

### Phase 1: Bayesian Active Inquiry (Data Collection)
*   **The Inquirer Agent (LLM)**: Proposes potential interview questions.
*   **The Optimizer (Bayesian)**:
    *   Maintains a "Belief State" about the user.
    *   Selects the question most likely to reduce uncertainty or test a specific hypothesis about the user's behavior.
    *   *Note: In this context, BO optimizes the "question selection" rather than hyperparameters.*
*   **The User**: Answers natural language questions.
*   **Belief Update**: The system updates the textual profile based on the answer.

### Phase 2: Reverse RAG (Data Processing)
*   **Input**: The rich textual profile collected in Phase 1 (The "Raw Data").
*   **Process** (Existing "Reverse RAG" pipeline):
    *   **Generator (Gemini 1.5 Flash)**: Takes the profile and generates thousands of synthetic "Scenario -> Reaction" (Q&A) pairs.
    *   **Dataset**: A JSONL file of specific behavioral examples rooted in the key profile.

### Phase 3: Fine-Tuning (Model Creation)
*   **Target Model (Gemini 1.0 Pro)**: Fine-tuned on the Reverse RAG dataset.
*   **Output**: A "Digital Twin" or "Cognitive Agent" that accurately reproduces the user's specific behavioral patterns, not just generic traits.

## 4. Implementation Strategy (Project Evolution)
We will evolve the current repository from a "Restaurant Review" demo to this "Behavior Reproduction" framework.

### New Components to Build:
1.  `inquiry_agent.py`: The loop for asking questions and updating the profile.
2.  `optimizer_bridge.py`: Connecting the inquiry step to a Bayesian objective (e.g., maximizing profile diversity or consistence).
3.  **Integration**: Feeding the final profile into the existing `Reverse RAG` notebook pipeline.

## 5. Reverse RAG Role
The **Reverse RAG** implementation remains the core engine for *transforming* the collected information into actionable training data. It bridges the gap between the "Rich Behavioral Description" and the "Computational Model."
