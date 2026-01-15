import optuna
import vertexai
from inquiry_system.user_sim import UserSimulator, EXAMPLE_PERSONA
from inquiry_system.belief import BeliefManager
from inquiry_system.agent import InquiryAgent
from inquiry_system.evaluator import ProfileEvaluator

# Configuration
# TODO: Update this to your Google Cloud Project ID
PROJECT_ID = "abis-345004" # Placeholder: Replace with your actual project ID
LOCATION = "us-central1"

def objective(trial):
    # 1. Hyperparameters to Optimize
    creativity = trial.suggest_float("creativity", 0.1, 0.9)
    depth_bias = trial.suggest_categorical("depth_bias", ["broad", "deep", "balanced"])
    focus_topic = trial.suggest_categorical("focus_topic", ["values", "habits", "childhood", "work", "hobbies"])
    
    agent_params = {
        "creativity": creativity,
        "depth_bias": depth_bias,
        "focus_topic": focus_topic
    }
    
    # 2. Setup System
    # We use a fixed Persona for consistent optimization
    user = UserSimulator(PROJECT_ID, LOCATION, EXAMPLE_PERSONA)
    belief = BeliefManager(PROJECT_ID, LOCATION)
    agent = InquiryAgent(PROJECT_ID, LOCATION)
    
    # 3. Collaborative Loop (Simulation)
    # Run a short session to see how well this strategy builds a profile
    NUM_TURNS = 3
    
    print(f"\n--- Trial {trial.number} Start (Strategy: {depth_bias}, {focus_topic}) ---")
    
    for i in range(NUM_TURNS):
        current_profile = belief.get_profile()
        question = agent.generate_question(current_profile, agent_params)
        answer = user.answer_question(question)
        belief.update_belief(question, answer)
        print(f"Turn {i+1}: Q='{question}' -> A='{answer[:50]}...'")
        
    # 4. Evaluation
    evaluator = ProfileEvaluator(PROJECT_ID, LOCATION)
    score, scenario, pred, actual = evaluator.evaluate_profile(belief.get_profile(), user)
    
    print(f"Trial {trial.number} Result: Score={score}")
    print(f"Scenario: {scenario}")
    print(f"Prediction: {pred[:50]}...")
    print(f"Actual: {actual[:50]}...")
    
    return score

if __name__ == "__main__":
    print("Starting Bayesian Behavioral Inquiry Optimization...")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5) # 5 trials for demo
    
    print("\noptimization Complete!")
    print("Best params:", study.best_params)
    print("Best value:", study.best_value)
