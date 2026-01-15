import vertexai
from vertexai.generative_models import GenerativeModel

class ProfileEvaluator:
    def __init__(self, project_id, location):
        self.project_id = project_id
        self.location = location
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel("gemini-1.5-pro-001")

    def evaluate_profile(self, profile, user_sim):
        """
        Tests the predictive power of the profile against the user simulator.
        Returns a score from 0.0 to 10.0.
        """
        
        # 1. Generate a Test Scenario
        scenario = self._generate_scenario()
        
        # 2. Predict Behavior using the Profile
        prediction = self._predict_behavior(profile, scenario)
        
        # 3. Get Ground Truth from Simulator
        ground_truth = user_sim.answer_question(f"Scenario: {scenario}\nHow would you react?")
        
        # 4. Judge the Accuracy
        score = self._judge_accuracy(scenario, prediction, ground_truth)
        
        return score, scenario, prediction, ground_truth

    def _generate_scenario(self):
        prompt = "Generate a short, specific, everyday scenario that would test a person's values or personality (e.g., finding a wallet, traffic jam, receiving criticism). Output ONLY the scenario text."
        response = self.model.generate_content(prompt)
        return response.text.strip()

    def _predict_behavior(self, profile, scenario):
        prompt = f"""
        Based on the following behavioral profile:
        {profile}
        
        Predict how this person would react in this scenario:
        {scenario}
        
        describe the reaction in the first person.
        """
        response = self.model.generate_content(prompt)
        return response.text.strip()

    def _judge_accuracy(self, scenario, prediction, ground_truth):
        prompt = f"""
        Scenario: {scenario}
        
        Predicted Reaction (based on profile):
        {prediction}
        
        Actual Reaction (Ground Truth):
        {ground_truth}
        
        Task:
        Rate the accuracy of the Prediction compared to the Actual Reaction.
        Consider:
        - Emotional tone match.
        - Decision similarity.
        - Reasoning alignment.
        
        Score from 0.0 (Completely wrong) to 10.0 (Perfect match).
        Output ONLY the numeric score.
        """
        try:
            response = self.model.generate_content(prompt)
            return float(response.text.strip())
        except:
            return 0.0
