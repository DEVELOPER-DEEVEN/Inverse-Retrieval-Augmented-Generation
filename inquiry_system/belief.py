import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting

class BeliefManager:
    def __init__(self, project_id, location):
        self.project_id = project_id
        self.location = location
        self.profile_text = "The user is an unknown individual. No specific traits are known yet."
        
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel("gemini-1.5-flash-001")

    def update_belief(self, question, answer):
        """
        Updates the profile text by incorporating the new Q&A information.
        """
        prompt = f"""
        You are a psychologist building a detailed behavioral profile of a subject.
        
        Current Profile:
        {self.profile_text}
        
        New Information from Interview:
        Q: {question}
        A: {answer}
        
        Task:
        Update the Current Profile to incorporate this new information.
        - Merge compatible information.
        - Refine general statements with the specific details provided.
        - Maintain a cohesive narrative description of the person's personality, values, habits, and quirks.
        - Do NOT simply append the Q&A. Synthesis it into the description.
        
        Updated Profile:
        """
        
        generation_config = {
            "max_output_tokens": 1024,
            "temperature": 0.4, # Lower temperature for stable summarization
            "top_p": 0.8,
        }

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            self.profile_text = response.text.strip()
            return self.profile_text
        except Exception as e:
            print(f"Error updating belief: {e}")
            return self.profile_text

    def get_profile(self):
        return self.profile_text
