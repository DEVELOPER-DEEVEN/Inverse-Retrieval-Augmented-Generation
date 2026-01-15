import vertexai
from vertexai.generative_models import GenerativeModel
from inquiry_system.state import BeliefState

class InquiryAgent:
    def __init__(self, project_id, location):
        self.project_id = project_id
        self.location = location
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel("gemini-1.5-flash-001")

    def generate_question(self, state: BeliefState, params: dict):
        """
        Generates a question to maximize information gain regarding the current uncertainties.
        """
        
        creativity = params.get('creativity', 0.6)
        
        # Format the context from the structured state
        context_str = state.to_prompt_context()
        
        prompt = f"""
        You are an intelligent, curious, and respectful interviewer.
        
        INTERNAL STATE (Do not reveal this to the user):
        {context_str}
        
        OBJECTIVE:
        We need to reduce uncertainty about the user.
        Detailed Goal: Identify which latent traits are most uncertain. Generate ONE question that maximizes expected information gain about these specific areas.
        
        CONSTRAINTS:
        - Ask only one question.
        - Tone: Natural, conversational, not clinical.
        - Style: Prefer scenario-based ("What would you do if...") or trade-off questions over direct "Rate yourself" survey questions.
        - Do NOT mention probabilities, "latent traits", or "internal model".
        - Do NOT repeat questions asked in the history.
        
        OUTPUT FORMAT:
        [QUESTION]
        <Your single adaptive question here>
        """
        
        generation_config = {
            "max_output_tokens": 128,
            "temperature": creativity,
            "top_p": 0.95,
        }

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            # Parse output to be safe, though prompt instruction is strong
            text = response.text.strip()
            if "[QUESTION]" in text:
                text = text.split("[QUESTION]")[1].strip()
            return text
        except Exception as e:
            print(f"Error generating question: {e}")
            return "Could you tell me a bit more about what drives your decisions?"
