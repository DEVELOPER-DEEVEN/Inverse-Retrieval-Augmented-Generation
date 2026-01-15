import vertexai
from vertexai.generative_models import GenerativeModel

class InquiryAgent:
    def __init__(self, project_id, location):
        self.project_id = project_id
        self.location = location
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel("gemini-1.5-flash-001")

    def generate_question(self, current_profile, params):
        """
        Generates the next question to ask the user to maximize information gain.
        params: dict containing 'creativity' (temperature), 'depth_bias', 'focus_topic'
        """
        
        creativity = params.get('creativity', 0.5)
        # Cap creativity for API safety if needed, or rely on normal range 0.0-1.0
        
        depth_bias = params.get('depth_bias', 'balanced') 
        # e.g., 'broad' (explore new topics) vs 'deep' (drill down on existing ones)
        
        focus_topic = params.get('focus_topic', 'general')
        
        prompt = f"""
        You are an expert interviewer designed to build a psychological profile of a user.
        
        Current Known Profile:
        {current_profile}
        
        Strategy:
        - Depth Bias: {depth_bias} (If 'broad', ask about untouched areas. If 'deep', follow up on specific details in the profile.)
        - Focus Topic: {focus_topic}
        
        Task:
        Generate the SINGLE most effective question to ask next to expand our understanding of this user.
        The question should be open-ended and thought-provoking.
        Do not ask things we already know.
        
        Question:
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
            return response.text.strip()
        except Exception as e:
            print(f"Error generating question: {e}")
            return "Tell me about your day."
