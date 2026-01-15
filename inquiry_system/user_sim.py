import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting

class UserSimulator:
    def __init__(self, project_id, location, persona_description):
        self.project_id = project_id
        self.location = location
        self.persona = persona_description
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel("gemini-1.5-pro-001")
        
    def answer_question(self, question):
        """
        Simulates the user answering a question based on their hidden persona.
        """
        prompt = f"""
        You are a role-playing actor. You are strictly embodying the following persona:
        
        {self.persona}
        
        A researcher is asking you questions to understand your behavior.
        Answer the following question naturally, in the first person, strictly consistent with your persona.
        Do not reveal that you are an AI or a simulation.
        
        Question: {question}
        Answer:
        """
        
        generation_config = {
            "max_output_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        
        safety_settings = [
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
            ),
        ]

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error simulating user answer: {e}")
            return "I'm not sure specifically."

# Example Persona for testing
EXAMPLE_PERSONA = """
Name: Alex Chen
Age: 34
Occupation: Software Architect
Personality:
- Highly analytical but prone to overthinking.
- Introverted, prefers asynchronous communication.
- Value efficiency and minimalism.
- Hobbies: Bouldering, Chess, Modular Synthesizers.
- Behavioral Quirk: Gets annoyed when people are vague or use buzzwords.
- Financial Habit: Extremely frugal on day-to-day items but spends lavishly on high-quality tools (e.g., keyboards, chairs).
- Social: Has a small circle of close friends, avoids large parties.
"""
