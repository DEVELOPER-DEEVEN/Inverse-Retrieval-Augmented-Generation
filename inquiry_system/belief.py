import vertexai
import json
from vertexai.generative_models import GenerativeModel
from inquiry_system.state import BeliefState, Variable

class BeliefManager:
    def __init__(self, project_id, location):
        self.project_id = project_id
        self.location = location
        self.state = BeliefState() 
        # Start with an empty, uncertain state
        self.state.profile_summary = "An unknown individual. No data available."
        self.state.top_uncertainties = ["Values", "Decision Making", "Risk Tolerance", "Social Preferences"]
        
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel("gemini-1.5-flash-001")

    def update_belief(self, question, answer):
        """
        Updates the BeliefState by analyzing the new interaction.
        """
        # 1. Update History
        self.state.conversation_history.append({"role": "agent", "content": question})
        self.state.conversation_history.append({"role": "user", "content": answer})
        
        # 2. Run Meta-Analysis
        self._analyze_state()
        
        return self.state

    def _analyze_state(self):
        """
        Uses LLM to infer latent traits and uncertainty from history.
        """
        history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in self.state.conversation_history])
        
        prompt = f"""
        You are a highly analytical psychologist. 
        Your goal is to build a probabilistic model of a user based on their interview transcript.
        
        TRANSCRIPT:
        {history_text}
        
        TASK:
        Analyze the user to update our internal belief state.
        
        1. **Latent Traits**: Identify 3-5 key personality traits (e.g., Risk Tolerance, Decision Style, Values). 
           - Assign a specific Value (e.g., "High", "Intuitive").
           - Assign a Confidence Score (0.0 to 1.0).
           - Provide brief Reasoning.
           
        2. **Profile Summary**: A 1-2 sentence narrative summary of who this person is.
        
        3. **Uncertainties**: What are the top 3 most critical things we DO NOT know yet? (e.g., "Political leanings", "Reaction to stress").
        
        4. **Overall Confidence**: How well do we know this person overall? (0.0 to 1.0).
        
        OUTPUT FORMAT (JSON ONLY):
        {{
            "traits": [
                {{"name": "...", "value": "...", "confidence": 0.8, "reasoning": "..."}}
            ],
            "profile_summary": "...",
            "uncertainties": ["..."],
            "overall_confidence": 0.5
        }}
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            data = json.loads(response.text)
            
            # Map back to internal State object
            self.state.latent_traits = {}
            for item in data.get("traits", []):
                self.state.latent_traits[item["name"]] = Variable(
                    name=item["name"],
                    value=item["value"],
                    confidence=item["confidence"],
                    reasoning=item["reasoning"]
                )
            
            self.state.profile_summary = data.get("profile_summary", "")
            self.state.top_uncertainties = data.get("uncertainties", [])
            self.state.overall_confidence = data.get("overall_confidence", 0.0)
            
        except Exception as e:
            print(f"Error analyzing belief state: {e}")

    def get_state(self):
        return self.state
