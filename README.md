# ai_feedback_assistant

'AI in the Loop' essay feedback assistant that prioritizes expert human judgement. Flask app that uses the OpenAI and Anthropic APIs to evaluate the degree of alignment between a marker's feedback and the marking rubric/descriptors for an assignment. 

# TODO 
[ ] There is currently a problem where the Evaluations table is being used instead of the CriterionFeedback table. For example, the export functionality uses Evaluations. However, the Evaluations table is no longer used. 