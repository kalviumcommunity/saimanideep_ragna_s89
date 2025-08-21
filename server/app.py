import os
from flask import Flask, request, jsonify
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Initialize Gemini client with API key
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Common Gemini config
def get_gemini_response(contents):
    tools = [types.Tool(googleSearch=types.GoogleSearch())]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
        tools=tools,
        temperature=0.7,
        top_p=0.9,
        top_k=40
    )
    response_text = ""
    try:
        for chunk in client.models.generate_content_stream(
            model="gemini-2.5-pro",
            contents=contents,
            config=generate_content_config
        ):
            if chunk.text:
                response_text += chunk.text
        if not response_text.strip():
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=contents,
                config=generate_content_config
            )
            print("[DEBUG] Raw Gemini response:", response)
            # Try to extract text from the response object
            if hasattr(response, "text") and response.text:
                response_text = response.text
            elif hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "text") and candidate.text:
                    response_text = candidate.text
                elif hasattr(candidate, "parts") and candidate.parts:
                    response_text = "".join([part.text for part in candidate.parts if hasattr(part, "text")])
                else:
                    response_text = ""
            elif hasattr(response, "parts") and response.parts:
                response_text = "".join([part.text for part in response.parts if hasattr(part, "text")])
            else:
                response_text = ""
    except Exception as e:
        print("Error:", e)
        return "", str(e)
    return response_text, None


# System and User Prompt (RTFC framework only)
@app.route("/cricket-rtfc", methods=["POST"])
def cricket_rtfc():
    data = request.json
    query = data.get("query", "")
    system_prompt = (
        "Role: You are CricketBot 🏏, a cricket expert assistant.\n"
        "Task: Answer all cricket-related questions with accuracy and enthusiasm.\n"
        "Format: Respond in a friendly, clear, and structured way, using bullet points for stats.\n"
        "Context: You have access to live scores, player stats, match schedules, and cricket history."
    )
    user_prompt = f"My question: {query}"
    contents = [
        types.Content(role="model", parts=[types.Part(text=system_prompt)]),
        types.Content(role="user", parts=[types.Part(text=user_prompt)])
    ]
    response_text, error = get_gemini_response(contents)
    if error:
        return jsonify({"response": "", "error": error}), 500
    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(debug=True)