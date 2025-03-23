#!/usr/bin/env python
import json
import base64
import os

def select_object_with_gpt(openai_client, text_prompt, detected_objects, is_auto_mode=False):
    """
    Use GPT-4o to select the object that best matches the text prompt.
    
    Args:
        openai_client: OpenAI client instance
        text_prompt: Text description of the object to find
        detected_objects: List of detected objects with bounding boxes, labels, and scores
        is_auto_mode: Whether we're in automatic mode where GPT identified the main dish
        
    Returns:
        dict: Selected object matching the prompt, or None if no match found
    """
    try:
        # If we only have one object, return it without consulting GPT
        if len(detected_objects) == 1:
            print("Only one object detected, selecting it automatically")
            return detected_objects[0]
        
        # Prepare context for GPT-4o to select the object
        objects_json = []
        for i, obj in enumerate(detected_objects):
            objects_json.append({
                "id": i,
                "label": obj["label"],
                "confidence_score": round(obj["score"], 2),
                "bounding_box": [round(x, 1) for x in obj["box"]],
                "is_from_heuristic": obj.get("from_heuristic", False)
            })
        
        # Create a system prompt to guide GPT-4o
        system_prompt = """You are a vision analysis system that helps select the correct object from a list of detected objects based on a user's description.
Your task is to analyze the list of detected objects and select the ONE that best matches the user's description.
For each object, you have its ID, label, confidence score, and bounding box dimensions.
Objects marked as "is_from_heuristic" were detected using traditional computer vision methods as a fallback.

Return your answer in this exact JSON format only, with no additional text:
{
  "selected_object_id": <id of the selected object>,
  "reasoning": "<brief explanation of why you selected this object>"
}

If none of the objects match the description, return:
{
  "selected_object_id": null,
  "reasoning": "<explanation of why no objects match>"
}"""

        # Define a prompt for selecting the object
        selection_prompt = f"""Here is the description of what I'm looking for:
"{text_prompt}"

Here are the detected objects:
{json.dumps(objects_json, indent=2)}

Please select the most appropriate object that matches the description."""

        # Add hint for auto mode
        if is_auto_mode:
            selection_prompt += """
Since we're in automatic mode, please focus on selecting the main dish or food item in the image.
"""

        print("Consulting GPT-4o to find the best match...")
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": selection_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        
        # Extract the response
        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        
        selected_id = result.get("selected_object_id")
        reasoning = result.get("reasoning", "No reasoning provided")
        
        # If no object was selected, return None
        if selected_id is None:
            print(f"GPT-4o couldn't find a matching object: {reasoning}")
            return None
        
        # Return the selected object
        print(f"GPT-4o selected object #{selected_id}: {reasoning}")
        return detected_objects[selected_id]
        
    except Exception as e:
        print(f"Error selecting object with GPT: {e}")
        # Fallback: return the highest confidence object
        print("Fallback: selecting the highest confidence object")
        if detected_objects:
            return detected_objects[0]
        return None

def identify_main_dish(openai_client, image_path):
    """
    Use GPT-4o to identify the main dish in the image.
    
    Args:
        openai_client: OpenAI client instance
        image_path: Path to the image file
        
    Returns:
        str: Description of the main dish, or None if identification failed
    """
    try:
        # Convert image to base64 for API
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        prompt = """This image shows a food dish. Identify what the main dish is and describe it in a brief phrase (2-5 words only).
Don't include any servingware in your description - just identify the main food.
Examples of good responses:
- "grilled salmon"
- "chicken curry"
- "chocolate cake"
- "vegetable pasta"
Be specific but concise."""
        
        # Send the request to GPT-4o
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        # Extract the response
        dish_description = response.choices[0].message.content.strip()
        print(f"GPT-4o identified: {dish_description}")
        
        return dish_description
        
    except Exception as e:
        print(f"Error identifying main dish: {e}")
        return None 