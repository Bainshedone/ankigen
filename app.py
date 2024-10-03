from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional
import gradio as gr
import json
import re



class Step(BaseModel):
    explanation: str
    output: str


class Subtopics(BaseModel):
    steps: List[Step]
    result: List[str]


class Topics(BaseModel):
    result: List[Subtopics]


class CardFront(BaseModel):
    question: Optional[str] = None


class CardBack(BaseModel):
    answer: Optional[str] = None
    explanation: str
    example: str


class Card(BaseModel):
    front: CardFront
    back: CardBack


class CardList(BaseModel):
    topic: str
    cards: List[Card]

def universal_response_handler(response_content):
    # Extract JSON content from the response
    json_match = re.search(r'\{.*\}|\[.*\]', response_content, re.DOTALL)
    if json_match:
        json_content = json_match.group()
    else:
        print("No JSON content found in the response")
        return None

    try:
        # Parse the JSON content
        parsed_response = json.loads(json_content)
        
        # If the parsed response is a list, return it directly
        if isinstance(parsed_response, list):
            return parsed_response
        # If it's a dict, check if it has a 'result' key
        elif isinstance(parsed_response, dict):
            if 'result' in parsed_response:
                return parsed_response['result']
            elif 'topic' in parsed_response and 'cards' in parsed_response:
                # Handle the case where we get a single topic with cards
                return [parsed_response]
            else:
                return [parsed_response]
        else:
            print("Unexpected response format:", parsed_response)
            return None

    except json.JSONDecodeError:
        print("Invalid JSON response:", json_content)
        return None
    except Exception as ex:
        print(f"An error occurred while parsing the response: {ex}")
        return None
    
def structured_output_completion(
    client, model, system_prompt, user_prompt
):
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
        )
        
        print("Raw API response:", completion)
        
        response_content = completion.choices[0].message.content
        return universal_response_handler(response_content)

    except Exception as e:
        print(f"An error occurred during the API call: {e}")
        return None

def generate_cards(
    api_key_input,
    model_name,
    subject,
    topic_number=1,
    cards_per_topic=2,
    preference_prompt="assume I'm a beginner",
):
    gr.Info("Starting process")

    if not api_key_input:
        return gr.Error("Error: OpenRouter API key is required.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key_input,
    )
    model = model_name


    all_card_lists = []

    system_prompt = f"""
    You are an expert in {subject}, assisting the user to master the topic while 
    keeping in mind the user's preferences: {preference_prompt}.
    Please provide your responses in valid JSON format.
    """

    topic_prompt = f"""
    Generate the top {topic_number} important subjects to know on {subject} in 
    order of ascending difficulty. Return ONLY a JSON array of objects, 
    each containing 'subject' and 'difficulty' keys. Do not include any additional text or explanations. For example:
    [
    {{"subject": "Basic Concept 1", "difficulty": 1}},
    {{"subject": "Advanced Concept 2", "difficulty": 2}}
    ]
    """
    try:
        topics_response = structured_output_completion(
            client, model_name, system_prompt, topic_prompt
        )
        if topics_response is None:
            raise gr.Error("Failed to generate topics. Please try again.")
        
        if isinstance(topics_response, list):
            topic_list = []
            for item in topics_response[:topic_number]:
                if isinstance(item, dict):
                    topic = item.get('subject') or item.get('topic')
                    if topic:
                        topic_list.append(topic)
                elif isinstance(item, str):
                    topic_list.append(item)
            
            if not topic_list:
                raise gr.Error("Unexpected response format. Please try again.")
        else:
            raise gr.Error("Unexpected response format. Please try again.")

    except Exception as e:
        raise gr.Error(f"An error occurred: {str(e)}. Please try again or check your API key.")

    for topic in topic_list:
        card_prompt = f"""
        Generate {cards_per_topic} cards on {subject}: "{topic}"
        keeping in mind the user's preferences: {preference_prompt}.
        Questions should cover both sample problems and concepts.
        Use the explanation field to help the user understand the reason behind things
        and maximize learning. Additionally, offer tips (performance, gotchas, etc.).
        Return the result as a JSON object with the following structure:
        {{
            "topic": "string",
            "cards": [
                {{
                    "front": {{ "question": "string" }},
                    "back": {{
                        "answer": "string",
                        "explanation": "string",
                        "example": "string"
                    }}
                }}
            ]
        }}
        """

        try:
            cards = structured_output_completion(
                client, model_name, system_prompt, card_prompt
            )
            if cards is None:
                print(f"Failed to generate cards for topic '{topic}'.")
                continue
            if isinstance(cards, dict) and 'topic' in cards and 'cards' in cards:
                all_card_lists.append(cards)
            elif isinstance(cards, list) and len(cards) > 0 and isinstance(cards[0], dict) and 'topic' in cards[0] and 'cards' in cards[0]:
                all_card_lists.extend(cards)
            else:
                print(f"Invalid card response format for topic '{topic}'.")
        except Exception as e:
            print(f"An error occurred while generating cards for topic '{topic}': {e}")
            continue

    flattened_data = []

    for card_list_index, card_list in enumerate(all_card_lists, start=1):
        try:
            topic = card_list['topic']
            # Get the total number of cards in this list to determine padding
            total_cards = len(card_list['cards'])
            # Calculate the number of digits needed for padding
            padding = len(str(total_cards))

            for card_index, card in enumerate(card_list['cards'], start=1):
                # Format the index with zero-padding
                index = f"{card_list_index}.{card_index:0{padding}}"
                question = card['front']['question']
                answer = card['back']['answer']
                explanation = card['back']['explanation']
                example = card['back']['example']
                row = [index, topic, question, answer, explanation, example]
                flattened_data.append(row)
        except Exception as e:
            print(f"An error occurred while processing card {index}: {e}")
            continue

    return flattened_data


def export_csv(d):
    MIN_ROWS = 2

    if len(d) < MIN_ROWS:
        gr.Warning(f"The dataframe has fewer than {MIN_ROWS} rows. Nothing to export.")
        return None

    gr.Info("Exporting...")
    d.to_csv("anki_deck.csv", index=False)
    return gr.File(value="anki_deck.csv", visible=True)


with gr.Blocks(
    gr.themes.Soft(), title="AnkiGen", css="footer{display:none !important}"
) as ankigen:
    gr.Markdown("# 📚 AnkiGen - Anki Card Generator")
    gr.Markdown("#### Generate an LLM generated Anki comptible csv based on your subject and preferences.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Configuration")

            api_key_input = gr.Textbox(
                label="OpenRouter API Key",
                type="password",
                placeholder="Enter your OpenRouter API key",
            )
            model_input = gr.Textbox(
                label="Model Name",
                placeholder="Enter the model name (e.g., nousresearch/hermes-3-llama-3.1-405b:free)",
                value="nousresearch/hermes-3-llama-3.1-405b:free"
            )
            subject = gr.Textbox(
                label="Subject",
                placeholder="Enter the subject, e.g., 'Basic SQL Concepts'",
            )
            topic_number = gr.Slider(
                label="Number of Topics", minimum=2, maximum=20, step=1, value=2
            )
            cards_per_topic = gr.Slider(
                label="Cards per Topic", minimum=2, maximum=30, step=1, value=3
            )
            preference_prompt = gr.Textbox(
                label="Preference Prompt",
                placeholder="Any preferences? For example: Learning level, e.g., \"Assume I'm a beginner\" or \"Target an advanced audience\" Content scope, e.g., \"Only cover up until subqueries in SQL\" or \"Focus on organic chemistry basics\"",
            )
            generate_button = gr.Button("Generate Cards")
        with gr.Column(scale=2):
            gr.Markdown("### Generated Cards")
            gr.Markdown(
                """
                Subject to change: currently exports a .csv with the following fields, you can
                create a new note type with these fields to handle importing.: 
                <b>Index, Topic, Question, Answer, Explanation, Example</b>
                """
            )
            output = gr.Dataframe(
                headers=[
                    "Index",
                    "Topic",
                    "Question",
                    "Answer",
                    "Explanation",
                    "Example",
                ],
                interactive=False,
                height=800,
            )
            export_button = gr.Button("Export to CSV")
            download_link = gr.File(interactive=False, visible=False)

    generate_button.click(
        fn=generate_cards,
        inputs=[
            api_key_input,
            model_input,
            subject,
            topic_number,
            cards_per_topic,
            preference_prompt,
        ],
        outputs=output,
    )

    export_button.click(fn=export_csv, inputs=output, outputs=download_link)

if __name__ == "__main__":
    ankigen.launch(share=False, favicon_path="./favicon.ico")
