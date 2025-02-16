
import asyncio
import openai
import backoff

def opro_prompt_output(client, prompt_string: str, model: str, temperature: float) -> list[str]:
    # Create the message for OpenAI
    messages = [{"role": "system", "content": prompt_string}]

    # Pass in the message into OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    # Reformat the string into a list of strings
    opro_sentences = response.choices[0].message.content
    opro_splits = opro_sentences.split("\n")

    # Clean each sentence by removing punctuation and blank spaces
    opro_cleaned = [s.strip("[]").rstrip() for s in opro_splits]
    return opro_cleaned


@backoff.on_exception(backoff.expo, openai.RateLimitError)
async def opro_prompt_evaluation(
    client, system_prompt_string: str, user_prompt_string: str, temperature: float, model: str,
) -> str:
    # Create the messages to be sent to OpenAI
    messages = [
        {"role": "system", "content": system_prompt_string},
        {"role": "user", "content": user_prompt_string},
    ]

    # Pass the message into OpenAI API
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    return response.choices[0].message.content
