# chatbot
chatbot code
import os
import sys
import argparse
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

from groq import Groq
import groq as groq_lib  # for exception classes if needed

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")  # default model on Groq
API_KEY = "you sercet code"

if not API_KEY:
    print("Error: set GROQ_API_KEY env var (or put it in a .env file).")
    sys.exit(1)

client = Groq(api_key=API_KEY)


def run_non_streaming(messages, max_tokens=512, temperature=0.2):
    resp = client.chat.completions.create(
        messages=messages,
        model=MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    # typical location of assistant content:
    content = resp.choices[0].message.content
    return content


def run_streaming(messages, max_tokens=512, temperature=0.2):
    # Uses the streaming helper from the Groq client.
    # The streaming protocol yields lines; behaviour may vary slightly depending on library version.
    with client.chat.completions.with_streaming_response.create(
        messages=messages,
        model=MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
    ) as stream_resp:
        # Print chunks as they arrive
        # iter_lines() yields raw lines; adapt if you have a newer client variant
        output = ""
        for line in stream_resp.iter_lines():
            # line may be bytes or already a string/object depending on version
            try:
                chunk = line.decode() if isinstance(line, (bytes, bytearray)) else str(line)
            except Exception:
                chunk = str(line)
            # Many Groq streaming payload lines are small json-like strings; print as-is
            print(chunk, end="", flush=True)
            output += chunk
        print()  # newline after stream
        # try to get final parsed content (if present)
        try:
            final_text = stream_resp.choices[0].message.content
            if final_text:
                return final_text
        except Exception:
            pass
        return output

def chat_loop(stream_mode=False):
    # Simple conversation memory. Trim as needed for token limits.
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    print(f"Model: {MODEL}  (type 'exit' or 'quit' to leave)\n")
    while True:
        try:
            user = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            print("Goodbye 👋")
            break

        messages.append({"role": "user", "content": user})

        try:
            if stream_mode:
                print("Bot: ", end="", flush=True)
                assistant_text = run_streaming(messages)
            else:
                assistant_text = run_non_streaming(messages)
                print("Bot:", assistant_text)

            # Append assistant reply to the history (so the model sees convo)
            messages.append({"role": "assistant", "content": assistant_text})

            # keep conversation from growing forever: optional simple trimming
            if len(messages) > 30:
                # keep system prompt + last 20 messages
                messages = [messages[0]] + messages[-20:]

        except groq_lib.APIConnectionError as e:
            print("\n[Network error]", e)
        except groq_lib.RateLimitError as e:
            print("\n[Rate limit]", e)
        except groq_lib.APIStatusError as e:
            print("\n[API error]", e.status_code, e.response)
        except Exception as e:
            print("\n[Unexpected error]", str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Terminal chatbot using Groq + GPT-OSS")
    parser.add_argument("--stream", action="store_true", help="use streaming output")
    args = parser.parse_args()
    chat_loop(stream_mode=args.stream)
