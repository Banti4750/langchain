
# CHANGE 1: Add re + inspect — we'll parse tool calls from raw text instead of structured JSON.
import re
import inspect
from dotenv import load_dotenv

load_dotenv()

from groq import Groq
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "openai/gpt-oss-120b"

client = Groq()


# --- Tools (LangChain @tool decorator) ---


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50
    }
    return prices.get(product.lower(), 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""

    print(
        f"    >> Executing apply_discount(price={price}, "
        f"discount_tier='{discount_tier}')"
    )

    price = float(price)

    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23
    }

    discount = discount_percentages.get(discount_tier.lower(), 0)

    return round(price * (1 - discount / 100), 2)


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}


# CHANGE 3: Delete the JSON schemas. Tools now live inside the prompt as plain text.
# We derive descriptions from the functions themselves using inspect.

def get_tool_descriptions(tools_dict):
    descriptions = []

    for tool_name, tool_function in tools_dict.items():

        # __wrapped__ bypasses decorator wrappers (e.g., @traceable)
        original_function = getattr(
            tool_function,
            "__wrapped__",
            tool_function
        )

        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function) or ""

        descriptions.append(
            f"{tool_name}{signature} - {docstring}"
        )

    return "\n".join(descriptions)


tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())


react_prompt = f"""
STRICT RULES — you must follow these exactly:

1. NEVER guess or assume any product price.
   You MUST call get_product_price first.

2. Only call apply_discount AFTER receiving a price
   from get_product_price.

3. Pass the exact price returned by get_product_price.
   Never invent a number.

4. NEVER calculate discounts yourself.
   Always use the apply_discount tool.

5. If the user does not specify a discount tier,
   ask them which tier to use.

Answer the following questions using these tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer

Thought: Briefly explain your next action.

Action: One of [{tool_names}]

Action Input: Arguments as comma separated values.

Observation: Result returned by the tool.

... (Thought/Action/Action Input/Observation can repeat)

Thought: I now know the final answer.

Final Answer: The final answer to the original question.

Begin!

Question: {{question}}
Thought:"""


# CHANGE 4: Drop tools= from Groq chat completions.
# The LLM has no native tool interface in this implementation.
# All agency comes from the prompt and our regex parsing below.

@traceable(name="Groq Chat", run_type="llm")
def groq_chat_traced(model, messages, options):

    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=options.get("temperature", 0),
        stop=options.get("stop")
    )


# --- Agent Loop ---


@traceable(name="Groq ReAct Agent Loop")
def run_agent(question: str):

    print(f"Question: {question}")
    print("=" * 60)

    # CHANGE 5: One prompt string replaces the system/user message split.
    prompt = react_prompt.format(question=question)

    scratchpad = ""

    for iteration in range(1, MAX_ITERATIONS + 1):

        print(f"\n--- Iteration {iteration} ---")

        full_prompt = prompt + scratchpad

        # Stop token prevents the LLM from generating its own Observation.
        # We inject the real tool result instead.

        response = groq_chat_traced(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            options={
                "stop": ["\nObservation:"],
                "temperature": 0
            }
        )

        output = response.choices[0].message.content or ""

        print(f"LLM Output:\n{output}")

        # Check if LLM generated a final answer.

        print("  [Parsing] Looking for Final Answer...")

        final_answer_match = re.search(
            r"Final Answer:\s*(.+)",
            output,
            re.DOTALL
        )

        if final_answer_match:

            final_answer = final_answer_match.group(1).strip()

            print(f"  [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")

            return final_answer

        # CHANGE 6: Parse tool calls from raw text with regex.

        print("  [Parsing] Looking for Action and Action Input...")

        action_match = re.search(
            r"^Action:\s*(.+)$",
            output,
            re.MULTILINE
        )

        action_input_match = re.search(
            r"^Action Input:\s*(.+)$",
            output,
            re.MULTILINE
        )

        if not action_match or not action_input_match:

            print("  [Parsing] ERROR: Could not parse Action/Action Input")
            break

        tool_name = action_match.group(1).strip()

        tool_input_raw = action_input_match.group(1).strip()

        print(f"  [Tool Selected] {tool_name}")
        print(f"  [Tool Input] {tool_input_raw}")

        # Split comma-separated arguments.

        raw_args = [
            x.strip()
            for x in tool_input_raw.split(",")
        ]

        # Strip key= prefix if model outputs key=value format.

        args = [
            x.split("=", 1)[-1].strip().strip("'\"")
            for x in raw_args
        ]

        print(f"  [Parsed Arguments] {args}")

        # Find and execute actual Python function.

        print(f"  [Tool Executing] {tool_name}({args})")

        if tool_name not in tools:

            observation = (
                f"Error: Tool '{tool_name}' not found. "
                f"Available tools: {list(tools.keys())}"
            )

        else:

            try:
                observation = str(tools[tool_name](*args))

            except Exception as error:
                observation = f"Tool execution error: {error}"

        print(f"  [Tool Result] {observation}")

        # CHANGE 7: History is one growing string re-sent every iteration.

        scratchpad += (
            f"{output}\n"
            f"Observation: {observation}\n"
            f"Thought:"
        )

    print("ERROR: Max iterations reached or parsing failed.")
    return None


if __name__ == "__main__":

    print("Hello Groq ReAct Agent!")
    print()

    result = run_agent(
        "What is the price of a laptop after applying a gold discount?"
    )
