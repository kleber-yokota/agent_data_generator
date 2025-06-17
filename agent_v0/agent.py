import json
import re

from langgraph.graph import StateGraph
from langchain_community.chat_models import ChatOllama
from typing import TypedDict, Optional
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from io import StringIO

class ExtractionAgent(TypedDict):
    user_input: str
    refactore: bool = False
    entities: Optional[dict]
    metadata: Optional[dict]
    specification_generation: Optional[str]

entity_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are an entity extraction system. Your task is to extract the following entities from a user input or prompt can be empty:
    theme – The main topic or subject for which metadata will be generated.
    columns – The total number of columns required in the metadata. 
     - columns – The total number of columns required in the metadata.  
     - If the user explicitly provides a number of columns, use that number.
    - If the user asks to "add" or "remove" a column and there is an existing number of columns in the old entities, update the number accordingly:
     - For example, if old entities specify `columns: 5` and the user says "add one column", the new value should be `6`.
     - If the user does NOT specify any number and there are no old entities, leave `columns` as null.
       - If the user simply asks to "change" columns without specifying a total number, this should NOT be captured in "columns". Instead, describe it fully in "specification".
    rows – The total number of rows to be generated.
    specification – A free-text field with any details or instructions related to the columns. This may include changes to column names or types, formatting requirements, or other information that could affect metadata generation. This field must always be included, even if empty.
    Return the extracted entities in a structured JSON format.

    old entities: {entities}
       """),
    ("human", "{user_input}")
])

llm = ChatOllama(model="qwen3:4b", temperature=0.0)
llm1 = ChatOllama(model="qwen3:4b", temperature=1.0)


def entity_extraction(state: ExtractionAgent) -> ExtractionAgent:
    prompt = entity_prompt.format(user_input=state["user_input"], entities=state.get("entities"))
    response = llm.invoke(prompt)  # prompt é do tipo 'str' ou 'ChatPromptValue'
    content = response.content
    match = re.search(r"</think>\s*(\{[\s\S?\}?,]+)", content)
    if match:
        json_str = match.group(1)
        try:
            entity = json.loads(json_str)
        except json.JSONDecodeError:
            entity = None
    else:
        entity = None
    old_entity = state.get("entities")
    if old_entity:
        for k, v in entity.items():
            if v is not None or v != "":
                old_entity[k] = v
    return {
        **state,
        "entities": entity,
    }


metadata_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
You are a JSON metadata generator for CSV files.

### Output format:
{{"columns":[{{"name":"...","type":"string|integer|float|boolean|date|datetime|time|decimal","description":"..."}}]}}

### Rules:
- Return **only valid JSON**, no comments or extra text.
- If metadata is null, generate new metadata.
- If metadata exists:
  - If it matches the theme, reuse columns when possible.
  - If not, discard and create new metadata.
- **If metadata exists and there is no request for change, keep the existing columns exactly as they are — including types, formats, and descriptions. Do not convert to standard formats (like ISO) or alter patterns unnecessarily.**
- If a pattern or format is specified for the columns (e.g., date format, phone number pattern, ID structure, address format), **you must apply the necessary modifications** to:
  - The description (reflecting the pattern clearly).
  - The type (change to `"string"` if the format is non-standard or region-specific).
  - And any examples, formats, or constraints implied by the pattern.

**Important:**  
- **When a column has a region-specific or custom format (like date formats, address formats, phone numbers, or IDs), its type must be `"string"` and the description must clearly specify the expected format.**  
- For example:
  - If a date is specified in US format (MM/DD/YYYY) or Brazilian format (DD/MM/YYYY), it is **not ISO 8601**, so it must be treated as `"string"` with the format explained in the description.
  - The same applies to address formats, which vary by country (e.g., US, Brazil, Europe). The description must capture the expected structure based on the region.

### Description Rules:
- Every column must include a clear and detailed description covering:
  - The purpose or meaning of the column.
  - Expected data format, pattern, or structure when applicable.
- For categorical columns, list all allowed values.
- For common field types, apply these pattern rules:

**Full Name:**
- String in the format 'Firstname Lastname'.
- Can include compound first names (e.g., 'Maria Clara') and/or compound last names (e.g., 'Silva Oliveira'), but compounds are optional.

**Email:**
- Format 'username@domain.com'.
- The username can include letters, digits, dots (.), and underscores.
- The domain follows standard email formats.

**Phone Number:**
- Specify the country format:
   - If unspecified, use international E.164 format: '+CountryCode AreaCode Number'.
   - Otherwise, indicate the pattern explicitly (e.g., Brazilian format: '+55 (XX) XXXXX-XXXX' or US format: '+1 (XXX) XXX-XXXX').
- If a country-specific format is specified, the type must be `"string"` and the description must explain the format.

**ID Fields:**
- Describe the pattern (e.g., 'Uxxxxx' where 'x' is a digit) or specify if it's a UUID.

**Address-related Fields:**
- Must include the expected structure based on the country. For example:
   - US format: 'Street Name, Number, City, State, ZIP Code'.
   - Brazilian format: 'Street Name, Number, Complement (optional), Neighborhood, City, State, ZIP Code'.
- Address fields are always `"string"` with the structure described.

**Rule for dates or timestamps:**
- Use ISO 8601 format by default (recommended).
- **If a non-ISO 8601 format is specified, ignore ISO format, set the type to `"string"`, and explicitly state the expected format in the description.**
- For non-ISO formats, follow either:
  - American format: 'MM/DD/YYYY' or 'MM/DD/YYYY HH:MM:SS'.
  - Brazilian format: 'DD/MM/YYYY' or 'DD/MM/YYYY HH:MM:SS'.
- **If the existing metadata already uses a non-ISO format, do not change it unless explicitly requested.**
- Always document the chosen format in the description.

### Syntax:
- The output must be valid JSON.
- No extra text, comments, or markdown.
- All brackets, commas, and quotation marks must be correct.
     """),
    ("human", """
 The metadata theme is "{theme}". 

 It must have {columns} columns. 
 The specification is "{specification}". 

Existing metadata is {metadata}. If it's null, generate a new one.
 """)
])


def generate_metadata(state: ExtractionAgent) -> ExtractionAgent:
    print("create metadata")
    prompt = metadata_prompt.format(
        theme=state["entities"]["theme"],
        columns=state["entities"]["columns"],
        specification=state["entities"]["specification"],
        metadata=state.get("metadata", "Null")
    )
    response = llm.invoke(prompt)  # prompt é do tipo 'str' ou 'ChatPromptValue'
    content = response.content
    print(content)
    match = re.search(r"(?s)(?<=</think>)(.*)", content)
    if match:
        json_str = match.group(1)
        try:
            metadata = json.loads(json_str)
        except json.JSONDecodeError:
            metadata = None
    else:
        metadata = None
    return {
        **state,
        "metadata": metadata,
    }


intention_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an intention detection system.

Given a user input, your task is to classify the intention into one of these two options:
Metadata generate: {metadata}

1. **modification** — if the user wants to change, update, correct, or adjust the metadata schema. This includes modifications to:
   - Column names
   - Column types
   - Column descriptions
   - Column constraints
   - Adding or removing columns
   - Any structural or schema-level changes
 - **Changing the theme to something different from the current schema (e.g., moving from "users" to "ecommerce")**
   - **If the theme provided does not match the current schema, it's considered a modification, because it requires creating or altering the schema to fit the new context.**
   - Any change in descriptions, which are used to guide data generation.
   - **Changes of quantity of columns.**
   - **Changes to the description of columns.**
     - The description is critical because it is used to guide data generation.
     - If the user modifies, clarifies, or updates any column description — whether to change meaning, improve clarity, or specify different formats or values — it must be classified as a **modification**.

**Important Note about descriptions:**  
If the user specifies or requests format constraints related to any column (for example, "phone number should be in US format", or "date should follow YYYY-MM-DD"), this is considered a **modification**, because it changes the **description** that defines how data should be generated. Descriptions are part of the metadata schema. Any clarification, change, or addition to descriptions is classified as **modification**, not generation.

2. **generation** — if the user wants to generate data (e.g., CSV rows) based on the current metadata, without modifying the schema itself.  
   This includes requests about:
   - How the values should be distributed (e.g., normal distribution, Poisson, uniform)
   - Category imbalance (e.g., 80% "active" and 20% "inactive")
   - Forcing certain values to be more frequent (e.g., more 'others' in gender)
   - Randomness constraints
   - Data formatting or output structure
   - Number of rows

**Important Rule:**  
If the user input contains both intentions (modifying metadata and generating data), always choose **"modification"** as the final intention.

Return only one word, either `"modification"` or `"generation"` for intention.  
No explanations, no extra text, no markdown — only the word.
"""),
    ("human", "{user_input}")
])


def detect_intention(state: ExtractionAgent) -> ExtractionAgent:
    print(state)
    user_input = input("Are the metadata okay? Let me know if you’d like me to change anything. \n")
    state["user_input"] = user_input
    prompt = intention_prompt.format(user_input=user_input, metadata=state["metadata"])
    response = llm.invoke(prompt)
    content = response.content.strip().lower()
    match = re.search(r"(?s)(?<=</think>)(.*)", content)
    if match:
        specification = match.group(1).strip()
    else:
        specification = None
    state["entities"]["specification"] = specification

    if "modification" == specification:
        state["refactore"] = True
    else:
        state["refactore"] = False
    return state


data_generate_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a CSV data generator.

### Goal:
Generate realistic, diverse, and constraint-respecting CSV data based on the provided schema.

### General rules:
- The CSV must:
  - Match the exact column names and types.
  - Follow formats, patterns, and constraints from descriptions.
  - Respect categorical values strictly.

### Data generation:
- Categorical fields: pick values randomly from the allowed set.
- Dates: generate valid, realistic dates.
- Datetime/time:
  - Randomize hours (00–23), minutes (00–59), and seconds (00–59) fully.
  - Avoid repeated patterns like 00:00:00 or 12:00:00.
- Strings:
  - Follow specified patterns (e.g., IDs like 'Uxxxxx' where x is a digit).
  - For free text, generate plausible, varied strings as described.
- Phone numbers:
  - Follow regional or international formats (e.g., '+55 (XX) 9XXXX-XXXX').
  - Fully randomize area codes and numbers, avoid sequential patterns.
- Numbers:
  - Generate within reasonable ranges unless a sequence is explicitly required.
- Booleans: randomly true/false unless specified.

### Examples in metadata:
- If examples define patterns (IDs, formats), follow the pattern.
- DO NOT copy examples directly.
- Use them only to understand structure and style.

### Output:
- Return ONLY the CSV content and just the data.
- No explanation, no commentary — only the CSV.
- Follow RFC 4180 (Common Format and MIME Type for Comma-Separated Values (CSV) Files):
    - Fields must be enclosed in double quotes if they contain commas, double-quotes, or line breaks.
    - Escape double-quotes inside fields by doubling them.
- If unable to generate all 10 rows, output as many as possible, but ensure that:
    - Each row is complete and valid.
    - Do not leave any row broken or cut-off at the end.
    - If the generation is interrupted, the CSV must end at the last complete line.
"""),
    ("human", """
Metadata: {metadata}

Generate exactly 10 rows.

Specification: "{specification}"
""")
])


def generate_data(state: ExtractionAgent) -> ExtractionAgent:
    prompt = data_generate_prompt.format(
        specification=state["entities"]["specification"],
        metadata=state["metadata"]
    )
    response = llm1.invoke(prompt)
    content = response.content.strip().lower()
    match = re.search(r"(?s)(?<=</think>)(.*)", content)
    r = match.group(1).strip()
    print(r)
    df = pd.read_csv(StringIO(r))
    df.to_csv("sample.csv", index=False)

    return state


graph_builder = StateGraph(ExtractionAgent)

graph_builder.add_node("detect_intention", detect_intention)
graph_builder.add_node("entity_extraction", entity_extraction)
graph_builder.add_node("generate_metadata", generate_metadata)
graph_builder.add_node("generate_data", generate_data)

graph_builder.set_entry_point("entity_extraction")
graph_builder.add_edge("entity_extraction", "generate_metadata")
graph_builder.add_edge("generate_metadata", "detect_intention")

graph_builder.add_conditional_edges(
    "detect_intention",
    lambda state: "entity_extraction" if state["refactore"] else "generate_data",
    {
        "entity_extraction": "entity_extraction",
        "generate_data": "generate_data",
    }
)

graph_builder.set_finish_point("generate_data")
graph = graph_builder.compile()
print(graph.get_graph().print_ascii())
result = graph.invoke({
    "user_input": "generate user registre with 15 cols"
})



