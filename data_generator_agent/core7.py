import json
import re

from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
from typing import TypedDict, Optional
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import datetime
from io import StringIO
from langgraph.constants import Send
import operator
from typing import Annotated
from pprint import pprint


class ExtractionAgent(TypedDict):
    user_input: str
    refactore: bool = False
    entities: Optional[dict]
    metadata: Optional[dict]
    specification_generation: Optional[str]
    csv_chuck: Annotated[list, operator.add]


llm = ChatOllama(model="qwen3:4b", temperature=0.0)
llm2 = ChatOllama(model="qwen3:8b", temperature=0.0)
llm1 = ChatOllama(model="qwen3:8b", temperature=0.65)

entity_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are an entity extraction system. Your task is to extract the following entities from a user input or prompt can be empty:
    theme – The main topic or subject for which metadata will be generated, such as "user profiles", "ecommerce products", or "bank transactions".

    Do not extract technical or statistical instructions (e.g., "normal distribution", "Poisson", or "categorical") as the theme.

    If the user does not mention a real-world subject or theme, set theme to empty.
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
    print(entity)
    old_entity = state.get("entities", {})
    if old_entity:
        for k, v in entity.items():
            v.strip() if isinstance(v, str) else v
            print(v)
            if v is not None or v != "":
                old_entity[k] = entity[k]
    else:
        old_entity = entity
    return {
        **state,
        "entities": old_entity,
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
    prompt = metadata_prompt.format(
        theme=state["entities"]["theme"],
        columns=state["entities"]["columns"],
        specification=state["entities"]["specification"],
        metadata=state.get("metadata", "Null")
    )
    response = llm1.invoke(prompt)  # prompt é do tipo 'str' ou 'ChatPromptValue'
    content = response.content
    # print(content)
    match = re.search(r"(?s)(?<=</think>)(.*)", content)
    if match:
        json_str = match.group(1)
        try:
            metadata = json.loads(json_str)
        except json.JSONDecodeError:
            metadata = state["metadata"]
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

    modification — if the user wants to change, update, correct, or adjust the metadata schema. This includes modifications to:

        Column names

        Column types

        Column descriptions

        Column constraints

        Adding or removing columns

        Any structural or schema-level changes

        Changing the theme to something different from the current schema (e.g., moving from "users" to "ecommerce")

            If the theme provided does not match the current schema, it's considered a modification because it requires creating or altering the schema to fit the new context.

        Any change in descriptions, which are used to guide data generation.

        Changes in the number of columns, including specifying a different quantity of columns or different number of columns of a given type (e.g., requesting 4 categorical columns when the schema has a different number).

        Changes to the number of rows are also considered a modification

            Changing the quantity of rows or requesting a different number of rows implies altering the generation parameters and is treated as a schema-level modification.

        Changes to the description of columns.

            The description is critical because it is used to guide data generation.

            If the user modifies, clarifies, or updates any column description—whether to change meaning, improve clarity, or specify different formats or values—it must be classified as a modification.

Important Note about descriptions:
If the user change quantity of columns will be considerate change of schema
If the user specifies or requests format constraints related to any column (for example, "phone number should be in US format", or "date should follow YYYY-MM-DD"), this is considered a modification, because it changes the description that defines how data should be generated. Descriptions are part of the metadata schema. Any clarification, change, or addition to descriptions is classified as modification, not generation.

    generation — if the user wants to generate data (e.g., CSV rows) based on the current metadata, without modifying the schema itself. This includes requests about:

        How the values should be distributed (e.g., normal distribution, Poisson, uniform)

        Category imbalance (e.g., 80% "active" and 20% "inactive")

        Forcing certain values to be more frequent (e.g., more 'others' in gender)

        Randomness constraints

        Data formatting or output structure

        Number of rows

Important Rule:
If the user input contains both intentions (modifying metadata and generating data), always choose "modification" as the final intention.

Return only one word, either "modification" or "generation" for intention.
"""),
    ("human", "{user_input}")
])


def detect_intention(state: ExtractionAgent) -> ExtractionAgent:
    # print(state)
    pprint(state['metadata'])
    user_input = input("Are the metadata okay? Let me know if you’d like me to change anything. \n")
    state["user_input"] = user_input
    prompt = intention_prompt.format(user_input=user_input, metadata=state["metadata"])
    response = llm2.invoke(prompt)
    content = response.content.strip().lower()
    match = re.search(r"(?s)(?<=</think>)(.*)", content)
    if match:
        specification = match.group(1).strip()
    else:
        specification = None
    state["entities"]["specification"] = specification
    print(specification)
    if "modification" == specification:
        state["refactore"] = True
    else:
        state["refactore"] = False
    # prompt = specification_generation_prompt.format(user_input=user_input)
    # response = llm.invoke(prompt)
    # content = response.content.strip().lower()
    # match = re.search(r"(?s)(?<=</think>)(.*)", content)
    # print(f"pegou a especificacao\n{match}\n\n")
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
- No explanation, no commentary
- Follow RFC 4180 (Common Format and MIME Type for Comma-Separated Values (CSV) Files):
    - Fields must be enclosed in double quotes if they contain commas, double-quotes, or line breaks.
    - Escape double-quotes inside fields by doubling them.
- If unable to generate all 20 rows, output as many as possible, but ensure that:
    - Each row is complete and valid.
    - Do not leave any row broken or cut-off at the end.
    - If the generation is interrupted, the CSV must end at the last complete line.
"""),
    ("human", """
Metadata: {metadata}

Generate exactly 20 

Specification: "{specification}"
""")
])


def split_generate_jobs(state: dict) -> list[Send]:
    rows = state["entities"].get("rows")
    return [Send("generate_chunk_data",
                 arg={
                     **state,
                     "col": i
                 })
            for i in state['metadata']['columns']
            ]


def generate_chunk_data(state: dict) -> dict:
    print(f"start time:{datetime.datetime.now()}")
    metadata = state["metadata"]
    specification = state["entities"]["specification"]

    prompt = data_generate_prompt.format(
        specification=specification,
        metadata=state["col"]
    )
    result = []
    for _ in range(50):
        response = llm1.invoke(prompt)
        content = response.content.strip().lower()
        match = re.search(r"(?s)(?<=</think>)(.*)", content)
        r = match.group(1).strip()
        result.append(r)
    print(result)
    return {"csv_chuck": [result]}


def dummy(state):
    print("Start generate")
    return state


def reduce_generated_data(states: list[dict]) -> dict:
    print(states)
    print(f"end time:{datetime.datetime.now()}")
    return {
        "csv_data": "end"
    }


graph_builder = StateGraph(ExtractionAgent)

graph_builder.add_node("detect_intention", detect_intention)
graph_builder.add_node("entity_extraction", entity_extraction)
graph_builder.add_node("generate_metadata", generate_metadata)
graph_builder.add_node("reduce_generated_data", reduce_generated_data)
graph_builder.add_node("dummy", dummy)
graph_builder.set_entry_point("entity_extraction")
graph_builder.add_edge("entity_extraction", "generate_metadata")
graph_builder.add_edge("generate_metadata", "detect_intention")
graph_builder.add_conditional_edges("dummy", split_generate_jobs, ["generate_chunk_data"])
graph_builder.add_edge("generate_chunk_data", "reduce_generated_data")

graph_builder.add_conditional_edges(
    "detect_intention",
    lambda state: "entity_extraction" if state["refactore"] else "split_generate_jobs",
    {
        "entity_extraction": "entity_extraction",
        "split_generate_jobs": "dummy",
    }
)

graph_builder.add_node("generate_chunk_data", generate_chunk_data)
graph_builder.set_finish_point("reduce_generated_data")
graph = graph_builder.compile()
print(graph.get_graph().print_ascii())
user_input = input("what kind of data do you want to generate: ")
result = graph.invoke({
    "user_input": user_input
})

specification_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Extract only the relevant data generation specifications from the text. Focus exclusively on the following aspects:

- Distribution: statistical distribution of values (e.g., normal, uniform, categorical).
- Proportion: value proportions in a column (e.g., 80% active, 20% inactive).
- Range: valid numerical intervals (e.g., age between 18 and 60).
- Conditional rules: only if the value of one column depends on the value of another (e.g., "column 'adult' is true if 'age' is over 18").

For each extracted specification:
- Include the column name if it is explicitly mentioned in the text.
- If no column is specified, group those specifications under a "general" key.
- Only include a field if the information is clearly provided.
- Ignore any unrelated details such as formatting rules (e.g., email format, CPF mask), number of records, data types, domain lists, or any kind of string formatting.

Return the results in JSON format, like this:

{
  "age": {
    "distribution": "normal",
    "range": [18, 60]
  },
  "status": {
    "proportion": {
      "active": 0.8,
      "inactive": 0.2
    }
  },
  "adult": {
    "conditional": "adult = true if age > 18"
  },
  "general": {
    "distribution": "uniform",
    "proportion": {
      "male": 0.5,
      "female": 0.5
    }
  }
}
Do not invent or infer values — only extract what's explicitly mentioned in the text.
"""),
    ("human", "{user_input}")
])

"start time:2025-06-13 12:53:53.449524"