import uuid

import openlit
from langgraph.graph import StateGraph

openlit.init(
    application_name="llm-test",
  otlp_endpoint="http://localhost:4318",
otlp_headers={"authorization": "7c4055bc-9213-41e3-aac8-fc7d7eb59472"},
    disabled_instrumentors=False,
    disable_batch=False,
    disable_metrics=False,

)
from nodes import *
graph_builder = StateGraph(ExtractionAgent)

graph_builder.add_node("entity_extraction", entity_extraction)
graph_builder.add_node("generate_metadata", generate_metadata)
graph_builder.add_node("detect_intention", detect_intention)
graph_builder.add_node("generate_data", generate_data)

graph_builder.add_edge("entity_extraction", "generate_metadata")
graph_builder.add_edge("generate_metadata", "detect_intention")
graph_builder.set_entry_point("entity_extraction")

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
session = str(uuid.uuid4())
with openlit.start_trace("session") as trace:
    trace.set_metadata({"user_session":session, "user_id":"user_111"})
    result = graph.invoke({"user_input":
            """
    generate logs with 5 cols
            """
                           })
print(result)