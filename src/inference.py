from langchain_community.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from utils.utils import *

# Load finetuned model and tokenizer
FULL_MODEL_PATH = "full_model"

model = AutoModelForCausalLM.from_pretrained(
    FULL_MODEL_PATH,
    device_map="auto",
    local_files_only=True
    )
tokenizer = AutoTokenizer.from_pretrained(FULL_MODEL_PATH, local_files_only=True)

# Create a pipline for LLM 
llm_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1
)

# Create output parser and obtain answer formating instructions
conclusion_schema = ResponseSchema(
    name="conclusion",
    description="Final legal interpretation."
)
related_laws_schema = ResponseSchema(
    name="related_laws",
    description="Relevant laws."
)

risk_level_schema = ResponseSchema(
    name="risk_level",
    description="Low, Medium, or High."
)

response_schemas = [conclusion_schema, related_laws_schema, risk_level_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# Create question prompt
prompt = PromptTemplate(
    template="""You are a legal reasoning assistant. 
Answer the following question in a structured JSON format:

Question:
{question}

You MUST follow these rules:
- Respond ONLY with a JSON object
- Follow EXACTLY the schema
- Do NOT include explanations
- Do NOT include any extra text
- Return ONLY valid JSON

{format_instructions}
""",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions}
)

def generate_answer(question):
    prompt_text = prompt.format(question=question)
    output = llm_pipe(prompt_text)[0]["generated_text"]
    json_block = extract_json_block(output)
    parsed_json = parse_json(json_block)
    answer = (
        f"{parsed_json.get('conclusion', 'N/A')}\n"
        f"- Related Laws: {parsed_json.get('related_laws', 'N/A')}\n"
        f"- Risk Level: {parsed_json.get('risk_level', 'N/A')}\n"
    )
    return answer
