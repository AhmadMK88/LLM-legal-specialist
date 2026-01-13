from config.configs import FULL_MODEL_PATH
from langdetect import detect
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from utils.utils import extract_json_block

#===================================
# Configuration
#===================================
LABELS = {
    'en':{
        'related_laws':'Related Laws',
        'risk_level':'Risk Level'
    },

    'ar':{
        'related_laws':'القوانين ذات صلة',
        'risk_level':'مستوى المخاطر'
    }
} 

def _create_llm_pipline() -> pipeline:
    """
    create LLM pipline using fully trained model and tokenizer

    Returns:
        - llm_pipline(Pipiline): full pipline used in answer generation
    """

    model = AutoModelForCausalLM.from_pretrained(
        FULL_MODEL_PATH,
        device_map="auto",
        local_files_only=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(FULL_MODEL_PATH, local_files_only=True)

    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1
    )

    return llm_pipeline

def _create_prompt() -> PromptTemplate:
    """
    Create a prompt used for generating answer in a specifc schema

    Returns:
        - prompt(PromptTemplate): final prompt to be used in answer generating
    """

    conclusion_schema = ResponseSchema(
        name="conclusion",
        description="Final legal interpretation."
    )
    related_laws_schema = ResponseSchema(
        name="related_laws",
        description=("Relevant laws."
                    "Must be in the same language as the conclusion ")
    )

    risk_level_schema = ResponseSchema(
        name="risk_level",
        description=("Risk severity."
                    "Must be in the same language as the conclusion ")
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
    - Include appropriate labels inside the text fields
    - Labels must be in the SAME language as the conclusion
    - Return ONLY valid JSON

    {format_instructions}
    """,
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions}
    )

    return prompt

def generate_answer(question: str) -> str :
    """
    Generate answer given a question using full llm pipline and a prompt for answer formatting

    Args:
        - question(str): user question to be passed to the pipeline
    
    Returns:
        - answer(str): model generated answer
    """
    prompt = _create_prompt()
    prompt_text = prompt.format(question=question)
    
    llm_pipeline = _create_llm_pipline()
    output = llm_pipeline(prompt_text)[0]["generated_text"]
    json_block = extract_json_block(output)

    language = detect(json_block.get("conclusion", "N/A"))
    labels = LABELS[language]
    
    conclusion = json_block.get('conclusion', 'N/A')
    related_laws = json_block.get('related_laws', 'N/A')
    risk_level = json_block.get('risk_level', 'N/A')

    answer = (
        f"{conclusion}\n"
        f"- {labels['related_laws']} : {related_laws}\n"
        f"- {labels['risk_level']} : {risk_level}"
    )
    return answer
