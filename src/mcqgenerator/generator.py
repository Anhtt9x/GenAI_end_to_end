import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.logger import logging
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from langchain.chat_models import ChatOpenAI
from src.mcqgenerator.utils import read_file , get_table_data
from langchain import HuggingFaceHub


load_dotenv()

huggingface_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
apiopenai_key = os.getenv("apiopenai_key")

llm = HuggingFaceHub(
    repo_id="EleutherAI/gpt-j-6b",
    model_kwargs={"temperature": 0.7},
    huggingfacehub_api_token=huggingface_key
)

TEMPLATE="""
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}

"""

quiz_generator = PromptTemplate(template=TEMPLATE, input_variables=["text", "number", "subject", "tone", "response_json"])

quiz_chain = LLMChain(llm=llm, prompt=quiz_generator, output_key="quiz", verbose=True)

TEMPLATE2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis.
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(template=TEMPLATE2, input_variables=["subject", "quiz"])

review_chain = LLMChain(llm=llm , prompt=quiz_evaluation_prompt, output_key="review", verbose=True)

generator_chain = SequentialChain(chains=[quiz_chain,review_chain],
                                  input_variables=["text", "number", "subject", "tone", "response_json"],
                                  output_variables=["subject", "quiz"],
                                  verbose=True)

