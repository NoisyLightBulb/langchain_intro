from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import argparse

#load environment variables
load_dotenv()

#create default values
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()


#create language model
llm = OpenAI()

#prompt template
code_prompt = PromptTemplate(
    template = "Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
)

#create chain
code_chain = LLMChain(
    llm = llm,
    prompt = code_prompt
)

#create dictionary
chain_dict = {
    "language": args.language,
    "task": args.task}


result = code_chain(chain_dict)

print(result["text"])
