from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
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


##prompt templates
#prompt for code creation
code_prompt = PromptTemplate(
    input_variables=["language", "task"],
    template = "Write a very short {language} function that will {task}."
)

#prompt for code testing
test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template = "Write a test for the following {language} code;\n{code}"
)


#create chains
#chain for code creation
code_chain = LLMChain(
    llm = llm,
    prompt = code_prompt,
    output_key = "code" #renames the output key to "code"
)

#chain for code testing
test_chain = LLMChain(
    llm = llm,
    prompt = test_prompt,
    output_key = "test" #renames the output key to "test"
)

#create a sequential chain
chain = SequentialChain(
    chains = [code_chain, test_chain],
    input_variables = ["task", "language"], #input to the first chain
    output_variables = ["test", "code"] #output of the final chain
)

#create dictionary
chain_dict = {
    "language": args.language,
    "task": args.task}


#run chain
result = chain(chain_dict)

print(">>>>>>>>>>>> Generated code:")
print(result["code"])


print(">>>>>>>>>>>> Generated test:")
print(result["test"])
