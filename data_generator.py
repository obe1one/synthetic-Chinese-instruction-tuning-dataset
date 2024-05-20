from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts.prompt import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from schema import Instruction, InstructionInputs
from prompts import *
import dotenv

class DataGenerator(ABC):
    """
    A data generator generating synthetic multi-turn instruction-following dataset
    """
    @abstractmethod
    def generate(self):
        """
        Generate a synthetic data
        """
        pass
    
    @abstractmethod
    def few_shot_generate(self, examples, turns):
        """
        Generate a synthetic data with some in-context examples
        """
        pass

class LLMDataGenerator(DataGenerator):
    def __init__(self, llm, model_name, api_key):
        self._validation(llm)

        self.llm = llm
        self.model_name = model_name

        if self.llm == "claude":
            self.agent = ChatAnthropic(model=self.model_name, anthropic_api_key=api_key, default_request_timeout=60)
        elif self.llm == "openai":
            self.agent = ChatOpenAI(model=self.model_name, openai_api_key=api_key, timeout=60)

    def _validation(self, llm):
        if not llm in ["claude", "openai"]:
            raise ValueError(
                f"The LLM {llm} doesn't exist"
            )
    
    def generate(self):
        pass

    def few_shot_generate(self, examples, turns):
        first_instruction = self.generate_first_instruction(examples)
        if first_instruction is None:
            return []
        
        dialogue = [{ "instruction": first_instruction.content }]
        first_response = self.response_to_multi_turn_instruction(dialogue=dialogue)

        if first_response is None:
            return []

        dialogue[0]["output"] = first_response.content

        print("Me => " + dialogue[0]["instruction"])
        print("AI => " + dialogue[0]["output"])

        for i in range(turns - 1):
            instruction = self.generate_multi_turn_instruction(dialogue=dialogue)
            if instruction is None:
                continue

            print("Me => " + instruction.content)

            dialogue.append({
                "instruction": instruction.content
            })

            response = self.response_to_multi_turn_instruction(dialogue=dialogue)

            if response is None:
                dialogue.pop()
                continue

            print("AI => " + response.content)

            dialogue[-1]["output"] = response.content
        
        return dialogue

    def generate_first_instruction(self, examples):
        prompt_template = create_first_instruction_prompt_v3(examples)

        chain = prompt_template | self.agent

        try:
            instr = chain.invoke({})
            return instr
        except Exception as e:
            print(e)
            return None
    
    def response_to_multi_turn_instruction(self, dialogue):
        prompt = []
        for context in dialogue:
            if "instruction" in context:
                prompt.append(HumanMessage(content=context["instruction"]))
            if "output" in context:
                prompt.append(AIMessage(content=context["output"]))
        
        try:
            output = self.agent.invoke(prompt)
            return output
        except Exception as e:
            print(e)
            return None
        
    def generate_multi_turn_instruction(self, dialogue):
        prompt_template = create_multi_turn_prompt_v3(dialogue)

        chain = prompt_template | self.agent
        
        try:
            output = chain.invoke({})
            return output
        except Exception as e:
            print(e)
            return None

class Claude3DataGenerator(DataGenerator):
    def __init__(self, model_name):
        config = dotenv.dotenv_values(".env")
        self.model_name = model_name
        self.agent = ChatAnthropic(model=model_name, anthropic_api_key=config["ANTHROPIC_API_KEY"])
    
    def generate(self):
        parser = PydanticOutputParser(pydantic_object=Instruction)
        prompt_template = PromptTemplate(
            template="你是一個好奇心旺盛的人類，你現在在和一個大型語言模型進行互動。請向這個大型語言模型進行提問\n"
                     "{format_instructions}",
            input_variables=[],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        chain = prompt_template | self.agent | parser
        output = chain.invoke({})
        print(output)
    
    def few_shot_generate(self, examples):
        first_instruction = self.generate_first_instruction(examples)
        if first_instruction is None:
            return None
        
        inputs = self.generate_first_inputs(examples, instruction=first_instruction.instruction)
        if inputs is None:
            return None

    def generate_first_instruction(self, examples):
        parser = PydanticOutputParser(pydantic_object=Instruction)

        example_sentences = "\n".join([f"問題 {i + 1}: {e['instruction']}" for i, e in enumerate(examples)])

        prompt_template = PromptTemplate(
            template="以下是一系列的問題：\n\n" + example_sentences + "\n\n" + "請仿造這些指示範例，生成一個新的指示。\n" + "{format}",
            input_variables=[],
            partial_variables={"format": parser.get_format_instructions()}
        )

        chain = prompt_template | self.agent | parser

        try:
            instr = chain.invoke({})
            return instr
        except Exception as e:
            print(e)
            return None
    
    def generate_first_inputs(self, examples, instruction):
        parser = PydanticOutputParser(pydantic_object=InstructionInputs)

        example_sentences = "\n\n".join([f"指示: {e['instruction']}\n輸入: {e['input']}" for i, e in enumerate(examples)])
        
        prompt_template = PromptTemplate(
            template = "以下是一系列的指示，以及他們的輸入：\n\n" + example_sentences + "\n\n"
                       "現在，我給你一個新的指示，請仿造以上的範例，生成這個指示的輸入。你的回覆必須是一個json格式\n\n指示: {instruction}\n{format}\njson: ",
            input_variables=["instruction"],
            partial_variables={"format": parser.get_format_instructions()}
        )

        # print(prompt_template.format(instruction=instruction))

        chain = prompt_template | self.agent | parser

        try:
            output = chain.invoke({"instruction": instruction})
            return output
        except Exception as e:
            print(e)
            return None

    def response_to_instruction(self, instruction):
        prompt = PromptTemplate(
            template=instruction,
            input_variables=[]
        )
        chain = prompt | self.agent

        try:
            output = chain.invoke({})
            return output
        except Exception as e:
            print(e)
            return None
    
    def response_to_multi_turn_instruction(self, dialogue):
        prompt = [
            HumanMessage(content=value) if key == "instruction" else AIMessage(content=value)
            for data in dialogue["conversation"] for key, value in data.items()
        ]

        try:
            output = self.agent.invoke(prompt)
            return output
        except Exception as e:
            print(e)
            return None
    
    def generate_multi_turn_instruction(self, dialogue):
        multi_turn = "\n\n".join([f"我:\n{data['instruction']}\n\n對方:\n{data['output']}" for data in dialogue['conversation']])
        
        prompt = PromptTemplate(
            template="我現在在和另一個人進行對話，以下為我們的對話內容:\n\n" + multi_turn +
                     "\n\n請幫我構思接下來我下一句要怎麼回應，才能讓對話進行下去，並且主題必須要和我們現在討論的主題連貫。這個問題裡面最多只能有一個問題。\n"
                     "我會把你的回應直接貼給對方。",
            input_variables=[]
        )
        chain = prompt | self.agent
        
        try:
            output = chain.invoke({})
            return output
        except Exception as e:
            print(e)
            return None