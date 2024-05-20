from langchain_core.pydantic_v1 import BaseModel, Field, validator

class Instruction(BaseModel):
    instruction: str = Field(description="The instruction to be asked, should be in Traditional Chinese format")

class InstructionInputs(BaseModel):
    inputs: str = Field(description="The input based on the instruction. If there's no input, return `None`")

