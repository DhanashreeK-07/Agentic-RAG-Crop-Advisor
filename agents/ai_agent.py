from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain.tools import Tool
from langchain import hub

from tools.rag_tool import rag_tool
from tools.crop_tool import crop_prediction_tool
from tools.soil_tool import soil_tool


llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key="xxxxxx-oFx0OmrjBSCEUmvDqcL2lnk66Z81Q90RBg570rYbA3irL7UEDFcj6pWicuF-7KjVg_MNeI7B0hT3BlbkFJpjJvL0Sj1PzR-ypxSjt1joJqaAkdAe-12BcDYkKH7SpGrSuKHN3Nx08QYoA1_jFWAsyCjh8WoA"
)

tools = [

    Tool(
        name="Crop Knowledge Retriever",
        func=rag_tool,
        description="Retrieve crop knowledge from dataset and crop documents"
    ),

    Tool(
        name="Crop Predictor",
        func=crop_prediction_tool,
        description="Predict best crop based on soil values"
    ),

    Tool(
        name="Soil Analyzer",
        func=soil_tool,
        description="Analyze soil pH"
    )

]


prompt = hub.pull("hwchase17/react")


agent = create_react_agent(
    llm,
    tools,
    prompt
)


agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)


def run_agent(query):

    response = agent_executor.invoke({"input": query})

    return response["output"]
