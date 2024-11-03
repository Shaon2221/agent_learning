import os
from crewai import Agent, Task, Crew, Process
# pip install -U duckduckgo-search
from langchain_community.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()
from langchain.tools import Tool
search_tool = Tool(
    name="DuckDuckGo Search",
    func=DuckDuckGoSearchRun().run,
    description="Search the internet for current information. Input should be a search query string."
)

from crewai import LLM
ollama_llm = LLM(
    model="ollama/llama3.1", 
    base_url="http://localhost:11434"
)


# Define your agents with roles and goals
researcher = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge developments in AI and data science',
  backstory="""You are an expert at a technology research group, 
  skilled in identifying trends and analyzing complex data.""",
  verbose=True,
  allow_delegation=False,
  tools=[search_tool],
  llm=ollama_llm
)
writer = Agent(
    role='Tech Content Strategist',
    goal='Craft compelling content on tech advancements',
    backstory="""You are a content strategist known for 
    making complex tech topics interesting and easy to understand.""",
    verbose=True,
    allow_delegation=True,
    llm=ollama_llm
)

# Create tasks for your agents
task1 = Task(
    description="""Analyze 2024's AI advancements. 
    Find major trends, new technologies, and their effects. 
    Provide a detailed report.""",
    expected_output="""A comprehensive report on 2024's AI advancements, 
    including major trends, breakthrough technologies, and their potential impact.""",
    agent=researcher
)
task2 = Task(
    description="""Create a blog post about major AI advancements using your insights. 
    Make it interesting, clear, and suited for tech enthusiasts. 
    It should be at least 4 paragraphs long.""",
    expected_output="""A well-structured, engaging blog post about AI advancements in 2024, 
    targeting tech enthusiasts, with at least 4 paragraphs of content.""",
    agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=True
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)