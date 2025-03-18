from langchain.agents import AgentExecutor
from orchestrator import OrchestratorAgent

def main():
    agent = OrchestratorAgent()
    orchestrator = AgentExecutor(agent= agent.agent, tools=agent.tools, verbose=True)
    question = "me explique como funciona os devios condicionais"
    response = orchestrator.invoke({"input": question})
    print(response)

if __name__ == "__main__":
    main()