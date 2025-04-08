You are tasked with building an AI Agent or Multiple agents if its need to help automate the research discover of new research papers published to https://arxiv.org/ -- I provided an example JSON of a similar type of agent in n8n which you can use for the system and user prompts adn to get an idea of the flow. Note, the output should be a newletter that is in markdown format.

When the AI Agent it triggered, it should take in the below payload
"user_info": {
"first_name": "test name",
"job_tittle": "Project Manager",
"company": "Twilio",
"goal": "learn more about how I can build new AI products in my job."
},
"date": "2024-04-02"

This AI Agent should follow the below steps:
1) Tool call: Fetch the lastest article for the previous day (use "date" property as qurrry param) GET https://arxiv.org/catchup/cs/2025-04-01
2) Parse the response which is in HTML to get all the important information. Example code @/example_parse.js 
3) Pass the data to an AI Agent to find the 5 most relevent articles to the users info and rank them as well as provide a reason for the ranking.
4) It should then pass this data in a structured way (json) to nother "Researcher" AI Agent that will iterate through each of the 5 articles to read and summarize them for the newsletter. It should output data like @/example_researcher_output.json 
5) Then all the data should be joined and passed to a final "Copy Writter" AI Agent that takes all this and formats in into a newsletter.