# Implementation Notes for Article Ranking Agent

## Current Issues

- The current implementation is overly complex with unnecessary tool registration and dependencies
- Error: `__init__() missing 3 required positional arguments: 'model', 'usage', and 'prompt'` suggests we're calling a tool directly rather than using the agent's core API
- The implementation relies on complex tool registration and RunContext which adds unnecessary complexity

## Learnings from Archon

1. **Simplification is Key**: We can dramatically simplify the implementation by:

   - Removing all tool registration and usage
   - Not using RunContext, Deps, or any tool imports
   - Placing both article data and user data directly into the main prompt

2. **Core Pydantic AI Approach**: The agent should use the clean, core API of Pydantic AI rather than complex tool registrations

3. **Data Handling**: Instead of using tools to fetch and process data, we can:

   - Load the JSON file directly in the main function
   - Pass the data directly in the user message
   - Let the LLM handle the processing and ranking

4. **Error Handling**: We need better error handling for JSON parsing and response validation

## Action Plan

1. **Simplify Agent Implementation**:

   - Remove all tool-related code
   - Create a clean Agent instance with just the LLM and system prompt
   - Remove unnecessary dependencies and imports

2. **Improve Data Handling**:

   - Load article data directly from file or use sample data
   - Format user information clearly in the prompt
   - Pass all data directly to the agent in the user message

3. **Enhance Response Processing**:

   - Add robust JSON parsing for the agent's response
   - Implement better error handling for various response formats
   - Format the output in a user-friendly way

4. **Future Improvements**:
   - Add input validation for user information
   - Implement chunking for large article datasets
   - Add more sophisticated ranking algorithms if needed
   - Consider adding a simple web interface for user input

## Next Steps

1. Implement the simplified agent.py as suggested by Archon
2. Test with both sample data and real arXiv data
3. Refine the system prompt to ensure consistent JSON responses
4. Add more robust error handling and response validation
