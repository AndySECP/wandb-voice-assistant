# core/agent/llm_agent.py
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
import json
import logging
from datetime import datetime
import httpx

from src.core.agent.tools import AVAILABLE_TOOLS
from src.core.agent.memory import MemoryManager
from src.services.data_manager import REQUIRED_COLUMNS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""You are an AI assistant specializing in analyzing an hallucination detection experiments. 
Speak naturally as if we're having a friendly conversation. Your responses should be easy to listen to and understand. Response should be concise and just respond to the question.

When analyzing:
- Focus on meaningful patterns and their implications
- Highlight key differences that drive performance changes
- Connect technical findings to practical improvements
- Proactively suggest next steps or areas to investigate

Communicate naturally using:
- Clear comparisons ("Model A outperforms Model B by 30 percent because...")
- Impact-focused statements ("The key finding is... which suggests...")
- Concrete recommendations ("Based on these results, consider...")
- Conversational transitions and relevant context
- Avoid listing metrics, the conversation should sounds natural

When discussing numbers:
- Round for clarity and provide context
- Explain what the metrics mean in practice
- Focus on meaningful differences and trends
- Connect metrics to actionable insights

Remember to:
- Keep it concise and clear
- Focus on the most relevant information
- Use natural transitions between ideas
- Speak as if you're explaining to a colleague


If you want to query the data, the available columns are {REQUIRED_COLUMNS}
"""


class HallucinationAnalysisAgent:
    def __init__(
        self,
        openai_api_key: str,
        data_manager,
        tools,
        memory_manager: Optional[MemoryManager] = None,
        use_realtime: bool = False,
    ):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.data_manager = data_manager
        self.tools = tools
        self.memory = memory_manager or MemoryManager()
        self.api_key = openai_api_key
        self.use_realtime = use_realtime

        # Store tools as both dict and list for different use cases
        self.available_functions = {
            "get_best_models": tools.get_best_models,
            "compare_hyperparams": tools.compare_hyperparams,
            "get_experiment_details": tools.get_experiment_details,
            "analyze_by_model_type": tools.analyze_by_model_type,
            "analyze_config_impact": tools.analyze_config_impact,
            "get_performance_distribution": tools.get_performance_distribution,
            "compare_architectures": tools.compare_architectures,
            "query_data": tools.query_data,
        }
        self.tool_descriptions = AVAILABLE_TOOLS

    async def get_realtime_session(self) -> Dict[str, Any]:
        """Generate an ephemeral token for WebRTC connection"""
        async with httpx.AsyncClient() as client:
            try:
                # Format tools for realtime API format
                formatted_tools = []
                for tool in self.tool_descriptions:
                    if tool["type"] == "function":
                        formatted_tool = {
                            "name": tool["function"]["name"],
                            "type": "function",
                            "description": tool["function"]["description"],
                            "parameters": tool["function"]["parameters"],
                        }
                        formatted_tools.append(formatted_tool)

                session_config = {
                    "model": "gpt-4o-realtime-preview-2024-12-17",
                    "voice": "alloy",
                    "tools": formatted_tools,
                    "temperature": 0.7,
                    "instructions": SYSTEM_PROMPT,
                }

                logger.info(
                    f"Sending realtime session request with config: {json.dumps(session_config, indent=2)}"
                )

                response = await client.post(
                    "https://api.openai.com/v1/realtime/sessions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=session_config,
                    timeout=30.0,
                )

                if response.status_code != 200:
                    logger.error(f"OpenAI response status: {response.status_code}")
                    logger.error(f"OpenAI response text: {response.text}")

                response.raise_for_status()
                return response.json()

            except Exception as e:
                logger.error(f"Error generating realtime session: {str(e)}")
                raise

    async def process_query(
        self, query: str, context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Process a user query using either realtime or standard mode"""
        if self.use_realtime:
            # Check if this is a session initialization request
            is_init = context and context.get("mode") == "realtime"
            if is_init:
                return await self._process_realtime_query(query, context)
            else:
                return await self._process_standard_query(query, context)
        else:
            return await self._process_standard_query(query, context)

    async def _process_realtime_query(
        self, query: str, context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Process a query using the realtime API"""
        try:
            # Get session details with properly formatted tools
            session = await self.get_realtime_session()

            # Store query in memory
            await self.memory.add_message("user", query)

            # If this is a function execution request from a previous realtime call
            if context and context.get("function_name"):
                function_name = context["function_name"]
                arguments = context["arguments"]

                # Execute the function if it exists
                if function_name in self.available_functions:
                    try:
                        result = await self.available_functions[function_name](
                            **arguments
                        )
                        return {
                            "response": "Function executed successfully",
                            "metadata": {
                                "function_name": function_name,
                                "success": True,
                            },
                            "function_result": result,
                        }
                    except Exception as e:
                        logger.error(f"Error executing function {function_name}: {e}")
                        return {
                            "response": f"Error executing function: {str(e)}",
                            "metadata": {
                                "function_name": function_name,
                                "success": False,
                                "error": str(e),
                            },
                        }

            # Return session info for regular realtime initialization
            return {
                "response": "Realtime session initialized",
                "metadata": {
                    "session_id": session["id"],
                    "client_secret": session["client_secret"],
                    "voice": session["voice"],
                    "expires_at": session["expires_at"],
                },
                "session": session,  # Full session details
            }

        except Exception as e:
            logger.error(f"Error in realtime processing: {str(e)}")
            raise

    async def _process_standard_query(
        self, query: str, context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Process a user query using the LLM agent with simplified conversation handling"""
        try:
            # Get conversation history
            conversation_history = self.memory.get_recent_context()

            # Initialize messages array
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(conversation_history)
            messages.append({"role": "user", "content": query})

            # Store user query
            await self.memory.add_message("user", query)

            # Get initial response
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                tools=AVAILABLE_TOOLS,
                tool_choice="auto",
                temperature=0.1,
            )

            message = response.choices[0].message

            # If no tool calls, store and return direct response
            if not message.tool_calls:
                await self.memory.add_message("assistant", message.content)
                return {
                    "response": message.content,
                    "metadata": {
                        "tools_used": [],
                        "timestamp": datetime.now().isoformat(),
                    },
                }

            # Handle tool calls
            tools_used = []
            tool_results = []

            # Process each tool call
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                if tool_name in self.available_functions:
                    try:
                        result = await self.available_functions[tool_name](**tool_args)
                        tool_results.append(result)
                        tools_used.append(tool_name)
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_name}: {e}")
                        tool_results.append({"error": str(e)})

            # Get final response using tool results
            final_messages = messages.copy()
            if tool_results:
                final_messages.append(
                    {
                        "role": "assistant",
                        "content": "I've gathered the requested information.",
                    }
                )
                final_messages.append(
                    {
                        "role": "user",
                        "content": f"Here are the results: {json.dumps(tool_results)}",
                    }
                )

            final_response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview", messages=final_messages, temperature=0.1
            )

            final_message = final_response.choices[0].message.content

            # Store final response
            await self.memory.add_message("assistant", final_message)

            return {
                "response": final_message,
                "metadata": {
                    "tools_used": tools_used,
                    "timestamp": datetime.now().isoformat(),
                },
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise

    async def _compare_experiments(
        self, experiment_ids: List[str], metrics: List[str]
    ) -> Dict[str, Any]:
        """Compare multiple experiments"""
        return await self.data_manager.compare_experiments(experiment_ids, metrics)

    async def _get_experiment_details(self, experiment_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific experiment"""
        return await self.data_manager.get_experiment_details(experiment_id)
