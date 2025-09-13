from fastapi import APIRouter, Body, HTTPException
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from agents.orchestrator.agent import OrchestratorAgent
import json
import logging

router = APIRouter()

APP_NAME = "fact-check-extension"
USER_ID = "user-001"
session_service = InMemorySessionService()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_orchestrator(query: str):
    try:
        session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
        runner = Runner(agent=OrchestratorAgent(), app_name=APP_NAME, session_service=session_service)

        content = Content(role="user", parts=[Part(text=query)])
        final_text = None

        async for event in runner.run_async(user_id=USER_ID, session_id=session.id, new_message=content):
            if event.is_final_response():
                final_text = event.content.parts[0].text
                logger.info(f"Final response received: {final_text}")

        return final_text or "No response generated"
        
    except Exception as e:
        logger.error(f"Error in run_orchestrator: {str(e)}")
        raise e

def process_agent_response(response_text: str) -> dict:
    if not response_text:
        return {"error": "No response from agent"}
    
    if response_text.strip().startswith('{') and response_text.strip().endswith('}'):
        try:
            parsed = json.loads(response_text)
            
            if isinstance(parsed, dict):
                if "name" in parsed and parsed.get("name") == "transfer_to_agent":
                    return {"error": "Agent transfer not properly handled", "raw_response": parsed}
                
                if "result" in parsed:
                    result = parsed["result"]
                    if isinstance(result, str) and result.startswith('{'):
                        try:
                            return {"result": json.loads(result)}
                        except json.JSONDecodeError:
                            return {"result": result}
                    return {"result": result}
                
                return parsed
            
        except json.JSONDecodeError:
            pass
    
    return {"result": response_text}

@router.post("/process")
async def process_query(query: str = Body(..., embed=True)):
    print("HIT ENDPOINT")
    try:
        logger.info(f"Processing query: {query}")
        raw_response = await run_orchestrator(query)
        logger.info(f"Raw response: {raw_response}")
        processed_response = process_agent_response(raw_response)
        logger.info(f"Processed response: {processed_response}")
        return processed_response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/process-text")
async def process_query_text_only(query: str = Body(..., embed=True)):
    try:
        response = await run_orchestrator(query)
        
        if response and response.strip().startswith('{'):
            try:
                parsed = json.loads(response)
                
                if isinstance(parsed, dict):
                    for field in ['answer', 'result', 'content', 'text', 'response']:
                        if field in parsed:
                            return {"result": parsed[field]}
                    
                    return {"result": str(parsed)}
                    
            except json.JSONDecodeError:
                pass
        
        return {"result": response}
        
    except Exception as e:
        logger.error(f"Error in process_query_text_only: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/debug")
async def debug_query(query: str = Body(..., embed=True)):
    try:
        response = await run_orchestrator(query)
        
        return {
            "query": query,
            "raw_response": response,
            "response_type": type(response).__name__,
            "is_json": response.strip().startswith('{') if response else False,
            "length": len(response) if response else 0
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "query": query
        }

@router.post("/simple")
async def simple_process(query: str = Body(..., embed=True)):
    try:
        raw_response = await run_orchestrator(query)
        print(f"DEBUG - Raw response: {raw_response}")
        print(f"DEBUG - Response type: {type(raw_response)}")
        
        if isinstance(raw_response, str):
            try:
                if raw_response.strip().startswith('{'):
                    parsed = json.loads(raw_response)
                    return parsed
            except:
                pass
            
            return {"answer": raw_response}
        
        return {"answer": str(raw_response)}
        
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}
