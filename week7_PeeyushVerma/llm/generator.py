import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from utils.logger import get_logger
from utils.metrics import track_latency
from config import Config

logger = get_logger(__name__)

class LLMGenerator:
    def __init__(self):
        self.model_name = Config.LLM_MODEL
        self.api_key = Config.GROQ_API_KEY
        self.llm = None
        
        if self.api_key:
            try:
                self.llm = ChatGroq(
                    temperature=0.0, 
                    groq_api_key=self.api_key, 
                    model_name=self.model_name
                )
                logger.info(f"Initialized Groq LLM: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Groq LLM: {e}")
        else:
            logger.warning("GROQ_API_KEY not found in environment variables.")

        # Define the prompt template according to requirements
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Context:
{context}

Question:
{question}

Instructions:
Answer ONLY using the provided context.
If the answer is unavailable, reply:
"I couldn't find that information in the uploaded documents."
"""
        )

    @track_latency("generation")
    def generate_answer(self, query, retrieved_chunks):
        """Generate an answer using the LLM based on retrieved chunks."""
        if not self.llm:
            return "Error: LLM not initialized. Check your GROQ_API_KEY."

        if not retrieved_chunks:
            return "I couldn't find that information in the uploaded documents."

        # Extract text from chunks (handling FAISS return format which is tuple of (doc, score))
        context_texts = []
        for item in retrieved_chunks:
            if isinstance(item, tuple):
                doc, _ = item
                context_texts.append(doc.page_content)
            else:
                context_texts.append(item.page_content)
                
        context_str = "\n\n---\n\n".join(context_texts)
        
        try:
            prompt = self.prompt_template.format(context=context_str, question=query)
            logger.info(f"Sending prompt to LLM (Context length: {len(context_str)})")
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error during generation: {str(e)}"
