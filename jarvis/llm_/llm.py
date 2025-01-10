from abc import ABC, abstractmethod
from langchain_community.llms.ollama import Ollama
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate



PROMPT_TEMPLATE = """
Baseado **apenas** no seguinte contexto:

{context}

---

Responda à seguinte pergunta: {question}

### Regras:
1. Não inclua informações que não estejam no contexto fornecido.
2. Se a resposta não puder ser dada, diga: "Não foi possível encontrar uma resposta no contexto fornecido."
3. Use um estilo claro e direto.


Responda agora:
"""

FOLLOW_UP_QUESTIONS_TEMPLATE = """
Baseado no seguinte contexto:

{context}

---

Sugira 2 perguntas relevantes de acompanhamento baseadas na questão e na resposta fornecidas acima:

Questão: {question}

A resposta de apenas incluir as questoes nada mais 
Perguntas de Acompanhamento:
"""
class LLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def invoke(self, prompt: str) -> str:
        pass

    def generate_response(self, context: str, question: str) -> str:
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context, question=question)
        return self.invoke(prompt)

    def generate_follow_up_questions(self, context: str, question: str) -> str:
        follow_up_template = ChatPromptTemplate.from_template(FOLLOW_UP_QUESTIONS_TEMPLATE)
        prompt = follow_up_template.format(context=context, question=question)
        return self.invoke(prompt)


class OllamaModel(LLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = Ollama(model=model_name)

    def invoke(self, prompt: str) -> str:
        return self.model.invoke(prompt)


class GPTModel(LLM):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.client = OpenAI(api_key=api_key)

    def invoke(self, prompt: str) -> str:
        messages = [
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
