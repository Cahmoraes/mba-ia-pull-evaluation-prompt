"""
Script para fazer pull de prompts do LangSmith Prompt Hub.

Este script:
1. Conecta ao LangSmith usando credenciais do .env
2. Faz pull dos prompts do Hub
3. Salva localmente em prompts/bug_to_user_story_v1.yml
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain import hub

from utils import check_env_vars, print_section_header, save_yaml

load_dotenv()

PROMPT_SOURCE = "leonanluppi/bug_to_user_story_v1"
OUTPUT_FILE = "prompts/bug_to_user_story_v1.yml"
RAW_OUTPUT_FILE = "prompts/raw_prompts.yml"


def pull_prompts_from_langsmith():
    """
    Faz pull do prompt do LangSmith Hub e retorna os dados estruturados.

    Returns:
        dict com dados do prompt ou None se erro
    """
    try:
        print(f"   Puxando prompt: {PROMPT_SOURCE}")
        prompt = hub.pull(PROMPT_SOURCE)
        print(f"   ✓ Prompt carregado com sucesso")

        # Extrair dados do prompt
        prompt_data = {
            "bug_to_user_story_v1": {
                "description": "Prompt para converter relatos de bugs em User Stories",
                "system_prompt": "",
                "user_prompt": "{bug_report}",
                "version": "v1",
                "created_at": "2025-01-15",
                "tags": ["bug-analysis", "user-story", "product-management"],
            }
        }

        # Extrair mensagens do ChatPromptTemplate
        if hasattr(prompt, "messages"):
            for msg in prompt.messages:
                msg_content = (
                    msg.prompt.template if hasattr(msg, "prompt") else str(msg)
                )
                if hasattr(msg, "__class__"):
                    class_name = msg.__class__.__name__.lower()
                    if "system" in class_name:
                        prompt_data["bug_to_user_story_v1"]["system_prompt"] = (
                            msg_content
                        )
                    elif "human" in class_name:
                        prompt_data["bug_to_user_story_v1"]["user_prompt"] = msg_content
        elif hasattr(prompt, "template"):
            prompt_data["bug_to_user_story_v1"]["system_prompt"] = prompt.template

        return prompt_data

    except Exception as e:
        print(f"   ❌ Erro ao fazer pull: {e}")
        return None


def main():
    """Função principal"""
    print_section_header("PULL DE PROMPTS DO LANGSMITH HUB")

    required_vars = ["LANGSMITH_API_KEY"]
    if not check_env_vars(required_vars):
        return 1

    prompt_data = pull_prompts_from_langsmith()

    if prompt_data is None:
        print("❌ Falha ao fazer pull dos prompts")
        return 1

    # Salvar em ambos os arquivos
    for output_path in [OUTPUT_FILE, RAW_OUTPUT_FILE]:
        if save_yaml(prompt_data, output_path):
            print(f"   ✓ Salvo em: {output_path}")
        else:
            print(f"   ❌ Erro ao salvar: {output_path}")
            return 1

    print("\n✅ Pull concluído com sucesso!")
    print(f"\nPróximos passos:")
    print(f"  1. Analise o prompt em {OUTPUT_FILE}")
    print(f"  2. Crie o prompt otimizado em prompts/bug_to_user_story_v2.yml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
