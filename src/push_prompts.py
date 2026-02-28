"""
Script para fazer push de prompts otimizados ao LangSmith Prompt Hub.

Este script:
1. Lê os prompts otimizados de prompts/bug_to_user_story_v2.yml
2. Valida os prompts
3. Faz push PÚBLICO para o LangSmith Hub
4. Adiciona metadados (tags, descrição, técnicas utilizadas)
"""

import os
import sys

from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate

from utils import check_env_vars, load_yaml, print_section_header

load_dotenv()

PROMPT_FILE = "prompts/bug_to_user_story_v2.yml"
PROMPT_KEY = "bug_to_user_story_v2"


def push_prompt_to_langsmith(prompt_name: str, prompt_data: dict) -> bool:
    """
    Faz push do prompt otimizado para o LangSmith Hub (PÚBLICO).

    Args:
        prompt_name: Nome do prompt
        prompt_data: Dados do prompt

    Returns:
        True se sucesso, False caso contrário
    """
    try:
        system_prompt = prompt_data.get("system_prompt", "").strip()
        user_prompt = prompt_data.get("user_prompt", "{bug_report}").strip()

        # Criar ChatPromptTemplate com system e human messages
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", user_prompt),
            ]
        )

        # Determinar se é push público ou privado
        username = os.getenv("USERNAME_LANGSMITH_HUB", "").strip()
        if username:
            full_name = f"{username}/{prompt_name}"
        else:
            full_name = prompt_name

        description = prompt_data.get(
            "description", "Prompt otimizado para Bug to User Story"
        )
        tags = prompt_data.get("tags", [])

        # Se não tem handle, push privado primeiro
        is_public = bool(username)

        print(f"   Fazendo push: {full_name}")
        print(f"   Público: {'Sim' if is_public else 'Não (sem handle configurado)'}")
        print(f"   Descrição: {description}")
        print(f"   Tags: {', '.join(tags)}")

        hub.push(
            full_name,
            prompt_template,
            new_repo_is_public=is_public,
            new_repo_description=description,
        )

        print(f"   ✓ Push realizado com sucesso!")
        print(f"   🔗 Verifique em: https://smith.langchain.com/hub/{full_name}")
        return True

    except Exception as e:
        print(f"   ❌ Erro ao fazer push: {e}")
        import traceback

        traceback.print_exc()
        return False


def validate_prompt(prompt_data: dict) -> tuple[bool, list]:
    """
    Valida estrutura básica de um prompt (versão simplificada).

    Args:
        prompt_data: Dados do prompt

    Returns:
        (is_valid, errors) - Tupla com status e lista de erros
    """
    errors = []

    required_fields = ["description", "system_prompt", "version"]
    for field in required_fields:
        if field not in prompt_data:
            errors.append(f"Campo obrigatório faltando: {field}")

    system_prompt = prompt_data.get("system_prompt", "").strip()
    if not system_prompt:
        errors.append("system_prompt está vazio")

    if "[TODO]" in system_prompt:
        errors.append("system_prompt ainda contém [TODO]")

    techniques = prompt_data.get("techniques_applied", [])
    if len(techniques) < 2:
        errors.append(
            f"Mínimo de 2 técnicas requeridas, encontradas: {len(techniques)}"
        )

    return (len(errors) == 0, errors)


def main():
    """Função principal"""
    print_section_header("PUSH DE PROMPTS OTIMIZADOS PARA LANGSMITH HUB")

    required_vars = ["LANGSMITH_API_KEY"]
    if not check_env_vars(required_vars):
        return 1

    # Carregar prompt do YAML
    print(f"Carregando prompt de: {PROMPT_FILE}")
    data = load_yaml(PROMPT_FILE)
    if data is None:
        print(f"❌ Não foi possível carregar {PROMPT_FILE}")
        return 1

    prompt_data = data.get(PROMPT_KEY)
    if prompt_data is None:
        print(f"❌ Chave '{PROMPT_KEY}' não encontrada no YAML")
        return 1

    # Validar prompt
    print("\nValidando prompt...")
    is_valid, errors = validate_prompt(prompt_data)

    if not is_valid:
        print("❌ Prompt com erros de validação:")
        for error in errors:
            print(f"   - {error}")
        return 1

    print("   ✓ Prompt válido")

    # Push para LangSmith
    print(f"\nFazendo push do prompt...")
    success = push_prompt_to_langsmith(PROMPT_KEY, prompt_data)

    if success:
        print("\n✅ Push concluído com sucesso!")
        print("\nPróximos passos:")
        print("  1. Verifique o prompt no dashboard do LangSmith")
        print("  2. Execute a avaliação: python src/evaluate.py")
        return 0
    else:
        print("\n❌ Falha no push")
        return 1


if __name__ == "__main__":
    sys.exit(main())
