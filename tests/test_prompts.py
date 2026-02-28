"""
Testes automatizados para validação de prompts.
"""

import sys
from pathlib import Path

import pytest
import yaml

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import validate_prompt_structure

PROMPT_V2_PATH = str(
    Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"
)


def load_prompts(file_path: str):
    """Carrega prompts do arquivo YAML."""
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def prompt_data():
    """Fixture que carrega e retorna os dados do prompt v2."""
    data = load_prompts(PROMPT_V2_PATH)
    return data["bug_to_user_story_v2"]


@pytest.fixture
def system_prompt(prompt_data):
    """Fixture que retorna o system_prompt como string."""
    return prompt_data.get("system_prompt", "")


class TestPrompts:
    def test_prompt_has_system_prompt(self, prompt_data):
        """Verifica se o campo 'system_prompt' existe e não está vazio."""
        assert "system_prompt" in prompt_data, (
            "Campo 'system_prompt' não encontrado no YAML"
        )
        system = prompt_data["system_prompt"].strip()
        assert len(system) > 0, "system_prompt está vazio"

    def test_prompt_has_role_definition(self, system_prompt):
        """Verifica se o prompt define uma persona (ex: 'Você é um Product Manager')."""
        lower = system_prompt.lower()
        role_keywords = [
            "você é",
            "voce é",
            "você é um",
            "product manager",
            "agile coach",
            "senior",
            "especialista",
        ]
        has_role = any(kw in lower for kw in role_keywords)
        assert has_role, (
            "O prompt não define uma persona/role. "
            "Deve conter expressões como 'Você é um Product Manager'."
        )

    def test_prompt_mentions_format(self, system_prompt):
        """Verifica se o prompt exige formato Markdown ou User Story padrão."""
        lower = system_prompt.lower()
        format_keywords = [
            "como um",
            "eu quero",
            "para que",
            "user story",
            "markdown",
            "given-when-then",
            "dado-quando-então",
            "critérios de aceitação",
            "dado que",
        ]
        has_format = any(kw in lower for kw in format_keywords)
        assert has_format, (
            "O prompt não menciona formato Markdown ou User Story padrão. "
            "Deve conter referências ao formato 'Como um... eu quero... para que...' "
            "ou critérios de aceitação."
        )

    def test_prompt_has_few_shot_examples(self, system_prompt):
        """Verifica se o prompt contém exemplos de entrada/saída (técnica Few-shot)."""
        lower = system_prompt.lower()
        # Verificar presença de exemplos
        example_indicators = [
            "exemplo",
            "example",
            "entrada",
            "saída",
            "bug report",
            "few-shot",
        ]
        has_examples = any(kw in lower for kw in example_indicators)
        assert has_examples, (
            "O prompt não contém exemplos de entrada/saída (Few-shot). "
            "Deve incluir pelo menos 2 exemplos de conversão bug → user story."
        )

        # Verificar que há pelo menos 2 exemplos
        example_count = lower.count("exemplo")
        assert example_count >= 2, (
            f"Encontrados apenas {example_count} exemplos. "
            f"Mínimo recomendado: 2 exemplos de Few-shot Learning."
        )

    def test_prompt_no_todos(self, system_prompt):
        """Garante que você não esqueceu nenhum `[TODO]` no texto."""
        assert "[TODO]" not in system_prompt, (
            "O prompt ainda contém marcações [TODO]. "
            "Remova todos os TODOs antes de usar o prompt."
        )
        assert "[todo]" not in system_prompt.lower(), (
            "O prompt contém variações de [TODO]. Remova todas."
        )

    def test_minimum_techniques(self, prompt_data):
        """Verifica (através dos metadados do yaml) se pelo menos 2 técnicas foram listadas."""
        techniques = prompt_data.get("techniques_applied", [])
        assert len(techniques) >= 2, (
            f"Mínimo de 2 técnicas requeridas no campo 'techniques_applied'. "
            f"Encontradas: {len(techniques)} ({techniques})"
        )

    def test_prompt_has_user_prompt(self, prompt_data):
        """Verifica se o campo 'user_prompt' existe e contém a variável {bug_report}."""
        assert "user_prompt" in prompt_data, "Campo 'user_prompt' não encontrado"
        user = prompt_data["user_prompt"]
        assert "{bug_report}" in user, "user_prompt deve conter {bug_report}"

    def test_prompt_structure_valid(self, prompt_data):
        """Verifica se a estrutura do prompt é válida usando a função utilitária."""
        is_valid, errors = validate_prompt_structure(prompt_data)
        assert is_valid, f"Estrutura do prompt inválida: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
