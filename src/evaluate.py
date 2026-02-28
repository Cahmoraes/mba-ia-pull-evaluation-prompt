"""
Script COMPLETO para avaliar prompts otimizados.

Este script:
1. Carrega dataset de avaliação de arquivo .jsonl (datasets/bug_to_user_story.jsonl)
2. Cria/atualiza dataset no LangSmith
3. Puxa prompts otimizados do LangSmith Hub (fonte única de verdade)
4. Executa prompts contra o dataset
5. Calcula 5 métricas (Helpfulness, Correctness, F1-Score, Clarity, Precision)
6. Publica resultados no dashboard do LangSmith
7. Exibe resumo no terminal

Suporta múltiplos providers de LLM:
- OpenAI (gpt-4o, gpt-4o-mini)
- Google Gemini (gemini-1.5-flash, gemini-1.5-pro)

Configure o provider no arquivo .env através da variável LLM_PROVIDER.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client

from metrics import (
    evaluate_acceptance_criteria_score,
    evaluate_completeness_score,
    evaluate_tone_score,
    evaluate_user_story_format_score,
)
from utils import check_env_vars, format_score, print_section_header
from utils import get_llm as get_configured_llm

load_dotenv()


def get_llm():
    return get_configured_llm(temperature=0)


def load_dataset_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    examples = []

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # Ignorar linhas vazias
                    example = json.loads(line)
                    examples.append(example)

        return examples

    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {jsonl_path}")
        print(
            "\nCertifique-se de que o arquivo datasets/bug_to_user_story.jsonl existe."
        )
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Erro ao parsear JSONL: {e}")
        return []
    except Exception as e:
        print(f"❌ Erro ao carregar dataset: {e}")
        return []


def create_evaluation_dataset(
    client: Client, dataset_name: str, jsonl_path: str
) -> str:
    print(f"Criando dataset de avaliação: {dataset_name}...")

    examples = load_dataset_from_jsonl(jsonl_path)

    if not examples:
        print("❌ Nenhum exemplo carregado do arquivo .jsonl")
        return dataset_name

    print(f"   ✓ Carregados {len(examples)} exemplos do arquivo {jsonl_path}")

    try:
        datasets = client.list_datasets(dataset_name=dataset_name)
        existing_dataset = None

        for ds in datasets:
            if ds.name == dataset_name:
                existing_dataset = ds
                break

        if existing_dataset:
            print(f"   ✓ Dataset '{dataset_name}' já existe, usando existente")
            return dataset_name
        else:
            dataset = client.create_dataset(dataset_name=dataset_name)

            for example in examples:
                client.create_example(
                    dataset_id=dataset.id,
                    inputs=example["inputs"],
                    outputs=example["outputs"],
                )

            print(f"   ✓ Dataset criado com {len(examples)} exemplos")
            return dataset_name

    except Exception as e:
        print(f"   ⚠️  Erro ao criar dataset: {e}")
        return dataset_name


def pull_prompt_from_langsmith(prompt_name: str) -> ChatPromptTemplate:
    try:
        print(f"   Puxando prompt do LangSmith Hub: {prompt_name}")
        prompt = hub.pull(prompt_name)
        print(f"   ✓ Prompt carregado com sucesso")
        return prompt

    except Exception as e:
        error_msg = str(e).lower()

        print(f"\n{'=' * 70}")
        print(f"❌ ERRO: Não foi possível carregar o prompt '{prompt_name}'")
        print(f"{'=' * 70}\n")

        if "not found" in error_msg or "404" in error_msg:
            print("⚠️  O prompt não foi encontrado no LangSmith Hub.\n")
            print("AÇÕES NECESSÁRIAS:")
            print("1. Verifique se você já fez push do prompt otimizado:")
            print(f"   python src/push_prompts.py")
            print()
            print("2. Confirme se o prompt foi publicado com sucesso em:")
            print(f"   https://smith.langchain.com/prompts")
            print()
            print(
                f"3. Certifique-se de que o nome do prompt está correto: '{prompt_name}'"
            )
            print()
            print("4. Se você alterou o prompt no YAML, refaça o push:")
            print(f"   python src/push_prompts.py")
        else:
            print(f"Erro técnico: {e}\n")
            print("Verifique:")
            print("- LANGSMITH_API_KEY está configurada corretamente no .env")
            print("- Você tem acesso ao workspace do LangSmith")
            print("- Sua conexão com a internet está funcionando")

        print(f"\n{'=' * 70}\n")
        raise


def evaluate_prompt_on_example(
    prompt_template: ChatPromptTemplate, example: Any, llm: Any
) -> Dict[str, Any]:
    try:
        inputs = example.inputs if hasattr(example, "inputs") else {}
        outputs = example.outputs if hasattr(example, "outputs") else {}

        chain = prompt_template | llm

        response = chain.invoke(inputs)
        answer = response.content

        reference = outputs.get("reference", "") if isinstance(outputs, dict) else ""

        if isinstance(inputs, dict):
            question = inputs.get(
                "question", inputs.get("bug_report", inputs.get("pr_title", "N/A"))
            )
        else:
            question = "N/A"

        return {"answer": answer, "reference": reference, "question": question}

    except Exception as e:
        print(f"      ⚠️  Erro ao avaliar exemplo: {e}")
        import traceback

        print(f"      Traceback: {traceback.format_exc()}")
        return {"answer": "", "reference": "", "question": ""}


def evaluate_prompt(
    prompt_name: str, dataset_name: str, client: Client
) -> Dict[str, float]:
    print(f"\n🔍 Avaliando: {prompt_name}")

    try:
        prompt_template = pull_prompt_from_langsmith(prompt_name)

        examples = list(client.list_examples(dataset_name=dataset_name))
        print(f"   Dataset: {len(examples)} exemplos")
        print(f"   Avaliando primeiros 3 exemplos (rate limiting)")

        llm = get_llm()

        tone_scores = []
        acceptance_scores = []
        format_scores = []
        completeness_scores = []

        print("   Avaliando exemplos...")

        for i, example in enumerate(examples[:3], 1):
            result = evaluate_prompt_on_example(prompt_template, example, llm)
            time.sleep(5)  # Rate limiting entre geração e avaliação

            if result["answer"]:
                time.sleep(8)  # Rate limiting para Gemini free tier
                tone = evaluate_tone_score(
                    result["question"], result["answer"], result["reference"]
                )
                time.sleep(8)
                acceptance = evaluate_acceptance_criteria_score(
                    result["question"], result["answer"], result["reference"]
                )
                time.sleep(8)
                fmt = evaluate_user_story_format_score(
                    result["question"], result["answer"], result["reference"]
                )
                time.sleep(8)
                completeness = evaluate_completeness_score(
                    result["question"], result["answer"], result["reference"]
                )

                tone_scores.append(tone["score"])
                acceptance_scores.append(acceptance["score"])
                format_scores.append(fmt["score"])
                completeness_scores.append(completeness["score"])

                print(
                    f"      [{i}/{min(3, len(examples))}] Tone:{tone['score']:.2f} Acceptance:{acceptance['score']:.2f} Format:{fmt['score']:.2f} Completeness:{completeness['score']:.2f}"
                )

        avg_tone = sum(tone_scores) / len(tone_scores) if tone_scores else 0.0
        avg_acceptance = (
            sum(acceptance_scores) / len(acceptance_scores)
            if acceptance_scores
            else 0.0
        )
        avg_format = sum(format_scores) / len(format_scores) if format_scores else 0.0
        avg_completeness = (
            sum(completeness_scores) / len(completeness_scores)
            if completeness_scores
            else 0.0
        )

        return {
            "tone_score": round(avg_tone, 4),
            "acceptance_criteria_score": round(avg_acceptance, 4),
            "user_story_format_score": round(avg_format, 4),
            "completeness_score": round(avg_completeness, 4),
        }

    except Exception as e:
        print(f"   ❌ Erro na avaliação: {e}")
        return {
            "tone_score": 0.0,
            "acceptance_criteria_score": 0.0,
            "user_story_format_score": 0.0,
            "completeness_score": 0.0,
        }


def display_results(prompt_name: str, scores: Dict[str, float]) -> bool:
    print("\n" + "=" * 50)
    print(f"Prompt: {prompt_name}")
    print("=" * 50)

    print("\nMétricas Específicas (Bug to User Story):")
    print(f"  - Tone Score: {format_score(scores['tone_score'], threshold=0.9)}")
    print(
        f"  - Acceptance Criteria Score: {format_score(scores['acceptance_criteria_score'], threshold=0.9)}"
    )
    print(
        f"  - User Story Format Score: {format_score(scores['user_story_format_score'], threshold=0.9)}"
    )
    print(
        f"  - Completeness Score: {format_score(scores['completeness_score'], threshold=0.9)}"
    )

    average_score = sum(scores.values()) / len(scores)

    # Verificar se TODAS as métricas >= 0.9
    all_above_threshold = all(v >= 0.9 for v in scores.values())

    print("\n" + "-" * 50)
    print(f"📊 MÉDIA GERAL: {average_score:.4f}")
    print("-" * 50)

    passed = all_above_threshold and average_score >= 0.9

    if passed:
        print(f"\n✅ STATUS: APROVADO ✓ - Todas as métricas atingiram o mínimo de 0.9")
    else:
        print(f"\n❌ STATUS: REPROVADO")
        print(f"⚠️  Média atual: {average_score:.4f} | Necessário: 0.9000")
        for key, val in scores.items():
            if val < 0.9:
                print(f"   ⚠️  {key}: {val:.4f} (abaixo de 0.9)")

    return passed


def main():
    print_section_header("AVALIAÇÃO DE PROMPTS OTIMIZADOS")

    provider = os.getenv("LLM_PROVIDER", "openai")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    eval_model = os.getenv("EVAL_MODEL", "gpt-4o")

    print(f"Provider: {provider}")
    print(f"Modelo Principal: {llm_model}")
    print(f"Modelo de Avaliação: {eval_model}\n")

    required_vars = ["LANGSMITH_API_KEY", "LLM_PROVIDER"]
    if provider == "openai":
        required_vars.append("OPENAI_API_KEY")
    elif provider in ["google", "gemini"]:
        required_vars.append("GOOGLE_API_KEY")

    if not check_env_vars(required_vars):
        return 1

    client = Client()
    project_name = os.getenv(
        "LANGCHAIN_PROJECT", "prompt-optimization-challenge-resolved"
    )

    jsonl_path = "datasets/bug_to_user_story.jsonl"

    if not Path(jsonl_path).exists():
        print(f"❌ Arquivo de dataset não encontrado: {jsonl_path}")
        print("\nCertifique-se de que o arquivo existe antes de continuar.")
        return 1

    dataset_name = f"{project_name}-eval"
    create_evaluation_dataset(client, dataset_name, jsonl_path)

    print("\n" + "=" * 70)
    print("PROMPTS PARA AVALIAR")
    print("=" * 70)
    print("\nEste script irá puxar prompts do LangSmith Hub.")
    print("Certifique-se de ter feito push dos prompts antes de avaliar:")
    print("  python src/push_prompts.py\n")

    username = os.getenv("USERNAME_LANGSMITH_HUB", "").strip()
    prompts_to_evaluate = [
        f"{username}/bug_to_user_story_v2" if username else "bug_to_user_story_v2",
    ]

    all_passed = True
    evaluated_count = 0
    results_summary = []

    for prompt_name in prompts_to_evaluate:
        evaluated_count += 1

        try:
            scores = evaluate_prompt(prompt_name, dataset_name, client)

            passed = display_results(prompt_name, scores)
            all_passed = all_passed and passed

            results_summary.append(
                {"prompt": prompt_name, "scores": scores, "passed": passed}
            )

        except Exception as e:
            print(f"\n❌ Falha ao avaliar '{prompt_name}': {e}")
            all_passed = False

            results_summary.append(
                {
                    "prompt": prompt_name,
                    "scores": {
                        "tone_score": 0.0,
                        "acceptance_criteria_score": 0.0,
                        "user_story_format_score": 0.0,
                        "completeness_score": 0.0,
                    },
                    "passed": False,
                }
            )

    print("\n" + "=" * 50)
    print("RESUMO FINAL")
    print("=" * 50 + "\n")

    if evaluated_count == 0:
        print("⚠️  Nenhum prompt foi avaliado")
        return 1

    print(f"Prompts avaliados: {evaluated_count}")
    print(f"Aprovados: {sum(1 for r in results_summary if r['passed'])}")
    print(f"Reprovados: {sum(1 for r in results_summary if not r['passed'])}\n")

    if all_passed:
        print("✅ Todos os prompts atingiram média >= 0.9!")
        print(f"\n✓ Confira os resultados em:")
        print(f"  https://smith.langchain.com/projects/{project_name}")
        print("\nPróximos passos:")
        print("1. Documente o processo no README.md")
        print("2. Capture screenshots das avaliações")
        print("3. Faça commit e push para o GitHub")
        return 0
    else:
        print("⚠️  Alguns prompts não atingiram média >= 0.9")
        print("\nPróximos passos:")
        print("1. Refatore os prompts com score baixo")
        print("2. Faça push novamente: python src/push_prompts.py")
        print("3. Execute: python src/evaluate.py novamente")
        return 1


if __name__ == "__main__":
    sys.exit(main())
