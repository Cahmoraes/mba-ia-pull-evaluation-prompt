# Pull, Otimização e Avaliação de Prompts com LangChain e LangSmith

## Objetivo

Software capaz de:

1. **Fazer pull de prompts** do LangSmith Prompt Hub contendo prompts de baixa qualidade
2. **Refatorar e otimizar** esses prompts usando técnicas avançadas de Prompt Engineering
3. **Fazer push dos prompts otimizados** de volta ao LangSmith
4. **Avaliar a qualidade** através de métricas customizadas (Tone, Acceptance Criteria, User Story Format, Completeness)
5. **Atingir pontuação mínima** de 0.9 (90%) em todas as métricas de avaliação

---

## Técnicas Aplicadas (Fase 2)

### 1. Role Prompting

**Por que escolhi:** Definir uma persona clara e especializada garante que o LLM gere respostas consistentes e com o tom profissional adequado para documentação. Um "Senior Product Manager e Agile Coach" tem o contexto necessário para transformar bugs técnicos em User Stories centradas no usuário.

**Como apliquei:**
- Defini a persona como "Senior Product Manager e Agile Coach com mais de 15 anos de experiência"
- Especifiquei a especialidade em metodologias Scrum/Kanban
- Estabeleci o tom profissional, empático e orientado a valor de negócio

### 2. Few-shot Learning

**Por que escolhi:** Fornecer exemplos concretos de entrada/saída é a forma mais efetiva de "ensinar" o modelo o formato exato esperado. Isso reduz ambiguidade e garante consistência na estrutura das User Stories geradas.

**Como apliquei:**
- Incluí 3 exemplos completos cobrindo diferentes níveis de complexidade:
  - **Exemplo 1 (Bug Simples):** Mostra formato básico com User Story + Critérios de Aceitação
  - **Exemplo 2 (Bug Médio):** Demonstra como incluir Contexto Técnico e exemplos de cálculo
  - **Exemplo 3 (Bug Complexo):** Mostra o formato completo com seções (=== SEÇÃO ===), múltiplas categorias de critérios, tasks técnicas sugeridas

### 3. Chain of Thought (CoT)

**Por que escolhi:** A conversão de bug para User Story requer raciocínio em múltiplas etapas — analisar o problema, identificar a persona, formular positivamente, articular valor. O CoT guia o modelo a seguir essa sequência lógica.

**Como apliquei:**
- Defini 8 passos sequenciais de raciocínio:
  1. Analisar o Bug
  2. Classificar a Complexidade
  3. Identificar a Persona
  4. Formular a Necessidade Positivamente
  5. Articular o Valor de Negócio
  6. Escrever os Critérios de Aceitação
  7. Adicionar Contexto Técnico
  8. Para bugs complexos: Sugerir tasks técnicas

---

## Resultados Finais

### Critérios de Aprovação

| Métrica | Mínimo | Status |
|---------|--------|--------|
| Tone Score | >= 0.9 | ✅ |
| Acceptance Criteria Score | >= 0.9 | ✅ |
| User Story Format Score | >= 0.9 | ✅ |
| Completeness Score | >= 0.9 | ✅ |
| **MÉDIA** | **>= 0.9** | ✅ |

### Comparativo v1 vs v2

| Métrica | v1 (antes) | v2 (depois) |
|---------|-----------|-------------|
| Tone Score | ~0.5 | >= 0.9 |
| Acceptance Criteria Score | ~0.4 | >= 0.9 |
| User Story Format Score | ~0.5 | >= 0.9 |
| Completeness Score | ~0.4 | >= 0.9 |

---

## Como Executar

### Pré-requisitos

- **Python** 3.9+
- **uv** (gerenciador de pacotes Python) ou pip
- Conta no **LangSmith** com API Key
- **API Key do Google** (Gemini) ou OpenAI

### Instalação

```bash
# Clonar repositório
git clone <url-do-repo>
cd mba-ia-pull-evaluation-prompt

# Criar ambiente virtual
uv venv venv
source venv/bin/activate

# Instalar dependências
uv pip install -r requirements.txt
```

### Configuração

1. Copie o `.env.example` para `.env`:
```bash
cp .env.example .env
```

2. Configure no `.env`:
```env
LANGSMITH_API_KEY=sua_chave_langsmith
GOOGLE_API_KEY=sua_chave_google
USERNAME_LANGSMITH_HUB=seu_username
LLM_PROVIDER=google
LLM_MODEL=gemini-2.5-flash
EVAL_MODEL=gemini-2.5-flash
```

### Execução

```bash
# 1. Pull dos prompts iniciais (v1)
python src/pull_prompts.py

# 2. Push do prompt otimizado (v2) para o LangSmith Hub
python src/push_prompts.py

# 3. Avaliação automática
python src/evaluate.py

# 4. Executar testes
pytest tests/test_prompts.py -v
```

---

## Tecnologias

- **Python 3.9+**
- **LangChain** + **LangSmith**
- **Google Gemini** (gemini-2.5-flash)
- **pytest** para testes
