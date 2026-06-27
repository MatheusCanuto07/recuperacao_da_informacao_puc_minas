"""
Protótipo didático para processamento de documentos e recuperação de informação.
Tema: Processamento de documentos
Disciplina: Recuperação de Informação na Web e Redes Sociais

O script executa:
1. leitura de documentos;
2. normalização;
3. tokenização;
4. remoção de stopwords;
5. stemming simplificado;
6. construção de índice invertido;
7. busca por TF-IDF e similaridade do cosseno.
"""

# "recuperação"  → remove -acao  → "recuper"
# "recuperar"    → remove -ar    → "recuper"
# "recuperado"   → remove -ado   → "recuper"
# "recuperações" → remove -acoes → "recuper"

# re — Expressões Regulares
# Usada na tokenização. O padrão [a-zA-Z0-9]+ extrai sequências de letras e números, descartando automaticamente pontuação, vírgulas, aspas e qualquer outro caractere que não seja palavra. É a forma mais precisa e concisa de fazer isso em Python.
# math
# Usada em dois momentos: math.log para o cálculo do IDF, e math.sqrt para calcular a norma dos vetores na similaridade do cosseno. São operações matemáticas puras que não justificam importar nenhuma biblioteca externa.
# collections

# Duas estruturas são usadas:
# Counter — conta a frequência de cada token numa lista com uma linha de código. É a forma que o python tem de calcular o TF.
# defaultdict — cria o índice invertido sem precisar verificar se a chave já existe antes de inserir. Evita código defensivo desnecessário.
# unicodedata
# Usada para remover acentos. O processo é: normalizar o texto no formato NFD (que separa a letra base do acento como caracteres distintos) e depois filtrar todos os caracteres da categoria Mn (Mark, Nonspacing — que são exatamente os acentos). É o método correto e robusto para fazer isso sem depender de substituições manuais letra por letra.


from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Se alguém pesquisa "o que é indexação", a intenção está em "indexação". O "o", o "que" e o "é" não dizem nada sobre o que a pessoa quer encontrar.
# Não faz muito sentido procurar 
STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas",
    "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas",
    "por", "para", "com", "sem", "sobre", "entre", "e", "ou", "que",
    "se", "ao", "aos", "mais", "menos", "como", "ser", "sao", "foi",
    "foram", "esse", "essa", "este", "esta", "isso", "tambem", "muito"
}

SUFIXOS = [
    "amentos", "imento", "imentos", "amento", "mente", "idades", "idade",
    "acoes", "acao", "sao", "coes", "logias", "logia", "ados", "adas",
    "ido", "ida", "ar", "er", "ir", "s"
]

DOCUMENTOS_EXEMPLO = {
    "doc1.txt": "O processamento de documentos transforma textos em termos de indexação para recuperação de informação.",
    "doc2.txt": "Redes sociais geram grandes volumes de documentos, mensagens e dados textuais para análise.",
    "doc3.txt": "A eliminação de stopwords e o stemming reduzem ruídos e melhoram a estrutura do índice invertido.",
}


def normalizar(texto: str) -> str:
    """Converte o texto para minúsculas e remove acentos."""
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def tokenizar(texto: str) -> List[str]:
    """Extrai tokens alfanuméricos do texto normalizado."""
    texto = normalizar(texto)
    return re.findall(r"[a-zA-Z0-9]+", texto)


def aplicar_stemming_simples(token: str) -> str:
    """Aplica uma remoção simples de sufixos para fins didáticos."""
    for sufixo in SUFIXOS:
        if token.endswith(sufixo) and len(token) > len(sufixo) + 3:
            return token[: -len(sufixo)]
    return token


def processar(texto: str) -> List[str]:
    """Executa tokenização, remoção de stopwords e stemming."""
    tokens = tokenizar(texto)
    tokens_filtrados = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return [aplicar_stemming_simples(t) for t in tokens_filtrados]


def carregar_documentos(pasta: Path) -> Dict[str, str]:
    """Carrega arquivos .txt da pasta. Se não houver arquivos, cria exemplos."""
    pasta.mkdir(exist_ok=True)
    if not list(pasta.glob("*.txt")):
        for nome, conteudo in DOCUMENTOS_EXEMPLO.items():
            (pasta / nome).write_text(conteudo, encoding="utf-8")

    return {arquivo.name: arquivo.read_text(encoding="utf-8") for arquivo in pasta.glob("*.txt")}


def construir_indice(documentos_processados: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    """Cria índice invertido: termo -> {documento: frequência}."""
    indice: Dict[str, Dict[str, int]] = defaultdict(dict)
    for nome_doc, tokens in documentos_processados.items():
        frequencias = Counter(tokens)
        for termo, freq in frequencias.items():
            indice[termo][nome_doc] = freq
    return dict(indice)


def calcular_idf(indice: Dict[str, Dict[str, int]], total_docs: int) -> Dict[str, float]:
    """Calcula IDF suavizado para cada termo."""
    return {
        termo: math.log((1 + total_docs) / (1 + len(postagens))) + 1
        for termo, postagens in indice.items()
    }


def vetor_tfidf(tokens: Iterable[str], idf: Dict[str, float]) -> Dict[str, float]:
    """Converte uma lista de tokens em vetor TF-IDF."""
    frequencias = Counter(tokens)
    total = sum(frequencias.values()) or 1
    return {
        termo: (freq / total) * idf.get(termo, 0.0)
        for termo, freq in frequencias.items()
    }


def similaridade_cosseno(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    """Calcula similaridade do cosseno entre dois vetores esparsos."""
    termos = set(v1) | set(v2)
    numerador = sum(v1.get(t, 0.0) * v2.get(t, 0.0) for t in termos)
    norma1 = math.sqrt(sum(valor * valor for valor in v1.values()))
    norma2 = math.sqrt(sum(valor * valor for valor in v2.values()))
    if norma1 == 0 or norma2 == 0:
        return 0.0
    return numerador / (norma1 * norma2)


def buscar(consulta: str, documentos_processados: Dict[str, List[str]], idf: Dict[str, float]) -> List[Tuple[str, float]]:
    """Processa a consulta e retorna documentos ranqueados."""
    vetor_consulta = vetor_tfidf(processar(consulta), idf)
    resultados = []

    for nome_doc, tokens in documentos_processados.items():
        vetor_doc = vetor_tfidf(tokens, idf)
        score = similaridade_cosseno(vetor_consulta, vetor_doc)
        if score > 0:
            resultados.append((nome_doc, score))

    return sorted(resultados, key=lambda item: item[1], reverse=True)


def main() -> None:
    pasta = Path("documentos")
    documentos = carregar_documentos(pasta)
    documentos_processados = {nome: processar(texto) for nome, texto in documentos.items()}

    indice = construir_indice(documentos_processados)
    idf = calcular_idf(indice, total_docs=len(documentos))

    print("Documentos processados:")
    for nome, tokens in documentos_processados.items():
        print(f"- {nome}: {tokens}")

    print("\nTermos do índice invertido:")
    for termo, postagens in sorted(indice.items()):
        print(f"- {termo}: {postagens}")

    consulta = input("\nDigite uma consulta: ").strip()
    resultados = buscar(consulta, documentos_processados, idf)

    print("\nResultados ranqueados:")
    if not resultados:
        print("Nenhum documento recuperado.")
        return

    for posicao, (documento, score) in enumerate(resultados, start=1):
        print(f"{posicao}. {documento} - score={score:.4f}")


if __name__ == "__main__":
    main()

# Vc tem um artigo gigante de 20 páginas e nele aparece a palavra documento 10 vezes, mas vc tem um artigo de 1 página que aparece a palavra documento 4 vezes, qual vai aparecer primeiro na busca? O artigo de 1 página, porque a frequência relativa é maior. A frequência relativa é a frequência do termo dividido pelo total de termos no documento.

# Ele percorre o texto inteiro e extrai todos os pedaços que contenham apenas letras de a a z, A a Z, ou dígitos de 0 a 9. Qualquer coisa fora disso — vírgula, ponto, espaço, hífen, aspas, exclamação — serve como separador e é descartada.

# O sistema procura exatamente a string "recuperação" no índice. Se o documento foi escrito com "recuperar", "recuperado", "recuperações" — não encontra nada. Zero resultado.

# Representação Vetorial dos Documentos (Número de exemplo)
# Para calcular o termo da consulta ele usa o texto da propria consulta
# "indexação de documentos"
# ["index", "document"]
# TF — frequência do termo no documento dividida pelo total de tokens do documento.
#                 index    document    red    stem
# consulta    →  [ 0.846,   0.846,     0.0,   0.0 ]
# doc1        →  [ 0.8,      0.6,      0.0,   0.2 ]
# doc2        →  [ 0.0,      0.4,      0.9,   0.0 ]
# doc3        →  [ 0.0,      0.0,      0.0,   0.8 ]