# -*- coding: utf-8 -*-
r"""
CÍVEL CONSUMIDOR: análise por PROCESSO (sentenças + acórdãos juntos) com
modelo LOCAL no Ollama.

Diferença para a versão do Juizado:
 - NÃO remove as SENTENÇAS.
 - Todos os documentos de um mesmo codProcesso são analisados JUNTOS, numa
   única chamada, em ORDEM CRONOLÓGICA (data_evento; empate -> codDocumento).
   Cada documento vai com um cabeçalho: nº, data, evento e codDocumento.
 - O modelo recebe regras de prevalência (a decisão posterior prevalece no
   que reformar; o que não foi reformado continua valendo; majoração de
   honorários em recurso etc.) e devolve o resultado FINAL do processo.
 - Saída: 1 linha por processo (CSV_SAIDA), com as 4 colunas de sempre +
   qual documento definiu os honorários + um resumo curto da cadeia.

Fluxo:
 1) Carrega o CSV (todas as linhas) e, opcionalmente, filtra eventos
 2) Copia/extrai os .zip e renomeia para {codProcesso}-{ordem}
    (checkpoint por documento em CSV_ARQUIVOS)
 3) Agrupa por processo, ordena por data, extrai o texto de cada documento,
    descarta documentos com texto duplicado e recorta cada um dentro de um
    orçamento de caracteres (sentenças recebem peso maior)
 4) Envia ao Ollama e pede JSON estruturado
 5) Valida, grava o CSV por processo (checkpoint a cada N processos)

Instalação:
    pip install pandas requests pdfplumber python-docx tqdm

Interromper: Ctrl+C (ou "Interrupt" do Jupyter) salva o checkpoint e sai.
Na próxima execução retoma dos processos sem resultado.
"""
import hashlib
import json
import logging
import os
import queue
import re
import shutil
import threading
import time
import unicodedata
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ----------------------------------------------------------------- CONFIG
CSV_ENTRADA = r"E:\ARQUIVOS_CIVEL_CONSUMIDOR\llama\CIVEL_CONSUMIDOR_documentos.csv"
# Checkpoint da etapa de arquivos (1 linha por documento)
CSV_ARQUIVOS = r"E:\ARQUIVOS_CIVEL_CONSUMIDOR\llama\CIVEL_CONSUMIDOR_arquivos.csv"
# Resultado (1 linha por PROCESSO)
CSV_SAIDA = r"E:\ARQUIVOS_CIVEL_CONSUMIDOR\llama\CIVEL_CONSUMIDOR_resultado_ollama.csv"
LOG_ARQUIVO = r"E:\ARQUIVOS_CIVEL_CONSUMIDOR\llama\CIVEL_CONSUMIDOR_resultado_ollama.log"

# Pasta onde estão os .zip baixados. Se o caminho da coluna "arquivo" não
# existir, o script procura o arquivo pelo nome nesta pasta.
PASTA_ORIGEM = Path(r"E:\ARQUIVOS_CIVEL_CONSUMIDOR")
PASTA_DESTINO = PASTA_ORIGEM / "CIVEL_EXTRAIDOS"
PASTA_ZIPS = PASTA_DESTINO / "_zips_originais"
PASTA_EXTRAIDOS = PASTA_DESTINO

# Eventos a considerar (regex sobre o texto do evento, sem acento, maiúsculo).
# None = todos. Ex.: r"^(SENTENCA|ACORDAO|DECISAO)"
FILTRO_EVENTOS = None

# --- Ollama
OLLAMA_URL = "http://localhost:11434"
MODELO = "qwen3.5:4b"
NUM_CTX = 8192           # vários documentos por processo: precisa de mais contexto
NUM_PREDICT = 700         # JSON + resumo curto
TEMPERATURA = 0
PENSAR = False            # True pode melhorar cadeias complexas (bem mais lento)
USAR_SCHEMA = True
KEEP_ALIVE = "30m"

LIMITE = 5000            # nº de PROCESSOS para teste; None = todos
# 1 GPU de 6 GB: 2 workers = um lê o PDF enquanto o outro usa a GPU.
# Mais que isso só divide a GPU (cada processo fica mais lento, vazão igual).
WORKERS = 2
# Ordem de processamento pela quantidade de anexos do processo:
#   "menor" = começa pelos de 1 anexo (resultados e validação mais rápidos)
#   "maior" = começa pelos maiores (evita um processo grande sozinho no fim)
ORDEM = "menor"

# --- Orçamento de texto por processo
MAX_CHARS_TOTAL = 14000   # soma do texto enviado por processo (~11k tokens)
MIN_CHARS_DOC = 1500      # piso por documento (dispositivo + honorários)
PESO_SENTENCA = 2.0       # sentença recebe o dobro do orçamento de um acórdão
JANELA_LINHAS = 2
FRACAO_FINAL = 0.4        # parte do orçamento de cada doc reservada ao FIM do documento

TIMEOUT_RESPOSTA = 900
TENTATIVAS = 3
TOLERANCIA_HONOR = 0.02

SALVAR_A_CADA = 10
SALVAR_ARQUIVOS_A_CADA = 200
LOG_DETALHADO = True

PROMPT_SISTEMA = (
    "Você é um analista jurídico que extrai dados de processos cíveis de "
    "direito do consumidor da Justiça Comum (CPC). Você recebe TODAS as "
    "decisões de um mesmo processo (sentenças, acórdãos e decisões de "
    "tribunais superiores) em ORDEM CRONOLÓGICA, cada uma com data e tipo de "
    "evento. Responda sempre e somente com JSON válido. Use apenas o que está "
    "no texto; se não houver informação, use null. Nunca invente valores.\n\n"
    "Regras para chegar ao resultado FINAL do processo:\n"
    "1. A decisão POSTERIOR prevalece sobre a anterior apenas naquilo que "
    "reformar. O que não foi reformado continua valendo.\n"
    "2. A sentença normalmente fixa a condenação e os honorários (art. 85, §2º, "
    "CPC: 10% a 20% sobre a condenação, o proveito econômico ou o valor da causa).\n"
    "3. Acórdão de apelação pode manter, reformar ou anular a sentença. Se "
    "anular, vale a decisão proferida depois.\n"
    "4. Honorários recursais (art. 85, §11, CPC): se o tribunal MAJORAR, o "
    "percentual final é o novo total (ex.: sentença 10%, 'majoro para 15%' = "
    "15; 'majoro em 2%' ou 'acrescento 2%' = 12).\n"
    "5. Embargos de declaração só alteram o resultado se acolhidos com efeito "
    "modificativo. Agravos e recursos ao STJ/STF não conhecidos ou desprovidos "
    "mantêm a decisão anterior (salvo majoração expressa de honorários).\n"
    "6. Se houver sucumbência invertida (o réu vencedor passa a ser credor dos "
    "honorários), considere os honorários finais fixados.\n"
    "7. Honorários EXISTEM (condenacao_honorarios = \"Sim\") mesmo quando: há "
    "sucumbência recíproca (cada parte paga honorários à outra); a "
    "exigibilidade está suspensa pela justiça gratuita (art. 98, §3º, CPC); "
    "ou são fixados por equidade em valor fixo. Só é \"Não\" se o texto disser "
    "expressamente que não há condenação em honorários.\n"
    "8. Percentuais são sempre em pontos percentuais: 15% = 15 (nunca 0.15).\n"
    "Os textos podem ter trechos omitidos, marcados com [...]."
)

PROMPT_USUARIO = (
    "Analise a cadeia de decisões abaixo e devolva um objeto JSON com "
    "exatamente estas chaves, refletindo a situação FINAL do processo:\n"
    "- valores_condenacao: lista com CADA valor da condenação que prevaleceu, "
    "um item por parcela, no formato {\"descricao\": \"danos morais\", "
    "\"valor\": 5000.00}. Para restituição em dobro, informe o valor já "
    "dobrado se o texto informar; senão, o valor e a descrição \"em dobro\". "
    "Não inclua honorários nem custas. Não some: liste. Lista vazia [] se "
    "não houver valores\n"
    "- condenacao_honorarios: \"Sim\" ou \"Não\" (há honorários advocatícios "
    "na decisão que prevaleceu?), ou null se não houver informação\n"
    "- percentual_aplicado: percentual FINAL dos honorários (já com eventual "
    "majoração), ou null\n"
    "- valor_aplicado: valor em reais dos honorários se o texto informar ou se "
    "for valor fixo, ou null\n"
    "- documento_base: número (1, 2, 3...) do DOCUMENTO que fixou os "
    "honorários finais, ou null\n"
    "- trecho_honorarios: copie a frase desse documento que trata dos "
    "honorários, ou null\n"
    "- resumo: no máximo 2 frases explicando como a cadeia chegou ao "
    "resultado final (o que a sentença fixou e o que os recursos mudaram)\n\n"
    "DOCUMENTOS DO PROCESSO (ordem cronológica):\n"
)

SCHEMA = {
    "type": "object",
    "properties": {
        "valores_condenacao": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"descricao": {"type": "string"},
                               "valor": {"type": ["number", "null"]}},
                "required": ["descricao", "valor"],
            },
        },
        "condenacao_honorarios": {"type": ["string", "null"]},
        "percentual_aplicado": {"type": ["number", "null"]},
        "valor_aplicado": {"type": ["number", "null"]},
        "documento_base": {"type": ["integer", "null"]},
        "trecho_honorarios": {"type": ["string", "null"]},
        "resumo": {"type": ["string", "null"]},
    },
    "required": ["valores_condenacao", "condenacao_honorarios",
                 "percentual_aplicado", "valor_aplicado", "documento_base",
                 "trecho_honorarios", "resumo"],
}

RELEVANTE = re.compile(
    r"honor[aá]ri|sucumb|condena|condeno|R\$|\d\s*%|por cento|art\.?\s*85|"
    r"§\s*11|major|provimento|provido|ACORDAM|ac[oó]rd[aã]o|ante o exposto|"
    r"isto posto|diante do exposto|pelo exposto|posto isso|dispositivo|"
    r"julgo|procedente|improcedente|danos? mora|danos? materia|restitui|"
    r"indeniza|repeti[cç][aã]o|em dobro|custas|reform|mant[eé]m|mantid|"
    r"anul|embargos|acolho|rejeito|n[aã]o conhe|conhe[cç]o|agravo",
    re.IGNORECASE,
)

NOVAS_COLUNAS = [
    "valor_total_condenacao",
    "condenacao_honorarios",
    "percentual_aplicado",
    "valor_aplicado",
]
COLUNAS_EXTRAS = ["itens_condenacao", "documento_base_cod", "documento_base_evento",
                  "documento_base_data", "resumo", "trecho_honorarios"]
COLUNAS_CONTROLE = ["qtd_documentos", "qtd_analisados", "eventos",
                    "id_req", "status", "alerta", "origem"]
CHAVE_DOC = ["codProcesso", "codDocumento"]
INVALIDOS = ("NAO_ENCONTRADO", "VAZIO", "<NA>", "nan", "None", "")
FALHAS = ("erro no modelo", "sem JSON", "erro")

PARAR = threading.Event()

logging.basicConfig(
    filename=LOG_ARQUIVO,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
    encoding="utf-8",
)
log = logging.getLogger("ollama_civel")


# ----------------------------------------------------------------- UTIL
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).upper().strip()


def ler_csv(caminho):
    for enc in ("utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(caminho, sep=None, engine="python",
                               encoding=enc, dtype=str,
                               keep_default_na=False, na_values=[""])
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Não consegui ler {caminho} (encoding).")


def salvar_csv(df, caminho):
    """Gravação atômica (.tmp -> replace)."""
    tmp = caminho + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8-sig", sep=";")
    for _ in range(10):
        try:
            os.replace(tmp, caminho)
            return
        except PermissionError:
            time.sleep(1)
    log.error(f"não consegui substituir {caminho} (arquivo aberto?); "
              f"o checkpoint mais recente está em {tmp}")
    tqdm.write(f"AVISO: {caminho} está aberto; checkpoint salvo em {tmp}")


def localizar_arquivo(row) -> Path | None:
    p = Path(str(row["arquivo"]).strip())
    if p.exists():
        return p
    nome = p.name if "\\" not in p.name else str(row["arquivo"]).split("\\")[-1]
    p2 = PASTA_ORIGEM / nome
    if p2.exists():
        return p2
    achados = list(PASTA_ORIGEM.glob(f"{row['codProcesso']}-*_{row['codDocumento']}_*"))
    return achados[0] if achados else None


def para_numero(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s.upper() in ("", "N/A", "NULL", "NONE"):
        return None
    s = re.sub(r"[^\d,.\-]", "", s)
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def normaliza_sim_nao(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return "Sim" if v else "Não"
    n = norm(v)
    if n in ("TRUE", "YES"):
        return "Sim"
    if n in ("FALSE", "NO"):
        return "Não"
    if n.startswith("S"):
        return "Sim"
    if n.startswith("N") and n not in ("N/A", "NULL", "NONE"):
        return "Não"
    return None


def eh_sentenca(evento) -> bool:
    return norm(evento).startswith("SENTENCA")


# ----------------------------------------------------------------- ARQUIVOS
def copiar_e_extrair(origem: Path, cod_processo: str, contador: list) -> list[Path]:
    PASTA_ZIPS.mkdir(parents=True, exist_ok=True)
    PASTA_EXTRAIDOS.mkdir(parents=True, exist_ok=True)

    copia = PASTA_ZIPS / origem.name
    if not copia.exists():
        shutil.copy2(origem, copia)

    resultado = []

    def salvar(conteudo_path: Path, ext: str):
        contador[0] += 1
        destino = PASTA_EXTRAIDOS / f"{cod_processo}-{contador[0]:06d}{ext}"
        shutil.move(str(conteudo_path), destino)
        resultado.append(destino)

    if zipfile.is_zipfile(copia):
        tmp = PASTA_DESTINO / "_tmp_extracao"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        with zipfile.ZipFile(copia) as z:
            z.extractall(tmp)
        for f in sorted(p for p in tmp.rglob("*") if p.is_file()):
            salvar(f, f.suffix.lower())
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        tmp_copy = PASTA_DESTINO / f"_tmp_{copia.name}"
        shutil.copy2(copia, tmp_copy)
        salvar(tmp_copy, copia.suffix.lower() or ".bin")
    return resultado


def extrair_texto(arq: Path) -> str:
    ext = arq.suffix.lower()
    try:
        if ext == ".pdf":
            import pdfplumber
            with pdfplumber.open(arq) as pdf:
                partes = []
                for pg in pdf.pages:
                    partes.append(pg.extract_text() or "")
                    pg.flush_cache()
                return "\n".join(partes)
        if ext == ".docx":
            import docx
            return "\n".join(p.text for p in docx.Document(arq).paragraphs)
        if ext in (".txt", ".html", ".htm", ".rtf", ".xml", ".csv"):
            raw = arq.read_bytes()
            txt = raw.decode("latin-1")
            for enc in ("utf-8", "cp1252", "latin-1"):
                try:
                    txt = raw.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            if ext in (".html", ".htm", ".xml"):
                txt = re.sub(r"<[^>]+>", " ", txt)
            return txt
    except Exception as e:
        return f"[ERRO AO LER {arq.name}: {e}]"
    return ""


def limpar(texto: str) -> str:
    texto = re.sub(r"[ \t]+", " ", texto)
    return re.sub(r"\n{3,}", "\n\n", texto).strip()


def recortar_texto(texto: str, limite: int) -> tuple[str, bool]:
    """Mantém as linhas relevantes (com contexto) + o fim do documento,
    dentro de `limite` caracteres. Devolve (texto, foi_recortado)."""
    if len(texto) <= limite:
        return texto, False
    linhas = texto.splitlines()
    final_chars = int(limite * FRACAO_FINAL)

    fim = set()
    acum = 0
    for n in range(len(linhas) - 1, -1, -1):
        fim.add(n)
        acum += len(linhas[n]) + 1
        if acum >= final_chars:
            break

    relevantes = set()
    for n, linha in enumerate(linhas):
        if RELEVANTE.search(linha):
            relevantes.update(range(max(0, n - JANELA_LINHAS),
                                    min(len(linhas), n + JANELA_LINHAS + 1)))

    def montar(conj):
        partes, anterior = [], -2
        for n in sorted(conj):
            if n != anterior + 1:
                partes.append("[...]")
            partes.append(linhas[n])
            anterior = n
        return "\n".join(partes)

    rec = montar(relevantes | fim)
    if len(rec) > limite:
        # relevantes demais: mantém o fim inteiro e os relevantes mais próximos dele
        txt_fim = montar(fim)
        resto = limite - len(txt_fim) - 10
        corpo = montar(relevantes - fim) if resto > 0 else ""
        rec = (corpo[-resto:] + "\n[...]\n" if resto > 0 and corpo else "") + txt_fim
        rec = rec[-limite:]
    return rec, True


# ----------------------------------------------------------------- JSON
def extrair_json(texto: str) -> dict | None:
    texto = (texto or "").strip()
    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL).strip()
    texto = re.sub(r"^```(?:json)?|```$", "", texto).strip()
    try:
        d = json.loads(texto)
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        pass
    # JSON com lista aninhada: pega do primeiro "{" ao último "}"
    ini, fim = texto.find("{"), texto.rfind("}")
    if ini != -1 and fim > ini:
        try:
            d = json.loads(texto[ini:fim + 1])
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            pass
    return None


# ----------------------------------------------------------------- LEGIBILIDADE
_PALAVRAS_PT = re.compile(r"\b(de|que|do|da|em|para|com|não|nao|o|a|os|as|e|no|na|por)\b",
                          re.IGNORECASE)


def texto_legivel(texto: str) -> bool:
    """Detecta texto que veio como lixo (PDF escaneado, fonte sem mapeamento,
    codificação quebrada): poucas letras ou quase nenhuma palavra comum."""
    amostra = texto[:20000]
    sem_espaco = re.sub(r"\s", "", amostra)
    if len(sem_espaco) < 200:
        return False
    letras = sum(ch.isalpha() for ch in sem_espaco)
    if letras / len(sem_espaco) < 0.6:
        return False
    palavras = len(_PALAVRAS_PT.findall(amostra))
    return palavras / max(1, len(amostra) / 1000) >= 15   # >= 15 por 1000 caracteres


# ----------------------------------------------------------------- OLLAMA
class AgenteOllama:
    def __init__(self, worker_id=1):
        self.worker_id = worker_id
        self.sessao = requests.Session()
        self.formato = SCHEMA if USAR_SCHEMA else "json"
        self.pensar = PENSAR

    def _payload(self, conteudo: str) -> dict:
        p = {
            "model": MODELO,
            "messages": [
                {"role": "system", "content": PROMPT_SISTEMA},
                {"role": "user", "content": PROMPT_USUARIO + conteudo},
            ],
            "stream": False,
            "format": self.formato,
            "keep_alive": KEEP_ALIVE,
            "options": {
                "temperature": TEMPERATURA,
                "num_ctx": NUM_CTX,
                "num_predict": NUM_PREDICT,
            },
        }
        if self.pensar is not None:
            p["think"] = self.pensar
        return p

    def perguntar(self, conteudo: str) -> tuple[dict | None, dict]:
        ultimo_erro = None
        for tentativa in range(1, TENTATIVAS + 1):
            if PARAR.is_set():
                return None, {}
            try:
                r = self.sessao.post(f"{OLLAMA_URL}/api/chat",
                                     json=self._payload(conteudo),
                                     timeout=TIMEOUT_RESPOSTA)
                if r.status_code >= 400:
                    msg = r.text[:300]
                    if self.pensar is not None and "think" in msg.lower():
                        log.warning(f"worker {self.worker_id}: 'think' não suportado; desligando")
                        self.pensar = None
                        continue
                    if self.formato != "json":
                        log.warning(f"worker {self.worker_id}: schema recusado ({msg}); usando format=json")
                        self.formato = "json"
                        continue
                    raise RuntimeError(f"HTTP {r.status_code}: {msg}")
                data = r.json()
                metricas = {
                    "prompt_tokens": data.get("prompt_eval_count") or 0,
                    "resp_tokens": data.get("eval_count") or 0,
                    "seg": (data.get("total_duration") or 0) / 1e9,
                    "done_reason": data.get("done_reason"),
                }
                return extrair_json(data.get("message", {}).get("content", "")), metricas
            except (requests.RequestException, RuntimeError) as e:
                ultimo_erro = e
                log.warning(f"worker {self.worker_id}: erro na tentativa "
                            f"{tentativa}/{TENTATIVAS}: {e}")
                time.sleep(3 * tentativa)
        raise RuntimeError(f"Ollama falhou após {TENTATIVAS} tentativas: {ultimo_erro}")

    def fechar(self):
        self.sessao.close()


def verificar_ollama():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama não respondeu em {OLLAMA_URL}. Ele está aberto? ({e})")
    nomes = {m["name"] for m in r.json().get("models", [])}
    if MODELO not in nomes and f"{MODELO}:latest" not in nomes:
        raise RuntimeError(f"Modelo '{MODELO}' não encontrado. Rode: ollama pull {MODELO}\n"
                           f"Disponíveis: {sorted(nomes)}")
    print(f"Carregando {MODELO} na memória...")
    requests.post(f"{OLLAMA_URL}/api/generate",
                  json={"model": MODELO, "keep_alive": KEEP_ALIVE}, timeout=300)


# ----------------------------------------------------------------- MONTAGEM DO PROCESSO
def montar_processo(docs: list[dict]) -> tuple[str, list[dict], list[str]]:
    """docs já em ordem cronológica. Extrai o texto, remove duplicados,
    distribui o orçamento e monta o conteúdo com cabeçalhos.
    Devolve (conteudo, docs_enviados, avisos)."""
    avisos = []
    candidatos = []
    vistos = {}
    for d in docs:
        arqs = str(d.get("arquivo_extraido", ""))
        if arqs in INVALIDOS or arqs.startswith("ERRO"):
            avisos.append(f"{d['codDocumento']}: sem arquivo")
            continue
        texto = limpar("\n\n".join(extrair_texto(Path(p)) for p in arqs.split("|")))
        if not texto:
            avisos.append(f"{d['codDocumento']}: sem texto (PDF escaneado?)")
            continue
        if not texto_legivel(texto):
            avisos.append(f"{d['codDocumento']}: texto ilegível (precisa de OCR)")
            continue
        h = hashlib.md5(re.sub(r"\s+", "", texto).encode("utf-8")).hexdigest()
        if h in vistos:
            avisos.append(f"{d['codDocumento']}: texto idêntico a {vistos[h]} (ignorado)")
            continue
        vistos[h] = d["codDocumento"]
        candidatos.append({**d, "texto": texto})

    if not candidatos:
        return "", [], avisos

    pesos = [PESO_SENTENCA if eh_sentenca(d["evento"]) else 1.0 for d in candidatos]
    soma = sum(pesos)
    blocos = []
    for n, (d, w) in enumerate(zip(candidatos, pesos), start=1):
        limite = max(MIN_CHARS_DOC, int(MAX_CHARS_TOTAL * w / soma))
        texto, cortado = recortar_texto(d["texto"], limite)
        d["numero"] = n
        d["recortado"] = cortado
        blocos.append(
            f"===== DOCUMENTO {n} de {len(candidatos)} | DATA: {d['data_evento']} | "
            f"EVENTO: {d['evento']} | codDocumento: {d['codDocumento']}"
            f"{' | (texto recortado)' if cortado else ''} =====\n{texto}"
        )
    return "\n\n".join(blocos), candidatos, avisos


# ----------------------------------------------------------------- VALIDAÇÃO
def somar_itens(itens) -> tuple[float | None, list[tuple[str, float]]]:
    """Soma em Python os valores listados pelo modelo (o modelo erra somas)."""
    if not isinstance(itens, list):
        return None, []
    validos = []
    for it in itens:
        if isinstance(it, dict):
            v = para_numero(it.get("valor"))
            desc = str(it.get("descricao") or "").strip()
        else:
            v, desc = para_numero(it), ""
        if v is not None and v > 0:
            validos.append((desc, v))
    if not validos:
        return None, []
    return round(sum(v for _, v in validos), 2), validos


def normalizar_resposta(d: dict) -> tuple[dict, list[str]]:
    alertas = []
    total, itens = somar_itens(d.get("valores_condenacao"))
    if total is None and d.get("valor_total_condenacao") is not None:
        total = para_numero(d.get("valor_total_condenacao"))   # resposta no formato antigo
    pct = para_numero(d.get("percentual_aplicado"))
    if pct is not None and 0 < pct < 1:
        pct = round(pct * 100, 4)                  # 0.15 -> 15
        alertas.append("percentual veio como fração (convertido)")
    dados = {
        "valor_total_condenacao": total,
        "condenacao_honorarios": normaliza_sim_nao(d.get("condenacao_honorarios")),
        "percentual_aplicado": pct,
        "valor_aplicado": para_numero(d.get("valor_aplicado")),
        "_itens": itens,
    }
    if dados["condenacao_honorarios"] is None and (
            dados["percentual_aplicado"] or dados["valor_aplicado"]):
        dados["condenacao_honorarios"] = "Sim"
        alertas.append("honorarios inferido como Sim (modelo não informou)")
    total, pct, val = (dados["valor_total_condenacao"],
                       dados["percentual_aplicado"], dados["valor_aplicado"])
    if dados["condenacao_honorarios"] == "Não" and (pct or val):
        alertas.append("honorarios=Não mas há % ou valor")
    if total and pct and val:
        esperado = total * pct / 100
        if esperado and abs(esperado - val) / esperado > TOLERANCIA_HONOR:
            alertas.append(f"valor_aplicado diverge de total*% (esperado {esperado:.2f})")
    if pct is not None and not (0 <= pct <= 100):
        alertas.append("percentual fora de 0-100")
    return dados, alertas


def _br(v: float, casas: int = 2) -> str:
    s = f"{v:,.{casas}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_moeda(v):
    return "N/A" if v is None else f"R$ {_br(v)}"


def fmt_pct(v):
    if v is None:
        return "N/A"
    return f"{_br(v, 0)}%" if float(v).is_integer() else f"{_br(v)}%"


def formatar_br(dados: dict) -> dict:
    return {
        "valor_total_condenacao": fmt_moeda(dados["valor_total_condenacao"]),
        "condenacao_honorarios": dados["condenacao_honorarios"] or "N/A",
        "percentual_aplicado": fmt_pct(dados["percentual_aplicado"]),
        "valor_aplicado": fmt_moeda(dados["valor_aplicado"]),
    }


def checar_pct_no_texto(pct, docs_env) -> list[str]:
    """O percentual final deve aparecer em algum documento, salvo majoração
    'em X%' (soma), que gera um número que não está escrito."""
    if pct is None:
        return []
    alvo = _br(pct, 0) if float(pct).is_integer() else _br(pct)
    padrao = re.compile(rf"(?<![\d,]){re.escape(alvo)}\s*(%|por cento)", re.I)
    if any(padrao.search(d["texto"]) for d in docs_env):
        return []
    return [f"{alvo}% não aparece literalmente nos textos (majoração somada? conferir)"]


# ----------------------------------------------------------------- WORKERS
_SENTINELA = None


def _analisar(agente, cod, docs) -> dict:
    res = {"cod": cod, "status": "erro", "dados": None, "extras": {},
           "controle": {"qtd_documentos": len(docs)}, "alerta": None,
           "mensagem": None}
    conteudo, docs_env, avisos = montar_processo(docs)
    res["controle"]["qtd_analisados"] = len(docs_env)
    res["controle"]["eventos"] = " | ".join(
        f"{d['data_evento']} {d['evento']}" for d in docs_env)
    if not docs_env:
        ilegivel = any("ilegível" in a for a in avisos)
        res.update(status="texto ilegível" if ilegivel else "sem texto",
                   dados={c: "N/A" for c in NOVAS_COLUNAS},
                   alerta="; ".join(avisos) or None)
        return res

    try:
        resp, met = agente.perguntar(conteudo)
    except Exception as e:
        res.update(status="erro no modelo", mensagem=f"{cod}: erro no modelo: {e}")
        return res
    if resp is None:
        res.update(status="sem JSON", mensagem=f"{cod}: sem JSON válido (reprocessa depois)")
        return res

    dados, alertas = normalizar_resposta(resp)
    alertas += checar_pct_no_texto(dados["percentual_aplicado"], docs_env)

    base = resp.get("documento_base")
    try:
        base = int(base) if base is not None else None
    except (TypeError, ValueError):
        base = None
    if len(docs_env) == 1 and (base is not None or dados["condenacao_honorarios"] == "Sim"):
        base = 1                                   # só há um documento
    doc_base = next((d for d in docs_env if d["numero"] == base), None)
    if base is not None and doc_base is None:
        alertas.append(f"documento_base={base} inexistente")
    if dados["condenacao_honorarios"] == "Sim" and doc_base is None:
        alertas.append("honorarios=Sim sem documento_base")

    if met.get("prompt_tokens", 0) >= NUM_CTX - NUM_PREDICT:
        alertas.append("texto pode ter sido truncado pelo contexto (aumente NUM_CTX "
                       "ou reduza MAX_CHARS_TOTAL)")
    if met.get("done_reason") == "length":
        alertas.append("resposta cortada (aumente NUM_PREDICT)")
    if avisos:
        alertas.append("docs: " + ", ".join(avisos))

    alerta = "; ".join(alertas)
    dados_br = formatar_br(dados)
    tok_s = met["resp_tokens"] / met["seg"] if met.get("seg") else 0
    res.update(
        status="OK", dados=dados_br, alerta=alerta or None,
        extras={
            "itens_condenacao": " + ".join(
                f"{desc or 'valor'}: {fmt_moeda(v)}" for desc, v in dados["_itens"]) or None,
            "documento_base_cod": doc_base["codDocumento"] if doc_base else None,
            "documento_base_evento": doc_base["evento"] if doc_base else None,
            "documento_base_data": doc_base["data_evento"] if doc_base else None,
            "resumo": resp.get("resumo"),
            "trecho_honorarios": resp.get("trecho_honorarios"),
        },
        mensagem=(f"{cod}: {len(docs_env)}/{len(docs)} docs | {dados_br} | "
                  f"itens: {dados['_itens']} | "
                  f"base: doc {base} | resumo: {resp.get('resumo')!r} | "
                  f"{met['prompt_tokens']} tok entrada, {met['seg']:.1f}s, {tok_s:.1f} tok/s"
                  + (f" ALERTA: {alerta}" if alerta else "")),
    )
    res["controle"]["id_req"] = uuid.uuid4().hex[:10]
    return res


def _worker_ia(worker_id: int, fila_tarefas, fila_resultados):
    agente = None
    item = None
    try:
        agente = AgenteOllama(worker_id=worker_id)
        while not PARAR.is_set():
            try:
                item = fila_tarefas.get(timeout=1)
            except queue.Empty:
                item = None
                continue
            if item is _SENTINELA:
                break
            cod, docs = item
            try:
                resultado = _analisar(agente, cod, docs)
            except Exception as e:
                resultado = {"cod": cod, "status": "erro", "dados": None,
                             "extras": {}, "controle": {}, "alerta": None,
                             "mensagem": f"processo {cod}: {e}"}
            if PARAR.is_set() and resultado["status"] in FALHAS:
                break
            fila_resultados.put(resultado)
            item = None
    except Exception as e:
        if item not in (None, _SENTINELA):
            fila_tarefas.put(item)
        fila_resultados.put({"worker_erro": f"Worker {worker_id} parou: {e}"})
    finally:
        if agente:
            agente.fechar()


# ----------------------------------------------------------------- ETAPA DE ARQUIVOS
def preparar_arquivos(df) -> pd.DataFrame:
    """Copia/extrai os arquivos de cada documento; retoma de CSV_ARQUIVOS."""
    df = df.reset_index(drop=True)
    df["arquivo_extraido"] = pd.NA
    p = Path(CSV_ARQUIVOS)
    if p.exists() and p.stat().st_size > 0:
        ant = ler_csv(p)
        if set(CHAVE_DOC + ["arquivo_extraido"]) <= set(ant.columns):
            ant = ant.drop_duplicates(subset=CHAVE_DOC, keep="last")
            m = df[CHAVE_DOC].merge(ant[CHAVE_DOC + ["arquivo_extraido"]],
                                    on=CHAVE_DOC, how="left")
            m.index = df.index
            df["arquivo_extraido"] = m["arquivo_extraido"]
            print(f"  {df['arquivo_extraido'].notna().sum()} documentos já extraídos "
                  f"({p.name})")

    contador = [0]
    nao_encontrados = 0
    novos = 0
    try:
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Arquivos",
                           unit="doc", dynamic_ncols=True, mininterval=0.5):
            if PARAR.is_set():
                break
            arq_ant = row["arquivo_extraido"]
            if pd.notna(arq_ant):
                m = re.search(r"-(\d+)$", Path(str(arq_ant).split("|")[-1]).stem)
                if m:
                    contador[0] = max(contador[0], int(m.group(1)))
                continue
            origem = localizar_arquivo(row)
            if not origem:
                df.at[i, "arquivo_extraido"] = "NAO_ENCONTRADO"
                nao_encontrados += 1
                log.warning(f"[{i + 1}] arquivo não encontrado: {row['arquivo']}")
            else:
                try:
                    extraidos = copiar_e_extrair(origem, str(row["codProcesso"]), contador)
                    df.at[i, "arquivo_extraido"] = "|".join(map(str, extraidos)) or "VAZIO"
                except Exception as e:
                    df.at[i, "arquivo_extraido"] = f"ERRO: {e}"
                    log.error(f"[{i + 1}] erro ao copiar/extrair {row['codProcesso']}: {e}")
            novos += 1
            if novos % SALVAR_ARQUIVOS_A_CADA == 0:
                salvar_csv(df, CSV_ARQUIVOS)
    finally:
        salvar_csv(df, CSV_ARQUIVOS)
    if nao_encontrados:
        print(f"{nao_encontrados} arquivos não encontrados (detalhes no log)")
    return df


def agrupar_processos(df) -> dict:
    """{codProcesso: [docs em ordem cronológica]}"""
    df = df.copy()
    df["_data"] = pd.to_datetime(df["data_evento"], dayfirst=True, errors="coerce")
    df["_cod"] = pd.to_numeric(df["codDocumento"], errors="coerce")
    df = df.sort_values(["codProcesso", "_data", "_cod"], na_position="first")
    grupos = {}
    for cod, g in df.groupby("codProcesso", sort=False):
        grupos[cod] = g[["codDocumento", "evento", "data_evento",
                         "arquivo_extraido"]].to_dict("records")
    return grupos


def carregar_resultados(processos: list) -> pd.DataFrame:
    colunas = ["codProcesso"] + NOVAS_COLUNAS + COLUNAS_EXTRAS + COLUNAS_CONTROLE
    res = pd.DataFrame({"codProcesso": processos})
    for c in colunas[1:]:
        res[c] = pd.NA
    p = Path(CSV_SAIDA)
    if p.exists() and p.stat().st_size > 0:
        ant = ler_csv(p).drop_duplicates(subset=["codProcesso"], keep="last")
        for c in colunas:
            if c not in ant.columns:
                ant[c] = pd.NA
        m = res[["codProcesso"]].merge(ant[colunas], on="codProcesso", how="left")
        m.index = res.index
        res = m
        print(f"  {res['valor_total_condenacao'].notna().sum()} processos já "
              f"analisados ({p.name})")
    # pandas 3 lê CSV com dtype "str" estrito; object aceita qualquer valor
    return res[colunas].astype(object)


def _cel(v):
    """Valor para gravar no DataFrame: texto, ou NA se vazio."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return pd.NA
    return str(v)


# ----------------------------------------------------------------- MAIN
def main():
    if not isinstance(WORKERS, int) or WORKERS < 1:
        raise ValueError("WORKERS precisa ser um inteiro maior ou igual a 1.")

    verificar_ollama()

    df = ler_csv(CSV_ENTRADA)
    if FILTRO_EVENTOS:
        df = df[df["evento"].map(norm).str.contains(FILTRO_EVENTOS, regex=True)].copy()
    print(f"{len(df)} documentos de {df['codProcesso'].nunique()} processos")

    if LIMITE is not None:
        # o teste respeita a ORDEM: com "menor", pega os processos com menos anexos
        tam = df.groupby("codProcesso").size().sort_values(
            ascending=(ORDEM != "maior"), kind="stable")
        manter = pd.Series(tam.index[:LIMITE])
        df = df[df["codProcesso"].isin(manter)].copy()
        print(f"MODO TESTE: {LIMITE} processos ({len(df)} documentos)")

    df = preparar_arquivos(df)
    if PARAR.is_set():
        print("Interrompido na etapa de arquivos. Checkpoint salvo.")
        return

    grupos = agrupar_processos(df)
    res_df = carregar_resultados(list(grupos))
    idx = {cod: i for i, cod in enumerate(res_df["codProcesso"])}
    print(f"Detalhes por processo em: {LOG_ARQUIVO}")

    fila_tarefas = queue.Queue()
    fila_resultados = queue.Queue()
    pendentes = [cod for cod in grupos
                 if pd.isna(res_df.at[idx[cod], "valor_total_condenacao"])]
    # ordem pela quantidade de anexos (ver ORDEM no topo)
    pendentes.sort(key=lambda c: len(grupos[c]), reverse=(ORDEM == "maior"))
    qtds = pd.Series([len(grupos[c]) for c in pendentes])
    if len(qtds):
        print("Anexos por processo: " + ", ".join(
            f"{k}: {v}" for k, v in qtds.value_counts().sort_index().items()))
    for cod in pendentes:
        fila_tarefas.put((cod, grupos[cod]))
    for _ in range(WORKERS):
        fila_tarefas.put(_SENTINELA)
    print(f"{len(pendentes)} processos para analisar com {MODELO} (WORKERS={WORKERS})")

    cont = {"ok": 0, "sem_json": 0, "erros": 0, "alertas": 0}

    def aplicar(r):
        if "worker_erro" in r:
            tqdm.write(r["worker_erro"])
            log.error(r["worker_erro"])
            return False
        i = idx[r["cod"]]
        status = r["status"]
        res_df.at[i, "status"] = _cel(status)
        for c, v in (r.get("controle") or {}).items():
            res_df.at[i, c] = _cel(v)
        if r.get("dados") is not None:
            res_df.at[i, "origem"] = "ollama"
            for c, v in r["dados"].items():
                res_df.at[i, c] = _cel(v)
            for c, v in (r.get("extras") or {}).items():
                res_df.at[i, c] = _cel(v)
            res_df.at[i, "alerta"] = _cel(r.get("alerta"))
        if status == "OK":
            cont["ok"] += 1
            if r.get("alerta"):
                cont["alertas"] += 1
                log.warning(r["mensagem"])
            elif LOG_DETALHADO:
                log.info(r["mensagem"])
        elif status == "sem JSON":
            cont["sem_json"] += 1
            log.warning(r["mensagem"])
        elif status in FALHAS:
            cont["erros"] += 1
            log.error(r["mensagem"])
        return True

    pool = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="ia-worker")
    futures = [pool.submit(_worker_ia, wid, fila_tarefas, fila_resultados)
               for wid in range(1, WORKERS + 1)]
    barra = tqdm(total=len(pendentes), desc=MODELO, unit="proc",
                 dynamic_ncols=True, mininterval=0.5, smoothing=0.05)
    recebidos = 0
    inicio = time.time()
    try:
        while recebidos < len(pendentes) and not PARAR.is_set():
            try:
                r = fila_resultados.get(timeout=1)
            except queue.Empty:
                if all(f.done() for f in futures):
                    tqdm.write("Todos os workers pararam. Encerrando (veja o log).")
                    break
                continue
            if not aplicar(r):
                continue
            recebidos += 1
            barra.update(1)
            barra.set_postfix(ok=cont["ok"], sem_json=cont["sem_json"],
                              erros=cont["erros"], alertas=cont["alertas"],
                              refresh=False)
            if recebidos % SALVAR_A_CADA == 0:
                salvar_csv(res_df, CSV_SAIDA)
    finally:
        PARAR.set()
        while True:
            try:
                aplicar(fila_resultados.get_nowait())
            except queue.Empty:
                break
        barra.close()
        salvar_csv(res_df, CSV_SAIDA)
        print(f"Checkpoint salvo ({cont['ok']} processos OK nesta execução).")
        pool.shutdown(wait=True, cancel_futures=True)

    seg = time.time() - inicio
    proc = (res_df["status"] == "OK").sum()
    print(f"\nConcluído: {CSV_SAIDA}")
    print(f"  processos analisados: {proc}/{len(res_df)}")
    print(f"  sem JSON (reprocessa na próxima): {cont['sem_json']}   erros: {cont['erros']}")
    print(f"  com alerta: {res_df['alerta'].notna().sum()}")
    if recebidos:
        print(f"  vazão: {recebidos / seg * 60:.1f} processos/min")


def executar():
    """Roda main() em thread própria: Ctrl+C/Interrupt vira parada limpa."""
    PARAR.clear()
    erros = []

    def alvo():
        try:
            main()
        except BaseException as e:  # noqa
            erros.append(e)

    t = threading.Thread(target=alvo, name="main")
    t.start()
    try:
        while t.is_alive():
            t.join(0.5)
    except KeyboardInterrupt:
        PARAR.set()
        print("\nInterrupção recebida: salvando o checkpoint. Aguarde...")
        while t.is_alive():
            try:
                t.join(0.5)
            except KeyboardInterrupt:
                print("Ainda salvando, aguarde...")
    if erros:
        raise erros[0]


if __name__ == "__main__":
    executar()
