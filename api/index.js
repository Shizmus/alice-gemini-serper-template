const DEFAULT_MODEL = "openai/gpt-oss-120b";
const DEFAULT_FALLBACK_MODEL = "llama-3.3-70b-versatile";
const MAX_HISTORY_MESSAGES = 8;
const MAX_HISTORY_MESSAGE_CHARS = 1200;
const MAX_ALICE_TEXT = 1024;
const TOTAL_RESPONSE_BUDGET_MS = 2900;
const SERPER_TIMEOUT_MS = 900;
const GROQ_TIMEOUT_MS = 2200;
const MIN_RETRY_MS = 700;

const SYSTEM_PROMPT = [
  "Ты — полезный голосовой помощник в навыке Яндекс Алисы.",
  "Отвечай естественно, по-русски, кратко и так, чтобы ответ было удобно слушать.",
  "Обычно используй 1–3 коротких предложения. Не используй Markdown, таблицы, списки с символами или длинные вступления.",
  "Если вопрос требует актуальных данных и переданы результаты поиска, опирайся на них; не выдумывай отсутствующие факты.",
  "Если не знаешь ответа, честно скажи об этом.",
  "У тебя нет инструментов и функций: никогда не пытайся вызывать поиск, браузер или любые функции, отвечай только текстом."
].join(" ");

function aliceResponse(text, endSession = false, history = []) {
  return {
    version: "1.0",
    response: {
      text: fitAliceText(text),
      end_session: endSession
    },
    session_state: { history }
  };
}

function fitAliceText(value) {
  const text = String(value ?? "")
    .replace(/[\r\n\t]+/g, " ")
    .replace(/\s{2,}/g, " ")
    .trim();

  if (text.length <= MAX_ALICE_TEXT) return text || "Не удалось подготовить ответ.";

  const candidate = text.slice(0, MAX_ALICE_TEXT - 1);
  const sentenceEnd = Math.max(
    candidate.lastIndexOf(". "),
    candidate.lastIndexOf("! "),
    candidate.lastIndexOf("? ")
  );
  if (sentenceEnd >= 300) return candidate.slice(0, sentenceEnd + 1).trim();

  const lastSpace = candidate.lastIndexOf(" ");
  return (lastSpace > 200 ? candidate.slice(0, lastSpace) : candidate).trim() + "…";
}

function isSearchIntent(text) {
  return /\b(latest|today|current|news|weather|price|exchange rate|recent|search|look up)\b|сегодня|сейчас|последн|новост|погод|курс валют|найди|поищи|актуальн|сколько стоит|свежие данные/i.test(text);
}

function safeHistory(history) {
  if (!Array.isArray(history)) return [];
  return history
    .filter(item => item && (item.role === "user" || item.role === "model" || item.role === "assistant") && typeof item.text === "string")
    .slice(-MAX_HISTORY_MESSAGES)
    .map(item => ({
      role: item.role === "model" ? "assistant" : item.role,
      text: item.text.slice(0, MAX_HISTORY_MESSAGE_CHARS)
    }));
}

async function fetchJson(url, options, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), Math.max(100, timeoutMs));
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    const data = await response.json().catch(() => ({}));
    return { response, data };
  } finally {
    clearTimeout(timer);
  }
}

async function searchWeb(query, timeoutMs) {
  const key = process.env.SERPER_API_KEY;
  if (!key || timeoutMs < 150) return "";

  try {
    const { response, data } = await fetchJson(
      "https://google.serper.dev/search",
      {
        method: "POST",
        headers: {
          "X-API-KEY": key,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ q: query, gl: "ru", hl: "ru", num: 3 })
      },
      Math.min(SERPER_TIMEOUT_MS, timeoutMs)
    );

    if (!response.ok) {
      console.warn("Serper request failed with HTTP", response.status);
      return "";
    }

    const items = (Array.isArray(data.organic) ? data.organic : [])
      .slice(0, 3)
      .map((item, i) => {
        const title = String(item.title || "").slice(0, 180);
        const snippet = String(item.snippet || "").slice(0, 350);
        const link = String(item.link || "").slice(0, 240);
        return `${i + 1}. ${title}: ${snippet} ${link}`.trim();
      });

    const answerBox = data.answerBox?.answer || data.answerBox?.snippet;
    if (answerBox) items.unshift(`Краткий ответ поисковика: ${String(answerBox).slice(0, 350)}`);
    return items.length ? items.join("\n") : "";
  } catch (error) {
    console.warn("Serper request failed or timed out; continuing without search.", error?.name || "");
    return "";
  }
}

function isModelName(model) {
  return /^[A-Za-z0-9._/-]+$/.test(model);
}

async function callGroq(model, messages, timeoutMs) {
  const key = process.env.GROQ_API_KEY;
  const isReasoning = model.startsWith("openai/gpt-oss");

  const payload = {
    model,
    messages,
    temperature: 0.5,
    max_tokens: isReasoning ? 1200 : 300
  };
  // gpt-oss: рассуждения тратят лимит токенов, поэтому минимизируем их
  if (isReasoning) payload.reasoning_effort = "low";

  const { response, data } = await fetchJson(
    "https://api.groq.com/openai/v1/chat/completions",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${key}`
      },
      body: JSON.stringify(payload)
    },
    timeoutMs
  );
  return { response, data };
}

async function askGroq(text, history, searchContext, timeoutMs) {
  const key = process.env.GROQ_API_KEY;
  if (!key) throw new Error("MISSING_GROQ_KEY");
  if (timeoutMs < 250) throw new Error("TIME_BUDGET_EXHAUSTED");

  const model = String(process.env.GROQ_MODEL || DEFAULT_MODEL).trim();
  const fallbackModel = String(process.env.GROQ_FALLBACK_MODEL || DEFAULT_FALLBACK_MODEL).trim();
  if (!isModelName(model) || !isModelName(fallbackModel)) throw new Error("INVALID_MODEL_NAME");

  const messages = [
    { role: "system", content: SYSTEM_PROMPT },
    ...safeHistory(history).map(item => ({
      role: item.role,
      content: item.text
    }))
  ];

  const prompt = searchContext
    ? `${text}\n\nРезультаты поиска в интернете:\n${searchContext}\nИспользуй их только там, где они относятся к вопросу.`
    : text;
  messages.push({ role: "user", content: prompt });

  const startedAt = Date.now();
  const left = () => timeoutMs - (Date.now() - startedAt);

  const attempts = [model];
  if (fallbackModel && fallbackModel !== model) attempts.push(fallbackModel);

  let lastError = new Error("GROQ_REQUEST_FAILED");

  for (let i = 0; i < attempts.length; i++) {
    const current = attempts[i];
    const budget = Math.min(GROQ_TIMEOUT_MS, left());
    if (i > 0 && budget < MIN_RETRY_MS) break;

    try {
      const { response, data } = await callGroq(current, messages, budget);

      if (!response.ok) {
        const status = Number(response.status);
        const code = data?.error?.code;
        console.error("Groq request failed", { model: current, status, code, message: data?.error?.message });
        if (status === 401 || status === 403) throw new Error("GROQ_KEY_REJECTED");
        if (status === 429) throw new Error("GROQ_QUOTA");
        // tool_use_failed: модель попыталась вызвать несуществующий инструмент -> пробуем запасную модель
        lastError = new Error(code === "tool_use_failed" ? "GROQ_TOOL_CALL" : "GROQ_REQUEST_FAILED");
        continue;
      }

      const choice = data?.choices?.[0];
      const answer = choice?.message?.content?.trim() || "";
      if (!answer) {
        console.error("Groq empty content", { model: current, finish_reason: choice?.finish_reason });
        lastError = new Error("GROQ_EMPTY_RESPONSE");
        continue;
      }
      return answer;
    } catch (error) {
      if (["GROQ_KEY_REJECTED", "GROQ_QUOTA"].includes(error?.message)) throw error;
      lastError = error?.name === "AbortError" ? new Error("TIME_BUDGET_EXHAUSTED") : error;
    }
  }

  throw lastError;
}

function errorForUser(error) {
  switch (error?.message) {
    case "MISSING_GROQ_KEY":
      return "Сервер пока не настроен. Проверьте переменную GROQ_API_KEY в настройках Vercel.";
    case "GROQ_KEY_REJECTED":
      return "Не удалось подключиться к Groq. Проверьте API-ключ в настройках Vercel.";
    case "GROQ_QUOTA":
      return "Сейчас превышен лимит запросов к Groq. Попробуйте немного позже.";
    case "TIME_BUDGET_EXHAUSTED":
      return "Не успела подготовить ответ. Попробуйте задать вопрос ещё раз, чуть короче.";
    default:
      return "Не получилось получить ответ от нейросети. Попробуйте ещё раз чуть позже.";
  }
}

export default async function handler(req, res) {
  if (req.method === "GET") {
    return res.status(200).json({
      ok: true,
      service: "alice-groq-serper",
      message: "Webhook is ready. Yandex Dialogs should send POST requests to /api."
    });
  }

  if (req.method !== "POST") {
    res.setHeader("Allow", "GET, POST");
    return res.status(405).json({ error: "Method not allowed" });
  }

  const startedAt = Date.now();
  const remainingMs = () => TOTAL_RESPONSE_BUDGET_MS - (Date.now() - startedAt);
  const body = req.body && typeof req.body === "object" ? req.body : {};
  const text = String(body?.request?.command || body?.request?.original_utterance || "").trim();
  const oldHistory = safeHistory(body?.state?.session?.history);

  if (!text) {
    return res.status(200).json(aliceResponse(
      "Привет! Я могу ответить на вопрос или найти актуальную информацию. Что хочешь узнать?",
      false,
      oldHistory
    ));
  }

  try {
    let searchContext = "";
    if (isSearchIntent(text) && process.env.SERPER_API_KEY) {
      searchContext = await searchWeb(text, Math.min(SERPER_TIMEOUT_MS, Math.max(0, remainingMs() - 1500)));
    }

    const answer = await askGroq(
      text,
      oldHistory,
      searchContext,
      Math.max(0, Math.min(GROQ_TIMEOUT_MS, remainingMs() - 150))
    );

    const history = [
      ...oldHistory,
      { role: "user", text: text.slice(0, MAX_HISTORY_MESSAGE_CHARS) },
      { role: "assistant", text: answer.slice(0, MAX_HISTORY_MESSAGE_CHARS) }
    ].slice(-MAX_HISTORY_MESSAGES);

    return res.status(200).json(aliceResponse(answer, false, history));
  } catch (error) {
    const safeError = error?.name === "AbortError" ? new Error("TIME_BUDGET_EXHAUSTED") : error;
    console.warn("Alice webhook returned a fallback response:", safeError?.message || "unknown error");
    return res.status(200).json(aliceResponse(errorForUser(safeError), false, oldHistory));
  }
}

export { aliceResponse, fitAliceText, isSearchIntent, safeHistory };
