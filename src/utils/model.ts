import { debug, userProvidedAPIKey } from '../constants'
import { terms } from '../constants'

export const ModelAvailable = 'codestral-2405'  //globalBestModelAvailable

export type ModelForMagic = 'codestral-2405' | 'codestral-latest'

export const models = {
  smarter: ModelAvailable as ModelForMagic,
  faster: ModelAvailable as ModelForMagic,
}

const temperatures = {
  response: 0.7,
  parsing: 0.3,
}

export interface AIChatCompletionResponseStream {
  id: string
  object: string
  created: number
  model: string
  choices: {
    delta: {
      content?: string
      role?: string
    }
  }[]
  index: number
  finish_reason: string
}

export interface Prompt {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export const sanitizePrompts = (prompts: Prompt[]): Prompt[] => {
  let systemPrompt: Prompt | undefined = undefined
  const rest: Prompt[] = []

  prompts.forEach((p) => {
    if (p.role === 'system') {
      if (!systemPrompt) {
        systemPrompt = p
      }
    } else {
      rest.push(p)
    }
  })

  return systemPrompt ? [systemPrompt, ...rest] : rest
}

export const getCompletionOptions = (
  prompts: Prompt[],
  model: ModelForMagic,
  temperature: number | undefined,
  token: number | undefined,
  stream = false,
) => {
  return {
    messages: prompts,
    model: model,
    temperature: temperature,
    max_tokens: token,
    n: 1,
    top_p: 1,
    frequency_penalty: 0,
    presence_penalty: 0,
    stream: stream,
  }
}

const getRequestOptions = (options: any) => {
  return {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization:
        'Bearer ' +
        String(
          debug
            ? "fw_3ZV2fDTtt7L2QcprNEkF5Afn"
            : userProvidedAPIKey.current,
        ),
    },
    body: JSON.stringify(options),
  }
}

export const getAICompletion = async (
  prompts: Prompt[],
  model: ModelForMagic,
  temperature = temperatures.response,
  token = 1024,
) => {
  const cleanedPrompts = sanitizePrompts(prompts)

  const resolvedModel = model // Assuming no mapping is needed; replace with actual logic if required
  console.log(`Requesting model: ${resolvedModel}`, cleanedPrompts)

  const options = getCompletionOptions(cleanedPrompts, resolvedModel, temperature, token)
  const requestOptions = getRequestOptions({ ...options, model: resolvedModel })

  const endpoint = 'https://api.fireworks.ai/inference/v1/chat/completions';

  const response = await fetch(endpoint, requestOptions)

  if (!response.ok) {
    const errorText = await response.text()
    console.error(`API Request failed: ${response.status} – ${errorText}`)
    return { error: errorText }
  }

  const data = await response.json()
  return data
}


export const streamAICompletion = async (
  prompts: Prompt[],
  model: ModelForMagic,
  streamFunction: (data: any, freshStream: boolean) => void,
  freshStream: boolean,
  temperature = temperatures.response,
  token = 2048,
) => {
  const cleanedPrompts = sanitizePrompts(prompts)
  const resolvedModel = terms[model] ?? model
  console.log(`Streaming from model: ${resolvedModel}`, cleanedPrompts)

  const options = getCompletionOptions(cleanedPrompts, resolvedModel as ModelForMagic, temperature, token, true)
  const requestOptions = getRequestOptions({ ...options, model: resolvedModel })

  const endpoint = 'https://api.fireworks.ai/inference/v1/chat/completions';

  const response = await fetch(endpoint, requestOptions)
  const reader = response.body?.getReader()
  if (!reader) return

  const dataParticle = { current: '' }

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    const data = new TextDecoder('utf-8').decode(value)
    const dataObjects = data
      .split('\n')
      .map(d => d.trim())
      .filter(d => d.length > 0)

    dataObjects.forEach(d => {
      d = d.replace(/^data: /, '').trim()
      if (d === '[DONE]') return

      let parsingSuccessful = false
      try {
        const dataObject = JSON.parse(d) as AIChatCompletionResponseStream
        if (
          dataObject.choices?.length > 0 &&
          dataObject.choices[0].delta.content
        ) {
          streamFunction(dataObject, freshStream)
        }
        parsingSuccessful = true
        dataParticle.current = ''
      } catch {
        dataParticle.current += d
      }

      if (!parsingSuccessful) {
        try {
          const dataObject = JSON.parse(dataParticle.current) as AIChatCompletionResponseStream
          if (
            dataObject.choices?.length > 0 &&
            dataObject.choices[0].delta.content
          ) {
            streamFunction(dataObject, freshStream)
          }
          dataParticle.current = ''
        } catch { }
      }
    })
  }
}

export const parseAIResponseToObjects = async (
  prompts: Prompt[],
  model: ModelForMagic,
  temperature = temperatures.parsing,
  token = 2048,
  retries = 3,
  delay = 2000,
) => {
  const cleanedPrompts = sanitizePrompts(prompts)
  const resolvedModel = terms[model] ?? model
  console.log("Prompts sent:", JSON.stringify(cleanedPrompts, null, 2))

  const options = getCompletionOptions(cleanedPrompts, resolvedModel as ModelForMagic, temperature, token, false)
  const requestOptions = getRequestOptions({ ...options, model: resolvedModel })

  const endpoint = 'https://api.fireworks.ai/inference/v1/chat/completions';

  let currentDelay = delay

  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const response = await fetch(endpoint, requestOptions)

      if (response.status === 429) {
        console.warn(`Rate limit exceeded. Retrying in ${currentDelay}ms...`)
        await new Promise((resolve) => setTimeout(resolve, currentDelay))
        currentDelay *= 2
        continue
      }

      if (!response.ok) {
        const errorText = await response.text()
        console.error('API Request failed:', response.status, errorText)
        return { error: `API request failed with status ${response.status}`, choices: [] }
      }

      const data = await response.json()

      if (!data || !data.choices || data.choices.length === 0) {
        console.error('Invalid API response structure:', data)
        return { error: 'Invalid API response structure', choices: [] }
      }

      return data
    } catch (error) {
      console.error('Error fetching AI response:', error)
      if (attempt === retries - 1) {
        return {
          error: (error as Error).message || 'Unknown error',
          choices: [],
        }
      }
    }
  }

  return { error: 'Failed to get response from API after retries', choices: [] }
}


export const getTextFromModelResponse = (response: any): string => {
  if (!response || !response.choices || response.choices.length === 0) {
    console.error("Invalid API response", response);
    return '';
  }
  return response.choices[0]?.message?.content ?? '';
};

export const getTextFromStreamResponse = (
  response: AIChatCompletionResponseStream,
): string => {
  return response.choices[0].delta.content ?? ''
} 
