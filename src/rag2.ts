import { llm, embeddings, vectorStore } from "../app.ts";
import "cheerio"; // Importa Cheerio per il web scraping
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import { z } from "zod"; // Libreria per la validazione degli schemi
import { tool } from "@langchain/core/tools";
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { BaseMessage, isAIMessage } from "@langchain/core/messages";
import { MemorySaver } from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

// 1 - Caricamento e suddivisione del testo
const pTagSelector = "p";
const cheerioLoader = new CheerioWebBaseLoader(
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  {
    selector: pTagSelector, // Selezioniamo solo i paragrafi
  }
);

const docs = await cheerioLoader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000, // Lunghezza massima di ogni segmento
  chunkOverlap: 200, // Sovrapposizione tra segmenti
});
const allSplits = await splitter.splitDocuments(docs);

// Creazione del grafo di stato
//const graph = new StateGraph(MessagesAnnotation);

// Definizione dello strumento di retrieval
// Schema per la query di retrieval
const retrieveSchema = z.object({ query: z.string() });

// Funzione per recuperare documenti rilevanti dalla memoria vettoriale
const retrieve = tool(
  async ({ query }) => {
    const retrievedDocs = await vectorStore.similaritySearch(query, 2); // Cerca documenti simili
    const serialized = retrievedDocs
      .map(
        (doc) => `Source: ${doc.metadata.source}\nContent: ${doc.pageContent}`
      )
      .join("\n");

    return [serialized, retrievedDocs]; //?modificato con tupla in output, tutorial non lo fa
  },
  {
    name: "retrieve",
    description: "Retrieve information related to a query.",
    schema: retrieveSchema,
    responseFormat: "content_and_artifact", 
  }
);


// Definizione delle funzioni per il flusso del grafo

// Funzione per generare una risposta dell'AI, 
// decide se rispondere direttamente o usare uno strumento per recuperare informazioni.
async function queryOrRespond(state) {
  const llmWithTools = llm.bindTools([retrieve]); // Associa lo strumento di retrieval
  const response = await llmWithTools.invoke(state.messages); // Ottiene la risposta dall'LLM
  return { messages: [...state.messages, response] }; // Aggiunge la risposta allo stato
}

// Nodo per l'esecuzione dello strumento di retrieval
const tools = new ToolNode([retrieve]);

// Funzione per generare una risposta basata sui documenti recuperati
async function generate(state) {
    let toolMessages = state.messages.filter(
    (message) => message instanceof ToolMessage
    );

    // Estrai il contenuto dei documenti recuperati
    const docsContent = toolMessages.map((doc) => doc.content).join("\n");
    const systemMessageContent =
    "You are an assistant for question-answering tasks. " +
    "Use the following pieces of retrieved context to answer " +
    "the question. If you don't know the answer, say that you " +
    "don't know. Use three sentences maximum and keep the " +
    "answer concise." +
    "\n\n" +
    `${docsContent}`;

    // Filtra i messaggi rilevanti per la generazione della risposta
    const conversationMessages = state.messages.filter(
    (message) =>
        message instanceof HumanMessage ||
        message instanceof SystemMessage ||
        (message instanceof AIMessage && (message.tool_calls?.length ?? 0) === 0)
    );

    // Costruisce il prompt
    const prompt = [new SystemMessage(systemMessageContent), ...conversationMessages];

    // Genera la risposta
    const response = await llm.invoke(prompt);
    return { messages: [...state.messages, response] };
}

// Creazione del grafo
const graphBuilder = new StateGraph(MessagesAnnotation)
    .addNode("queryOrRespond", queryOrRespond)
    .addNode("tools", tools)
    .addNode("generate", generate)
    .addEdge("__start__", "queryOrRespond")
    .addConditionalEdges("queryOrRespond", toolsCondition, {
    __end__: "__end__",
    tools: "tools",
    })
    .addEdge("tools", "generate")
    .addEdge("generate", "__end__");

const graph = graphBuilder.compile();

// Funzione di stampa formattata dei risultati
const prettyPrint = (message: BaseMessage) => {
    let txt = `[${message._getType()}]: ${message.content}`;
    if ((isAIMessage(message) && message.tool_calls?.length) || 0 > 0) {
      const tool_calls = (message as AIMessage)?.tool_calls
        ?.map((tc) => `- ${tc.name}(${JSON.stringify(tc.args)})`)
        .join("\n");
      txt += ` \nTools: \n${tool_calls}`;
    }
    console.log(txt);
};

// test
let inputs1 = { messages: [new HumanMessage("Hello")] };

for await (const step of await graph.stream(inputs1, {
  streamMode: "values",
})) {
  const lastMessage = step.messages[step.messages.length - 1];
  prettyPrint(lastMessage);
  console.log("-----\n");
}

// test con retrieval
let inputs2 = {
    messages: [{ role: "user", content: "What is Task Decomposition?" }],
  };
  
  for await (const step of await graph.stream(inputs2, {
    streamMode: "values",
  })) {
    const lastMessage = step.messages[step.messages.length - 1];
    prettyPrint(lastMessage);
    console.log("-----\n");
  }

//memory
const checkpointer = new MemorySaver();
const graphWithMemory = graphBuilder.compile({ checkpointer });

const threadConfig = {
  configurable: { thread_id: "davidu" },
  streamMode: "values" as const,
};


let inputs3 = {
  messages: [{ role: "user", content: "What is Task Decomposition?" }],
};

for await (const step of await graphWithMemory.stream(inputs3, threadConfig)) {
  const lastMessage = step.messages[step.messages.length - 1];
  prettyPrint(lastMessage);
  console.log("-----\n");
}

let inputs4 = {
  messages: [
    { role: "user", content: "Can you look up some common ways of doing it?" },
  ],
};

for await (const step of await graphWithMemory.stream(inputs4, threadConfig)) {
  const lastMessage = step.messages[step.messages.length - 1];
  prettyPrint(lastMessage);
  console.log("-----\n");
}
/*
//agents
const agent = createReactAgent({ llm: llm, tools: [retrieve] });

let inputMessage = `What is the standard method for Task Decomposition?
Once you get the answer, look up common extensions of that method.`;

let inputs5 = { messages: [{ role: "user", content: inputMessage }] };

for await (const step of await agent.stream(inputs5, {
  streamMode: "values",
})) {
  const lastMessage = step.messages[step.messages.length - 1];
  prettyPrint(lastMessage);
  console.log("-----\n");
}
*/