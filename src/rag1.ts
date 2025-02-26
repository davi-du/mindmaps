import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { llm, embeddings, vectorStore } from "../app.ts";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { Document } from "@langchain/core/documents";
import { Annotation } from "@langchain/langgraph";
import { StateGraph } from "@langchain/langgraph";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { z } from "zod";

    /* INDEXING */
// Document Loader: Carica il contenuto di una pagina web utilizzando Cheerio
console.log("\n\n DOCUMENT LOADER \n");
const pTagSelector = "p";
const cheerioLoader = new CheerioWebBaseLoader(
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  {
    selector: pTagSelector, // Selettore CSS per estrarre i paragrafi
  }
);
const docs = await cheerioLoader.load();    // Carica il contenuto della pagina


console.log(docs[0].pageContent.slice(0, 500));
console.assert(docs.length === 1);
console.log(`Numero caratteri: ${docs[0].pageContent.length}`);

// Text Splitter: Divide il contenuto caricato in sottodocumenti più piccoli
console.log("\n\n TEXT SPLITTER \n");
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,  // Dimensione massima in caratteri di un sottodocumento
  chunkOverlap: 200,    // Sovrapposizione tra i sottodocumenti?
});
const allSplits = await splitter.splitDocuments(docs);
console.log(`Split blog post into ${allSplits.length} sub-documents.`);
console.log(allSplits[0].pageContent.slice(0, 500));

// Memorizzazione dei documenti nel Vector Store
console.log("\n\n VECTOR STORE\n");
await vectorStore.addDocuments(allSplits); 
console.log("Numero totale di documenti:", vectorStore.memoryVectors.length);


    /* RETRIEVAL AND GENERATE */
// Prompt Template per generare risposte basate sul contesto
//const promptTemplate = ragPrompt;
const template = `Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "grazie compare!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:`;

const promptTemplate = ChatPromptTemplate.fromMessages([
    ["user", template],
]);

// Stato iniziale per la gestione delle annotazioni nel flusso dati
const InputStateAnnotation = Annotation.Root({
    question: Annotation, 
});

const StateAnnotation = Annotation.Root({
    question: Annotation,
    context: Annotation,
    answer: Annotation,
});

// Funzione di recupero dei documenti affini alla domanda che l'utente ha posto
const retrieve = async (state) => {
    const retrievedDocs = await vectorStore.similaritySearch(state.question);
    return { context: retrievedDocs };
};

// Funzione per generare una risposta in base al contesto recuperato
const generate = async (state) => {
    const docsContent = state.context.map((doc) => doc.pageContent).join("\n");
    const messages = await promptTemplate.invoke({
        question: state.question,   
        context: docsContent,   
    });
    const response = await llm.invoke(messages);
    return { answer: response.content };
};

// Creazione di un grafo di stati per orchestrare il processo di retrieval e generazione
const graph = new StateGraph(StateAnnotation)
    .addNode("retrieve", retrieve)
    .addNode("generate", generate)
    .addEdge("__start__", "retrieve")
    .addEdge("retrieve", "generate")
    .addEdge("generate", "__end__")
    .compile();

// Esempio di test dell'intero sistema
let inputs = { question: "What is Task Decomposition?" };
const result = await graph.invoke(inputs);
if (Array.isArray(result.context)) {
    console.log(result.context.slice(0, 2));
} else {
    console.error("Unexpected type for result.context");
}
console.log(`\nAnswer: ${result["answer"]}`);

// Streaming delle risposte generate
console.log(inputs);
console.log("\n====\n");
for await (const chunk of await graph.stream(inputs, {
  streamMode: "updates",
})) {
  console.log(chunk);
  console.log("\n====\n");
}

// Streaming dei messaggi generati dal grafo
const stream = await graph.stream(inputs, { streamMode: "messages" });
for await (const [message, _metadata] of stream) {
  process.stdout.write(message.content + "|");
}

// Classificazione dei documenti in sezioni per una migliore analisi delle query
const totalDocuments = allSplits.length;
const third = Math.floor(totalDocuments / 3);
allSplits.forEach((document, i) => {
  if (i < third) {
    document.metadata["section"] = "beginning";
  } else if (i < 2 * third) {
    document.metadata["section"] = "middle";
  } else {
    document.metadata["section"] = "end";
  }
});
allSplits[0].metadata;

// Creazione di un nuovo Vector Store per le domande e risposte
const vectorStoreQA = new MemoryVectorStore(embeddings);
await vectorStoreQA.addDocuments(allSplits);

// Schema di validazione per la ricerca di informazioni
const searchSchema = z.object({
  query: z.string().describe("Search query to run."),
  section: z.enum(["beginning", "middle", "end"]).describe("Section to query."),
});
const structuredLlm = llm.withStructuredOutput(searchSchema);

// Definizione di un grafo con analisi della query, retrieval e generazione della risposta
const StateAnnotationQA = Annotation.Root({
  question: Annotation<string>,
  search: Annotation<z.infer<typeof searchSchema>>,
  context: Annotation<Document[]>,
  answer: Annotation<string>,
});

const analyzeQuery = async (state: typeof InputStateAnnotation.State) => {
  const result = await structuredLlm.invoke(state.question);
  return { search: result };
};

const retrieveQA = async (state: typeof StateAnnotationQA.State) => {
  const filter = (doc) => doc.metadata.section === state.search.section;
  const retrievedDocs = await vectorStore.similaritySearch(
    state.search.query,
    2,
    filter
  );
  return { context: retrievedDocs };
};

const generateQA = async (state: typeof StateAnnotationQA.State) => {
  const docsContent = state.context.map((doc) => doc.pageContent).join("\n");
  const messages = await promptTemplate.invoke({
    question: state.question,
    context: docsContent,
  });
  const response = await llm.invoke(messages);
  return { answer: response.content };
};

const graphQA = new StateGraph(StateAnnotationQA)
  .addNode("analyzeQuery", analyzeQuery)
  .addNode("retrieveQA", retrieveQA)
  .addNode("generateQA", generateQA)
  .addEdge("__start__", "analyzeQuery")
  .addEdge("analyzeQuery", "retrieveQA")
  .addEdge("retrieveQA", "generateQA")
  .addEdge("generateQA", "__end__")
  .compile();

  //fine prima parte
  let inputsQA = {
    question: "What does the end of the post say about Task Decomposition?",
  };
  
  console.log(inputsQA);
  console.log("\n====\n");
  for await (const chunk of await graphQA.stream(inputsQA, {
    streamMode: "updates",
  })) {
    console.log(chunk);
    console.log("\n====\n");
  }
