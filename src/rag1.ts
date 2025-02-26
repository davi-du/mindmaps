import "cheerio";
//import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { llm, embeddings, vectorStore } from "../app.ts";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { Document } from "@langchain/core/documents";
import { Annotation } from "@langchain/langgraph";
import { StateGraph } from "@langchain/langgraph";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { z } from "zod";

/* 1. DOCUMENT LOADING: Caricamento PDF dalla directory */
console.log("\n\n DOCUMENT LOADER \n");
const dataPath = "./pdf_files"; // Carico la directory direttamente
const directoryLoader = new DirectoryLoader(dataPath, {
  ".pdf": (path: string) => new PDFLoader(path),
});
const directoryDocs = await directoryLoader.load();
console.log("Pagine caricate:", directoryDocs.length);

/*
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

// Text Splitter: Divide il contenuto caricato in sottodocumenti
console.log("\n\n TEXT SPLITTER \n");
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,  // Dimensione massima in caratteri di un sottodocumento
  chunkOverlap: 200,    // Sovrapposizione tra i sottodocumenti?
});
const allSplits = await splitter.splitDocuments(docs);
console.log(`Split blog post into ${allSplits.length} sub-documents.`);
console.log(allSplits[0].pageContent.slice(0, 500));
*/

/*
// Carica il contenuto di un file PDF
const prova = "./pdf_files/prova.pdf";

const loader = new PDFLoader(prova);
const docs = await loader.load();
docs[0];
console.log(docs[0].metadata);
console.log(docs[0].pageContent.slice(0, 500));
*/

/* 2. TEXT SPLITTING: Divido i documenti in chunk */
console.log("\n\n TEXT SPLITTER \n");
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const allSplits = await splitter.splitDocuments(directoryDocs);
console.log(allSplits[0]);

/* 3. VECTOR STORE: Memorizzo i documenti nel Vector Store */
console.log("\n\n VECTOR STORE\n");
await vectorStore.addDocuments(allSplits);
console.log("Numero totale di documenti:", vectorStore.memoryVectors.length);

/* 4. PROMPT TEMPLATE per generare risposte basate sul contesto */
const template = `Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "Grazie per averlo chiesto!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:`;
const promptTemplate = ChatPromptTemplate.fromMessages([
  ["user", template],
]);

/* 5. STATE GRAPH: Definizione del grafo per retrieval e generazione */
const StateAnnotation = Annotation.Root({
  question: Annotation,
  context: Annotation,
  answer: Annotation,
});
const retrieve = async (state) => {
  const retrievedDocs = await vectorStore.similaritySearch(state.question);
  return { context: retrievedDocs };
};
const generate = async (state) => {
  const docsContent = state.context.map((doc) => doc.pageContent).join("\n");
  const messages = await promptTemplate.invoke({
    question: state.question,
    context: docsContent,
  });
  const response = await llm.invoke(messages);
  return { answer: response.content };
};
const graph = new StateGraph(StateAnnotation)
  .addNode("retrieve", retrieve)
  .addNode("generate", generate)
  .addEdge("__start__", "retrieve")
  .addEdge("retrieve", "generate")
  .addEdge("generate", "__end__")
  .compile();

/* 6. DOCUMENT CLASSIFICATION: Classifico i documenti in sezioni */
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

/* 7. VECTOR STORE QA: Creo un nuovo Vector Store per QA */
const vectorStoreQA = new MemoryVectorStore(embeddings);
await vectorStoreQA.addDocuments(allSplits);
const searchSchema = z.object({
  query: z.string().describe("Search query to run."),
  section: z.enum(["beginning", "middle", "end"]).describe("Section to query."),
});
const structuredLlm = llm.withStructuredOutput(searchSchema);

/* 8. STATE GRAPH QA: Definizione del grafo con analisi della query */
const StateAnnotationQA = Annotation.Root({
  question: Annotation<string>,
  search: Annotation<z.infer<typeof searchSchema>>,
  context: Annotation<Document[]>,
  answer: Annotation<string>,
});

const analyzeQuery = async (state) => {
  const result = await structuredLlm.invoke(state.question);
  return { search: result };
};

const retrieveQA = async (state) => {
  const filter = (doc) => doc.metadata.section === state.search.section;
  const retrievedDocs = await vectorStore.similaritySearch(
    state.search.query,
    2,
    filter
  );
  return { context: retrievedDocs };
};

const generateQA = async (state) => {
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

/* 9. TEST RETRIEVAL E GENERAZIONE */
let inputs = { question: "What is an interpreter?" };
const result = await graph.invoke(inputs);
console.log(`\nQuestion: ${result["question"]}`);
console.log(`\nAnswer: ${result["answer"]}`);

/* 10. TEST QA SYSTEM */
let inputsQA = {
  question: "Tell me how does it work and what's the paragraph name where it is in the file.",
};
let outputQA;

console.log("\n====\n");
for await (const chunk of await graphQA.stream(inputsQA, { streamMode: "updates" })) {
  console.log(chunk);
  outputQA = chunk;
  console.log("\n====\n");
}
console.log(`\nQuestion: ` + inputsQA.question);
if (outputQA && outputQA.generateQA && outputQA.generateQA.answer) {
  console.log(`\nAnswer: ${outputQA.generateQA.answer}`);
} else {
  console.log(`\nAnswer: Unable to extract answer from response`);
}