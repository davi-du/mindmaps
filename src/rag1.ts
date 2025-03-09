import { indexDocuments } from "./indexing.ts";
import { llm, embeddings, vectorStore } from "../app.ts";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { Document } from "@langchain/core/documents";
import { Annotation } from "@langchain/langgraph";
import { StateGraph } from "@langchain/langgraph";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { z } from "zod";
import { template } from "./prompts/rag_template.js";

/* Indexing */
const allSplits = await indexDocuments("./pdf_files", "https://en.wikipedia.org/wiki/Interpreter_(computing)");

/* Template del prompt per generare risposte basate sul contesto */
const promptTemplate = ChatPromptTemplate.fromMessages([
  ["user", template],
]);

/* DOCUMENT CLASSIFICATION: Classifico i documenti in sezioni */
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

/* VECTOR STORE QA: Creo un nuovo Vector Store per QA */
const vectorStoreQA = new MemoryVectorStore(embeddings);
await vectorStoreQA.addDocuments(allSplits);
const searchSchema = z.object({
  query: z.string().describe("Search query to run."),
  section: z.enum(["beginning", "middle", "end"]).describe("Section to query."),
});
const structuredLlm = llm.withStructuredOutput(searchSchema);

// Variabile globale per tenere traccia delle domande e risposte precedenti
let previousQA = "";

/* STATE GRAPH QA: Definizione del grafo con analisi della query */
const StateAnnotationQA = Annotation.Root({
  question: Annotation<string>,
  search: Annotation<z.infer<typeof searchSchema>>,
  context: Annotation<Document[]>,
  answer: Annotation<string>,
  previous_qa: Annotation<string>, // Aggiungiamo una annotazione per le Q&A precedenti
});

/*
//funziona con tutti tranne groq
const analyzeQuery = async (state) => {
  const result = await structuredLlm.invoke(state.question);
  return { search: result, previous_qa: previousQA };
};
*/

//da usare con groq
const analyzeQuery = async (state) => {
  // Use a simple approach without structured output
  const prompt = `Analyze this question: "${state.question}"
  
  Based on the question, determine:
  1. The key search terms to use for finding relevant information
  2. Which section to search in (choose only from: beginning, middle, end)
  
  Respond exactly in this format:
  Query: [search terms]
  Section: [beginning/middle/end]`;
  
  const response = await llm.invoke(prompt);
  
  // Extract data from the model's response
  const content = typeof response.content === 'string' ? response.content : response.text;
  const queryMatch = content.match(/Query:\s*(.+?)(?=\n|$)/i);
  const sectionMatch = content.match(/Section:\s*(beginning|middle|end)/i);
  
  const query = queryMatch ? queryMatch[1].trim() : state.question;
  const section = sectionMatch ? sectionMatch[1].toLowerCase() : "beginning";
  
  //console.log(`Query analysis: '${query}', section: '${section}'`);
  
  return { search: { query, section }, previous_qa: previousQA }; //groq vuole anche la sezione
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
    previous_qa: state.previous_qa || "No previous questions.",
  });
  const response = await llm.invoke(messages);
  
  // Aggiorniamo la variabile globale delle Q&A precedenti
  previousQA += `Question: ${state.question}\nAnswer: ${response.content}\n\n`;
  
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

//TEST
/* TEST RETRIEVAL E GENERAZIONE */
/* attraverso previousQA ora tiene traccia della conversazione, viene aggiunta al contesto */
//prima domanda
let inputs = { question: "What is an interpreter?" };
const result = await graphQA.invoke(inputs);
console.log(`\nQuestion: ${result["question"]}`);
console.log(`\nAnswer: ${result["answer"]}`);

/* TEST QA SYSTEM */
//seconda domanda
let inputsQA = {
  question: "During the software development cycle, programmers make frequent changes to the source code, how can an interpreter help?",
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
  console.log(`\nAnswer: ${outputQA.generateQA.answer} \n\n`);
} else {
  console.log(`\nAnswer: Unable to extract answer from response`);
}

//terza domanda
let inputsQA1 = {
  question: "What is the name of the technique where a program in L0 executes programs written in L1 by interpreting each instruction without first generating a new program in L0?",
  //question: "Tell me how does it work and what's the paragraph name where it is in the file.",
};
let outputQA1;

console.log("\n====\n");
for await (const chunk of await graphQA.stream(inputsQA1, { streamMode: "updates" })) {
  console.log(chunk);
  outputQA1 = chunk;
  console.log("\n====\n");
}
console.log(`\nQuestion: ` + inputsQA1.question);
if (outputQA1 && outputQA1.generateQA && outputQA1.generateQA.answer) {
  console.log(`\nAnswer: ${outputQA1.generateQA.answer} \n\n`);
} else {
  console.log(`\nAnswer: Unable to extract answer from response`);
}

//quarta domanda
let inputsQA2 = {
  question: "Is MATLAB an interpreted language or a compiled language?",
};
let outputQA2;

console.log("\n====\n");
for await (const chunk of await graphQA.stream(inputsQA2, { streamMode: "updates" })) {
  console.log(chunk);
  outputQA2 = chunk;
  console.log("\n====\n");
}
console.log(`\nQuestion: ` + inputsQA2.question);
if (outputQA2 && outputQA2.generateQA && outputQA2.generateQA.answer) {
  console.log(`\nAnswer: ${outputQA2.generateQA.answer} \n\n`);
} else {
  console.log(`\nAnswer: Unable to extract answer from response`);
}

//quinta domanda
let inputsQA3 = {
  question: "Can you summarize the main benefits of using it and tell me how many questions I asked you?",
  //question: "Can you summarize the key advantages of using it?",
};
let outputQA3;

console.log("\n====\n");
for await (const chunk of await graphQA.stream(inputsQA3, { streamMode: "updates" })) {
  console.log(chunk);
  outputQA3 = chunk;
  console.log("\n====\n");
}
console.log(`\nQuestion: ` + inputsQA3.question);
if (outputQA3 && outputQA3.generateQA && outputQA3.generateQA.answer) {
  console.log(`\nAnswer: ${outputQA3.generateQA.answer} \n\n`);
} else {
  console.log(`\nAnswer: Unable to extract answer from response`);
}