import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatMistralAI, MistralAIEmbeddings } from "@langchain/mistralai";
import { fileURLToPath } from "url";
import path, { dirname } from "path";

import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import cosineSimilarity from "compute-cosine-similarity";

//
const Reset = "\x1b[0m";
const Bright = "\x1b[1m";

const FgRed = "\x1b[31m";
const FgGreen = "\x1b[32m";
const FgYellow = "\x1b[33m";
const FgBlue = "\x1b[34m";

// Simulazione di __dirname per ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const mistralApiKey = "TwLq6jM7zTwo6qcE3vNXokD3MuofaMOA"; //in produzione va tolta eh

//modello di chat 
const llm = new ChatMistralAI({
  model: "mistral-large-latest",
  temperature: 0,
  apiKey: mistralApiKey,
});

const embeddings = new MistralAIEmbeddings({
  model: "mistral-embed",
  apiKey: mistralApiKey,
});

let vectorStore: MemoryVectorStore | null = null;

async function loadAndIndexData() {
  console.log(`${FgGreen}Loading and indexing data...`);
  console.log(`${Reset}`);

  const pdfDir = path.resolve(__dirname, "./data/pdf_files");

  const pdfLoader = new DirectoryLoader(pdfDir, {
  ".pdf": (filePath) => new PDFLoader(filePath),
});


  const webLoader = new CheerioWebBaseLoader(
    "https://en.wikipedia.org/wiki/Interpreter_(computing)",
    { selector: "p" }
  );
  
  //caricamento dei documenti puliti
  const docs = (await pdfLoader.load()).map(doc => ({
    ...doc,
    pageContent: normalizeText(doc.pageContent),
    metadata: { ...doc.metadata, source: "pdf" },
  }));

  const webDocs = (await webLoader.load()).map(doc => ({
    ...doc,
    pageContent: normalizeText(doc.pageContent),
    metadata: { ...doc.metadata, source: "wikipedia" },
  }));

  //const allDocs = [...docs, ...webDocs];
  const allDocsRaw = [...docs, ...webDocs];
  console.log("\n--- All documents before removing duplicates ---\n");
  console.log(allDocsRaw);
  const allDocs = await removeNearDuplicates(allDocsRaw, embeddings);
  console.log("\n--- All documents after removing duplicates ---\n");
  console.log(allDocs);

  //new 
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const splitDocs = await splitter.splitDocuments(allDocs);
  vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);

}





//senza reranking
/*
export async function buildRagContext(question: string): Promise<string> {
  if (!vectorStore) {
    await loadAndIndexData();
  }

  const results = await vectorStore!.similaritySearch(question, 3);
  return results.map((doc) => doc.pageContent).join("\n");
}
*/

//con reranking
export async function buildRagContext(question: string): Promise<string> {
  if (!vectorStore) {
    await loadAndIndexData();
  }

  // faccio fare l'embedding della query
  const queryEmbedding = await embeddings.embedQuery(question);
  // le farò recuperare più documenti del dovuto, una top 10 per poter scegliere dopo
  const results = await vectorStore!.similaritySearch(question, 10);
/*
  console.log("\n--- Docs before reranking) ---\n");
  results.forEach((doc, i) => {
    console.log(`Doc ${i + 1} (source: ${doc.metadata.source}):\n${doc.pageContent}\n`);
  });

  console.log("\n--- Docs after reranking) ---\n");
*/
  // controllo la similarità tra query e ogni documento
  const scored = await Promise.all(results.map(async (doc) => {
    const docEmbedding = await embeddings.embedQuery(doc.pageContent);
    const score = cosineSimilarity(queryEmbedding, docEmbedding); //similarità del coseno
    return { doc, score };
  }));
  // riordino i documenti in base a quanto sono simili
  const reranked = scored
    .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
    .map((item) => item.doc);
  // dai 3 più rilevanti faccio la return del contesto
  const topDocs = reranked.slice(0, 3);

  // log dettagliato per ogni documento
  console.log();
  console.log(`${FgYellow} ### Reranked context ###`);
  topDocs.forEach((doc, i) => {
    console.log(`${FgBlue}--- Document ${i + 1} (source: ${doc.metadata?.source}) ---`);
    console.log(`${FgGreen} ${doc.pageContent}`);
    console.log();
  });
  console.log(`${FgYellow} ### End reranked context ###`);
  console.log();
  console.log(`${Reset}`);
  // restituisco il contesto concatenato
  /*
  //return topDocs.map((doc) => doc.pageContent).join("\n");

  // Costruisco il prompt da passare all'LLM

  const prompt = `
    Dato questo materiale recuperato da PDF e Wikipedia:\n\n${topDocs.map(d => d.pageContent).join("\n\n")}\n\n
    Scrivi una spiegazione tecnica e coerente che risponda alla domanda: "${question}"
  `;
  */

  const prompt = `Based on the following material retrieved from PDF documents and Wikipedia:\n\n
        ${topDocs.map(d => d.pageContent).join("\n\n")}\n\n
        Answer the user's question in a clear, coherent, and technically accurate way: "${question}". 
        Structure the response in a single paragraph with a maximum of eight sentences. 
        Each sentence should express a single, essential idea to help the user build a concept map. 
        Avoid annotations, symbols, or formatting. Use natural but concise language.
        If the answer is not present in the material, say "I don't know".`;

  // Eseguo la generazione
  const response = await llm.invoke(prompt);
  console.log(`${FgBlue}### Answer generated by the LLM ###`);
  console.log(response.content);
  console.log();
  // Ritorno il testo riformulato come risultato finale
  return response.content as string;
}

//provare?
async function removeNearDuplicates(
  docs: { pageContent: string; metadata: any }[],
  embeddings: MistralAIEmbeddings,
  threshold = 0.9 // soglia di similarità, 0.95 è molto simile?
): Promise<typeof docs> {
  const uniqueDocs: typeof docs = [];
  const seenEmbeddings: number[][] = [];

  for (const doc of docs) {
    const currentEmbedding = await embeddings.embedQuery(doc.pageContent);

    const isDuplicate = seenEmbeddings.some((existingEmbedding) => {
      const similarity = cosineSimilarity(existingEmbedding, currentEmbedding);
      return similarity !== null && similarity > threshold;
    });

    if (!isDuplicate) {
      uniqueDocs.push(doc);
      seenEmbeddings.push(currentEmbedding);
    }
  }

  return uniqueDocs;
}

//sistemare il testo dovrebbe aiutare?
/*
function normalizeText(text: string): string {
  return text
    .replace(/\s+/g, " ")
    .replace(/\n+/g, "\n")
    .trim();
}
*/
function normalizeText(text: string): string {
  return text
    .toLocaleLowerCase()
    // Rimuove spazi multipli tranne nei newline doppi (paragrafi)
    .replace(/[ \t]+/g, " ")
    // Rimuove spazi prima di andare a capo
    .replace(/ +\n/g, "\n")
    // Pulisce gli spazi all'inizio di ogni riga
    .replace(/\n[ \t]+/g, "\n")
    // Rende uniforme la punteggiatura (es. "ciao , mondo ." -> "ciao, mondo.")
    .replace(/ ?([.,;:!?]) ?/g, "$1 ")
    // Mantiene massimo due newline consecutivi (un solo spazio tra paragrafi)
    .replace(/\n{3,}/g, "\n\n")
    // Rimuove spazi multipli
    .replace(/ {2,}/g, " ")
    .trim();
}