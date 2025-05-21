import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { ChatMistralAI, MistralAIEmbeddings } from "@langchain/mistralai";
import { fileURLToPath } from "url";
import path, { dirname } from "path";
import fs from "fs";
import dotenv from 'dotenv';
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import cosineSimilarity from "compute-cosine-similarity";
import { FaissStore } from "@langchain/community/vectorstores/faiss";

import { CohereEmbeddings, CohereRerank } from "@langchain/cohere";
import { Document } from "@langchain/core/documents";

////////////////////////////////////////////////////////////////////////////////////

// costanti colori
const Reset = "\x1b[0m";
const FgGreen = "\x1b[32m";
const FgYellow = "\x1b[33m";
const FgBlue = "\x1b[34m";
const Bright = "\x1b[1m";
const FgRed = "\x1b[31m";


const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

dotenv.config();

const mistralApiKey = process.env.MISTRAL_API_KEY;
if (!mistralApiKey) {
  throw new Error("MISTRAL_API_KEY not defined in .env");
}

const llm = new ChatMistralAI({
  model: "mistral-large-latest",
  temperature: 0,
  apiKey: mistralApiKey,
});

const embeddings = new MistralAIEmbeddings({
  model: "mistral-embed",
  apiKey: mistralApiKey,
});

/*
const embeddings = new CohereEmbeddings({
  apiKey: process.env.COHERE_API_KEY!,    
  model: "embed-english-v2.0",
});
*/
//reranker già pronto (il mio non funziona bene, da punteggi troppo simili tra loro)
const cohereRerank = new CohereRerank({
  apiKey: process.env.COHERE_API_KEY!,
  model: "rerank-english-v3.0",
});

// Path di salvataggio dell'indice FAISS
const INDEX_DIR = path.resolve(__dirname, './faiss_index');
if (fs.existsSync(INDEX_DIR)) {
  fs.rmSync(INDEX_DIR, { recursive: true, force: true });
}

let vectorStore: FaissStore | null = null;

async function loadAndIndexData() {
  console.log(`\n\n\n`);
  console.time("FAISS indexing time: "); //timer start
  console.log(`${FgGreen}Loading and indexing data...${Reset}`);

  const pdfDir = path.resolve(__dirname, "./data/pdf_files");
  // Prendi solo file .pdf, ignora .DS_Store e simili
  const filePaths = fs
    .readdirSync(pdfDir)
    .filter(f => f.toLowerCase().endsWith(".pdf"))
    .map(f => path.join(pdfDir, f));

  // Carica ogni PDF uno per uno
  const rawDocs: { pageContent: string; metadata: any }[] = [];

  console.log("Files to be indexed:", filePaths);

  for (const filePath of filePaths) {
    const docsPage = await new PDFLoader(filePath).load();
    rawDocs.push(
      ...docsPage.map(doc => ({
        ...doc,
        pageContent: normalizeText(doc.pageContent),
        //metadata: { ...doc.metadata, source: "pdf" },
        metadata: { ...doc.metadata, source: path.basename(filePath) }, //nome file eni metadati, sta pescando solo dal primo?
      }))
    );
  }

  rawDocs.forEach((doc, i) => {
    console.log(
      `${FgBlue}Chunk ${i + 1}${Reset} from ${FgYellow}${doc.metadata.source}${Reset}: ` +
      `"${doc.pageContent.replace(/\n/g, ' ').slice(0, 60)}..."`
    );
  });


  const uniqueDocs = await removeNearDuplicates(rawDocs, embeddings);

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    //chunkOverlap: 200,
    //provo a diminuire la dim dei chunk per non dividere le frasi
    //chunkSize: 800,
    chunkOverlap: 300,
  });
  const splitDocs = await splitter.splitDocuments(uniqueDocs);

  // Costruisci o ricarica indice FAISS con fallback a rebuild
  if (fs.existsSync(INDEX_DIR)) {
    try {
      vectorStore = await FaissStore.load(INDEX_DIR, embeddings);
      console.log(`${FgYellow}FAISS index loaded from disk${Reset}`);
    } catch (err) {
      console.warn(`${FgRed}Failed to load FAISS index, rebuilding…${Reset}`, err);
      fs.rmSync(INDEX_DIR, { recursive: true, force: true });
    }
  }

  if (!vectorStore) {
    fs.mkdirSync(INDEX_DIR, { recursive: true });
    vectorStore = await FaissStore.fromDocuments(splitDocs, embeddings);
    await vectorStore.save(INDEX_DIR);
    console.log(`${FgYellow}FAISS index created and saved to disk${Reset}`);
  }

  console.timeEnd("FAISS indexing time: "); //timer stop
}

                            export async function buildRagContext(question: string): Promise<string> {
                              if (!vectorStore) await loadAndIndexData();

                              console.log(`${FgGreen}`, question)
                              const queryEmbedding = await embeddings.embedQuery(question);
                              //console.log(`${FgYellow}Query embedding vector:${Reset}`, queryEmbedding);                              

                              const results = await vectorStore!.similaritySearch(question, 10);
                              
                              console.log("Similarity scores:");
                              const scoredAll = await Promise.all(results.map(async (doc, idx) => {
                                const embDoc = await embeddings.embedQuery(doc.pageContent);
                                const score = cosineSimilarity(queryEmbedding, embDoc) || 0;
                                console.log(
                                  `${String(idx+1).padStart(2,' ')}.` +
                                  ` [${score.toFixed(4)}] ` +
                                  `"${doc.pageContent.replace(/\n/g,' ').slice(0,50)}…"`
                                );
                                return { doc, score };
                              }));
                              console.log("Similarity scores end\n");
                              
                              const docs = results.map((doc) => new Document({ pageContent: doc.pageContent }));

                              const rerankResults = await cohereRerank.rerank(docs, question, { topN: 10 });  //chiamata al reranker

                              //classifica
                              console.log(`${FgYellow} ### Rerank Results ###${Reset}`);
                              rerankResults.forEach((r, idx) => {
                                const originalDoc = results[r.index];
                                const preview = originalDoc.pageContent
                                  .replace(/\n/g, " ")
                                  .slice(0, 60);
                                console.log(
                                  `${FgBlue}${String(idx + 1).padStart(2, "0")}.` + 
                                  ` (orig idx ${r.index})` +
                                  ` ${FgYellow}Score:${Reset} ${r.relevanceScore.toFixed(4)}` +
                                  ` — "${preview}..."`
                                );
                              });
                              console.log(`${FgYellow} ### End Rerank ###${Reset}\n`);

                              // 5) Prendi i top-3 documenti ordinati
                              const topDocs = rerankResults
                                .slice(0, 3)
                                .map((r) => results[r.index]);

                              // 6) Log dei top-3 in dettaglio (come prima)
                              console.log(`${FgYellow} ### Top-3 Reranked Context ###${Reset}`);
                              topDocs.forEach((doc, i) => {
                                console.log(`${FgBlue}--- Document ${i + 1} (source: ${doc.metadata.source}) ---`);
                                console.log(`${FgGreen}${doc.pageContent}\n`);
                              });
                              console.log(`${FgYellow} ### End Top-3 ###${Reset}\n`);

                              const prompt = `
                                Based on the following material retrieved from the PDF documents:
                                ${topDocs.map(d => d.pageContent).join("\n\n")}
                                Write a structured, technically accurate explanation that addresses the question: "${question}".
                                Do not include introductory or concluding phrases. Avoid numbered lists or bullet points.
                                The user's goal is to build a concept map to visually explain the response.
                                To support this goal, provide a well-structured response in multiple paragraphs.
                                Each paragraph should cover a central aspect or topic of the answer.
                                Each paragraph must contain fewer than 10 sentences, and the full response should consist of 2 paragraphs total.
                                When multiple facts refer to the same entity or are logically connected (e.g., observations, causes, consequences),
                                write them as a single compound sentence. Use explicit conjunctions or subordinating structures to preserve the connection.
                                Do not split logically linked ideas into separate sentences, especially when they share the same subject or concept.
                                Avoid pronouns, relative clauses, and vague references such as “this”, “which”, “they”, or “such”.
                                Instead, always restate the subject explicitly to maintain clarity and enable consistent annotation.
                                If the same concept appears in multiple sentences, use exactly the same wording to refer to it, to ensure proper linking in the concept map.
                                Use precise, noun-based terminology for key entities, and clear, verb-based phrases for relationships.
                                If the answer is not present in the material, you must say only "I DON'T KNOW" and nothing else.
                              `;

                              const response = await llm.invoke(prompt);
                              console.log(`${FgBlue}### Answer generated by the LLM ###`);
                              console.log(response.content);


                              console.log();

                              return response.content as string;
                            }

async function removeNearDuplicates(
  docs: { pageContent: string; metadata: any }[],
  embeddings: MistralAIEmbeddings,
  //embeddings: CohereEmbeddings,
  threshold = 0.7 //stava a 0.9
): Promise<{ pageContent: string; metadata: any }[]> {
  const uniqueDocs: { pageContent: string; metadata: any }[] = [];
  const seenEmbeddings: number[][] = [];

  for (const doc of docs) {
    if (!doc.pageContent) {
      continue;
    }
    const emb: number[] = await embeddings.embedQuery(doc.pageContent);
    const isDup = seenEmbeddings.some((existingEmb: number[]) => {
      const sim = cosineSimilarity(existingEmb, emb);
      return sim !== null && sim > threshold;
    });
    if (!isDup) {
      uniqueDocs.push(doc);
      seenEmbeddings.push(emb);
    }
  }
  return uniqueDocs;
}

function normalizeText(text: string): string {
  return text
    .replace(/[ \t]+/g, " ")
    .replace(/ +\n/g, "\n")
    .replace(/\n[ \t]+/g, "\n")
    .replace(/ ?([.,;:!?]) ?/g, "$1 ")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/ {2,}/g, " ")
    .trim();
}
