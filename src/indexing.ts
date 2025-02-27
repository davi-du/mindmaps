import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { vectorStore } from "../app.ts";

export async function indexDocuments(dataPath, link) {

    /* DOCUMENT LOADING */
    //Prima carica i documenti dalla directory
    const directoryLoader = new DirectoryLoader(dataPath, {
        ".pdf": (path: string) => new PDFLoader(path),
    });
    const directoryDocs = await directoryLoader.load();
    // Carica il contenuto della paginaweb
    const cheerioLoader = new CheerioWebBaseLoader(
    link,
    {
        selector: "p", // Selettore CSS per estrarre i paragrafi
    }
    );
    const docs = await cheerioLoader.load(); 

    //console.log(docs[0].pageContent.slice(0, 500));
    //console.assert(docs.length === 1);
    //console.log(`Numero caratteri da sito: ${docs[0].pageContent.length}`);

    const allDocs = directoryDocs.concat(docs); //unione dei dati da web e da pdf

    /* TEXT SPLITTER */
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
    });
    const allSplits = await splitter.splitDocuments(allDocs);

    /* VECTOR STORE */
    await vectorStore.addDocuments(allSplits);
    //console.log("Numero totale di documenti:", vectorStore.memoryVectors.length);

    return allSplits;
}