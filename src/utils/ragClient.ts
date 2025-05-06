// fa chiamata al server che sta fuori da react

export async function getRagContext(question: string): Promise<string> {
    try {
      const response = await fetch("http://localhost:3001/rag", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      });
  
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();

      return data.context || "";

    } catch (err) {
      console.error("RAG fetch error", err);
      return "";
    }
  }
  