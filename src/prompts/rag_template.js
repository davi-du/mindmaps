/*

//vecchio template
const template = `Use the following pieces of context to answer the question at the end.  
If you don't know the answer, just say that you don't know, don't try to make up an answer.  
Use three sentences maximum and keep the answer as concise as possible.  

For each relevant term, append a token in the format $N1 directly next to the term.  
If a term belongs to a broader category, assign it a subtoken in the format $N1C1 (for categories) or $N1N1 (for specific instances).  
For example, 'car $N2' 'SUV $N2C1', and 'van $N2C2'.  
Similarly, specific items inherit their category, such as 'Ferrari $N2N1' and 'Ford $N2N2'.  

Always say "Grazie per averlo chiesto!" at the end of the answer.  

{context}  

{previous_qa}

Question: {question}  

Helpful Answer:`;

export { template };
*/

const template = `Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

IMPORTANT: You MUST tag key terms in your answer with tokens.
For each relevant technical term in your answer, append a token in the format $N1 directly next to the term (no space between).
If a term belongs to a broader category, assign it a subtoken in the format $N1C1 (for categories) or $N1N1 (for specific instances).
Example: 'car [$N2]' 'SUV [$N2C1]' 'van [$N2C2]' 'Ferrari [$N2N1]' 'Ford [$N2N2]'

Always say "Grazie per averlo chiesto!" at the end of the answer.

Context:
{context}

Previous Q&A:
{previous_qa}

Question: {question}

Helpful Answer (REMEMBER TO ADD TOKENS TO KEY TERMS):`;

export { template };
