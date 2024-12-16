# ================================
# Prompts for generate the answer.
# ================================
answer_prompt_template = """
Based on the following context, please only output the answer to the question. If there are multiple terms, use the complete noun phrase or full name instead of abbreviations or shortened forms.

For example, If the answer is 'The British acceptation of the type of sedan the Maruti Suzuki Dzire is a "small family car."', then the output should be 'small family car.'

Only provide the relevant noun phrase or term that directly answers the query.
Do not include any additional sentences, context, or information—only the specific answer.
If both an abbreviation and its full form appear (e.g., "Extended Play" and "EP"), provide the full form (e.g., "Extended Play").
For locations, use the complete place name as given in the context (e.g., "Tallaght, South Dublin" instead of just "Tallaght").
Example:

Context:

The creator of Circe Invidiosa is John William Waterhouse.
In 1883, Waterhouse married Esther Kenworthy Waterhouse, the daughter of an art schoolmaster from Ealing.
Query: Who is the spouse of the person who made The Circe Invidiosa?

Answer: Esther Kenworthy Waterhouse

Context: {context}

Query: {query}

Answer:
"""

answer_prompt_instruction="""
Based on the following context, please only output the answer to the question. If there are multiple terms, use the complete noun phrase or full name instead of abbreviations or shortened forms.

For example, If the answer is 'The British acceptation of the type of sedan the Maruti Suzuki Dzire is a "small family car."', then the output should be 'small family car.'

Only provide the relevant noun phrase or term that directly answers the query.
Do not include any additional sentences, context, or information—only the specific answer.
If both an abbreviation and its full form appear (e.g., "Extended Play" and "EP"), provide the full form (e.g., "Extended Play").
For locations, use the complete place name as given in the context (e.g., "Tallaght, South Dublin" instead of just "Tallaght").
Example:

Context:

The creator of Circe Invidiosa is John William Waterhouse.
In 1883, Waterhouse married Esther Kenworthy Waterhouse, the daughter of an art schoolmaster from Ealing.
Query: Who is the spouse of the person who made The Circe Invidiosa?

Answer: Esther Kenworthy Waterhouse
"""

answer_prompt_input_template="""
Context: {context}

Query: {query}

Answer:
"""

# ===============================================================
# Prompts for generate the answer for quality dataset.
# ===============================================================
answer_prompt_template_for_quality_dataset = """
Read the following article and answer a multiple choice question.
For example, if (C) is correct, answer with \"Answer: (C) ...\"

Context:
{context}

Question:
{query}
"""

answer_prompt_instruction_for_quality_dataset="""
Read the following article and answer a multiple choice question.
For example, if (C) is correct, answer with \"Answer: (C) ...\"
"""

answer_prompt_input_template_for_quality_dataset="""
Context:
{context}

Question:
{query}
"""

# ========================================
# Prompts for extracting the atomic facts.
# ========================================
key_elements_atomic_facts_extraction_prompt_template = """
You are now an intelligent assistant tasked with meticulously extracting both key elements and atomic facts from a conversation history.. 

1. Key Elements: The essential nouns (e.g., characters, times, events, places, numbers), verbs (e.g., actions), and adjectives (e.g., states, feelings) that are pivotal to the text's narrative. 
2. Atomic Facts: The smallest, indivisible facts, presented as concise sentences. These include propositions, theories, existences, concepts, and implicit elements like logic, causality, event sequences, interpersonal relationships, timelines, etc. 

Requirements: ##### 
1. Ensure that all the atomic facts contain full and complete information, reflecting the entire context of the sentence without omitting any key details. 
2. Ensure that all identified key elements are reflected within the corresponding atomic facts. 
3. You should extract key elements and atomic facts comprehensively, especially those that are important and potentially query-worthy and do not leave out details. 
4. Whenever applicable, replace pronouns with their specific noun counterparts (e.g., change I, He, She to actual names). 
5. Ensure that the key elements and atomic facts you extract are presented in the same language as the original text (e.g., English or Chinese). 
6. You should output a total of key elements and atomic facts that do not exceed 1024 tokens. 
7. Your answer format for each line should be: [Serial Number], [Atomic Facts], [List of Key Elements, separated with '|']
##### 

Example: 
##### 
Conversation:
1. Caroline said, "Woohoo Melanie! I passed the adoption agency interviews last Friday! I'm so excited and thankful. This is a big move towards my goal of having a family."
2. Melanie said, "Congrats, Caroline! Adoption sounds awesome. These figurines I bought yesterday remind me of family love. Tell me, what's your vision for the future?"
 and shared a photo of a couple of wooden dolls sitting on top of a table.

Atomic Facts and Key Elements:
1. Caroline passed the adoption agency interviews last Friday. | Caroline | adoption agency interviews | last Friday 
2. Caroline is excited and thankful for passing the adoption agency interviews. | Caroline | excited | thankful | adoption agency interviews
3. Passing the adoption agency interviews is a big move towards Caroline's goal of having a family. | Caroline | adoption agency interviews | goal | having a family
4. Melanie congratulated Caroline on passing the adoption agency interviews. | Melanie | Caroline | adoption agency interviews | Congratulations
5. Melanie thinks that adoption sounds awesome. | Melanie | Adoption | awesome
6. Melanie bought figurines yesterday. | Melanie | figurines | yesterday
7. The figurines Melanie bought remind her of family love. | Melanie | figurines | family love
8. Melanie asked Caroline about her vision for the future. | Melanie | Caroline | vision for the future
9. Melanie shared a photo of wooden dolls sitting on a table. | Melanie | wooden dolls | table | photo
# ##### 
# 
Please strictly follow the above format. Let's begin.

Conversation: 
{conversation}

Atomic Facts and Key Elements:
"""

# ========================================
# Prompts for extracting the triples.
# ========================================
triple_extraction_prompt_template = """You are now an intelligent assistant tasked with meticulously extracting both key elements and triples from a long text. 

1. Key Elements: The essential nouns (e.g., characters, times, events, places, numbers), verbs (e.g., actions), and adjectives (e.g., states, feelings) that are pivotal to the text’s narrative. 
2. Triples: Structured triplets in the format of "subject, relation, object". Each triple should represent a clear and concise fact, relation, or interaction within the observation. You should aim for simplicity and clarity, ensuring that each triplet has no more than 7 words.

Requirements: 
##### 
1. Ensure that all identified key elements are reflected within the corresponding atomic facts. 
2. You should extract key elements and atomic facts comprehensively, especially those that are important and potentially query-worthy and do not leave out details. 
3. Whenever applicable, replace pronouns with their specific noun counterparts (e.g., change I, He, She to actual names). 
4. Ensure that the key elements and triples you extract are presented in the same language as the original text (e.g., English or Chinese).  
5. Avoid Redundant Triples: Do not include irrelevant information like the current location of the agent (e.g., "you, are in, location") or placeholder entities such as "none."
6. Your answer format for each line should be: [Serial Number], [Atomic Facts], [List of Key Elements, separated with ‘|’] 
##### 

Example: 
##### 
# User: One day, a father and his little son ...... 
# 
# Assistant: 
1. Father, went to, home | father | went to | home
2. Son, went to, home | son | went to | home
3. Father, accompanied by, son | father | accompanied by | son
4. ...
##### 
# 
Please strictly follow the above format. Let’s begin.

Context: 
{context}

"""

# ========================================
# Prompts for atomic_fact_reasoning
# ========================================
atomic_fact_reasoning_prompt_template = """
Given a question provided contexts, evaluate whether additional context from the given candidate contexts would improve the answer to the question. 
Follow the instructions below to decide whether the current contexts are sufficient or if additional context is necessary.

Instructions:
- Thoroughly read the provided contexts and determine if they fully answer the question. 
- If the provided contexts already provide a complete and accurate answer, select Option A to indicate that no additional context is needed. 
- If the provided contexts are not sufficient to answer the question, carefully evaluate the candidate contexts and select the most helpful candidate context from the options. 
- You must only output the option (e.g., A, B, or C) and do not output any other content including the texts of selected context or introductory phrases. 

Example:
###
Example 1: 
Question: What are three alternate names of the University where W.G. Godfrey received his BA and MA degrees?

Context:
1. W.G. Godfrey received his BA and MA from the University of Waterloo and Ph.D. from Queen's University.

Candidate Context:
A. Existing contexts are sufficient to answer the question, no additional context is required.
B. The University of Waterloo (UWaterloo, UW, or Waterloo) is a public research university with a main campus in Waterloo, Ontario, Canada. 
C. The University of Waterloo is a public research university.
D. W. G. Godfrey worked as the department chair at Mount Allison University.

Selection: B

Example 2: 
Question: Who is the spouse of the person who made The Circe Invidiosa?

Context:
1. The creator of Circe Invidiosa is John William Waterhouse. 
2. In 1883, Waterhouse married Esther Kenworthy Waterhouse, the daughter of an art schoolmaster from Ealing. 

Candidate Context:
A. No Need for Additional Context
B. Circe Invidiosa is a painting by John William Waterhouse completed in 1892. 
C. Circe Invidiosa is based on Ovid's tale in Metamorphoses, wherein Circe turns Scylla into a sea monster.
D. Circe Invidiosa is part of the collection of the Art Gallery of South Australia.

Selection: A
###

Question: {query}

Context: 
{contexts}

Candidate Context:
{candidates}

Selection: 
"""

# ===============================================================
# Prompts for generating the summary from atomic facts or triples.
# ===============================================================
summary_template = """
You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two atomic facts, and its original descriptions, all related to the atomic facts.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the names so we have the full context.

#######
-Data-
Atomic facts:
{elements}

Original Description List:
{description_list}
#######

Output:
"""

# ===============================================================
# Prompts for batch rerank.
# ===============================================================
batch_rerank_template = """
A list of documents is shown below. Each document has a number next to it along with a summary of the document. A question is also provided. 
Respond with the numbers of the documents you should consult to answer the question, in order of relevance, as well 
as the relevance score. The relevance score is a number from 1-10 based on how relevant you think the document is to the question.
Respond with the numbers of **all** the documents along with a relevance score.
Example format: 
Document 1:
<summary of document 1>

Document 2:
<summary of document 2>

...

Document 10:
<summary of document 10>

Question: <question>
Answer:
Doc: 9, Relevance Score: 7
Doc: 3, Relevance Score: 4
Doc: 7, Relevance Score: 3

Let's try this now: 

{context_str}
Question: {query}
Answer:
"""

batch_rerank_instruction="""
A list of documents is shown below. Each document has a number next to it along with a summary of the document. A question is also provided. 
Respond with the numbers of the documents you should consult to answer the question, in order of relevance, as well 
as the relevance score. The relevance score is a number from 1-10 based on how relevant you think the document is to the question.
Respond with the numbers of **all** the documents along with a relevance score.
Example format: 
Document 1:
<summary of document 1>

Document 2:
<summary of document 2>

...

Document 10:
<summary of document 10>

Question: <question>
Answer:
Doc: 9, Relevance Score: 7
Doc: 3, Relevance Score: 4
Doc: 7, Relevance Score: 3
...
"""

batch_rerank_input_template="""
You MUST strictly follow the output format to output the relevance score for Each document. DON'T output any contents except doc number and relevance score. 

{context_str}
Question: {query}
Answer:
"""

# ===============================================================
# Prompts for Iterative Retrieval.
# ===============================================================
iterative_retrieval_template = """
Follow the examples to answer the input question by reasoning step-by-step. Output both reasoning steps and the answer. 

Examples:
#####
Question: Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from Yoko Ono?
Thought: The album issued by Apple Records, and written, recorded, and released during John Lennon's 18 month separation from Yoko Ono is Walls and Bridges. Nobody Loves You was written by John Lennon on Walls and Bridges album. So the answer is: Walls and Bridges.

Question: What is known as the Kingdom and has National Route 13 stretching towards its border?
Thought: Cambodia is officially known as the Kingdom of Cambodia. National Route 13 streches towards border to Cambodia. So the answer is: Cambodia.

Question: Jeremy Theobald and Christopher Nolan share what profession?
Thought: Jeremy Theobald is an actor and producer. Christopher Nolan is a director, producer, and screenwriter. Therefore, they both share the profession of being a producer. So the answer is: producer.

Question: What film directed by Brian Patrick Butler was inspired by a film directed by F.W. Murnau?
Thought: Brian Patrick Butler directed the film The Phantom Hour. The Phantom Hour was inspired by the films such as Nosferatu and The Cabinet of Dr. Caligari. Of these Nosferatu was directed by F.W. Murnau. So the answer is: The Phantom Hour.

Question: Vertical Limit stars which actor who also played astronaut Alan Shepard in \"The Right Stuff\"?
Thought: The actor who played astronaut Alan Shepard in \"The Right Stuff\" is Scott Glenn. The movie Vertical Limit also starred Scott Glenn. So the answer is: Scott Glenn.
#####

Input:

Context: 
{context}

Question: {question}
Thought: 
"""

iterative_retrieval_instruction="""
Follow the examples to answer the input question by reasoning step-by-step. Output both reasoning steps and the answer. 

Examples:

Question: Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from Yoko Ono?
Thought: The album issued by Apple Records, and written, recorded, and released during John Lennon's 18 month separation from Yoko Ono is Walls and Bridges. Nobody Loves You was written by John Lennon on Walls and Bridges album. So the answer is: Walls and Bridges.

Question: What is known as the Kingdom and has National Route 13 stretching towards its border?
Thought: Cambodia is officially known as the Kingdom of Cambodia. National Route 13 streches towards border to Cambodia. So the answer is: Cambodia.

Question: Jeremy Theobald and Christopher Nolan share what profession?
Thought: Jeremy Theobald is an actor and producer. Christopher Nolan is a director, producer, and screenwriter. Therefore, they both share the profession of being a producer. So the answer is: producer.

Question: What film directed by Brian Patrick Butler was inspired by a film directed by F.W. Murnau?
Thought: Brian Patrick Butler directed the film The Phantom Hour. The Phantom Hour was inspired by the films such as Nosferatu and The Cabinet of Dr. Caligari. Of these Nosferatu was directed by F.W. Murnau. So the answer is: The Phantom Hour.

Question: Vertical Limit stars which actor who also played astronaut Alan Shepard in \"The Right Stuff\"?
Thought: The actor who played astronaut Alan Shepard in \"The Right Stuff\" is Scott Glenn. The movie Vertical Limit also starred Scott Glenn. So the answer is: Scott Glenn.
"""

iterative_retrieval_input_template="""
Context: 
{context}

Question: {question}
Thought: 
"""