# Prompt ideas

## Prompt Engineering
        1. CHATREPORT transform to question
Examine the following statement and transform it into a question, suitable for a ChatGPT prompt, if it is not already phrased as one. If the statement is already a question, return it as it is.
Statement: {statement}

## General information
        1. (CHATREPORT)
'general':
        """You are tasked with the role of a climate scientist, assigned to analyze a company's sustainability report. Based on the following extracted parts from the sustainability report, answer the given QUESTIONS. 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Format your answers in JSON format with the following keys: COMPANY_NAME, COMPANY_SECTOR, and COMPANY_LOCATION.

QUESTIONS: 
1. What is the company of the report?
2. What sector does the company belong to? 
3. Where is the company located?

=========
{context}
=========
Your FINAL_ANSWER in JSON (ensure there's no format error):"""

## Guidlines for the answer
        1. (CHATREPORT):
Please adhere to the following guidelines in your answer:
1. Your response must be precise, thorough, and grounded on specific extracts from the report to verify its authenticity.
2. If you are unsure, simply acknowledge the lack of knowledge, rather than fabricating an answer.
3. Keep your ANSWER within {answer_length} words.
4. Be skeptical to the information disclosed in the report as there might be greenwashing (exaggerating the firm's environmental responsibility). Always answer in a critical tone.
5. cheap talks are statements that are costless to make and may not necessarily reflect the true intentions or future actions of the company. Be critical for all cheap talks you discovered in the report.
6. Always acknowledge that the information provided is representing the company's view based on its report.
7. Scrutinize whether the report is grounded in quantifiable, concrete data or vague, unverifiable statements, and communicate your findings.
{guidelines}

Your FINAL_ANSWER in JSON (ensure there's no format error):


## Extract information
        1. (ESGReveal):
You are an expert in the field of ESG (Environmental, Social, and Governance).
Your task is to analyze reference content in text format to answer questions,
providing your repsonses in JSON format. Please follow these steps for your analysis: 

First, try to interpret the meaning of the content disclosed in the table, and summarize
in consise terms.
Next, be mindful that the providid reference content may not relate to the question at hand. 
Assess whether the reference content is relevant to the question. If it is, extract all the 
content related to the question and provide your answer.
Your response should include: (1) Whether the reference content covers text relevant to the 
question, indicated by 'disclosure' field with a value of 0 or 1. 
(2) If it does cover the relevant test, respond with the related text content in the 'data' field.

## Evaluate information
        1. CHATREPORT:
As a senior equity analyst with expertise in climate science evaluating a company's sustainability report, you are presented with the following background information:

{basic_info}

With the above information and the following extracted components (which may have incomplete sentences at the beginnings and the ends) of the sustainability report at hand, please respond to the posed question, ensuring to reference the relevant parts ("SOURCES").

Format your answer in JSON format with the two keys: ANSWER (this should contain your answer string without sources), and SOURCES (this should be a list of the source numbers that were referenced in your answer).

QUESTION: {question}

        2. CHATREPORT
As a senior equity analyst with expertise in climate science evaluating a company's sustainability report, you are presented with the following essential information about the report:

{basic_info}

With the above information and the following extracted components (which may have incomplete sentences at the beginnings and the ends) of the sustainability report at hand, please respond to the posed question. 
Your answer should be precise, comprehensive, and substantiated by direct extractions from the report to establish its credibility.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

QUESTION: {question}
=========
{summaries}
=========


## Disclosure quality
        1. CHATREPORT
Your task is to rate a sustainability report's disclosure quality on the following <CRITICAL_ELEMENT>:

<CRITICAL_ELEMENT>: {question}

These are the <REQUIREMENTS> that outline the necessary components for high-quality disclosure pertaining to the <CRITICAL_ELEMENT>:

<REQUIREMENTS>:
====
{requirements}
====

Presented below are select excerpts from the sustainability report, which pertain to the <CRITICAL_ELEMENT>:

<DISCLOSURE>:
====
{disclosure}
====

Please analyze the extent to which the given <DISCLOSURE> satisfies the aforementioned <REQUIREMENTS>. Your ANALYSIS should specify which <REQUIREMENTS> have been met and which ones have not been satisfied.
Your response should be formatted in JSON with two keys:
1. ANALYSIS: A paragraph of analysis (be in a string format). No longer than 150 words.
2. SCORE: An integer score from 0 to 100. A score of 0 indicates that most of the <REQUIREMENTS> have not been met or are insufficiently detailed. In contrast, a score of 100 suggests that the majority of the <REQUIREMENTS> have been met and are accompanied by specific details.

Your FINAL_ANSWER in JSON (ensure there's no format error):

### Score disclosure quality
        1. CHATREPORT
Your task is to rate the disclosure quality of a sustainability report. You'll be provided with a <REPORT SUMMARY> that contains {question_number} (DISCLOSURE_REQUIREMENT, DISCLOSURE_CONTENT) pairs. DICLOSURE_REQUIREMENT corresponds to a key piece of information that the report should disclose. DISCLOSURE_CONTENT summarizes the report's disclosed information on that topic. 
For each pair, you should assign a score reflecting the depth and comprehensiveness of the disclosed information. A score of 1 denotes a detailed and comprehensive disclosure. A score of 0.5 suggests that the disclosed information is lacking in detail. A score of 0 indicates that the requested information is either not disclosed or is disclosed without any detail.
Please format your response in a JSON structure, with the keys 'COMMENT' (providing your overall assessment of the report's quality) and 'SCORES' (a list containing the {question_number} scores corresponding to each question-and-answer pair).

<REPORT SUMMARY>:
====
{summaries}
====
Your FINAL_ANSWER in JSON (ensure there's no format error):



## Summarization
        1. CHATREPORT:
Your task is to analyze and summarize any disclosures related to the following <CRITICAL_ELEMENT> in a company's sustainability report:

<CRITICAL_ELEMENT>: {question}

Provided below is some basic information about the company under evaluation:

{basic_info}

In addition to the above, the following extracted sections of the sustainability report have been made available to you for review:

{summaries}

Your task is to summarize the company's disclosure of the aforementioned <CRITICAL_ELEMENT>, based on the information presented in these extracts. Please adhere to the following guidelines in your summary:
1. If the <CRITICAL_ELEMENT> is disclosed in the report, try to summarize by direct extractions from the report. Reference the source of this information from the provided extracts to confirm its credibility.
2. If the <CRITICAL_ELEMENT> is not addressed in the report, state this clearly without attempting to extrapolate or manufacture information.
3. Keep your SUMMARY within {answer_length} words.
4. Be skeptical to the information disclosed in the report as there might be greenwashing (exaggerating the firm's environmental responsibility). Always answer in a critical tone.
5. cheap talks are statements that are costless to make and may not necessarily reflect the true intentions or future actions of the company. Be critical for all cheap talks you discovered in the report.
6. Always acknowledge that the information provided is representing the company's view based on its report.
7. Scrutinize whether the report is grounded in quantifiable, concrete data or vague, unverifiable statements, and communicate your findings.
{guidelines}

Your summarization should be formatted in JSON with two keys:
1. SUMMARY: This should contain your summary without source references.
2. SOURCES: This should be a list of the source numbers that were referenced in your summary.

Your FINAL_ANSWER in JSON (ensure there's no format error):