import time

import networkx as nx
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import ast
import os
import sys
from langchain_openai import OpenAIEmbeddings
import pickle
import argparse
import torch

from prompts import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd

sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))  
from helper_functions import *

from utils import rewrite_query

from loguru import logger
from langchain_core.pydantic_v1 import BaseModel, Field
from tqdm import tqdm

from prompts import *
import tiktoken
import time
import re

from generator import load_llm_tokenizer_and_model, Generator


class RatingScore(BaseModel):
    document: int
    relevance_score: float = Field(..., description="The relevance score of a document to a query.")


# Define a MultiRatingScore schema to handle multiple ratings
class MultiRatingScore(BaseModel):
    ratings: List[RatingScore] = Field(..., description="A list of document relevance scores.")


class MemoBox:

    def __init__(self, dataset, reader_model="gpt-4o-mini", reader=None):

        self.llm = ChatOpenAI(temperature=0,
                              model_name="gpt-4o-mini",
                              max_tokens=4096)
        self.embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
        self.atomic_fact_graph = nx.Graph()  # self.atomic_fact_graph.nodes(data=True): (i, {"text": , "key_elements"})
        self.token_encoder = tiktoken.encoding_for_model("gpt-4o-mini")
        self.reader_model = reader_model
        self.reader = reader

        self.dataset = dataset


    def build_vector_store(self, memory_units):
        memory_units_docs = []
        memory_units_count = 0
        for memory_unit in memory_units:
            doc = Document(page_content=memory_unit['text'], metadata={'global_id': memory_units_count})
            memory_units_docs.append(doc)
            memory_units_count += 1
        new_vector_store = FAISS.from_documents(memory_units_docs, self.embedding_model)
        return new_vector_store

    def load_dataset(self, memory_repr, dataset, qa_order, num_noise_docs=0):
        main_path = f"../../datasets/{dataset}/"

        self.noise_memory_units = []
        self.noise_memory_vector_stores = []

        # Load primary memory units and vector store
        if num_noise_docs == 0:
            self.memory_units, self.memory_units2docs = self._load_memory_units(main_path, memory_repr, qa_order)
            logger.info(f"|> We are totally loading {len(self.memory_units)} memory units")

            self.memory_units_vector_store = self._load_vector_store(main_path, memory_repr, qa_order)
            logger.info(
                f"|> We are totally loading {self.memory_units_vector_store.index.ntotal} memory units in the vector store")
        else:
            self.memory_units, self.memory_units2docs = self._load_noise_memory_units(main_path, memory_repr, qa_order, num_noise_docs)
            logger.info(f"|> We are totally loading {len(self.memory_units)} memory units")

            self.memory_units_vector_store = self._load_noise_vector_store(main_path, memory_repr, qa_order, num_noise_docs)
            logger.info(
                f"|> We are totally loading {self.memory_units_vector_store.index.ntotal} memory units in the vector store")


        logger.info(f"|> Total memory units after adding noise: {len(self.memory_units)}")
        logger.info(
            f"|> Total memory units in vector store after adding noise: {self.memory_units_vector_store.index.ntotal}")

    def _load_noise_memory_units(self, main_path, memory_repr, order, num_noise_docs):
        """Load memory units for a specified order."""
        logger.info(f"|> Loading memory units for order {order}")
        memory_units_path = os.path.join(main_path, f"{memory_repr}/memory_units_noise_docs_{num_noise_docs}/{order}.pkl")
        _memory_units = pickle.load(open(memory_units_path, "rb"))

        memory_units = []
        memory_units2docs = {}

        for unit in _memory_units:
            memory_units.extend(unit[memory_repr])
            cluster_id = unit['cluster_id']

            for _unit in unit[memory_repr]:
                text = _unit['text']
                memory_units2docs[text] = {'cluster_id': cluster_id}

        return memory_units, memory_units2docs

    def _load_memory_units(self, main_path, memory_repr, order):
        """Load memory units for a specified order."""
        logger.info(f"|> Loading memory units for order {order}")
        memory_units_path = os.path.join(main_path, f"{memory_repr}/memory_units/{order}.pkl")
        _memory_units = pickle.load(open(memory_units_path, "rb"))

        memory_units = []
        memory_units2docs = {}

        for unit in _memory_units:
            memory_units.extend(unit[memory_repr])
            cluster_id = unit['cluster_id']

            for _unit in unit[memory_repr]:
                text = _unit['text']
                memory_units2docs[text] = {'cluster_id': cluster_id}

        return memory_units, memory_units2docs

    def _load_vector_store(self, main_path, memory_repr, order):
        """Load vector store for a specified order."""
        logger.info(f"|> Loading vector store for order {order}")
        vector_store_path = os.path.join(main_path, f"{memory_repr}/vector_stores/vector_store_{order}")
        return FAISS.load_local(vector_store_path, self.embedding_model, allow_dangerous_deserialization=True)

    def _load_noise_vector_store(self, main_path, memory_repr, order, num_noise_docs):
        """Load vector store for a specified order."""
        logger.info(f"|> Loading vector store for order {order}")
        vector_store_path = os.path.join(main_path, f"{memory_repr}/vector_stores_noise_docs_{num_noise_docs}/vector_store_{order}")
        return FAISS.load_local(vector_store_path, self.embedding_model, allow_dangerous_deserialization=True)

    def _add_noise_memory_units(self, main_path, memory_repr, qa_order, num_noise_docs):
        """Add specified number of noise memory units and vector stores."""
        noise_orders = random.sample([i for i in range(191) if i != qa_order], num_noise_docs)
        logger.info(f"|> Adding {num_noise_docs} noise documents")

        for noise_order in tqdm(noise_orders, desc="Adding noise memory units"):
            logger.info(f"|> Adding noise memory units for order {noise_order}")
            try:
                noise_memory_units, _ = self._load_memory_units(main_path, memory_repr, noise_order)
                # noise_vector_store = self._load_vector_store(main_path, memory_repr, noise_order)
                # self.noise_memory_vector_stores.append(noise_vector_store)
                # self.memory_units_vector_store.merge_from(noise_vector_store)
                self.memory_units.extend(noise_memory_units)
            except:
                logger.info(f"|> Error merging vector store for order {noise_order}")

    def get_corresponding_raw_documents_by_memory_units(self,
                                                        memory_units_text):
        raw_documents_ids = [self.memory_units2docs[text]['cluster_id'] for text in memory_units_text]
        raw_documents_ids = list(set(raw_documents_ids))

        return [self.raw_documents[ids] for ids in raw_documents_ids]

    def query_with_only_memory_units(self,
                                     query,
                                     maximum_tokens):

        context = [ele['text'] for ele in self.memory_units]
        if maximum_tokens != -1 and maximum_tokens > 0:
            context = self.trim_context_by_tokens(context, maximum_tokens)

        context = self.get_generate_answer_context(context)
        return self.generate_answer(query=query, context=context)

    def retrieve_top_k_memory_units(self,
                                    query: str,
                                    top_k: int):

        return self.memory_units_vector_store.similarity_search_with_score(query,
                                                                           k=top_k)

    def get_memory_units_texts(self, memory_unit_ids_to_scores: dict, memory_unit_ids_to_unit: dict):

        sorted_memory_unit_ids_scores = sorted(memory_unit_ids_to_scores.items(), key=lambda x: x[1])
        memory_units_texts = [memory_unit_ids_to_unit[unit_id].page_content for unit_id, _ in
                              sorted_memory_unit_ids_scores]
        return memory_units_texts

    def generate_iterative_retrieval_thought(self, query: str, context: List[str]):

        if self.reader_model == "gpt-4o-mini":
            thought_prompt = PromptTemplate(input_variables=["context", "question"],
                                            template=iterative_retrieval_template)
            thought_chain = thought_prompt | self.llm | StrOutputParser()
            input_data = {
                "context": "\n\n".join(["{}. {}".format(i + 1, text) for i, text in enumerate(context)]),
                "question": query
            }
            thought = thought_chain.invoke(input_data)
        else:
            if "instruct" in self.reader_model:
                instruction = iterative_retrieval_instruction
                input_text = iterative_retrieval_input_template.format(
                    context="\n\n".join(["{}. {}".format(i + 1, text) for i, text in enumerate(context)]),
                    question=query
                )
                prompt_chat_template = self.reader.get_generator_prompts_chat_format([instruction], [input_text])
                inputs = self.reader.tokenizer_encode_chat_format(prompt_chat_template)
            else:
                raise NotImplementedError(
                    f"generate_iterative_retrieval_thought using f{self.reader_model} is not implemented!")

            generated_token_ids, _ = self.reader.generate(inputs)
            generated_text = self.reader.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)[0]
            thought = generated_text

        return thought

    def query_with_iterative_retrieve_top_k_memory_units(self,
                                                         query: str,
                                                         top_k: int,
                                                         top_t: int = 10,
                                                         num_turns: int = 4,
                                                         maximum_tokens: int = -1):

        thoughts = []
        memory_unit_ids_to_scores, memory_unit_ids_to_unit = {}, {}
        for _ in tqdm(range(num_turns)):
            retrieval_query = query if len(thoughts) == 0 else query + " " + thoughts[-1]
            retrieval_result = self.retrieve_top_k_memory_units(retrieval_query, top_t)
            for unit, score in retrieval_result:
                unit_id = unit.metadata['global_id']
                memory_unit_ids_to_scores[unit_id] = min(memory_unit_ids_to_scores.get(unit_id, 1e9), score)
                memory_unit_ids_to_unit[unit_id] = unit

            memory_units_texts = self.get_memory_units_texts(memory_unit_ids_to_scores, memory_unit_ids_to_unit)
            thought = self.generate_iterative_retrieval_thought(
                query=query, context=memory_units_texts
            )
            thoughts.append(thought.strip())

        sorted_memory_unit_ids_scores = sorted(memory_unit_ids_to_scores.items(), key=lambda x: x[1])
        retrieved_memory_units = [memory_unit_ids_to_unit[unit_id] for unit_id, _ in
                                  sorted_memory_unit_ids_scores[:top_k]]

        if maximum_tokens != -1 and maximum_tokens > 0:
            retrieved_memory_units = self.trim_context_by_tokens(retrieved_memory_units, maximum_tokens)

        context = self.get_generate_answer_context(retrieved_memory_units)

        return self.generate_answer(query=query, context=context)

    def iterative_retrieve_top_k_memory_units(self,
                                              query: str,
                                              top_k: int,
                                              top_t: int = 10,
                                              num_turns: int = 4):
        thoughts = []
        memory_unit_ids_to_scores, memory_unit_ids_to_unit = {}, {}
        for _ in tqdm(range(num_turns)):
            retrieval_query = query if len(thoughts) == 0 else query + " " + thoughts[-1]
            retrieval_result = self.retrieve_top_k_memory_units(retrieval_query, top_t)
            for unit, score in retrieval_result:
                unit_id = unit.metadata['global_id']
                memory_unit_ids_to_scores[unit_id] = min(memory_unit_ids_to_scores.get(unit_id, 1e9), score)
                memory_unit_ids_to_unit[unit_id] = unit

            memory_units_texts = self.get_memory_units_texts(memory_unit_ids_to_scores, memory_unit_ids_to_unit)
            thought = self.generate_iterative_retrieval_thought(
                query=query, context=memory_units_texts
            )
            thoughts.append(thought.strip())

        sorted_memory_unit_ids_scores = sorted(memory_unit_ids_to_scores.items(), key=lambda x: x[1])
        retrieved_memory_units = [memory_unit_ids_to_unit[unit_id] for unit_id, _ in
                                  sorted_memory_unit_ids_scores[:top_k]]

        return retrieved_memory_units

    def trim_context_by_tokens(self, context, maximum_tokens=1000):
        logger.info(f"We are going to trim the context by {maximum_tokens} tokens")
        context = "\n".join(context)
        encodings = self.token_encoder.encode(context)
        num_tokens = len(encodings)
        if num_tokens > maximum_tokens:
            context = self.token_encoder.decode(encodings[:maximum_tokens])
        return context.split("\n")

    def query_with_top_k_memory_units(self,
                                      query,
                                      top_k: int,
                                      top_t: int,
                                      num_turns: int,
                                      answer_with_memory_units: int,
                                      maximum_tokens: int,
                                      use_iterative_retrieval: int):
        if use_iterative_retrieval:
            retrieved_memory_units = self.iterative_retrieve_top_k_memory_units(query,
                                                                                top_k,
                                                                                top_t,
                                                                                num_turns)
            top_k_memory_units_text = [ele.page_content for ele in retrieved_memory_units]
        else:
            retrieved_top_k_memory_units = self.retrieve_top_k_memory_units(query, top_k)
            top_k_memory_units_text = [ele.page_content for ele, _ in retrieved_top_k_memory_units]

        # if maximum_tokens != -1 and maximum_tokens > 0:
        #     top_k_memory_units_text = self.trim_context_by_tokens(top_k_memory_units_text, maximum_tokens)

        if answer_with_memory_units:
            logger.info(f"We are going to answer with memory units")
            context = self.get_generate_answer_context(top_k_memory_units_text)
        else:
            logger.info(f"We are going to answer with documents")
            corresponding_raw_documents = self.get_corresponding_raw_documents_by_memory_units(top_k_memory_units_text)
            context = self.get_generate_answer_context(corresponding_raw_documents)
        return self.generate_answer(query=query, context=context)

    def query_with_top_k_with_rerank_top_r_memory_units(self,
                                                        query: str,
                                                        top_k: int,
                                                        top_r: int,
                                                        top_t: int,
                                                        num_turns: int,
                                                        answer_with_memory_units: int,
                                                        maximum_tokens: int,
                                                        use_iterative_retrieval: int):
        if use_iterative_retrieval:
            retrieved_memory_units = self.iterative_retrieve_top_k_memory_units(query,
                                                                                top_k,
                                                                                top_t,
                                                                                num_turns)
            top_k_memory_units = [ele for ele in retrieved_memory_units]
        else:
            retrieved_top_k_memory_units = self.retrieve_top_k_memory_units(query, top_k)
            top_k_memory_units = [ele for ele, _ in retrieved_top_k_memory_units]

        reranked_top_r_memory_units = self.rerank_documents_batch(query,
                                                                  top_k_memory_units,
                                                                  top_r=top_r)
        top_r_memory_units_texts = [fact.page_content for fact in reranked_top_r_memory_units]

        if answer_with_memory_units:
            logger.info(f"We are going to answer with memory units")
            # if maximum_tokens != -1 and maximum_tokens > 0:
            #     top_r_memory_units_texts = self.trim_context_by_tokens(top_r_memory_units_texts, maximum_tokens)
            context = self.get_generate_answer_context(top_r_memory_units_texts)
        else:
            logger.info(f"We are going to answer with documents")
            corresponding_raw_documents = self.get_corresponding_raw_documents_by_memory_units(top_r_memory_units_texts)
            # if maximum_tokens != -1 and maximum_tokens > 0:
            #     corresponding_raw_documents = self.trim_context_by_tokens(corresponding_raw_documents, maximum_tokens)

            context = self.get_generate_answer_context(corresponding_raw_documents)
        return self.generate_answer(query=query, context=context)

    def query_with_cluster_with_raw_documents(self,
                                              query: str):

        all_raw_documents = []
        for ids in self.summaries_ids:
            all_raw_documents.append(self.raw_documents[ids])

        context = self.get_generate_answer_context(all_raw_documents)
        return self.generate_answer(query=query, context=context)

    def query_with_memory_units_with_common_cluster(self,
                                                    query: str,
                                                    top_k: int,
                                                    top_s: int,
                                                    answer_with_memory_units: int,
                                                    maximum_tokens: int,
                                                    top_t: int,
                                                    num_turns: int,
                                                    use_iterative_retrieval: int):
        top_s_cluster_from_fasiss = self.summary_vector_store.similarity_search(query, k=top_s)
        top_s_cluster_id_from_faiss = [ele.metadata['cluster_id'] for ele in top_s_cluster_from_fasiss]

        if use_iterative_retrieval:
            retrieved_memory_units = self.iterative_retrieve_top_k_memory_units(query,
                                                                                top_k,
                                                                                top_t,
                                                                                num_turns)
            top_k_memory_units = [ele for ele in retrieved_memory_units]
            cluster_id_from_memory_units = list(set([fact.metadata['cluster_id'] for fact in top_k_memory_units]))
        else:
            retrieved_top_k_memory_units = self.retrieve_top_k_memory_units(query, top_k)
            top_k_memory_units = [ele for ele, _ in retrieved_top_k_memory_units]

            cluster_id_from_memory_units = list(set([fact.metadata['cluster_id'] for fact, _ in top_k_memory_units]))
        common_cluster_ids = list(set(top_s_cluster_id_from_faiss) & set(cluster_id_from_memory_units))

        if answer_with_memory_units:
            filtered_memory_units = [ele for ele, _ in top_k_memory_units if
                                     ele.metadata['cluster_id'] in common_cluster_ids]
            filtered_memory_units_text = [ele.page_content for ele in filtered_memory_units]

            # if maximum_tokens != -1 and maximum_tokens > 0:
            #     filtered_memory_units_text = self.trim_context_by_tokens(filtered_memory_units_text, maximum_tokens)

            context = self.get_generate_answer_context(filtered_memory_units_text)
        else:
            raw_documents = [self.raw_documents[ids] for ids in common_cluster_ids]
            # if maximum_tokens != -1 and maximum_tokens > 0:
            #     raw_documents = self.trim_context_by_tokens(raw_documents, maximum_tokens)

            context = self.get_generate_answer_context(raw_documents)
        return self.generate_answer(query=query, context=context)

    def get_generate_answer_context(self, text_list):
        return "\n".join([f"{i + 1}. {text}" for i, text in enumerate(text_list)])

    def parse_reader_answers(self, text: str):

        candidate_answers = text.split("\n")
        answer = ""
        i = 0
        while len(answer) < 1 and i < len(candidate_answers):
            answer = candidate_answers[i].strip()
            i += 1
        if "answer is" in answer:
            idx = answer.find("answer is")
            answer = answer[idx + len("answer is"):].strip()
            if answer.startswith(":"):
                answer = answer[1:].strip()
        return answer

    def generate_answer(self, query, context):

        if self.reader_model == "gpt-4o-mini":
            if self.dataset == "quality":
                answer_prompt = PromptTemplate(input_variables=["query", "context"],
                                               template=answer_prompt_template_for_quality_dataset)
            else:
                answer_prompt = PromptTemplate(input_variables=["query", "context"], template=answer_prompt_template)
            answer_chain = answer_prompt | self.llm | StrOutputParser()
            input_data = {"query": query, "context": context}
            answer = answer_chain.invoke(input_data)
        else:
            if "instruct" in self.reader_model:
                instruction = answer_prompt_instruction_for_quality_dataset if self.dataset == "quality" else answer_prompt_instruction
                input_template = answer_prompt_input_template_for_quality_dataset if self.dataset == "quality" else answer_prompt_input_template
                input_text = input_template.format(query=query, context=context)
                prompt_chat_template = self.reader.get_generator_prompts_chat_format([instruction], [input_text])
                inputs = self.reader.tokenizer_encode_chat_format(prompt_chat_template)
            else:
                prompt_template = answer_prompt_template_for_quality_dataset if self.dataset == "quality" else answer_prompt_template
                prompt = prompt_template.format(query=query, context=context)
                inputs = self.reader.tokenizer_encode([prompt])
            generated_token_ids, _ = self.reader.generate(inputs)
            generated_text = self.reader.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)[0]
            answer = self.parse_reader_answers(generated_text)

        context_length = self.count_tokens(context)

        return {"answer": answer, "context_length": context_length}

    def count_tokens(self, context):
        context = "\n".join(context)
        encodings = self.token_encoder.encode(context)
        return len(encodings)

    def parse_reader_rerank_scores(self, text: str):
        pattern = r'Doc: (\d+), Relevance Score: (\d+)'
        matches = re.findall(pattern, text)
        ratings = [RatingScore(document=int(match[0]), relevance_score=float(match[1])) for match in matches]
        return MultiRatingScore(ratings=ratings)

    def generate_rerank_scores(self, input_data, use_reader_model=True):

        if not use_reader_model or self.reader_model == "gpt-4o-mini":
            prompt_template = PromptTemplate(
                input_variables=["context_str", "query"],
                template=batch_rerank_template
            )
            llm_chain = prompt_template | self.llm.with_structured_output(MultiRatingScore)
            structured_output = llm_chain.invoke(input_data)
        else:
            context = input_data["context_str"]
            query = input_data["query"]
            if "instruct" in self.reader_model:
                instruction = batch_rerank_instruction
                input_text = batch_rerank_input_template.format(context_str=context, query=query)
                prompt_chat_template = self.reader.get_generator_prompts_chat_format([instruction], [input_text])
                inputs = self.reader.tokenizer_encode_chat_format(prompt_chat_template)
            else:
                raise NotImplementedError(f"batch_rerank using f{self.reader_model} is not implemented!")

            generated_token_ids, _ = self.reader.generate(inputs)
            generated_text = self.reader.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)[0]
            structured_output = self.parse_reader_rerank_scores(generated_text)

        return structured_output

    def rerank_documents_batch(self, query: str, docs: List[Document], top_r: int = 3, batch_num: int = 10) -> List[
        Document]:

        # First we need to split the documents into batches
        doc_batches = [docs[i:i + batch_num] for i in range(0, len(docs), batch_num)]

        doc_batches_str = []
        for i, doc_batch in enumerate(doc_batches):
            context_str = ""
            for j, doc in enumerate(doc_batch):
                context_str += f"Document {i * batch_num + j}:\n{doc.page_content}\n\n"
            doc_batches_str.append(context_str)


        scored_docs = []
        for doc_batch_str in doc_batches_str:
            input_data = {"context_str": doc_batch_str, "query": query}
            structured_output = self.generate_rerank_scores(
                input_data=input_data,
                use_reader_model=True  
            )
            logger.info(structured_output)
            for rating in structured_output.ratings:
                doc_id = rating.document
                score = rating.relevance_score
                scored_docs.append((docs[doc_id], score))

        reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked_docs[:top_r]]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key",
                        type=str, default="", help="OpenAI API Key")
    parser.add_argument("--memory_repr",
                        type=str,
                        choices=['raw_documents',
                                 'chunks',
                                 'triples',
                                 'atomic_facts',
                                 'summary',
                                 'mix',
                                 'mix2'],
                        default='mix',
                        help="memory_representation")
    parser.add_argument("--dataset",
                        type=str,
                        choices=['hotpotqa',
                                 'musique',
                                 '2wikimultihopqa',
                                 'locomo',
                                 'narrativeqa',
                                 'quality'],
                        default='2wikimultihopqa',
                        help='The qa dataset that you want to experiments')
    parser.add_argument("--ablation_type",
                        type=str,
                        choices=[
                            'only_memory_units',
                            'memory_units_with_top_k',
                            'memory_units_with_top_k_with_rerank_top_r',
                            'cluster_summary_with_raw_documents',
                            'memory_units_with_common_cluster',
                        ],
                        default='memory_units_with_top_k', )
    parser.add_argument("--answer_with_memory_units",
                        type=int,
                        choices=[0, 1],
                        default=1,
                        help='Answer with memory units or its corresponding raw documents')
    parser.add_argument("--use_iterative_retrieval",
                        type=int,
                        choices=[0, 1],
                        default=0,
                        help='Whether use iterative retrieval')
    parser.add_argument("--top_k", type=int, default=100, help="Top k memory units to retrieve")
    parser.add_argument("--top_r", type=int, default=0, help="Top r memory units to rerank"),
    parser.add_argument("--top_s", type=int, default=0, help="Top s clusters summary to retrieve")
    parser.add_argument("--top_t", type=int, default=0, help='Top r memory units to retrieve in each iteration')
    parser.add_argument("--num_turns", type=int, default=0, help="Number of turns for iterative retrieval")
    parser.add_argument("--maximum_tokens", type=int, default=4096, help="Maximum tokens for the context")
    parser.add_argument("--debug", type=int, default=0, choices=[0, 1], help="1 represents debug mode")
    parser.add_argument("--reader_model",
                        type=str,
                        choices=['gpt-4o-mini',
                                 'llama3_8b_instruct',
                                 'llama3.1_8b_instruct',
                                 'llama3.1_70b_instruct',
                                 'qwen2.5_7b_instruct',
                                 'qwen2.5_32b_instruct',
                                 'qwen2.5_72b_instruct',
                                 'gemma2_9b_instruct',
                                 "gemma2_27b_instruct"],
                        default="gpt-4o-mini",
                        help="LLM models used to generate answer")
    parser.add_argument("--num_noise_docs", type=int, default=0, help="Number of noise documents to add")

    args = parser.parse_args()

    os.environ["OPENAI_API_KEY"] = args.api_key

    args_dict = vars(args)  
    logger.info("|> Running with the following parameters:")
    for key, value in args_dict.items():
        logger.info(f"|> {key}: {value}")

    if args_dict['dataset'] == "hotpotqa":
        load_data_file = "../../datasets/hotpotqa/hotpotqa_200_sample.csv"
    elif args_dict['dataset'] == 'musique':
        load_data_file = "../../datasets/musique/musique_200_sample.csv"
    elif args_dict['dataset'] == '2wikimultihopqa':
        load_data_file = "../../datasets/2wikimultihopqa/2wikimultihopqa_200_sample.csv"
    elif args_dict['dataset'] == 'locomo':
        load_data_file = "../../datasets/locomo/locomo_sample_multi_hop_191.csv"
    elif args_dict['dataset'] == 'narrativeqa':
        load_data_file = "../../datasets/narrativeqa/narrativeqa_200_sample.csv"
    elif args_dict['dataset'] == 'quality':
        load_data_file = "../../datasets/quality/quality_200_sample.csv"
    else:
        raise ValueError("Dataset not supported.")

    dataset = args.dataset
    memory_repr = args.memory_repr
    top_k = args.top_k
    top_r = args.top_r
    top_s = args.top_s
    top_t = args.top_t
    num_turns = args.num_turns
    ablation_type = args.ablation_type
    answer_with_memory_units = args.answer_with_memory_units
    use_iterative_retrieval = args.use_iterative_retrieval
    maximum_tokens = args.maximum_tokens
    debug_mode = args.debug
    reader_model_name = args.reader_model
    num_noise_docs = args.num_noise_docs

    results_path = f"./results/{dataset}"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    save_file_name = f"{results_path}/reader_model_[{reader_model_name}]_ablation_type_[{ablation_type}]_memory_repr_[{memory_repr}]_topk_{top_k}_topr_{top_r}_tops_{top_s}_topt_{top_t}_num_turns_{num_turns}_answer_with_memory_units_{answer_with_memory_units}_use_iterative_retrieval_{use_iterative_retrieval}_maximum_tokens_{maximum_tokens}_num_noise_docs_{num_noise_docs}.csv"
    logger.info(f"|> The save file path is located in {save_file_name}")

    if os.path.exists(save_file_name):
        logger.info(f"Loading existing results from {save_file_name}")
        df = pd.read_csv(save_file_name)
        processed_questions = df.loc[df['predictions'] != 'BLANK'].shape[0]
        logger.info(f"|> {processed_questions} questions have been processed. Resuming from the next question.")
    
    else:
        df = pd.read_csv(load_data_file)
        df['predictions'] = ["BLANK"] * df.shape[0]
        df['total_tokens'] = ["BLANK"] * df.shape[0]
        df['total_prompt_tokens'] = ["BLANK"] * df.shape[0]
        df['total_completion_tokens'] = ["BLANK"] * df.shape[0]
        df['total_cost'] = ["BLANK"] * df.shape[0]

    num_of_questions = df.shape[0] if debug_mode != 1 else 1

    if args.reader_model == "gpt-4o-mini":
        reader = None
    else:
        if args.reader_model in ["llama3.1_70b_instruct", "qwen2.5_32b_instruct", "qwen2.5_72b_instruct",
                                 "gemma2_27b_instruct"]:
            reader_tokenizer, reader_llm = load_llm_tokenizer_and_model(args.reader_model, load_in_4bit=True)
        else:
            reader_tokenizer, reader_llm = load_llm_tokenizer_and_model(args.reader_model, load_in_4bit=False)
        reader = Generator(reader_tokenizer, reader_llm, max_length=8000, max_new_tokens=64)

    for qa_order in tqdm(range(0, num_of_questions)):

        logger.info(f"Processing question order {qa_order}")

        # try:
        current_prediction = df.at[qa_order, 'predictions']
        if current_prediction != "BLANK":
            logger.info(f"Question {qa_order} has already been processed. Skipping...")
            continue

        memobox = MemoBox(dataset=dataset, reader_model=args.reader_model, reader=reader)

        memobox.load_dataset(memory_repr, dataset, qa_order, num_noise_docs)

        logger.info(f"Answering questions for order {qa_order}")
        query = df['question'].tolist()[qa_order]
        gold_answer = df['answer'].tolist()[qa_order]
        assert "formatted_sentences" in df.columns, "formatted_sentences column is not in the dataframe"
        memobox.raw_documents = ast.literal_eval(df['formatted_sentences'].tolist()[qa_order])

        with get_openai_callback() as cb:

            if ablation_type == 'only_memory_units':
                logger.info(f"We are using only memory units")
                final_answer = memobox.query_with_only_memory_units(query=query,
                                                                    maximum_tokens=maximum_tokens)
            elif ablation_type == 'memory_units_with_top_k':
                logger.info(f"We are using memory units with top k")
                final_answer = memobox.query_with_top_k_memory_units(query=query,
                                                                     top_k=top_k,
                                                                     top_t=top_t,
                                                                     num_turns=num_turns,
                                                                     answer_with_memory_units=answer_with_memory_units,
                                                                     maximum_tokens=maximum_tokens,
                                                                     use_iterative_retrieval=use_iterative_retrieval)
            elif ablation_type == 'memory_units_with_top_k_with_rerank_top_r':
                logger.info(f"We are using memory units with top k with rerank top r")
                final_answer = memobox.query_with_top_k_with_rerank_top_r_memory_units(query=query,
                                                                                       top_k=top_k,
                                                                                       top_r=top_r,
                                                                                       top_t=top_t,
                                                                                       num_turns=num_turns,
                                                                                       use_iterative_retrieval=use_iterative_retrieval,
                                                                                       answer_with_memory_units=answer_with_memory_units,
                                                                                       maximum_tokens=maximum_tokens)
            elif ablation_type == 'cluster_summary_with_raw_documents':
                logger.info(f"We are using cluster summary with raw documents")
                final_answer = memobox.query_with_cluster_with_raw_documents(query=query)
            elif ablation_type == 'memory_units_with_common_cluster':
                logger.info(f"We are using memory units with common cluster")
                final_answer = memobox.query_with_memory_units_with_common_cluster(query=query,
                                                                                   top_k=top_k,
                                                                                   top_s=top_s,
                                                                                   top_t=top_t,
                                                                                   num_turns=num_turns,
                                                                                   maximum_tokens=maximum_tokens,
                                                                                   use_iterative_retrieval=use_iterative_retrieval,
                                                                                   answer_with_memory_units=answer_with_memory_units)

            logger.info(f"Predicted Answer: {final_answer['answer']} | Gold Answer: {gold_answer}")
            logger.info(f"Total Tokens: {cb.total_tokens}")
            logger.info(f"Prompt Tokens: {cb.prompt_tokens}")
            logger.info(f"Completion Tokens: {cb.completion_tokens}")
            logger.info(f"Total Cost (USD): ${cb.total_cost}")
            logger.info(f"Context Length: {final_answer['context_length']}")

            df.at[qa_order, 'predictions'] = final_answer['answer']
            df.at[qa_order, 'total_tokens'] = cb.total_tokens
            df.at[qa_order, 'total_prompt_tokens'] = cb.prompt_tokens
            df.at[qa_order, 'total_completion_tokens'] = cb.completion_tokens
            df.at[qa_order, 'total_cost'] = cb.total_cost
            df.at[qa_order, 'context_length'] = final_answer['context_length']

            df.to_csv(save_file_name, index=False)

