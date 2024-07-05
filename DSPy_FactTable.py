import dspy
from openai import OpenAI
import json
from pydantic import Field
from instructor import OpenAISchema
import instructor

api_key = open("/home/srijan/Documents/api.txt", "r").read(51)

class LLMResponseFactTables(OpenAISchema):

    class FactPhrase(OpenAISchema):
        """
        This class captures the name of the fact table which is
relevant to the user's request.
        It also stores a phrase in the user's request that matches
best to the fact table.
        """
        fact: str = Field(description="Name of the fact table relevant to the user's request.")
        phrase: str = Field(description="Phrase in the user's request that indicates the fact table.")

    input_facts: list[FactPhrase] = Field(
        description="List of FactPhrase objects. Each FactPhrase object contains the name of a fact table which is "
                    "relevant to the user's request, and the corresponding phrase in the user's request."
    )


class output(dspy.Signature):
  """You are an expert in business intelligence, SQL and databases. Analyze the user\'s request to understand the input fact tables that are needed to answer the user\'s request. This request needs to be answered from data stored in a warehouse. How many fact tables are needed to answer this request? The names and description of the available fact tables are provided in the following json structure: {"catalog returns": "The Catalog Returns Fact captures data related to returns processed by the company, including details such as return quantity, return amount, tax, fees, and other financial metrics.", "catalog sales": "Catalog Sales Fact quantifies the sales transactions recorded in the tpcds_1.tpcds.catalog_sales table. It provides insights into various aspects of sales performance and customer demographics.", "inventory management": "Inventory Management fact based on data in table tpcds_1.tpcds.inventory, quantified by Quantity measure", "store returns": "A business activity based on store returns data, quantified by return quantity, return amount, fee, and net loss.", "store sales data": "Store Sales Data quantifies the sales transactions in the store, capturing details such as quantity, costs, prices, discounts, and profits.", "web returns": "The Web Returns Fact captures data related to returns made by customers on the web platform, including details such as return quantities, amounts, taxes, fees, and customer demographics.", "web sales": "Fact Web Sales quantifies the sales data primarily based on the table tpcds_1.tpcds.web_sales. It includes various measures to analyze sales performance and customer transactions."}.  The key in the json is the name of the fact table, and the value is the description. Use the description of the fact tables to determine which of them are relevant for this user\'s request. Some of the phrases used in the user\'s request indicate the fact tables that are needed. The phrases related to measures and calculations are more important to indicate the fact table. The fact tables provide measures and calculations important to answer the user\'s question.Please pick the names of the fact tables that most relevant to the user\'s request from the given list of facts. The list of fact tables are: ["catalog returns", "catalog sales", "inventory management", "store returns", "store sales data", "web returns", "web sales"], which are same as the keys in the json structure for descriptions. Store sales table contains information only about purchases made by customers at stores.  Store returns contains information about returns made by customers at stores. Any mention of catalog is a strong indication of catalog sales or catalog returns fact tables. Any mention of web site or web page is a strong indication of web sales or web returns fact tables. Any mention of store is a strong indication of store sales or store returns fact tables. Any mention of warehouse indicates catalog. Any mention of call center indicates catalog. A channel is the medium through which sales and return transactions are performed by customers. There are 3 channels, web, catalog and store. All 3 channels indicate web sales, catalog sales and store sales. Sales Transactions related to the web channel are recorded in web_sales table. Return transactions related to the web channel are recorded in the web_returns table. Sales Transactions related to the catalog channel are recorded in catalog_sales table. Return transactions related to the catalog channel are recorded in the catalog_returns table. Sales Transactions related to the store channel are recorded in store_sales table. Return transactions related to the store channel are recorded in the store_returns table. Please output only those fact tables that you are absolutely sure are needed to answer the query. Ignore the facts tables that you are doubtful or not too sure about.For the relevant fact tables, please also extract the phrases and words in the user\'s request that refer to the fact table. The phrases and words must be present in the user query. The extracted phrase and words semantically match the corresponding fact table picked and indicate why the fact table is relevant for the user\'s question."""

  question: str = dspy.InputField()
  outline: LLMResponseFactTables = dspy.OutputField(desc="List of FactPhrase objects. Each FactPhrase object contains the name of a fact table which is relevant to the user's request, and the corresponding phrase in the user's request.")

class TypedBlog2Outline(dspy.Module):
    def __init__(self):
        self.question_outline = dspy.functional.TypedPredictor(output)

    def forward(self, question):
        question_outputs = self.question_outline(question=question)
        return question_outputs.outline
    
outline = TypedBlog2Outline()

question = "User's request: Analyze, for each state, all items that were sold in stores in a particular quarter and returned in the next three quarters and then repurchased by the customer through the catalog channel in the three following quarters."


turbo = dspy.OpenAI(model='gpt-3.5-turbo',max_tokens=1000,api_key=api_key)
dspy.settings.configure(lm = turbo)
print(outline(question=question))