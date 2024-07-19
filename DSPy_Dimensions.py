import dspy
from openai import OpenAI
import json
from pydantic import Field
from instructor import OpenAISchema
import instructor

api_key = open("/home/srijan/Documents/api.txt", "r").read(51)

class LLMResponseDimensionTables(OpenAISchema):

    class DimensionPhrase(OpenAISchema):
        """
        This class captures the name of the dimension table which is
relevant to the user's request.
        It also stores a phrase in the user's request that matches
best to the dimension table.
        """
        dimension: str = Field(description="Name of the dimension table relevant to the user's request.")
        phrase: str = Field(description="Phrase in the user's request that indicates the dimension table.")

    input_dimensions: list[DimensionPhrase] = Field(
        description="List of DimensionPhrase objects. Each DimensionPhrase object contains the name of a dimension table which is"
                    "relevant to the user's request, and the corresponding phrase in the user's request."
    )


class output(dspy.Signature):
  """You are an expert in business intelligence, SQL and databases. Analyze the user's request to understand the input dimension tables that are needed to answer the user's request. What are the most important dimension tables that are relevant to this request. The names and description of the available dimension tables are provided in the following json structure: {'call center': 'The Call Center Dimension provides information about different call centers, including their names, managers, counties, and unique identifiers.', 'catalog page': 'The Catalog Page dimension provides details about catalog pages in the tpcds_1.tpcds.catalog_page table.', 'customer': 'The Customer Dimension provides detailed information about customers, including personal details and preferences.', 'customer address': 'The Customer Address dimension provides details about the address of customers, including street information, location type, ZIP code, country, state, city, county, and GMT offset.', 'customer demographics': 'Customer Demographics dimension provides information about the demographic details of customers, including gender, marital status, education status, purchase estimate, credit rating, dependency count, dependents employed count, and dependents college count.', 'date': 'The Date Dimension provides detailed information about dates, including attributes such as day, month, year, and various sequences related to time.', 'household demographics': 'Dimension based on the table tpcds_1.tpcds.household_demographics', 'income bands': 'The Income Bands dimension provides information about different income ranges, with details on the lower and upper bounds of each band.', 'tpcds item': 'The TPCDS Item Dimension is based on the table tpcds_1.tpcds.item and contains information about various attributes related to items such as category, manufacturer, brand, price, and more.', 'promotion data': 'Dimension containing information about different promotion channels and promo SKs', 'reason': 'The Reason Dimension provides information about different reasons associated with transactions or events.', 'shipping modes': 'The Shipping Modes dimension provides details about different modes of shipping available, including unique identifiers for each mode.', 'store': 'The Store Dimension contains information about different stores, including their IDs, names, locations, and company details.', 'time': 'The Time Dimension represents various time-related attributes such as hour, minute, and meal time. It is based on the tpcds_1.tpcds.time_dim table.', 'warehouse details': 'This dimension provides information about warehouses, including their names, sizes, locations, and related details.', 'web page data': 'This dimension is based on the table tpcds_1.tpcds.web_page and contains information about web pages.', 'web site': 'A dimension based on the table tpcds_1.tpcds.web_site, containing information about different websites.'}. The key in the json is the name of the dimension table, and the value is the description. Use the description of the dimension tables to determine which of them are relevant for this user's request. Some of the phrases used in the user's request indicate the dimension tables that are needed. The phrases related to attributes are more important to indicate the dimension table. The dimension tables provide attributes and dimensions important to answer the user's question.Please pick the names of the dimensions tables that most relevant to the user's request from the given list of dimensions. The list of dimension tables are: ['call center', 'catalog page', 'customer', 'customer address', 'customer demographics', 'date', 'household demographics', 'income bands', 'tpcds item', 'promotion data', 'reason', 'shipping modes', 'store', 'time', 'warehouse details', 'web page data', 'web site'], which are same as the keys in the json structure for descriptions. You must distinguish between Store address and customer address as two different types of addresses. customer address is a separate dimension, but the store address implies store dimension. Please output only those dimension tables that you are absolutely sure are needed to answer the query. Ignore the dimension tables that you are doubtful or not too sure about.For the relevant dimension tables, please also extract the phrases and words in the user's request that refer to the dimension table. The extracted phrase and words semantically match the corresponding dimension table picked and indicate why the dimension table is relevant for the user's question."""
  question: str = dspy.InputField()
  outline: LLMResponseDimensionTables = dspy.OutputField(desc="List of DimensionPhrase objects. Each DimensionPhrase object contains the name of a dimension table which is relevant to the user's request, and the corresponding phrase in the user's request.")

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