import dspy
from openai import OpenAI
import json
from pydantic import Field
from instructor import OpenAISchema
import instructor

api_key = open("/home/srijan/Documents/api.txt", "r").read(51)

schema = """
Tables:
1. store_returns (sr_customer_sk, sr_store_sk, sr_returned_date_sk, SR_FEE)
2. date_dim (d_date_sk, d_year)
3. store (s_store_sk, s_state)
4. customer (c_customer_sk, c_customer_id)
"""

class SQLResponse(OpenAISchema):
    """
    This class captures the essential elements to construct an SQL query based on a text-based query.
    It includes the SQL components structured in a way that can be processed to form a valid SQL query.
    """

    class SQLComponent(OpenAISchema):
        """
        This class represents a SQL Code Snippet according to the user's query
        """
        sql_code: str = Field(description="Code snippet for the SQL component which will be able to query and retrieve from the database according to the user's request.")

    query_components: list[SQLComponent] = Field(
        description="List of SQLComponent objects. Each object contains the SQL code for the query according to the request done by the user."
    )


class output(dspy.Signature):
    """
    Schema : 
    Tables:
    1. store_returns (sr_customer_sk, sr_store_sk, sr_returned_date_sk, SR_FEE)
    2. date_dim (d_date_sk, d_year)
    3. store (s_store_sk, s_state)
    4. customer (c_customer_sk, c_customer_id)

    Example Output : 
    with
        customer_total_return
        as
        (
            select sr_customer_sk as ctr_customer_sk
                , sr_store_sk as ctr_store_sk
                , sum(SR_FEE) as ctr_total_return
            from store_returns
                , date_dim
            where sr_returned_date_sk = d_date_sk and d_year =2000
            group by sr_customer_sk
                ,sr_store_sk
        )
    select c_customer_id
    from customer_total_return ctr1
        , store
        , customer
    where ctr1.ctr_total_return > (
            select avg(ctr_total_return)*1.2
            from customer_total_return ctr2
            where ctr1.ctr_store_sk = ctr2.ctr_store_sk
        )
        and s_store_sk = ctr1.ctr_store_sk
        and s_state = 'NM'
        and ctr1.ctr_customer_sk = c_customer_sk
    order by c_customer_id
    limit 100;
    """

    question: str = dspy.InputField()
    outline: SQLResponse = dspy.OutputField(desc="List of SQLComponent objects. Each object contains the SQL code for the query according to the request done by the user.")

class TypedBlog2Outline(dspy.Module):
    def __init__(self):
        self.question_outline = dspy.functional.TypedPredictor(output)

    def forward(self, question):
        question_outputs = self.question_outline(question=question)
        return question_outputs.outline
    
outline = TypedBlog2Outline()

question = "User's request: Find customers who have returned items more than 20% more often than the average customer returns for a store in a given state for a given year."


turbo = dspy.OpenAI(model='gpt-3.5-turbo',max_tokens=1000,api_key=api_key)
dspy.settings.configure(lm = turbo)
print(outline(question=question))

print('-'*30)

question = "User's request: Analyze, for each state, all items that were sold in stores in a particular quarter and returned in the next three quarters and then repurchased by the customer through the catalog channel in the three following quarters."


turbo = dspy.OpenAI(model='gpt-3.5-turbo',max_tokens=1000,api_key=api_key)
dspy.settings.configure(lm = turbo)
print(outline(question=question))