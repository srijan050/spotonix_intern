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


# Prompt
prompt_data = [
    {"role": "system", "content": "You are an expert in business intelligence, SQL and databases."},
    {"role": "user", "content": f"Schema:\n{schema}"},
    {"role": "user", "content": "User's request: Find customers who have returned items more than 20% more often than the average customer returns for a store in a given state for a given year."},
]

class SQLResponse(OpenAISchema):
    """
    This class captures the essential elements to construct an SQL query based on a text-based query.
    It includes the SQL components structured in a way that can be processed to form a valid SQL query.
    """

    class SQLComponent(OpenAISchema):
        """
        This class represents an individual component of an SQL query.
        Each component corresponds to a specific part of the SQL statement.
        """
        sql_code: str = Field(description="Code snippet for the SQL component.")

    query_components: list[SQLComponent] = Field(
        description="List of SQLComponent objects. Each object contains the SQL code for the query according to the request done by the user."
    )

client = instructor.from_openai(OpenAI(api_key = api_key))

def get_openai_response(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        response_model = SQLResponse,
        max_tokens=1000
    )
    return response

if __name__ == "__main__":
    response = get_openai_response(prompt_data)
    print("Response from OpenAI:\n", response)
