from openai import OpenAI
import json
from pydantic import Field
from instructor import OpenAISchema
import instructor
import dspy

api_key = open("/home/srijan/Documents/api.txt", "r").read(51)

#Questions : 

tpcds_questions = ["Find customers who have returned items more than 20% more often than the average customer returns for a store in a given state for a given year.",
        "Report the ratios of weekly web and catalog sales increases from one year to the next year for each week. That is, compute the increase of Monday, Tuesday, ... Sunday sales from one year to the following. ",
        "Report the total extended sales price per item brand of a specific manufacturer for all sales in a specific month of the year.",
        "Find customers who spend more money via catalog than in stores. Identify preferred customers and their country of origin.",
        "Report sales, profit, return amount, and net loss in the store, catalog, and web channels for a 14 -day window. Rollup results by sales channel and channel specific sales method (store for store sales, catalog page for catalog sales and web site for web sales)",
        "List all the states with at least 10 customers who during a given month bought items with the price tag at least 20% higher than the average price of items in the same category.",
        "Compute the average quantity, list price, discount, and sales price for promotional items sold in stores where the promotion is not offered by mail or a special event. Restrict the results to a specific gender, marital and educational status.",
        "Compute the net profit of stores located in 400 Metropolitan areas with more than 10 preferred customers.",
        "Categorize store sales transactions into 5 buckets according to the number of items sold. Each bucket contains the average discount amount, sales price, list price, tax, net paid, paid price including tax, or net profit.",
        "Count the customers with the same gender, marital status, education status, purchase estimate, credit rating, dependent count, employed dependent count and college dependent count who live in certain counties and who have purchased from both stores and another sales channel during a three month time period of a given year.",
        "Find customers whose increase in spending was large over the web than in stores this year compared to last year.",
        "Compute the revenue ratios across item classes: For each item in a list of given categories, during a 30 day time period, sold through the web channel compute the ratio of sales of that item to the sum of all of the sales in that item's class.",
        "Calculate the average sales quantity, average sales price, average wholesale cost, total wholesale cost for store sales of different customer types (e.g., based on marital status, education status) including their household demographics, sales price and different combinations of state and sales profit for a given year.",
        "First identify items in the same brand, class and category that are sold in all three sales channels in two consecutive years. Then compute the average sales (quantity*list price) across all sales of all three sales channels in the same three years (average sales). Finally, compute the total sales and the total number of sales rolled up for each channel, brand, class and category. Only consider sales of cross channel sales that had sales larger than the average sale.",
        "Report the total catalog sales for customers in selected geographical regions or who made large purchases for a given year and quarter.",
        "Report number of orders, total shipping costs and profits from catalog sales of particular counties and states for a given 60 day period for non-returned sales filled from an alternate warehouse.",
        "Analyze, for each state, all items that were sold in stores in a particular quarter and returned in the next three quarters and then repurchased by the customer through the catalog channel in the three following quarters.",
        "Compute, for each county, the average quantity, list price, coupon amount, sales price, net profit, age, and number of dependents for all items purchased through catalog sales in a given year by customers who were born in a given list of six months and living in a given list of seven states and who also belong to a given gender and education demographic.",
        "Select the top revenue generating products bought by out of zip code customers for a given year, month and manager.",
        "Compute the total revenue and the ratio of total revenue to revenue by item class for specified item categories and time periods.",
        "For all items whose price was changed on a given date, compute the percentage change in inventory between the 30-day period BEFORE the price change and the 30-day period AFTER the change. Group this information by warehouse.",
        "For each product name, brand, class, category, calculate the average quantity on hand. Rollup data by product name, brand, class and category.",
        "Find frequently sold items that were sold more than 4 times on any day during four consecutive years through the store sales channel.  Compute the maximum store sales made by any given customer in a period of four consecutive years.  Compute the best store customers that are in the [Ed: top] 5th percentile of sales. Compute the total web and catalog sales in a particular month made by our best store customers buying our most frequent store items.",
        "Calculate the total specified monetary value of items in a specific color for store sales transactions by customer name and store, in a specific market, from customers who currently don't live in their birth countries and in the neighborhood of the store, and list only those customers for whom the total specified monetary value is greater than 5% of the average value",
        "Get all items that were (i) sold in stores in a particular month and year and (ii) returned and re-purchased by the customer through the catalog channel in the same month and in the six following months. For these items, compute the sum of net profit of store sales, net loss of store loss and net profit of catalog. Group this information by item and store.",
        "Computes the average quantity, list price, discount, sales price for promotional items sold through the catalog channel where the promotion was not offered by mail or in an event for given gender, marital status and educational status.",
        "For all items sold in stores located in six states during a given year, find the average quantity, average list price, average list sales price, average coupon amount for a given gender, marital status, education and customer demographic.",
        "Calculate the average list price, number of non empty (null) list prices and number of distinct list prices of six different sales buckets of the store sales channel. Each bucket is defined by a range of distinct items and information about list price, coupon amount and wholesale cost.",
        "Get all items that were sold in stores in a specific month and year and which were returned in the next six months of the same year and re-purchased by the returning customer afterwards through the catalog sales channel in the following three years. For those these items, compute the total quantity sold through the store, the quantity returned and the quantity purchased through the catalog. Group this information by item and store.",
        "Find customers and their detailed customer data who have returned items, which they bought on the web, for an amount that is 20% higher than the average amount a customer returns in a given state in a given time period across all items. Order the output by customer data.",
        "List counties where the percentage growth in web sales is consistently higher compared to the percentage growth in store sales in the first three consecutive quarters for a given year.",
        "Compute the total discounted amount for a particular manufacturer in a particular 90 day period for catalog sales whose discounts exceeded the average discount by at least 30%.",
        "What is the monthly sales figure based on extended price for a specific month in a specific year, for manufacturers in a specific category in a given time zone. Group sales by manufacturer identifier and sort output by sales amount, by channel, and give Total sales.",
        "Display all customers with specific buy potentials and whose dependent count to vehicle count ratio is larger than 1.2, who in three consecutive years made purchases with between 15 and 20 items in the beginning or the end of each month in stores located in 8 counties.",
        "For the groups of customers living in the same state, having the same gender and marital status who have purchased from stores and from either the catalog or the web during a given year.",
        "Compute store sales gross profit margin ranking for items in a given year for a given list of states.",
        "List all items and current prices sold through the catalog channel from certain manufacturers in a given $30 price range and consistently had a quantity between 100 and 500 on hand in a 60 -day period.",
        "Display count of customers with purchases from all 3 channels in a given year.",
        "Calculate the coefficient of variation and mean of every item and warehouse of two consecutive months. Find items that had a coefficient of variation in the first months of 1.5 or large",
        "Compute the impact of an item price change on the sales by computing the total sales for items in a 30 day period before and after the price change. Group the items by location of warehouse where they were delivered from.",
        "How many items do we carry with specific combinations of color, units, size and category.",
        "For each item and a specific year and month calculate the sum of the extended sales price of store transactions.",
        "Report the sum of all sales from Sunday to Saturday for stores in a given data range by stores.",
        "List the best and worst performing products measured by net profit.",
        "Report the total web sales for customers in specific zip codes, cities, counties or states, or specific items for a given year and quarter. .",
        "Compute the per-customer coupon amount and net profit of all 'out of town' customers buying from stores located in 5 cities on weekends in three consecutive years. The customers need to fit the profile of having a specific dependent count and vehicle count. For all these customers print the city they lived in at the time of purchase, the city in which the store is located, the coupon amount and net profit",
        "Find the item brands and categories for each store and company, the monthly sales figures for a specified year, where the monthly sales figure deviated more than 10% of the average monthly sales for the year, sorted by deviation and store. Report deviation of sales from the previous and the following monthly sales.",
        "Calculate the total sales by different types of customers (e.g., based on marital status, education status), sales price and different combinations of state and sales profit.",
        "Report the worst return ratios (sales to returns) of all items for each channel by quantity and currency sorted by ratio. Quantity ratio is defined as total number of sales to total number of returns. Currency ratio is defined as sum of return amount to sum of net paid .",
        "For each store count the number of items in a specified month that were returned after 30, 60, 90, 120 and more than 120 days from the day of purchase.",
        "Compute the count of store sales resulting from promotions, the count of all store sales and their ratio for specific categories in a particular time zone a nd for a given year and month.",
        "Report the total of extended sales price for all items of a specific brand in a specific year and month.",
        "Find the ID, quarterly sales and yearly sales of those manufacturers who produce items with specific characteristics and whose average monthly sales are larger than 10% of their monthly sales.",
        "Find all customers who purchased items of a given category and class on the web or through catalog in a given month and year that was followed by an in -store purchase at a store near their residence in the three consecutive months. Calculate a histogram of the revenue by these customers in $50 segments showing the number of customers in each of these revenue generated segments.",
        "For a given year, month and store manager calculate the total store sales of any combination all brands.",
        "Compute the monthly sales amount for a specific month in a specific year, for items with three specific colors across all sales channels. Only consider sales of customers residing in a specific time zone. Group sales by item and sort output by sales amount.",
        "Find the item brands and categories for each call center and their monthly sales figures for a specified year, where the monthly sales figure deviated more than 10% of the average monthly sales for the year, sorted by deviation and call center. Report the sales deviation from the previous and following month.",
        "Retrieve the items generating the highest revenue and which had a revenue that was approximately equivalent across all of store, catalog and web within the week ending a given date.",
        "Report the increase of weekly store sales from one year to the next year for each store and day of the week.",
        "What is the monthly sales amount for a specific month in a specific year, for items in a specific category, purchased by customers residing in a specific time zone. Group sales by item and sort output by sales amount.",
        "Find the ratio of items sold with and without promotions in a given month and year. Only items in certain categories sold to customers living in a specific time zone are considered.",
        "For web sales, create a report showing the counts of orders shipped within 30 days, from 31 to 60 days, from 61 to 90 days, from 91 to 120 days and over 120 days within a given year, grouped by warehouse, shipping mode and web site.",
        "For a given year calculate the monthly sales of items of specific categories, classes and brands that were sold in stores and group the results by store manager. Additionally, for every month and manager print the yearly average sales of those items.",
        "Find those stores that sold more cross-sales items from one year to another. Cross-sale items are items that are sold over the Internet, by catalog and in store.",
        "In a given period, for each store, report the list of items with revenue less than 10% the average revenue for all the items in that store.",
        "Compute web and catalog sales and profits by warehouse. Report results by month for a given year during a given 8 -hour period.",
        "Find top stores for each category based on store sales in a specific year.",
        "Compute the per customer extended sales price, extended list price and extended tax for 'out of town' shoppers buying from stores located in two cities in the first two days of each month of three consecutive years. Only consider customers with specific dependent and vehicle counts.",
        "Count the customers with the same gender, marital status, education status, education status, purchase estimate and credit rating who live in certain states and who have purchased from stores but neither form the catalog nor from the web during a two month time period of a given year.",
        "Compute store sales net profit ranking by state and county for a given year and determine the five most profitable states.",
        "Select the top revenue generating products, sold during breakfast or dinner time for one month managed by a given manager across all three sales channels.",
        "Count the number of customers with specific buy potentials and whose dependent count to vehicle count ratio is larger than 1 and who in three consecutive years bought in stores located in 4 counties between 1 and 5 items in one purchase. Only purchases in the first 2 days of the months are considered.",
        "Display customers with both store and web sales in consecutive years for whom the increase in web sales exceeds the increase in store sales for a specified year.",
        "For two consecutive years track the sales of items by brand, class and category.",
        "Computes the average quantity, list price, discount, sales price for promotional items sold through the web channel where the promotion is not offered by mail or in an event for given gender, marital status and educational status.",
        "Report the total sales, returns and profit for all three sales channels for a given 30 day period.  Roll up the results by channel and a unique channel location identifier.",
        "Report the top customer / item combinations having the highest ratio of store channel sales to all other channel sales (minimum 2 to 1 ratio), for combinations with at least one store sale and one other channel sale.  Order the output by highest ratio.",
        "Compute the per customer coupon amount and net profit of Monday shoppers. Only purchases of three consecutive years made on Mondays in large stores by customers with a certain dependent count and with a large vehicle count are considered.",
        "Report extended sales, extended net profit and returns in the store, catalog, and web channels for a 30 day window for items with prices larger than $50 not promoted on television, rollup results by sales channel and channel specific sales means (store for store sales, catalog page for catalog sales and web site for web sales)",
        "Find customers and their detailed customer data who have returned items bought from the catalog more than 20 percent the average customer returns for customers in a given state in a given time period. Order output by customer data.",
        "Find customers who tend to spend more money (net -paid) on -line than in stores.",
        "Retrieve the items with the highest number of returns where the number of returns was approximately equivalent across all store, catalog and web channels (within a tolerance of +/ - 10%), within the week ending a given date.",
        "List all customers living in a specified city, with an income between 2 values.",
        "For all web return reason calculate the average sales, average refunded cash and average return fee by different combinations of customer and sales types (e.g., based on marital status, education status, state and sales profit).",
        "Rollup the web sales for a given year by category and class, and rank the sales among peers within the parent, for each group compute sum of sales, location with the hierarchy and rank within the group.",
        "Count how many customers have ordered on the same day items on the web and the catalog and on the same day have bought items in a store.",
        "How many items do we sell between pacific times of a day in certain stores to customers with one dependent count and 2 or less vehicles registered or 2 dependents with 4 or fewer vehicles registered or 3 dependents and five or less vehicles registered. In one row break the counts into sells from 8:30 to 9, 9 to 9:30, 9:30 to 10 ... 12 to 12:30",
        "Within a year list all month and combination of item categories, classes and brands that have had monthly sales larger than 0.1 percent of the total yearly sales.",
        "What is the ratio between the number of items sold over the internet in the morning (8 to 9am) to the number of items sold in the evening (7 to 8pm) of customers with a specified number of dependents. Consider only websites with a high amount of content.",
        "Display total returns of catalog sales by call center and manager in a particular month for male customers of unknown education or female customers with advanced degrees with a specified buy potential and from a particular time zone.",
        "Compute the total discount on web sales of items from a given manufacturer over a particular 90 day period for sales whose discount exceeded 30% over the average discount of items from that manufacturer in that period of time.",
        "For a given merchandise return reason, report on customers' total cost of purchases minus the cost of returned items.",
        "Produce a count of web sales and total shipping cost and net profit in a given 60 day period to customers in a given state from a named web site for non returned orders shipped from more than one warehouse.",
        "Produce a count of web sales and total shipping cost and net profit in a given 60 day period to customers in a given state from a named web site for returned orders shipped from more than one warehouse.",
        "Compute a count of sales from a named store to customers with a given number of dependents made in a specified half hour period of the day.",
        "Generate counts of promotional sales and total sales, and their ratio from the web channel for a particular item category and month to customers in a given time zone.",
        "Report on items sold in a given 30 day period, belonging to the specified category.",
        "For catalog sales, create a report showing the counts of orders shipped within 30 days, from 31 to 60 days, from 61 to 90 days, from 91 to 120 days and over 120 days within a given year, grouped by warehouse, call center and shipping mode."
]

l = [0, 9, 17, 24, 37]

#Response
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


#OpenAI & Instructor 

client = instructor.from_openai(OpenAI(api_key = api_key))
print('OpenAI : ')

for i in l:
  user_info = client.chat.completions.create(
      model="gpt-3.5-turbo",
      max_tokens=1000,
      response_model=LLMResponseDimensionTables,
      messages=[{'role': 'system', 'content': 'You are an expert in business intelligence, SQL and databases.'}, {'role': 'user', 'content': "User's request: "+tpcds_questions[i]}, {'role': 'user', 'content': """Analyze the user's request to understand the input dimension tables that are needed to answer the user's request. What are the most important dimension tables that are relevant to this request. The names and description of the available dimension tables are provided in the following json structure: {'call center': 'The Call Center Dimension provides information about different call centers, including their names, managers, counties, and unique identifiers.', 'catalog page': 'The Catalog Page dimension provides details about catalog pages in the tpcds_1.tpcds.catalog_page table.', 'customer': 'The Customer Dimension provides detailed information about customers, including personal details and preferences.', 'customer address': 'The Customer Address dimension provides details about the address of customers, including street information, location type, ZIP code, country, state, city, county, and GMT offset.', 'customer demographics': 'Customer Demographics dimension provides information about the demographic details of customers, including gender, marital status, education status, purchase estimate, credit rating, dependency count, dependents employed count, and dependents college count.', 'date': 'The Date Dimension provides detailed information about dates, including attributes such as day, month, year, and various sequences related to time.', 'household demographics': 'Dimension based on the table tpcds_1.tpcds.household_demographics', 'income bands': 'The Income Bands dimension provides information about different income ranges, with details on the lower and upper bounds of each band.', 'tpcds item': 'The TPCDS Item Dimension is based on the table tpcds_1.tpcds.item and contains information about various attributes related to items such as category, manufacturer, brand, price, and more.', 'promotion data': 'Dimension containing information about different promotion channels and promo SKs', 'reason': 'The Reason Dimension provides information about different reasons associated with transactions or events.', 'shipping modes': 'The Shipping Modes dimension provides details about different modes of shipping available, including unique identifiers for each mode.', 'store': 'The Store Dimension contains information about different stores, including their IDs, names, locations, and company details.', 'time': 'The Time Dimension represents various time-related attributes such as hour, minute, and meal time. It is based on the tpcds_1.tpcds.time_dim table.', 'warehouse details': 'This dimension provides information about warehouses, including their names, sizes, locations, and related details.', 'web page data': 'This dimension is based on the table tpcds_1.tpcds.web_page and contains information about web pages.', 'web site': 'A dimension based on the table tpcds_1.tpcds.web_site, containing information about different websites.'}. The key in the json is the name of the dimension table, and the value is the description. Use the description of the dimension tables to determine which of them are relevant for this user's request. Some of the phrases used in the user's request indicate the dimension tables that are needed. The phrases related to attributes are more important to indicate the dimension table. The dimension tables provide attributes and dimensions important to answer the user's question.Please pick the names of the dimensions tables that most relevant to the user's request from the given list of dimensions. The list of dimension tables are: ['call center', 'catalog page', 'customer', 'customer address', 'customer demographics', 'date', 'household demographics', 'income bands', 'tpcds item', 'promotion data', 'reason', 'shipping modes', 'store', 'time', 'warehouse details', 'web page data', 'web site'], which are same as the keys in the json structure for descriptions. You must distinguish between Store address and customer address as two different types of addresses. customer address is a separate dimension, but the store address implies store dimension. Please output only those dimension tables that you are absolutely sure are needed to answer the query. Ignore the dimension tables that you are doubtful or not too sure about.For the relevant dimension tables, please also extract the phrases and words in the user's request that refer to the dimension table. The extracted phrase and words semantically match the corresponding dimension table picked and indicate why the dimension table is relevant for the user's question."""}],
  )
  print(f'Question : {tpcds_questions[i]}')
  print('Answer : ')
  print(user_info)
  print('\n')

#DSpy

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
turbo = dspy.OpenAI(model='gpt-3.5-turbo',max_tokens=1000,api_key=api_key)
dspy.settings.configure(lm = turbo)
print('\n\n\n\n\n')
print('DSPy : ')


for i in l:
  question_n = tpcds_questions[i]
  print(f'Question : {tpcds_questions[i]}')
  print('Answer : ')
  print(outline(question=question_n))
  print('\n')

